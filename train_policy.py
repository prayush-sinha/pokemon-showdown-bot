#!/usr/bin/env python3
"""
train_policy.py
================
Phase 6, Step 6D -- Training loop for the Set-Transformer policy network
defined in policy_net.py (Step 6C).

policy_net.py deliberately has no optimizer, data loader, or checkpointing
(see its module docstring) -- that is this file's job. train_policy.py:

  1. Loads the dataset produced by dataset_parser.py (Step 6B) from
     data/dataset_gen9ou.pt, and the vocab-size schema from
     data/feature_schema_gen9ou.json.
  2. Builds FutureSightPolicyNet via PolicyNetConfig.from_feature_schema,
     so embedding table sizes can never drift out of sync with the dataset.
  3. Trains with AdamW + ReduceLROnPlateau + early stopping on validation
     loss, using the exact compute_loss/compute_accuracy functions from
     policy_net.py so training-time metrics match Step 6C's self-test.
  4. Checkpoints the best (lowest val loss) weights to data/policy_net.pth,
     and can optionally export that checkpoint to data/policy_net.onnx.

DATASET SHAPE ASSUMPTIONS (matches policy_net.py's make_synthetic_batch /
load_real_batch and SlotEncoder/ContextEncoder field lists exactly):
  torch.load(dataset_path) is expected to be one of:
    - {"train": {...tensors...}, "val": {...tensors...}}   (pre-split)
    - {"train": {...tensors...}}                             (we split it)
    - {...tensors...}                                        (flat, we split it)
  Each per-split dict must contain every key SlotEncoder/ContextEncoder/
  compute_loss reference: species_ids, item_ids, ability_ids, tera_type_ids,
  status_ids, is_active, hp_fraction, fainted, terastallized, boosts,
  turn_norm, weather_id, field_effects, my_hazards, opp_hazards,
  my_screens, opp_screens, action_type, action_move_id, action_switch_slot.

USAGE (identical locally or in Colab -- just prefix with `!` in a cell):
    python3 train_policy.py --epochs 100 --batch-size 32
    python3 train_policy.py --epochs 100 --batch-size 64 --export-onnx

Must live alongside dataset_parser.py and policy_net.py (imports the latter,
which in turn imports the former).

Dependencies: Python 3.9+, `torch`. `onnx`/`onnxruntime` only needed if
--export-onnx is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from policy_net import (
    FutureSightPolicyNet,
    PolicyNetConfig,
    compute_accuracy,
    compute_loss,
)

logger = logging.getLogger("train_policy")

# Every tensor key FutureSightPolicyNet's forward pass consumes, in a fixed
# order -- used both for flat-dict detection and for building the ONNX
# wrapper's positional signature. Deliberately excludes the three label
# keys (action_type, action_move_id, action_switch_slot), which are
# training targets, not model inputs.
MODEL_INPUT_KEYS = [
    "species_ids", "item_ids", "ability_ids", "tera_type_ids", "status_ids",
    "is_active", "hp_fraction", "fainted", "terastallized", "boosts",
    "turn_norm", "weather_id", "field_effects",
    "my_hazards", "opp_hazards", "my_screens", "opp_screens",
]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset loading / splitting / batching
# ---------------------------------------------------------------------------

def load_dataset(path: Path):
    """Returns (train_raw, val_raw_or_None). See module docstring for the
    three dataset shapes this handles."""
    raw = torch.load(path, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected dataset format in {path}: expected a dict, got {type(raw)}")

    if isinstance(raw.get("train"), dict) and isinstance(raw.get("val"), dict):
        logger.info("Dataset has pre-split train/val sets -- using them as-is.")
        return raw["train"], raw["val"]
    if isinstance(raw.get("train"), dict):
        logger.info("Dataset has a 'train' set but no 'val' set -- splitting train ourselves.")
        return raw["train"], None
    if "species_ids" in raw:
        logger.info("Dataset is a flat tensor dict -- splitting into train/val ourselves.")
        return raw, None
    raise ValueError(f"Could not recognize dataset structure in {path}. Top-level keys: {list(raw.keys())}")


def split_train_val(flat: dict, val_split: float, seed: int):
    n = flat["action_type"].shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_split))) if n > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_data = {k: v[train_idx] for k, v in flat.items()}
    val_data = {k: v[val_idx] for k, v in flat.items()}
    return train_data, val_data


def iterate_batches(data: dict, batch_size: int, shuffle: bool, generator: Optional[torch.Generator] = None):
    n = data["action_type"].shape[0]
    device = data["action_type"].device
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(device)
    else:
        idx = torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        b_idx = idx[start:start + batch_size]
        yield {k: v[b_idx] for k, v in data.items()}


def compute_action_type_class_weight(train_data: dict, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weight for {move, switch}, so training doesn't
    just learn to always predict 'move' because it's the majority class."""
    counts = torch.bincount(train_data["action_type"], minlength=2).float().clamp(min=1.0)
    weight = counts.sum() / (2.0 * counts)
    return weight.to(device)


def weighted_mean(pairs: list) -> float:
    total_weight = sum(w for _, w in pairs)
    if total_weight == 0:
        return float("nan")
    return sum(v * w for v, w in pairs) / total_weight


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    data: dict,
    batch_size: int,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    action_type_class_weight: Optional[torch.Tensor] = None,
    loss_weights: tuple = (1.0, 1.0, 1.0),
    generator: Optional[torch.Generator] = None,
) -> dict:
    model.train(mode=train)
    loss_total_p, loss_type_p, loss_move_p, loss_switch_p = [], [], [], []
    acc_type_p, acc_move_p, acc_switch_p = [], [], []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in iterate_batches(data, batch_size, shuffle=train, generator=generator):
            bsz = batch["action_type"].shape[0]
            output = model(batch)
            loss = compute_loss(
                output, batch,
                action_type_weight=loss_weights[0],
                move_weight=loss_weights[1],
                switch_weight=loss_weights[2],
                action_type_class_weight=action_type_class_weight,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                optimizer.step()

            acc = compute_accuracy(output, batch)
            action_type_target = batch["action_type"]
            move_count = int((action_type_target == 0).sum().item())
            switch_count = int((action_type_target == 1).sum().item())

            loss_total_p.append((loss.total.item(), bsz))
            loss_type_p.append((loss.action_type.item(), bsz))
            acc_type_p.append((acc["action_type"], bsz))
            if loss.move is not None:
                loss_move_p.append((loss.move.item(), move_count))
                acc_move_p.append((acc["move"], move_count))
            if loss.switch is not None:
                loss_switch_p.append((loss.switch.item(), switch_count))
                acc_switch_p.append((acc["switch"], switch_count))

    return {
        "loss_total": weighted_mean(loss_total_p),
        "loss_action_type": weighted_mean(loss_type_p),
        "loss_move": weighted_mean(loss_move_p) if loss_move_p else float("nan"),
        "loss_switch": weighted_mean(loss_switch_p) if loss_switch_p else float("nan"),
        "acc_action_type": weighted_mean(acc_type_p),
        "acc_move": weighted_mean(acc_move_p) if acc_move_p else float("nan"),
        "acc_switch": weighted_mean(acc_switch_p) if acc_switch_p else float("nan"),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: nn.Module, cfg: PolicyNetConfig, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# ONNX export (best-effort; checkpoint is always saved regardless)
# ---------------------------------------------------------------------------

class _ONNXWrapper(nn.Module):
    """FutureSightPolicyNet.forward takes a dict, which torch.onnx.export
    can't trace directly. This wrapper exposes the same computation as a
    fixed positional signature over MODEL_INPUT_KEYS instead."""

    def __init__(self, model: FutureSightPolicyNet):
        super().__init__()
        self.model = model

    def forward(self, *tensors):
        batch = dict(zip(MODEL_INPUT_KEYS, tensors))
        out = self.model(batch)  # unmasked move logits, fainted-masked switch logits -- matches policy_net.py's documented scope
        return out.action_type_logits, out.move_logits, out.switch_logits


def export_onnx(model: FutureSightPolicyNet, sample_batch: dict, output_path: Path, opset: int = 17) -> None:
    wrapper = _ONNXWrapper(model).eval()
    args = tuple(sample_batch[k] for k in MODEL_INPUT_KEYS)
    dynamic_axes = {k: {0: "batch"} for k in MODEL_INPUT_KEYS}
    dynamic_axes.update({
        "action_type_logits": {0: "batch"},
        "move_logits": {0: "batch"},
        "switch_logits": {0: "batch"},
    })
    export_kwargs = dict(
        input_names=MODEL_INPUT_KEYS,
        output_names=["action_type_logits", "move_logits", "switch_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    try:
        # Newer torch (>=2.5) defaults to the dynamo-based exporter, which
        # needs the optional `onnxscript` package. Force the older
        # TorchScript-based tracer instead so --export-onnx works without
        # extra dependencies; older torch versions don't have this kwarg
        # at all, so fall back to the plain call for those.
        torch.onnx.export(wrapper, args, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, args, str(output_path), **export_kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 6D: train the Set-Transformer policy network.")
    p.add_argument("--dataset", default="data/dataset_gen9ou.pt", help="Path written by dataset_parser.py (default: %(default)s)")
    p.add_argument("--feature-schema", default="data/feature_schema_gen9ou.json", help="Path written by dataset_parser.py (default: %(default)s)")
    p.add_argument("--output", default="data/policy_net.pth", help="Where to save the best checkpoint (default: %(default)s)")
    p.add_argument("--export-onnx", action="store_true", help="Also export the best checkpoint to ONNX after training")
    p.add_argument("--onnx-output", default="data/policy_net.onnx", help="ONNX export path (default: %(default)s)")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--val-split", type=float, default=0.15, help="Only used if the dataset has no pre-made val set")
    p.add_argument("--patience", type=int, default=15, help="Early-stopping patience, in epochs with no val-loss improvement")
    p.add_argument("--lr-patience", type=int, default=5, help="ReduceLROnPlateau patience, in epochs")
    p.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau LR multiplier on plateau")
    p.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val-loss improvement to reset early-stopping patience")

    p.add_argument("--action-type-weight", type=float, default=1.0)
    p.add_argument("--move-weight", type=float, default=1.0)
    p.add_argument("--switch-weight", type=float, default=1.0)
    p.add_argument("--disable-class-weight", action="store_true", help="Disable automatic move/switch class balancing")

    p.add_argument("--d-model", type=int, default=None, help="Override d_model")
    p.add_argument("--num-heads", type=int, default=None, help="Override num_heads")
    p.add_argument("--num-layers", type=int, default=None, help="Override num_layers")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'; default auto-detects")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    set_seed(args.seed)

    device = resolve_device(args.device)
    logger.info("Using device: %s%s", device, " (GPU)" if device.type == "cuda" else " (CPU -- training will be slower)")

    schema_path = Path(args.feature_schema)
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}. Run dataset_parser.py first.")
    overrides = {k: v for k, v in (
        ("d_model", args.d_model), ("num_heads", args.num_heads), ("num_layers", args.num_layers),
    ) if v is not None}
    cfg = PolicyNetConfig.from_feature_schema(schema_path, **overrides)
    logger.info("Config: %s", asdict(cfg))

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run dataset_parser.py first.")
    train_raw, val_raw = load_dataset(dataset_path)
    if val_raw is None:
        train_data, val_data = split_train_val(train_raw, args.val_split, args.seed)
    else:
        train_data, val_data = train_raw, val_raw
    train_data = {k: v.to(device) for k, v in train_data.items()}
    val_data = {k: v.to(device) for k, v in val_data.items()}
    logger.info("Train samples: %d | Val samples: %d", train_data["action_type"].shape[0], val_data["action_type"].shape[0])

    model = FutureSightPolicyNet(cfg).to(device)
    logger.info("Total params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    action_type_class_weight = None
    if not args.disable_class_weight:
        action_type_class_weight = compute_action_type_class_weight(train_data, device)
        logger.info("action_type class weight [move, switch]: %s", [round(w, 3) for w in action_type_class_weight.tolist()])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-6,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    batch_gen = torch.Generator().manual_seed(args.seed)
    loss_weights = (args.action_type_weight, args.move_weight, args.switch_weight)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_m = run_epoch(model, train_data, args.batch_size, train=True, optimizer=optimizer,
                             action_type_class_weight=action_type_class_weight, loss_weights=loss_weights,
                             generator=batch_gen)
        val_m = run_epoch(model, val_data, args.batch_size, train=False,
                           action_type_class_weight=action_type_class_weight, loss_weights=loss_weights)
        scheduler.step(val_m["loss_total"])
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | lr=%.2e | train: loss=%.4f (type=%.4f move=%.4f switch=%.4f) acc[type=%.3f move=%.3f switch=%.3f] "
            "| val: loss=%.4f (type=%.4f move=%.4f switch=%.4f) acc[type=%.3f move=%.3f switch=%.3f] | %.1fs",
            epoch, args.epochs, lr_now,
            train_m["loss_total"], train_m["loss_action_type"], train_m["loss_move"], train_m["loss_switch"],
            train_m["acc_action_type"], train_m["acc_move"], train_m["acc_switch"],
            val_m["loss_total"], val_m["loss_action_type"], val_m["loss_move"], val_m["loss_switch"],
            val_m["acc_action_type"], val_m["acc_move"], val_m["acc_switch"],
            epoch_time,
        )

        improved = val_m["loss_total"] < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_m["loss_total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_path, model, cfg, epoch, val_m, args)
            logger.info("  -> new best val loss %.4f, checkpoint saved to %s", best_val_loss, output_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping: no val-loss improvement for %d epochs.", args.patience)
                break

    total_time = time.time() - start_time
    logger.info("Training finished in %.1fs. Best val loss=%.4f at epoch %d. Checkpoint: %s",
                total_time, best_val_loss, best_epoch, output_path)

    if args.export_onnx:
        if best_epoch == -1:
            logger.warning("No checkpoint was ever saved (best_epoch=-1) -- skipping ONNX export.")
            return 0
        logger.info("Reloading best checkpoint before ONNX export...")
        best_ckpt = torch.load(output_path, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        sample = {k: v[: min(2, v.shape[0])] for k, v in val_data.items()}
        onnx_path = Path(args.onnx_output)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            export_onnx(model, sample, onnx_path)
            logger.info("ONNX model exported to %s", onnx_path)
        except Exception as exc:
            logger.warning("ONNX export failed (checkpoint at %s is still saved and usable): %s", output_path, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
