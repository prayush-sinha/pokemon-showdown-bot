#!/usr/bin/env python3
"""
policy_net.py
==================
Phase 6, Step 6C — Set-Transformer Policy Model Architecture.

Consumes the exact tensor schema written by dataset_parser.py (Step 6B) and
predicts what a strong player would do next: switch or move, and which one.

ARCHITECTURE (genuinely Set Transformer, Lee et al. 2019 -- not a generic
transformer wearing a Set-Transformer label):

  1. SlotEncoder embeds each of the 12 pokemon slots (species/item/ability/
     tera/status via lookup tables with padding_idx=0, matching 6B's "0 =
     unrevealed/pad" convention everywhere) plus their continuous features
     (hp/boosts/flags) into one [N,12,D] set of per-slot tokens. This is a
     genuine SET: within a side, bench order is just alphabetical -- an
     artifact of the parser, not real information -- so nothing here should
     depend on slot position except "mine vs theirs" and "active vs bench",
     both of which are explicit features, not positions.

  2. ContextEncoder turns global, non-per-pokemon state (turn count,
     weather, field effects, hazards, screens) into one [N,D] vector, which
     is broadcast-added onto every slot token. This conditions every
     pokemon's representation on the shared battle context without
     pretending the context is itself a 13th "pokemon".

  3. A stack of Set Attention Blocks (SAB = full multi-head self-attention
     + FFN, pre/post-LayerNorm residual -- exactly nn.TransformerEncoder)
     lets every slot attend to every other slot (mine and the opponent's),
     e.g. "is my answer to their Iron Valiant still healthy".

  4. Two different heads read the SAB output two different ways:
       - The pointer/switch head reads the 6 contextualized "mine" tokens
         directly and scores each as a switch target -- this preserves
         per-slot identity, which pooling would destroy.
       - The move/action-type heads read a Pooling-by-Multihead-Attention
         (PMA) summary: a single learnable seed vector attends over all 12
         SAB outputs to produce one permutation-invariant [N,D] battle
         summary. This is the paper's actual pooling mechanism, not a
         BERT-style CLS token or a naive mean.

SCOPE (matches the boundaries documented in dataset_parser.py):
  * move_logits is a flat softmax over the global move vocabulary. It is
    NOT masked to the acting pokemon's actual known moveset/PP/Choice-lock
    -- 6B doesn't track movesets, so this model can't know them either.
    Legal-move masking belongs to Step 6E's search integration, which DOES
    have ground-truth access to the battle engine's legal action list.
  * switch_logits IS masked by default using the batch's own `fainted`
    flags (that ground truth is available and unambiguous), so the model
    never learns to reward switching into a dead slot. A caller-supplied
    `switch_mask` overrides this default for e.g. simulating Trapped state.

This file has NO training loop, optimizer, or data loader -- that is
Step 6D. It defines the nn.Module, a matching loss function, and a
self-contained runnable check that the shapes are consistent end-to-end.

Dependencies: Python 3.9+, `torch`. Must live alongside dataset_parser.py
(imports shared constants from it so the two files can never silently drift
out of sync on vocab/slot-count assumptions).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_parser import MAX_TEAM_SIZE, STATUS_LIST, TERA_TYPES, FIELD_EFFECTS

STATUS_VOCAB_SIZE = len(STATUS_LIST)
TERA_VOCAB_SIZE = len(TERA_TYPES) + 1
NUM_FIELD_EFFECTS = len(FIELD_EFFECTS)
HAZARD_DIM = 4   # hazards_vec() in dataset_parser.py always returns length 4
SCREEN_DIM = 4   # screens_vec() in dataset_parser.py always returns length 4
CONTINUOUS_PER_SLOT_DIM = 4 + len(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"])  # is_active,hp_fraction,fainted,terastallized + 7 boosts

logger = logging.getLogger("policy_net")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PolicyNetConfig:
    species_vocab_size: int
    moves_vocab_size: int
    items_vocab_size: int
    abilities_vocab_size: int
    weathers_vocab_size: int

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

    species_dim: int = 32
    item_dim: int = 16
    ability_dim: int = 16
    tera_dim: int = 8
    status_dim: int = 8
    side_dim: int = 4
    weather_dim: int = 8

    @classmethod
    def from_feature_schema(cls, schema_path, **overrides) -> "PolicyNetConfig":
        """Load vocab sizes straight from the feature_schema.json Step 6B
        writes, so this file never hardcodes a vocab size that could drift
        out of sync with the actual dataset."""
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        vs = schema["vocab_sizes"]
        cfg = cls(
            species_vocab_size=vs["species"],
            moves_vocab_size=vs["moves"],
            items_vocab_size=vs["items"],
            abilities_vocab_size=vs["abilities"],
            weathers_vocab_size=vs["weathers"],
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def default_synthetic(cls, **overrides) -> "PolicyNetConfig":
        """Placeholder vocab sizes for a self-test when no real
        feature_schema.json exists yet (e.g. Step 6B hasn't finished)."""
        cfg = cls(species_vocab_size=100, moves_vocab_size=200, items_vocab_size=50, abilities_vocab_size=40, weathers_vocab_size=6)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Slot + context encoders
# ---------------------------------------------------------------------------

class SlotEncoder(nn.Module):
    """Embeds each of the 12 pokemon slots into a [N,12,D] set of tokens."""

    def __init__(self, cfg: PolicyNetConfig):
        super().__init__()
        self.cfg = cfg
        self.species_emb = nn.Embedding(cfg.species_vocab_size, cfg.species_dim, padding_idx=0)
        self.item_emb = nn.Embedding(cfg.items_vocab_size, cfg.item_dim, padding_idx=0)
        self.ability_emb = nn.Embedding(cfg.abilities_vocab_size, cfg.ability_dim, padding_idx=0)
        self.tera_emb = nn.Embedding(TERA_VOCAB_SIZE, cfg.tera_dim, padding_idx=0)
        self.status_emb = nn.Embedding(STATUS_VOCAB_SIZE, cfg.status_dim, padding_idx=0)
        self.side_emb = nn.Embedding(2, cfg.side_dim)  # 0 = mine, 1 = opponent's -- no reserved pad value

        raw_dim = (
            cfg.species_dim + cfg.item_dim + cfg.ability_dim + cfg.tera_dim
            + cfg.status_dim + cfg.side_dim + CONTINUOUS_PER_SLOT_DIM
        )
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        species_ids = batch["species_ids"]  # [N,12]
        n, s = species_ids.shape
        side_ids = torch.zeros(n, s, dtype=torch.long, device=species_ids.device)
        side_ids[:, MAX_TEAM_SIZE:] = 1  # slots [0:6]=mine, [6:12]=theirs, per dataset_parser's schema

        parts = [
            self.species_emb(species_ids),
            self.item_emb(batch["item_ids"]),
            self.ability_emb(batch["ability_ids"]),
            self.tera_emb(batch["tera_type_ids"]),
            self.status_emb(batch["status_ids"]),
            self.side_emb(side_ids),
            batch["is_active"].unsqueeze(-1),
            batch["hp_fraction"].unsqueeze(-1),
            batch["fainted"].unsqueeze(-1),
            batch["terastallized"].unsqueeze(-1),
            batch["boosts"],  # [N,12,7]
        ]
        raw = torch.cat(parts, dim=-1)  # [N,12,raw_dim]
        return self.proj(raw)


class ContextEncoder(nn.Module):
    """Turns global (non-per-pokemon) battle state into one [N,D] vector."""

    def __init__(self, cfg: PolicyNetConfig):
        super().__init__()
        self.weather_emb = nn.Embedding(cfg.weathers_vocab_size, cfg.weather_dim, padding_idx=0)
        raw_dim = 1 + cfg.weather_dim + NUM_FIELD_EFFECTS + HAZARD_DIM * 2 + SCREEN_DIM * 2
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        parts = [
            batch["turn_norm"].unsqueeze(-1),
            self.weather_emb(batch["weather_id"]),
            batch["field_effects"],
            batch["my_hazards"], batch["opp_hazards"],
            batch["my_screens"], batch["opp_screens"],
        ]
        raw = torch.cat(parts, dim=-1)  # [N,raw_dim]
        return self.proj(raw)  # [N,D]


# ---------------------------------------------------------------------------
# Pooling by Multihead Attention (Set Transformer, Lee et al. 2019)
# ---------------------------------------------------------------------------

class PMA(nn.Module):
    """A learnable seed vector attends over the SAB-processed set to
    produce ONE permutation-invariant summary vector -- the paper's actual
    pooling mechanism, as opposed to mean-pooling or a BERT-style CLS
    token bolted onto the input."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, num_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model) * (d_model ** -0.5))
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N,S,D] SAB output -> [N,num_seeds,D]"""
        n = x.size(0)
        seeds = self.seeds.expand(n, -1, -1)
        attn_out, _ = self.attn(seeds, x, x)
        h = self.ln1(seeds + attn_out)
        return self.ln2(h + self.ff(h))


# ---------------------------------------------------------------------------
# Full policy network
# ---------------------------------------------------------------------------

@dataclass
class PolicyOutput:
    action_type_logits: torch.Tensor  # [N,2]      0=move, 1=switch
    move_logits: torch.Tensor         # [N,moves]
    switch_logits: torch.Tensor       # [N,6]


class FutureSightPolicyNet(nn.Module):
    def __init__(self, cfg: PolicyNetConfig):
        super().__init__()
        self.cfg = cfg
        self.slot_encoder = SlotEncoder(cfg)
        self.context_encoder = ContextEncoder(cfg)

        sab_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.num_heads, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.sab_stack = nn.TransformerEncoder(sab_layer, num_layers=cfg.num_layers)
        self.pma = PMA(cfg.d_model, cfg.num_heads, dropout=cfg.dropout, num_seeds=1)

        self.dropout = nn.Dropout(cfg.dropout)
        self.action_type_head = nn.Linear(cfg.d_model, 2)
        self.move_head = nn.Linear(cfg.d_model, cfg.moves_vocab_size)
        self.switch_head = nn.Linear(cfg.d_model, 1)  # shared across the 6 mine-slots (pointer-style)

    def forward(
        self,
        batch: dict,
        move_mask: Optional[torch.Tensor] = None,
        switch_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        slot_tokens = self.slot_encoder(batch)          # [N,12,D]
        context_vec = self.context_encoder(batch)        # [N,D]
        tokens = slot_tokens + context_vec.unsqueeze(1)   # broadcast-condition every slot on global state

        sab_out = self.sab_stack(tokens)                  # [N,12,D]
        mine_out = sab_out[:, :MAX_TEAM_SIZE, :]           # [N,6,D] -- keep per-slot identity for the pointer head
        pooled = self.pma(sab_out).squeeze(1)              # [N,D]   -- permutation-invariant summary
        pooled = self.dropout(pooled)

        action_type_logits = self.action_type_head(pooled)
        move_logits = self.move_head(pooled)
        switch_logits = self.switch_head(mine_out).squeeze(-1)  # [N,6]

        if move_mask is not None:
            move_logits = move_logits.masked_fill(~move_mask, float("-inf"))

        if switch_mask is not None:
            switch_logits = switch_logits.masked_fill(~switch_mask, float("-inf"))
        else:
            # Safe universal default: never point at an already-fainted slot.
            # This is ground truth we DO have from 6B; full legality
            # (trapping, Choice lock, PP) is Step 6E's job.
            fainted_mine = batch["fainted"][:, :MAX_TEAM_SIZE].bool()
            switch_logits = switch_logits.masked_fill(fainted_mine, float("-inf"))

        return PolicyOutput(action_type_logits, move_logits, switch_logits)


# ---------------------------------------------------------------------------
# Loss + metrics (architecture-adjacent utilities; the actual training loop
# with optimizer/scheduler/checkpointing is Step 6D)
# ---------------------------------------------------------------------------

@dataclass
class LossOutput:
    total: torch.Tensor
    action_type: torch.Tensor
    move: Optional[torch.Tensor]
    switch: Optional[torch.Tensor]


def compute_loss(
    output: PolicyOutput,
    batch: dict,
    action_type_weight: float = 1.0,
    move_weight: float = 1.0,
    switch_weight: float = 1.0,
    action_type_class_weight: Optional[torch.Tensor] = None,
) -> LossOutput:
    """action_type_class_weight lets 6D's training loop counteract the fact
    that real battles have far more moves than switches, without this file
    needing to know anything about class balance itself."""
    action_type_target = batch["action_type"]  # [N]
    loss_type = F.cross_entropy(output.action_type_logits, action_type_target, weight=action_type_class_weight)

    move_mask = action_type_target == 0
    switch_mask = action_type_target == 1

    loss_move = None
    if move_mask.any():
        loss_move = F.cross_entropy(output.move_logits[move_mask], batch["action_move_id"][move_mask])

    loss_switch = None
    if switch_mask.any():
        loss_switch = F.cross_entropy(output.switch_logits[switch_mask], batch["action_switch_slot"][switch_mask])

    total = action_type_weight * loss_type
    if loss_move is not None:
        total = total + move_weight * loss_move
    if loss_switch is not None:
        total = total + switch_weight * loss_switch

    return LossOutput(total=total, action_type=loss_type.detach(),
                       move=loss_move.detach() if loss_move is not None else None,
                       switch=loss_switch.detach() if loss_switch is not None else None)


def compute_accuracy(output: PolicyOutput, batch: dict) -> dict:
    action_type_target = batch["action_type"]
    action_type_pred = output.action_type_logits.argmax(dim=-1)
    acc = {"action_type": (action_type_pred == action_type_target).float().mean().item()}

    move_mask = action_type_target == 0
    if move_mask.any():
        move_pred = output.move_logits[move_mask].argmax(dim=-1)
        acc["move"] = (move_pred == batch["action_move_id"][move_mask]).float().mean().item()

    switch_mask = action_type_target == 1
    if switch_mask.any():
        switch_pred = output.switch_logits[switch_mask].argmax(dim=-1)
        acc["switch"] = (switch_pred == batch["action_switch_slot"][switch_mask]).float().mean().item()

    return acc


def count_parameters(model: nn.Module) -> dict:
    breakdown = {}
    for name, module in model.named_children():
        breakdown[name] = sum(p.numel() for p in module.parameters())
    breakdown["TOTAL"] = sum(p.numel() for p in model.parameters())
    return breakdown


# ---------------------------------------------------------------------------
# Self-test / CLI
# ---------------------------------------------------------------------------

def make_synthetic_batch(cfg: PolicyNetConfig, batch_size: int) -> dict:
    """A random-but-shape-correct batch, for verifying the architecture
    runs end-to-end even before Step 6B has produced real data."""
    n = batch_size
    boost_dim = len(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"])
    action_type = torch.randint(0, 2, (n,))
    action_move_id = torch.where(
        action_type == 0, torch.randint(0, cfg.moves_vocab_size, (n,)), torch.full((n,), -1),
    )
    action_switch_slot = torch.where(
        action_type == 1, torch.randint(0, MAX_TEAM_SIZE, (n,)), torch.full((n,), -1),
    )
    return {
        "species_ids": torch.randint(0, cfg.species_vocab_size, (n, 2 * MAX_TEAM_SIZE)),
        "is_active": torch.zeros(n, 2 * MAX_TEAM_SIZE),
        "hp_fraction": torch.rand(n, 2 * MAX_TEAM_SIZE),
        "fainted": torch.zeros(n, 2 * MAX_TEAM_SIZE),
        "status_ids": torch.randint(0, STATUS_VOCAB_SIZE, (n, 2 * MAX_TEAM_SIZE)),
        "item_ids": torch.randint(0, cfg.items_vocab_size, (n, 2 * MAX_TEAM_SIZE)),
        "ability_ids": torch.randint(0, cfg.abilities_vocab_size, (n, 2 * MAX_TEAM_SIZE)),
        "terastallized": torch.zeros(n, 2 * MAX_TEAM_SIZE),
        "tera_type_ids": torch.randint(0, TERA_VOCAB_SIZE, (n, 2 * MAX_TEAM_SIZE)),
        "boosts": (torch.rand(n, 2 * MAX_TEAM_SIZE, boost_dim) * 2 - 1),
        "turn_norm": torch.rand(n),
        "weather_id": torch.randint(0, cfg.weathers_vocab_size, (n,)),
        "field_effects": torch.randint(0, 2, (n, NUM_FIELD_EFFECTS)).float(),
        "my_hazards": torch.rand(n, HAZARD_DIM),
        "opp_hazards": torch.rand(n, HAZARD_DIM),
        "my_screens": torch.randint(0, 2, (n, SCREEN_DIM)).float(),
        "opp_screens": torch.randint(0, 2, (n, SCREEN_DIM)).float(),
        "action_type": action_type,
        "action_move_id": action_move_id,
        "action_switch_slot": action_switch_slot,
    }


def load_real_batch(dataset_path: Path, batch_size: int) -> Optional[dict]:
    dataset = torch.load(dataset_path, weights_only=False)
    train = dataset.get("train")
    if not train:
        return None
    n = min(batch_size, train["species_ids"].shape[0])
    return {k: v[:n] for k, v in train.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 6C: Set-Transformer policy network self-test.")
    parser.add_argument("--feature-schema", default="data/feature_schema_gen9ou.json",
                         help="Path written by dataset_parser.py; used to size the embedding tables (default: %(default)s)")
    parser.add_argument("--dataset", default="data/dataset_gen9ou.pt",
                         help="Path written by dataset_parser.py; if present, run a real forward pass on it (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the self-test (default: %(default)s)")
    parser.add_argument("--d-model", type=int, default=None, help="Override d_model")
    parser.add_argument("--num-heads", type=int, default=None, help="Override num_heads")
    parser.add_argument("--num-layers", type=int, default=None, help="Override num_layers")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    overrides = {k: v for k, v in (("d_model", args.d_model), ("num_heads", args.num_heads),
                                    ("num_layers", args.num_layers)) if v is not None}

    schema_path = Path(args.feature_schema)
    if schema_path.exists():
        cfg = PolicyNetConfig.from_feature_schema(schema_path, **overrides)
        logger.info("Loaded vocab sizes from %s", schema_path)
    else:
        cfg = PolicyNetConfig.default_synthetic(**overrides)
        logger.warning("%s not found -- using placeholder vocab sizes for a synthetic self-test.", schema_path)

    logger.info("Config: %s", asdict(cfg))
    model = FutureSightPolicyNet(cfg)
    model.eval()

    params = count_parameters(model)
    for name, count in params.items():
        logger.info("Params[%s] = %s", name, f"{count:,}")

    dataset_path = Path(args.dataset)
    batch = None
    if dataset_path.exists():
        batch = load_real_batch(dataset_path, args.batch_size)
        if batch is not None:
            logger.info("Loaded a real batch of %d samples from %s", batch["species_ids"].shape[0], dataset_path)
    if batch is None:
        batch = make_synthetic_batch(cfg, args.batch_size)
        logger.info("Using a synthetic random batch of %d samples (no usable real dataset found).", args.batch_size)

    with torch.no_grad():
        output = model(batch)
        loss = compute_loss(output, batch)
        acc = compute_accuracy(output, batch)

    logger.info("Forward pass shapes: action_type_logits=%s move_logits=%s switch_logits=%s",
                tuple(output.action_type_logits.shape), tuple(output.move_logits.shape), tuple(output.switch_logits.shape))
    logger.info("Loss (untrained, random-init weights -- expected to be high): total=%.4f action_type=%.4f move=%s switch=%s",
                loss.total.item(), loss.action_type.item(),
                f"{loss.move.item():.4f}" if loss.move is not None else "n/a",
                f"{loss.switch.item():.4f}" if loss.switch is not None else "n/a")
    logger.info("Accuracy (untrained, expect ~chance level): %s", {k: round(v, 3) for k, v in acc.items()})
    logger.info("Self-test OK: architecture runs end-to-end with %s vocab sizes.",
                "real" if dataset_path.exists() else "synthetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
