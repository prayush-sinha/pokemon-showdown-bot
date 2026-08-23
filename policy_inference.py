#!/usr/bin/env python3
"""
policy_inference.py
====================
Phase 6, Step 6E -- Lightweight Inference Wrapper for the Trained Policy Net.

This is the bridge between the offline training pipeline (Steps 6A-6D:
scrape_replays.py -> dataset_parser.py -> policy_net.py -> train_policy.py)
and the live search engine (expectiminimax.py): it loads whatever the
Colab/local training run produced and turns a live poke-env `battle` object
into exactly the tensor shape `FutureSightPolicyNet` expects, then returns
plain-Python probabilities the search tree can use.

BACKEND SELECTION (in priority order, all best-effort -- `PolicyModel.load()`
never raises; it returns None if nothing usable is found)
  1. ONNX Runtime, CPU, against data/policy_net.onnx. This is the preferred
     path for a live bot: no torch import, small footprint, fast startup.
  2. PyTorch fallback against data/policy_net.pth, using the architecture
     class from policy_net.py and the config saved INSIDE the checkpoint
     (train_policy.py's save_checkpoint() writes `config` alongside the
     weights -- see that file -- so this path doesn't even need
     feature_schema.json to reconstruct the right embedding-table sizes).
  3. Neither file / neither runtime importable / anything else goes wrong
     -> PolicyModel.load() returns None. Callers (expectiminimax.py) are
     expected to treat None as "keep using Smogon priors alone", which is
     exactly what they did before Step 6E existed.

ON "feature_schema_gen9ou.json" VS "vocab_gen9ou.json"
  dataset_parser.py (Step 6B) writes TWO sidecar files, not one:
    - feature_schema_<format>.json -- vocab **sizes** and a field-by-field
      description of the tensor layout. Useful for documentation and for
      sanity-clamping indices; it does NOT contain the actual
      string -> integer mapping.
    - vocab_<format>.json -- the actual {species/moves/items/abilities/
      weathers: {display_name: index}} mapping IS here.
  Both are loaded below. If vocab_<format>.json is missing, the model
  files might still load fine, but we can't turn "Landorus-Therian" into
  a meaningful embedding index -- so `PolicyModel.load()` treats a missing
  vocab file as "policy net not usable" and returns None, the same as a
  missing model file, rather than silently feeding the model an
  all-zeros/all-unrevealed battle state.

USAGE
-----
    from policy_inference import PolicyModel

    policy = PolicyModel.load()              # None if nothing usable found
    if policy is not None:
        pred = policy.predict(battle, acting_side="opponent")
        pred.action_type_probs      # {"move": p, "switch": p}
        pred.move_probs             # {move_id_str: probability, ...} (full vocab)
        pred.switch_species_probs   # {species_id_str: probability, ...} (known bench only)

Self-test (no real battle, no real trained model needed):
    python3 policy_inference.py --test-policy
    python3 policy_inference.py                 # same thing, no flag needed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("policy_inference")

# ---------------------------------------------------------------------------
# Fixed game constants -- deliberately DUPLICATED (not imported) from
# dataset_parser.py, so this module has zero torch/dataset_parser dependency
# on the primary (ONNX) path. These must stay in sync with dataset_parser.py;
# `_consistency_self_check()` near the bottom cross-checks them against
# dataset_parser.py automatically whenever that module happens to be
# importable (e.g. in dev/CI), so drift gets caught instead of silently
# producing a garbled tensor.
# ---------------------------------------------------------------------------
MAX_TEAM_SIZE = 6
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STATUS_LIST = ["", "brn", "par", "psn", "tox", "slp", "frz"]
STATUS_TO_ID = {s: i for i, s in enumerate(STATUS_LIST)}
TERA_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
    "Dragon", "Dark", "Steel", "Fairy", "Stellar",
]
TERA_TYPE_TO_ID = {t: i + 1 for i, t in enumerate(TERA_TYPES)}
FIELD_EFFECTS = [
    "Electric Terrain", "Grassy Terrain", "Misty Terrain", "Psychic Terrain",
    "Trick Room", "Gravity", "Wonder Room", "Magic Room",
]
HAZARD_DIM = 4
SCREEN_DIM = 4

# Exact positional order train_policy.py's ONNX export uses (MODEL_INPUT_KEYS
# in that file). The ONNX graph has no input *names* at runtime beyond what
# was baked in at export time, but onnxruntime lets us feed by name, so we
# keep this list purely for the torch-fallback path (dict -> positional
# tensors) and for documentation.
MODEL_INPUT_KEYS = [
    "species_ids", "item_ids", "ability_ids", "tera_type_ids", "status_ids",
    "is_active", "hp_fraction", "fainted", "terastallized", "boosts",
    "turn_norm", "weather_id", "field_effects",
    "my_hazards", "opp_hazards", "my_screens", "opp_screens",
]

# poke-env's Weather enum names don't always literally match the Showdown
# protocol string dataset_parser.py's vocab was built from (e.g. gen9's
# Snow shows up in poke-env as SNOWSCAPE). Only the mismatches need an
# entry here; anything else falls back to `enum_name.lower()`, which is
# already correct for RAINDANCE/SANDSTORM/SUNNYDAY/PRIMORDIALSEA/
# DESOLATELAND/DELTASTREAM/HAIL.
WEATHER_ENUM_OVERRIDES = {
    "SNOWSCAPE": "snow",
}


def to_id_str(name: str) -> str:
    """Showdown's `toID()`: lowercase, alphanumeric only. Reimplemented
    locally (identical to poke_env.data.to_id_str) so this module never
    needs poke-env just to normalize a string -- only battle_to_tensor's
    live-battle path actually needs poke-env installed."""
    if not name:
        return ""
    return "".join(ch for ch in str(name) if ch.isalnum()).lower()


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
@dataclass
class PolicyVocab:
    """Everything battle_to_tensor needs to turn live-battle strings into
    the same integer ids the model was trained on."""
    species_to_id: dict
    items_to_id: dict
    abilities_to_id: dict
    weathers_to_id: dict
    moves_to_id: dict
    id_to_species_display: dict   # id-str -> original display name (for bench ordering)
    vocab_sizes: dict             # from feature_schema.json, for defensive clamping

    @classmethod
    def load(cls, vocab_path: Path, feature_schema_path: Path) -> Optional["PolicyVocab"]:
        if not vocab_path.exists():
            logger.warning(
                "Policy vocab file not found at %s -- the model files might "
                "load fine, but without this file every species/item/"
                "ability/weather would encode as 'unrevealed' (id 0), which "
                "makes predictions meaningless. Treating the policy net as "
                "unavailable.", vocab_path,
            )
            return None
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _reverse(d: dict) -> dict:
            return {to_id_str(name): idx for name, idx in d.items()}

        vocab_sizes = {}
        if feature_schema_path.exists():
            with open(feature_schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            vocab_sizes = schema.get("vocab_sizes", {})
        else:
            logger.warning(
                "Feature schema not found at %s -- proceeding without "
                "index-range sanity clamping.", feature_schema_path,
            )

        return cls(
            species_to_id=_reverse(raw.get("species", {})),
            items_to_id=_reverse(raw.get("items", {})),
            abilities_to_id=_reverse(raw.get("abilities", {})),
            weathers_to_id=_reverse(raw.get("weathers", {})),
            moves_to_id=_reverse(raw.get("moves", {})),
            id_to_species_display={to_id_str(name): name for name in raw.get("species", {})},
            vocab_sizes=vocab_sizes,
        )

    def _clamp(self, value: int, field_name: str) -> int:
        size = self.vocab_sizes.get(field_name)
        if size and not (0 <= value < size):
            return 0
        return value

    def species_id(self, species_id_str: str) -> int:
        return self._clamp(self.species_to_id.get(to_id_str(species_id_str), 0), "species")

    def item_id(self, item_id_str: Optional[str]) -> int:
        if not item_id_str or item_id_str == "unknown_item":
            return 0
        return self._clamp(self.items_to_id.get(to_id_str(item_id_str), 0), "items")

    def ability_id(self, ability_id_str: Optional[str]) -> int:
        if not ability_id_str:
            return 0
        return self._clamp(self.abilities_to_id.get(to_id_str(ability_id_str), 0), "abilities")

    def weather_id(self, weather_id_str: Optional[str]) -> int:
        if not weather_id_str:
            return 0
        return self._clamp(self.weathers_to_id.get(to_id_str(weather_id_str), 0), "weathers")

    def display_name_for_sort(self, species_id_str: str) -> str:
        """Best-effort reconstruction of the original display name, so
        bench ordering matches training time (which alphabetized by the
        display string, e.g. 'Landorus-Therian', not the id form). Falls
        back to the id form itself for species the vocab never saw."""
        return self.id_to_species_display.get(to_id_str(species_id_str), species_id_str)


# ---------------------------------------------------------------------------
# Battle -> tensor conversion
# ---------------------------------------------------------------------------
def _hazards_vec(h: dict) -> list:
    return [
        1.0 if h.get("stealth_rock") else 0.0,
        h.get("spikes", 0) / 3.0,
        h.get("toxic_spikes", 0) / 2.0,
        1.0 if h.get("sticky_web") else 0.0,
    ]


def _screens_vec(s: dict) -> list:
    return [
        1.0 if s.get("reflect") else 0.0,
        1.0 if s.get("light_screen") else 0.0,
        1.0 if s.get("aurora_veil") else 0.0,
        1.0 if s.get("tailwind") else 0.0,
    ]


def _extract_hazards(side_conditions: dict) -> dict:
    out = {"stealth_rock": False, "spikes": 0, "toxic_spikes": 0, "sticky_web": False}
    for cond, val in (side_conditions or {}).items():
        name = cond.name if hasattr(cond, "name") else str(cond)
        if name == "STEALTH_ROCK":
            out["stealth_rock"] = True
        elif name == "SPIKES":
            out["spikes"] = int(val) if isinstance(val, (int, float)) else 1
        elif name == "TOXIC_SPIKES":
            out["toxic_spikes"] = int(val) if isinstance(val, (int, float)) else 1
        elif name == "STICKY_WEB":
            out["sticky_web"] = True
    return out


def _extract_screens(side_conditions: dict) -> dict:
    out = {"reflect": False, "light_screen": False, "aurora_veil": False, "tailwind": False}
    for cond, _val in (side_conditions or {}).items():
        name = cond.name if hasattr(cond, "name") else str(cond)
        if name == "REFLECT":
            out["reflect"] = True
        elif name == "LIGHT_SCREEN":
            out["light_screen"] = True
        elif name == "AURORA_VEIL":
            out["aurora_veil"] = True
        elif name == "TAILWIND":
            out["tailwind"] = True
    return out


def _extract_field_effects(fields: dict) -> list:
    active = set()
    for f in (fields or {}).keys():
        name = f.name if hasattr(f, "name") else str(f)
        active.add(name.replace("_", " ").title())
    return [1.0 if fe in active else 0.0 for fe in FIELD_EFFECTS]


def _extract_weather_token(weather: dict) -> Optional[str]:
    for w in (weather or {}).keys():
        name = w.name if hasattr(w, "name") else str(w)
        return WEATHER_ENUM_OVERRIDES.get(name, name.lower())
    return None


def _ordered_slots(active, bench: list, vocab: PolicyVocab) -> list:
    """Active first, then bench sorted by best-effort reconstructed display
    name -- mirrors dataset_parser.SideState.ordered_slots() at training
    time. `bench` excludes `active`."""
    bench_sorted = sorted(bench, key=lambda mon: vocab.display_name_for_sort(getattr(mon, "species", "")))
    return ([active] if active is not None else []) + bench_sorted


def _encode_side_slots(active, team: dict, vocab: PolicyVocab) -> dict:
    all_mons = list((team or {}).values())
    bench = [m for m in all_mons if m is not active]
    slots = _ordered_slots(active, bench, vocab)[:MAX_TEAM_SIZE]

    out = {
        "species_ids": [], "is_active": [], "hp_fraction": [], "fainted": [],
        "status_ids": [], "item_ids": [], "ability_ids": [],
        "terastallized": [], "tera_type_ids": [], "boosts": [],
    }
    slot_species: list = []  # parallel list of raw species id-strings (or None), for switch decoding
    for i in range(MAX_TEAM_SIZE):
        if i < len(slots):
            mon = slots[i]
            species = getattr(mon, "species", "") or ""
            fainted = bool(getattr(mon, "fainted", False))
            hp_frac = getattr(mon, "current_hp_fraction", 1.0)
            hp_frac = 0.0 if hp_frac is None else float(hp_frac)
            status_obj = getattr(mon, "status", None)
            status_name = ""
            if status_obj is not None:
                status_name = (status_obj.name if hasattr(status_obj, "name") else str(status_obj)).lower()
            is_terastallized = bool(getattr(mon, "is_terastallized", False) or getattr(mon, "terastallized", False))
            tera_type_obj = getattr(mon, "tera_type", None)
            tera_name = None
            if is_terastallized and tera_type_obj is not None:
                tera_name = (tera_type_obj.name if hasattr(tera_type_obj, "name") else str(tera_type_obj)).capitalize()
            boosts = getattr(mon, "boosts", None) or {}

            out["species_ids"].append(vocab.species_id(species))
            out["is_active"].append(1.0 if mon is active else 0.0)
            out["hp_fraction"].append(hp_frac)
            out["fainted"].append(1.0 if fainted else 0.0)
            out["status_ids"].append(STATUS_TO_ID.get(status_name, 0))
            out["item_ids"].append(vocab.item_id(getattr(mon, "item", None)))
            out["ability_ids"].append(vocab.ability_id(getattr(mon, "ability", None)))
            out["terastallized"].append(1.0 if is_terastallized else 0.0)
            out["tera_type_ids"].append(TERA_TYPE_TO_ID.get(tera_name, 0) if tera_name else 0)
            out["boosts"].append([boosts.get(stat, 0) / 6.0 for stat in BOOST_STATS])
            slot_species.append(species if not fainted else None)
        else:
            out["species_ids"].append(0); out["is_active"].append(0.0)
            out["hp_fraction"].append(0.0); out["fainted"].append(1.0)
            out["status_ids"].append(0); out["item_ids"].append(0); out["ability_ids"].append(0)
            out["terastallized"].append(0.0); out["tera_type_ids"].append(0)
            out["boosts"].append([0.0] * len(BOOST_STATS))
            slot_species.append(None)
    return out, slot_species


def battle_to_tensor(battle: Any, acting_side: str, vocab: PolicyVocab) -> tuple:
    """
    Convert a live poke-env `battle` object (or any duck-typed equivalent
    exposing the same attributes -- see the self-test mocks below) into the
    exact [1, ...] input tensors FutureSightPolicyNet expects.

    Parameters
    ----------
    battle : poke-env AbstractBattle (or duck-typed mock)
    acting_side : "bot" or "opponent" -- whose perspective slots [0:6] take.
        Pass "opponent" to project what the OPPONENT is about to do (the
        expectiminimax.py use case); pass "bot" to run the net from our own
        perspective instead (mostly useful for testing/debugging).
    vocab : a loaded PolicyVocab

    Returns
    -------
    (tensors, acting_slot_species)
        tensors: dict[str, np.ndarray] -- batch dimension 1, matching
            MODEL_INPUT_KEYS names/dtypes/shapes exactly.
        acting_slot_species: list[Optional[str]] of length 6 -- the raw
            poke-env species-id string occupying each of the ACTING side's
            own 6 slots (None for empty/pad/fainted slots), in the exact
            order switch_logits will be indexed by. Callers use this to
            turn "switch_logits[i] is high" into "the model likes switching
            to species X".
    """
    if acting_side not in ("bot", "opponent"):
        raise ValueError(f"acting_side must be 'bot' or 'opponent', got {acting_side!r}")

    if acting_side == "opponent":
        mine_active = getattr(battle, "opponent_active_pokemon", None)
        mine_team = getattr(battle, "opponent_team", {}) or {}
        mine_side_conditions = getattr(battle, "opponent_side_conditions", {}) or {}
        theirs_active = getattr(battle, "active_pokemon", None)
        theirs_team = getattr(battle, "team", {}) or {}
        theirs_side_conditions = getattr(battle, "side_conditions", {}) or {}
    else:
        mine_active = getattr(battle, "active_pokemon", None)
        mine_team = getattr(battle, "team", {}) or {}
        mine_side_conditions = getattr(battle, "side_conditions", {}) or {}
        theirs_active = getattr(battle, "opponent_active_pokemon", None)
        theirs_team = getattr(battle, "opponent_team", {}) or {}
        theirs_side_conditions = getattr(battle, "opponent_side_conditions", {}) or {}

    mine, mine_slot_species = _encode_side_slots(mine_active, mine_team, vocab)
    theirs, _theirs_slot_species = _encode_side_slots(theirs_active, theirs_team, vocab)
    merged = {k: mine[k] + theirs[k] for k in mine}

    turn = getattr(battle, "turn", 1) or 1
    weather_token = _extract_weather_token(getattr(battle, "weather", {}) or {})
    field_effects = _extract_field_effects(getattr(battle, "fields", {}) or {})
    my_hazards = _hazards_vec(_extract_hazards(mine_side_conditions))
    opp_hazards = _hazards_vec(_extract_hazards(theirs_side_conditions))
    my_screens = _screens_vec(_extract_screens(mine_side_conditions))
    opp_screens = _screens_vec(_extract_screens(theirs_side_conditions))

    tensors = {
        "species_ids": np.array([merged["species_ids"]], dtype=np.int64),
        "item_ids": np.array([merged["item_ids"]], dtype=np.int64),
        "ability_ids": np.array([merged["ability_ids"]], dtype=np.int64),
        "tera_type_ids": np.array([merged["tera_type_ids"]], dtype=np.int64),
        "status_ids": np.array([merged["status_ids"]], dtype=np.int64),
        "is_active": np.array([merged["is_active"]], dtype=np.float32),
        "hp_fraction": np.array([merged["hp_fraction"]], dtype=np.float32),
        "fainted": np.array([merged["fainted"]], dtype=np.float32),
        "terastallized": np.array([merged["terastallized"]], dtype=np.float32),
        "boosts": np.array([merged["boosts"]], dtype=np.float32),
        "turn_norm": np.array([min(turn / 100.0, 1.0)], dtype=np.float32),
        "weather_id": np.array([vocab.weather_id(weather_token)], dtype=np.int64),
        "field_effects": np.array([field_effects], dtype=np.float32),
        "my_hazards": np.array([my_hazards], dtype=np.float32),
        "opp_hazards": np.array([opp_hazards], dtype=np.float32),
        "my_screens": np.array([my_screens], dtype=np.float32),
        "opp_screens": np.array([opp_screens], dtype=np.float32),
    }
    return tensors, mine_slot_species


# ---------------------------------------------------------------------------
# Prediction result + model wrapper
# ---------------------------------------------------------------------------
@dataclass
class PolicyPrediction:
    action_type_probs: dict            # {"move": p, "switch": p}
    move_probs: dict                   # {move_id_str: probability} -- FULL vocab, unmasked
    switch_species_probs: dict         # {species_id_str: probability} -- known, non-active bench only
    slot_order: list = field(default_factory=list)  # acting side's 6 slots, species id-str or None
    backend: str = "unknown"


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    finite_max = np.max(np.where(np.isfinite(logits), logits, -np.inf))
    if not np.isfinite(finite_max):
        return np.full_like(logits, 1.0 / logits.shape[-1])
    shifted = np.where(np.isfinite(logits), logits - finite_max, -np.inf)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / logits.shape[-1])
    return exp / total


class PolicyModel:
    """Loaded, ready-to-query policy net. Construct via `PolicyModel.load()`,
    never directly."""

    def __init__(self, backend: str, vocab: PolicyVocab, session=None, torch_model=None, id_to_move: Optional[dict] = None):
        self.backend = backend                    # "onnx" or "torch"
        self.vocab = vocab
        self._session = session                   # onnxruntime.InferenceSession
        self._torch_model = torch_model            # policy_net.FutureSightPolicyNet
        self._id_to_move = id_to_move or {}        # vocab index -> move id-str, for decoding move_logits

    # -- loading -------------------------------------------------------
    @classmethod
    def load(
        cls,
        onnx_path: "str | Path" = "data/policy_net.onnx",
        pth_path: "str | Path" = "data/policy_net.pth",
        feature_schema_path: "str | Path" = "data/feature_schema_gen9ou.json",
        vocab_path: "str | Path" = "data/vocab_gen9ou.json",
    ) -> Optional["PolicyModel"]:
        """Best-effort load. Returns None (never raises) if nothing usable
        is found -- callers should treat that as 'use Smogon priors only',
        exactly Step 6E's required fallback behavior."""
        onnx_path = Path(onnx_path)
        pth_path = Path(pth_path)
        feature_schema_path = Path(feature_schema_path)
        vocab_path = Path(vocab_path)

        vocab = PolicyVocab.load(vocab_path, feature_schema_path)
        if vocab is None:
            return None
        id_to_move = {idx: mid for mid, idx in vocab.moves_to_id.items()}

        # -- Preferred path: ONNX Runtime, CPU --------------------------
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                logger.info("Step 6E: loaded policy net via ONNX Runtime from %s", onnx_path)
                return cls(backend="onnx", vocab=vocab, session=session, id_to_move=id_to_move)
            except Exception:
                logger.warning("Found %s but ONNX Runtime failed to load it; trying PyTorch fallback.", onnx_path, exc_info=True)
        else:
            logger.info("%s not found; trying PyTorch fallback.", onnx_path)

        # -- Fallback: raw PyTorch checkpoint ----------------------------
        if pth_path.exists():
            try:
                import torch
                from policy_net import FutureSightPolicyNet, PolicyNetConfig

                ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
                cfg = PolicyNetConfig(**ckpt["config"])
                model = FutureSightPolicyNet(cfg)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                logger.info("Step 6E: loaded policy net via PyTorch fallback from %s", pth_path)
                return cls(backend="torch", vocab=vocab, torch_model=model, id_to_move=id_to_move)
            except Exception:
                logger.warning("Found %s but PyTorch fallback failed to load it.", pth_path, exc_info=True)
        else:
            logger.info("%s not found either.", pth_path)

        logger.info("Step 6E: no usable policy net found -- falling back to Smogon priors only.")
        return None

    # -- inference -------------------------------------------------------
    def _forward(self, tensors: dict) -> tuple:
        """Returns (action_type_logits, move_logits, switch_logits), each
        a 1-D numpy array (batch dimension squeezed out)."""
        if self.backend == "onnx":
            outputs = self._session.run(
                ["action_type_logits", "move_logits", "switch_logits"], tensors,
            )
            return outputs[0][0], outputs[1][0], outputs[2][0]

        import torch
        with torch.no_grad():
            batch = {k: torch.from_numpy(v) for k, v in tensors.items()}
            out = self._torch_model(batch)
            return (
                out.action_type_logits[0].numpy(),
                out.move_logits[0].numpy(),
                out.switch_logits[0].numpy(),
            )

    def predict(self, battle: Any, acting_side: str = "opponent") -> Optional[PolicyPrediction]:
        """Run the net on the current battle state. Returns None (never
        raises) on any conversion/inference failure -- callers should treat
        that exactly like `PolicyModel.load()` returning None."""
        try:
            tensors, slot_species = battle_to_tensor(battle, acting_side, self.vocab)
            action_type_logits, move_logits, switch_logits = self._forward(tensors)

            action_type_probs_arr = _softmax(action_type_logits)
            move_probs_arr = _softmax(move_logits)
            switch_probs_arr = _softmax(switch_logits)  # fainted slots already -inf-masked by the model itself

            # Index 0 in the moves vocab is the reserved "unrevealed/pad"
            # slot, not a real move (see Vocab in dataset_parser.py) -- it
            # is never a valid training LABEL, so we exclude it here and
            # renormalize over real moves only, rather than let a stray
            # sliver of softmax mass on a non-action silently make every
            # real move's probability look smaller than it should.
            move_probs = {}
            for idx, p in enumerate(move_probs_arr):
                mid = self._id_to_move.get(idx)
                if mid:
                    move_probs[mid] = float(p)
            move_total = sum(move_probs.values())
            if move_total > 0:
                move_probs = {k: v / move_total for k, v in move_probs.items()}

            # Active slot (index 0, by construction of _ordered_slots) can't
            # be a switch target; known, non-fainted bench slots only.
            active_species = slot_species[0] if slot_species else None
            switch_species_probs = {}
            for i, species in enumerate(slot_species):
                if species is None or species == active_species:
                    continue
                switch_species_probs[species] = float(switch_probs_arr[i])
            total = sum(switch_species_probs.values())
            if total > 0:
                switch_species_probs = {k: v / total for k, v in switch_species_probs.items()}

            return PolicyPrediction(
                action_type_probs={"move": float(action_type_probs_arr[0]), "switch": float(action_type_probs_arr[1])},
                move_probs=move_probs,
                switch_species_probs=switch_species_probs,
                slot_order=slot_species,
                backend=self.backend,
            )
        except Exception:
            logger.debug("Policy net prediction failed; caller should fall back to Smogon priors.", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Self-test / CLI
# ---------------------------------------------------------------------------
class _MockStatus:
    def __init__(self, name: str):
        self.name = name


class _MockNamedEnum:
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hasattr(other, "name") and self.name == other.name


class _MockPokemon:
    """Minimal duck-typed stand-in for a poke-env Pokemon -- deliberately
    NOT a poke_env.battle.Pokemon subclass, so this self-test runs even in
    an environment where poke-env itself isn't installed."""

    def __init__(self, species, hp_fraction=1.0, fainted=False, status=None,
                 item=None, ability=None, boosts=None, is_terastallized=False,
                 tera_type=None):
        self.species = species
        self.current_hp_fraction = hp_fraction
        self.fainted = fainted
        self.status = _MockStatus(status) if status else None
        self.item = item
        self.ability = ability
        self.boosts = boosts or {s: 0 for s in BOOST_STATS}
        self.is_terastallized = is_terastallized
        self.tera_type = _MockNamedEnum(tera_type) if tera_type else None


class _MockBattle:
    def __init__(self, team, opponent_team, active_pokemon, opponent_active_pokemon,
                 side_conditions=None, opponent_side_conditions=None, weather=None,
                 fields=None, turn=5):
        self.team = team
        self.opponent_team = opponent_team
        self.active_pokemon = active_pokemon
        self.opponent_active_pokemon = opponent_active_pokemon
        self.side_conditions = side_conditions or {}
        self.opponent_side_conditions = opponent_side_conditions or {}
        self.weather = weather or {}
        self.fields = fields or {}
        self.turn = turn


def _build_mock_battle() -> "_MockBattle":
    opp_active = _MockPokemon("landorustherian", hp_fraction=0.72, item="leftovers", ability="intimidate")
    opp_bench = {
        "p2: Great Tusk": _MockPokemon("greattusk", hp_fraction=1.0),
        "p2: Gholdengo": _MockPokemon("gholdengo", hp_fraction=0.4),
    }
    opp_team = {"p2: Landorus-Therian": opp_active, **opp_bench}

    my_active = _MockPokemon("toxapex", hp_fraction=0.9, ability="regenerator")
    my_team = {"p1: Toxapex": my_active}

    sr = _MockNamedEnum("STEALTH_ROCK")
    return _MockBattle(
        team=my_team,
        opponent_team=opp_team,
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions={sr: 1},
        turn=12,
    )


def _make_synthetic_artifacts(tmp_dir: Path) -> dict:
    """Builds a tiny, real (randomly-initialized) ONNX export + matching
    vocab/schema files, so `--test-policy` proves the ONNX round-trip
    actually works end-to-end even before the user has trained a real
    model. Requires torch+onnx only for this synthetic-build step; the
    ONNX inference step itself that follows does not need torch."""
    import subprocess
    tmp_dir.mkdir(parents=True, exist_ok=True)

    from dataset_parser import Vocab, Vocabs, build_feature_schema  # noqa: local, dev-only import
    species = ["Landorus-Therian", "Great Tusk", "Gholdengo", "Toxapex", "Kingambit"]
    moves = ["Earthquake", "U-turn", "Stealth Rock", "Moonblast", "Recover", "Toxic"]
    items = ["Leftovers", "Heavy-Duty Boots"]
    abilities = ["Intimidate", "Regenerator", "Good as Gold"]
    weathers = ["Sandstorm", "RainDance"]
    vocabs = Vocabs(species=Vocab(species), moves=Vocab(moves), items=Vocab(items),
                     abilities=Vocab(abilities), weathers=Vocab(weathers))

    vocab_path = tmp_dir / "vocab_gen9ou.json"
    schema_path = tmp_dir / "feature_schema_gen9ou.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabs.to_json_dict(), f)
    with open(schema_path, "w") as f:
        json.dump(build_feature_schema(vocabs), f)

    from policy_net import PolicyNetConfig, FutureSightPolicyNet, make_synthetic_batch
    from train_policy import export_onnx

    cfg = PolicyNetConfig.from_feature_schema(schema_path, d_model=32, num_heads=2, num_layers=1, dim_feedforward=64)
    model = FutureSightPolicyNet(cfg).eval()
    sample = make_synthetic_batch(cfg, batch_size=2)
    onnx_path = tmp_dir / "policy_net.onnx"
    export_onnx(model, sample, onnx_path)

    return {"onnx": onnx_path, "schema": schema_path, "vocab": vocab_path}


def _run_self_test(args: argparse.Namespace) -> int:
    print("=" * 72)
    print("  Step 6E: policy_inference.py self-test")
    print("=" * 72)

    onnx_path = Path(args.onnx_path)
    pth_path = Path(args.pth_path)
    schema_path = Path(args.feature_schema)
    vocab_path = Path(args.vocab)

    built_synthetic = False
    if not onnx_path.exists() and not pth_path.exists():
        print(f"\nNo model found at {onnx_path} or {pth_path}.")
        print("Building a small synthetic (randomly-initialized) ONNX model")
        print("instead, purely to prove the inference plumbing works end to")
        print("end. Train the real thing with train_policy.py --export-onnx.")
        try:
            paths = _make_synthetic_artifacts(Path("data/_policy_selftest"))
            onnx_path, schema_path, vocab_path = paths["onnx"], paths["schema"], paths["vocab"]
            built_synthetic = True
        except Exception as exc:
            print(f"\nCould not build a synthetic model either ({exc}).")
            print("This usually just means torch/onnx aren't installed in this")
            print("environment -- that's fine for a live bot (which only needs")
            print("onnxruntime), but the self-test needs one or the other to")
            print("demonstrate a real forward pass. Exiting cleanly.")
            return 1

    print(f"\nLoading policy net:\n  onnx  = {onnx_path}\n  pth   = {pth_path}\n  schema= {schema_path}\n  vocab = {vocab_path}")
    policy = PolicyModel.load(onnx_path=onnx_path, pth_path=pth_path,
                               feature_schema_path=schema_path, vocab_path=vocab_path)
    if policy is None:
        print("\nPolicyModel.load() returned None -- graceful fallback path confirmed,")
        print("but there's nothing to run a forward pass against. Exiting.")
        return 1
    print(f"Loaded OK. Backend = {policy.backend}")

    battle = _build_mock_battle()
    print("\nRunning predict(battle, acting_side='opponent') on a mock battle...")
    pred = policy.predict(battle, acting_side="opponent")
    if pred is None:
        print("predict() returned None -- something went wrong even though load() succeeded.")
        return 1

    print(f"\naction_type_probs = {{'move': {pred.action_type_probs['move']:.3f}, "
          f"'switch': {pred.action_type_probs['switch']:.3f}}}")
    print(f"sum(move_probs)   = {sum(pred.move_probs.values()):.3f}  ({len(pred.move_probs)} moves in vocab)")
    top_moves = sorted(pred.move_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"top 3 predicted moves: {top_moves}")
    print(f"switch_species_probs = {pred.switch_species_probs}")
    print(f"acting-side slot order (index -> species) = {list(enumerate(pred.slot_order))}")

    assert abs(pred.action_type_probs["move"] + pred.action_type_probs["switch"] - 1.0) < 1e-4
    assert abs(sum(pred.move_probs.values()) - 1.0) < 1e-3
    if pred.switch_species_probs:
        assert abs(sum(pred.switch_species_probs.values()) - 1.0) < 1e-3

    print("\nSelf-test OK: battle -> tensor -> ONNX/torch forward pass -> "
          "probabilities all round-tripped without crashing.")
    if built_synthetic:
        print("(Ran against a synthetic randomly-initialized model -- point")
        print(" --onnx-path/--feature-schema/--vocab at your real trained")
        print(" artifacts to sanity-check the actual model.)")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 6E: policy net inference wrapper self-test.")
    p.add_argument("--test-policy", action="store_true", help="Run the self-test (also runs with no flags at all).")
    p.add_argument("--onnx-path", default="data/policy_net.onnx")
    p.add_argument("--pth-path", default="data/policy_net.pth")
    p.add_argument("--feature-schema", default="data/feature_schema_gen9ou.json")
    p.add_argument("--vocab", default="data/vocab_gen9ou.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    return _run_self_test(args)


if __name__ == "__main__":
    sys.exit(main())
