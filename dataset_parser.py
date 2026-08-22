#!/usr/bin/env python3
"""
dataset_parser.py
==================
Phase 6, Step 6B — Replay Parser & Tensor State-Action Encoder.

Consumes the raw replay JSONs produced by Step 6A (scrape_replays.py) and
turns them into a supervised (state, action) dataset for the Set-Transformer
policy network built in Step 6C.

PIPELINE
--------
1. First pass over every replay: harvest every species/move/item/ability/
   weather token that appears anywhere in the corpus, and build stable,
   sorted vocabularies from them (so ids are reproducible across runs).
2. Second pass: replay each battle log line-by-line through a small state
   machine (BattleState) that tracks both teams' HP, status, boosts, items,
   abilities, tera state, hazards, screens, weather, and field effects.
   Every time a player MOVES or SWITCHES, we snapshot the state immediately
   BEFORE that action is applied and record it as one training sample.
3. Optionally keep only the winning side's decisions (--winner-only,
   default on) — standard practice for imitation-learning a policy net,
   since we want to clone good play, not average it with the loser's.
4. Split by REPLAY (not by sample) into train/val, so no single battle's
   states leak across the split.
5. Save everything as a single dataset.pt, plus a vocab.json and a
   feature_schema.json describing exactly what every tensor field means —
   Step 6C should treat feature_schema.json as the source of truth for
   input dimensions rather than hardcoding shapes.

STATE REPRESENTATION
---------------------
Each sample encodes 12 "pokemon slots" (6 belonging to the acting side,
ordered [active, then bench alphabetically by species], followed by 6
belonging to the opponent, same ordering). This is deliberately NOT
flattened into one giant vector: it's shaped as a per-slot SET of features
so Step 6C's Set-Transformer can attend over slots directly, with species/
item/ability/tera looked up via embedding tables rather than one-hot.

Perspective is always canonicalized to "mine vs opponent's" from the acting
player's point of view, so the same model weights work regardless of
whether the acting side was p1 or p2 in the original replay.

KNOWN SCOPE LIMITATIONS (intentional, documented rather than silently
getting them wrong):
  * Singles formats only. Replays tagged with a non-"singles" gametype are
    skipped outright rather than mis-parsed.
  * Abilities/items are only marked "revealed" when an explicit
    |-ability|/|-item|/|-enditem| protocol line fires. Reveals that are only
    implied by a bracketed "[from] ability: X" annotation on an unrelated
    line (e.g. a damage line) are not parsed in this pass.
  * Mid-battle forme changes (Ogerpon masks, Zamazenta-Crowned, etc.) update
    the active slot's displayed species for the rest of the battle. Forme
    reversion-on-switch edge cases (Mimikyu Busted, Cramorant Gorging) are
    not modeled.
  * action_move_id is a flat index into a global move vocabulary, not
    masked to the acting pokemon's actual (PP-limited, Choice-locked, etc.)
    legal move set. Legal-action masking belongs to Step 6C/6D, since it
    needs move-slot tracking this parser doesn't attempt.
  * Ditto/Mew `|-transform|` is not modeled; the transformed-into identity
    is ignored and the pre-transform species/stats are kept as-is.

Dependencies: Python 3.9+, `torch`. No network access needed -- this step
only reads local files written by scrape_replays.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

# ---------------------------------------------------------------------------
# Fixed game constants (hardcoded because they're stable game rules, not
# corpus-dependent vocabulary)
# ---------------------------------------------------------------------------

MAX_TEAM_SIZE = 6

BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Index 0 is reserved to mean "no status" (also doubles as the pad value for
# empty bench slots).
STATUS_LIST = ["", "brn", "par", "psn", "tox", "slp", "frz"]
STATUS_TO_ID = {s: i for i, s in enumerate(STATUS_LIST)}

# Index 0 reserved for "not terastallized / unrevealed".
TERA_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
    "Dragon", "Dark", "Steel", "Fairy", "Stellar",
]
TERA_TYPE_TO_ID = {t: i + 1 for i, t in enumerate(TERA_TYPES)}

# Global field effects tracked as an independent multi-hot vector, since
# terrains, Trick Room, and Gravity can all be active simultaneously.
FIELD_EFFECTS = [
    "Electric Terrain", "Grassy Terrain", "Misty Terrain", "Psychic Terrain",
    "Trick Room", "Gravity", "Wonder Room", "Magic Room",
]

logger = logging.getLogger("dataset_parser")


# ---------------------------------------------------------------------------
# Small parsing helpers for Showdown's protocol lines
# ---------------------------------------------------------------------------

def parse_pokemon_ident(field: str) -> Tuple[str, str]:
    """'p1a: Sylveon' -> ('p1', 'Sylveon'). Doubles slot letters (a/b) are
    intentionally discarded -- this parser is singles-only."""
    prefix, _, nickname = field.partition(": ")
    side = prefix[:2] if len(prefix) >= 2 else prefix
    return side, nickname.strip()


def parse_side_id(field: str) -> str:
    """'p1' or 'p1: SomeUsername' -> 'p1'."""
    return field.split(":", 1)[0].strip()[:2]


def strip_source_prefix(field: str) -> str:
    """'move: Stealth Rock' -> 'Stealth Rock'; 'Stealth Rock' -> unchanged."""
    return field.split(": ", 1)[1] if ": " in field else field


def parse_details(details: str) -> Tuple[str, int, Optional[str]]:
    """'Landorus-Therian, L100, M' -> ('Landorus-Therian', 100, 'M')."""
    parts = [p.strip() for p in details.split(",")]
    species = parts[0]
    level = 100
    gender = None
    for p in parts[1:]:
        if p.startswith("L") and p[1:].isdigit():
            level = int(p[1:])
        elif p in ("M", "F"):
            gender = p
    return species, level, gender


def parse_hp_status(hp_field: str) -> Tuple[float, str]:
    """'48/100 par' -> (0.48, 'par'); '0 fnt' -> (0.0, 'fnt'); '100/100' -> (1.0, '')."""
    hp_field = hp_field.strip()
    head, _, status = hp_field.partition(" ")
    if status == "fnt" or head == "0" and "fnt" in hp_field:
        return 0.0, "fnt"
    if "/" in head:
        num_s, _, den_s = head.partition("/")
        try:
            num, den = float(num_s), float(den_s)
            frac = (num / den) if den else 0.0
        except ValueError:
            frac = 1.0
    else:
        try:
            frac = float(head) / 100.0
        except ValueError:
            frac = 1.0
    return max(0.0, min(1.0, frac)), status


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """A stable string->int vocabulary. Index 0 is always reserved for
    UNK/PAD/unrevealed so downstream embeddings can treat 0 as a safe
    "nothing here" default."""

    def __init__(self, tokens: Optional[Sequence[str]] = None):
        tokens = sorted(set(tokens or []))
        self._token_to_id = {tok: i + 1 for i, tok in enumerate(tokens)}

    def get(self, token: Optional[str], default: int = 0) -> int:
        if not token:
            return default
        return self._token_to_id.get(token, default)

    def __len__(self) -> int:
        return len(self._token_to_id) + 1  # +1 for the reserved UNK/PAD slot

    def to_dict(self) -> Dict[str, int]:
        return dict(self._token_to_id)


@dataclass
class Vocabs:
    species: Vocab
    moves: Vocab
    items: Vocab
    abilities: Vocab
    weathers: Vocab

    def to_json_dict(self) -> dict:
        return {
            "species": self.species.to_dict(),
            "moves": self.moves.to_dict(),
            "items": self.items.to_dict(),
            "abilities": self.abilities.to_dict(),
            "weathers": self.weathers.to_dict(),
            "statuses": STATUS_TO_ID,
            "tera_types": TERA_TYPE_TO_ID,
            "field_effects": FIELD_EFFECTS,
        }


def harvest_tokens(log_text: str, tokens: Dict[str, Set[str]]) -> None:
    """Lightweight first-pass scan: pull every species/move/item/ability/
    weather string out of a raw log, without simulating full battle state.
    Mutates `tokens` in place."""
    for raw_line in log_text.split("\n"):
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        try:
            if cmd == "poke" and len(parts) > 3:
                species, _, _ = parse_details(parts[3])
                tokens["species"].add(species)
            elif cmd in ("switch", "drag") and len(parts) > 3:
                species, _, _ = parse_details(parts[3])
                tokens["species"].add(species)
            elif cmd == "-formechange" and len(parts) > 3:
                species, _, _ = parse_details(parts[3])
                tokens["species"].add(species)
            elif cmd == "move" and len(parts) > 3:
                tokens["moves"].add(parts[3])
            elif cmd in ("-item", "-enditem") and len(parts) > 3 and parts[3]:
                tokens["items"].add(parts[3])
            elif cmd == "-ability" and len(parts) > 3 and parts[3]:
                tokens["abilities"].add(parts[3])
            elif cmd == "-weather" and len(parts) > 2 and parts[2] != "none":
                tokens["weathers"].add(parts[2])
        except (IndexError, ValueError):
            continue


# ---------------------------------------------------------------------------
# Mutable battle state
# ---------------------------------------------------------------------------

@dataclass
class PokemonSlot:
    base_species: str
    display_species: str
    level: int = 100
    hp_fraction: float = 1.0
    status: str = ""
    fainted: bool = False
    item: Optional[str] = None
    ability: Optional[str] = None
    terastallized: bool = False
    tera_type: Optional[str] = None
    boosts: Dict[str, int] = dataclass_field(default_factory=lambda: {s: 0 for s in BOOST_STATS})

    def reset_boosts(self) -> None:
        self.boosts = {s: 0 for s in BOOST_STATS}


class SideState:
    def __init__(self, username: str = ""):
        self.username = username
        self.team: Dict[str, PokemonSlot] = {}       # keyed by base_species
        self.active_species: Optional[str] = None
        self.nickname_map: Dict[str, str] = {}        # nickname -> base_species
        self.hazards = {"stealth_rock": False, "spikes": 0, "toxic_spikes": 0, "sticky_web": False}
        self.screens = {"reflect": False, "light_screen": False, "aurora_veil": False, "tailwind": False}

    def get_or_create(self, base_species: str) -> PokemonSlot:
        if base_species not in self.team:
            self.team[base_species] = PokemonSlot(base_species=base_species, display_species=base_species)
        return self.team[base_species]

    def get_active(self) -> Optional[PokemonSlot]:
        return self.team.get(self.active_species) if self.active_species else None

    def resolve_by_nickname(self, nickname: str) -> Optional[PokemonSlot]:
        species = self.nickname_map.get(nickname)
        return self.team.get(species) if species else None

    def ordered_slots(self) -> List[PokemonSlot]:
        """Active first, then bench sorted alphabetically -- deterministic
        regardless of team-sheet order or switch history."""
        active = self.get_active()
        bench = sorted(
            (m for m in self.team.values() if m is not active),
            key=lambda m: m.base_species,
        )
        return ([active] if active else []) + bench


class BattleState:
    def __init__(self):
        self.turn = 0
        self.weather = "none"
        self.field_effects: Set[str] = set()
        self.sides: Dict[str, SideState] = {}

    def side(self, side_id: str) -> SideState:
        if side_id not in self.sides:
            self.sides[side_id] = SideState()
        return self.sides[side_id]


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def hazards_vec(h: dict) -> List[float]:
    return [
        1.0 if h["stealth_rock"] else 0.0,
        h["spikes"] / 3.0,
        h["toxic_spikes"] / 2.0,
        1.0 if h["sticky_web"] else 0.0,
    ]


def screens_vec(s: dict) -> List[float]:
    return [
        1.0 if s["reflect"] else 0.0,
        1.0 if s["light_screen"] else 0.0,
        1.0 if s["aurora_veil"] else 0.0,
        1.0 if s["tailwind"] else 0.0,
    ]


def encode_side_slots(side_state: SideState, vocabs: Vocabs) -> dict:
    slots = side_state.ordered_slots()[:MAX_TEAM_SIZE]
    out = {
        "species_ids": [], "is_active": [], "hp_fraction": [], "fainted": [],
        "status_ids": [], "item_ids": [], "ability_ids": [],
        "terastallized": [], "tera_type_ids": [], "boosts": [],
    }
    for i in range(MAX_TEAM_SIZE):
        if i < len(slots):
            s = slots[i]
            out["species_ids"].append(vocabs.species.get(s.display_species))
            out["is_active"].append(1.0 if side_state.active_species == s.base_species else 0.0)
            out["hp_fraction"].append(s.hp_fraction)
            out["fainted"].append(1.0 if s.fainted else 0.0)
            out["status_ids"].append(STATUS_TO_ID.get(s.status, 0))
            out["item_ids"].append(vocabs.items.get(s.item))
            out["ability_ids"].append(vocabs.abilities.get(s.ability))
            out["terastallized"].append(1.0 if s.terastallized else 0.0)
            out["tera_type_ids"].append(TERA_TYPE_TO_ID.get(s.tera_type, 0) if s.tera_type else 0)
            out["boosts"].append([s.boosts[stat] / 6.0 for stat in BOOST_STATS])
        else:
            out["species_ids"].append(0); out["is_active"].append(0.0)
            out["hp_fraction"].append(0.0); out["fainted"].append(1.0)
            out["status_ids"].append(0); out["item_ids"].append(0); out["ability_ids"].append(0)
            out["terastallized"].append(0.0); out["tera_type_ids"].append(0)
            out["boosts"].append([0.0] * len(BOOST_STATS))
    return out


def encode_state(battle: BattleState, acting_side: str, vocabs: Vocabs) -> dict:
    """Encode the CURRENT battle state from `acting_side`'s perspective.
    Call this BEFORE applying the action's own line effects, so the sample
    reflects exactly what the acting player could see when they decided."""
    other_side = "p2" if acting_side == "p1" else "p1"
    mine = encode_side_slots(battle.side(acting_side), vocabs)
    theirs = encode_side_slots(battle.side(other_side), vocabs)

    merged = {k: mine[k] + theirs[k] for k in mine}  # 6 + 6 = 12 slots

    return {
        **merged,
        "turn_norm": min(battle.turn / 100.0, 1.0),
        "weather_id": vocabs.weathers.get(battle.weather if battle.weather != "none" else None),
        "field_effects": [1.0 if fe in battle.field_effects else 0.0 for fe in FIELD_EFFECTS],
        "my_hazards": hazards_vec(battle.side(acting_side).hazards),
        "opp_hazards": hazards_vec(battle.side(other_side).hazards),
        "my_screens": screens_vec(battle.side(acting_side).screens),
        "opp_screens": screens_vec(battle.side(other_side).screens),
    }


def compute_switch_slot(side_state: SideState, target_species: str) -> int:
    """Which of the acting side's 12-set slot indices (0-5, since it's
    always their own side's block) the switch target lands on. Ensures the
    target exists in the team dict first, since in formats without a team
    preview a pokemon may be revealed for the first time by this very
    switch."""
    side_state.get_or_create(target_species)
    slots = side_state.ordered_slots()[:MAX_TEAM_SIZE]
    for i, s in enumerate(slots):
        if s.base_species == target_species:
            return i
    return -1


# ---------------------------------------------------------------------------
# Hazard/screen bookkeeping
# ---------------------------------------------------------------------------

HAZARD_NAMES = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web"}
SCREEN_NAMES = {"Reflect", "Light Screen", "Aurora Veil", "Tailwind"}


def apply_side_condition_start(side_state: SideState, condition: str) -> None:
    if condition == "Stealth Rock":
        side_state.hazards["stealth_rock"] = True
    elif condition == "Spikes":
        side_state.hazards["spikes"] = clamp(side_state.hazards["spikes"] + 1, 0, 3)
    elif condition == "Toxic Spikes":
        side_state.hazards["toxic_spikes"] = clamp(side_state.hazards["toxic_spikes"] + 1, 0, 2)
    elif condition == "Sticky Web":
        side_state.hazards["sticky_web"] = True
    elif condition == "Reflect":
        side_state.screens["reflect"] = True
    elif condition == "Light Screen":
        side_state.screens["light_screen"] = True
    elif condition == "Aurora Veil":
        side_state.screens["aurora_veil"] = True
    elif condition == "Tailwind":
        side_state.screens["tailwind"] = True


def apply_side_condition_end(side_state: SideState, condition: str) -> None:
    if condition == "Stealth Rock":
        side_state.hazards["stealth_rock"] = False
    elif condition == "Spikes":
        side_state.hazards["spikes"] = 0
    elif condition == "Toxic Spikes":
        side_state.hazards["toxic_spikes"] = 0
    elif condition == "Sticky Web":
        side_state.hazards["sticky_web"] = False
    elif condition == "Reflect":
        side_state.screens["reflect"] = False
    elif condition == "Light Screen":
        side_state.screens["light_screen"] = False
    elif condition == "Aurora Veil":
        side_state.screens["aurora_veil"] = False
    elif condition == "Tailwind":
        side_state.screens["tailwind"] = False


# ---------------------------------------------------------------------------
# Per-replay simulation
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    samples: List[dict]
    skip_reason: Optional[str] = None
    parse_errors: int = 0


def simulate_replay(log_text: str, replay_id: str, vocabs: Vocabs, min_turns: int) -> ParseResult:
    battle = BattleState()
    usernames: Dict[str, str] = {}
    winner_username: Optional[str] = None
    saw_win_line = False
    samples: List[dict] = []
    parse_errors = 0

    for raw_line in log_text.split("\n"):
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        try:
            if cmd == "player" and len(parts) > 3:
                side_id, uname = parts[2], parts[3]
                usernames[side_id] = uname
                battle.side(side_id).username = uname

            elif cmd == "gametype" and len(parts) > 2:
                if parts[2] != "singles":
                    return ParseResult(samples=[], skip_reason=f"non_singles_format:{parts[2]}")

            elif cmd == "poke" and len(parts) > 3:
                side_id = parts[2]
                species, level, _ = parse_details(parts[3])
                slot = battle.side(side_id).get_or_create(species)
                slot.level = level

            elif cmd == "turn" and len(parts) > 2:
                battle.turn = int(parts[2])

            elif cmd in ("switch", "drag") and len(parts) > 4:
                side_id, nickname = parse_pokemon_ident(parts[2])
                species, level, _ = parse_details(parts[3])
                hp_frac, status = parse_hp_status(parts[4])
                side_state = battle.side(side_id)

                if cmd == "switch":
                    slot_idx = compute_switch_slot(side_state, species)
                    if slot_idx >= 0:
                        samples.append({
                            "state": encode_state(battle, side_id, vocabs),
                            "side": side_id,
                            "username": usernames.get(side_id, ""),
                            "turn": battle.turn,
                            "action_type": 1,  # switch
                            "action_move_id": -1,
                            "action_switch_slot": slot_idx,
                            "raw_action": species,
                            "replay_id": replay_id,
                        })

                prev_active = side_state.get_active()
                if prev_active is not None:
                    prev_active.reset_boosts()
                slot = side_state.get_or_create(species)
                slot.level = level
                slot.display_species = species
                if status == "fnt":
                    slot.fainted = True
                    slot.hp_fraction = 0.0
                else:
                    slot.hp_fraction = hp_frac
                    slot.status = status
                side_state.nickname_map[nickname] = species
                side_state.active_species = species

            elif cmd == "move" and len(parts) > 3:
                side_id, nickname = parse_pokemon_ident(parts[2])
                if side_id not in battle.sides:
                    continue
                move_name = parts[3]
                samples.append({
                    "state": encode_state(battle, side_id, vocabs),
                    "side": side_id,
                    "username": usernames.get(side_id, ""),
                    "turn": battle.turn,
                    "action_type": 0,  # move
                    "action_move_id": vocabs.moves.get(move_name),
                    "action_switch_slot": -1,
                    "raw_action": move_name,
                    "replay_id": replay_id,
                })

            elif cmd in ("-damage", "-heal", "-sethp") and len(parts) > 3:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is None:
                    continue
                hp_frac, status = parse_hp_status(parts[3])
                if status == "fnt":
                    slot.fainted = True
                    slot.hp_fraction = 0.0
                else:
                    slot.hp_fraction = hp_frac
                    slot.status = status

            elif cmd == "-status" and len(parts) > 3:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.status = parts[3]

            elif cmd == "-curestatus" and len(parts) > 2:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.status = ""

            elif cmd in ("-boost", "-unboost") and len(parts) > 4:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None and parts[3] in slot.boosts:
                    sign = 1 if cmd == "-boost" else -1
                    slot.boosts[parts[3]] = clamp(slot.boosts[parts[3]] + sign * int(parts[4]), -6, 6)

            elif cmd == "-setboost" and len(parts) > 4:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None and parts[3] in slot.boosts:
                    slot.boosts[parts[3]] = clamp(int(parts[4]), -6, 6)

            elif cmd == "-clearboost" and len(parts) > 2:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.reset_boosts()

            elif cmd == "-clearallboost":
                for side_state in battle.sides.values():
                    active = side_state.get_active()
                    if active is not None:
                        active.reset_boosts()

            elif cmd == "-weather" and len(parts) > 2:
                battle.weather = parts[2]

            elif cmd == "-fieldstart" and len(parts) > 2:
                name = strip_source_prefix(parts[2])
                if name in FIELD_EFFECTS:
                    battle.field_effects.add(name)

            elif cmd == "-fieldend" and len(parts) > 2:
                name = strip_source_prefix(parts[2])
                battle.field_effects.discard(name)

            elif cmd == "-sidestart" and len(parts) > 3:
                side_id = parse_side_id(parts[2])
                condition = strip_source_prefix(parts[3])
                apply_side_condition_start(battle.side(side_id), condition)

            elif cmd == "-sideend" and len(parts) > 3:
                side_id = parse_side_id(parts[2])
                condition = strip_source_prefix(parts[3])
                apply_side_condition_end(battle.side(side_id), condition)

            elif cmd == "-terastallize" and len(parts) > 3:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.terastallized = True
                    slot.tera_type = parts[3]

            elif cmd == "-formechange" and len(parts) > 3:
                side_id, nickname = parse_pokemon_ident(parts[2])
                new_species, _, _ = parse_details(parts[3])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.display_species = new_species

            elif cmd in ("-item",) and len(parts) > 3 and parts[3]:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.item = parts[3]

            elif cmd == "-enditem" and len(parts) > 3 and parts[3]:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.item = None  # consumed; gone for all future samples

            elif cmd == "-ability" and len(parts) > 3 and parts[3]:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.ability = parts[3]

            elif cmd == "faint" and len(parts) > 2:
                side_id, nickname = parse_pokemon_ident(parts[2])
                slot = battle.side(side_id).resolve_by_nickname(nickname)
                if slot is not None:
                    slot.fainted = True
                    slot.hp_fraction = 0.0

            elif cmd == "win" and len(parts) > 2:
                winner_username = parts[2]
                saw_win_line = True

            elif cmd == "tie":
                winner_username = None
                saw_win_line = True

        except (IndexError, ValueError) as exc:
            parse_errors += 1
            logger.debug("Parse error on %r in %s: %s", raw_line, replay_id, exc)
            continue

    if battle.turn < min_turns:
        return ParseResult(samples=[], skip_reason="too_short", parse_errors=parse_errors)

    if not saw_win_line:
        return ParseResult(samples=samples, skip_reason="no_win_line", parse_errors=parse_errors)

    winner_side_id = next((sid for sid, uname in usernames.items() if uname == winner_username), None)
    for s in samples:
        s["is_winner"] = (s["side"] == winner_side_id) if winner_side_id else False

    return ParseResult(samples=samples, skip_reason=None, parse_errors=parse_errors)


# ---------------------------------------------------------------------------
# Corpus-level driver
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    replays_total: int = 0
    replays_used: int = 0
    replays_skipped_non_singles: int = 0
    replays_skipped_too_short: int = 0
    replays_skipped_no_winner: int = 0
    replays_skipped_load_error: int = 0
    parse_errors: int = 0
    samples_total: int = 0
    samples_kept: int = 0

    def log_summary(self) -> None:
        logger.info(
            "CORPUS SUMMARY | replays_total=%d used=%d skipped(non_singles=%d "
            "too_short=%d no_winner=%d load_error=%d) parse_errors=%d "
            "samples_total=%d samples_kept=%d",
            self.replays_total, self.replays_used,
            self.replays_skipped_non_singles, self.replays_skipped_too_short,
            self.replays_skipped_no_winner, self.replays_skipped_load_error,
            self.parse_errors, self.samples_total, self.samples_kept,
        )


def load_replay_log(path: Path) -> Optional[Tuple[str, str]]:
    """Returns (replay_id, log_text) or None if the file can't be read."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None
    log_text = data.get("log")
    if not log_text:
        logger.warning("%s has no 'log' field, skipping.", path)
        return None
    replay_id = data.get("id", path.stem)
    return replay_id, log_text


def build_vocabs(replay_files: List[Path]) -> Vocabs:
    tokens: Dict[str, Set[str]] = {
        "species": set(), "moves": set(), "items": set(), "abilities": set(), "weathers": set(),
    }
    for path in replay_files:
        loaded = load_replay_log(path)
        if loaded is None:
            continue
        _, log_text = loaded
        harvest_tokens(log_text, tokens)

    logger.info(
        "Vocab harvested | species=%d moves=%d items=%d abilities=%d weathers=%d",
        len(tokens["species"]), len(tokens["moves"]), len(tokens["items"]),
        len(tokens["abilities"]), len(tokens["weathers"]),
    )
    return Vocabs(
        species=Vocab(tokens["species"]),
        moves=Vocab(tokens["moves"]),
        items=Vocab(tokens["items"]),
        abilities=Vocab(tokens["abilities"]),
        weathers=Vocab(tokens["weathers"]),
    )


def stack_samples(samples: List[dict]) -> Dict[str, torch.Tensor]:
    """Convert a list of per-sample dicts into batched tensors."""
    if not samples:
        return {}

    def col(key):
        return [s["state"][key] for s in samples]

    return {
        "species_ids": torch.tensor(col("species_ids"), dtype=torch.long),
        "is_active": torch.tensor(col("is_active"), dtype=torch.float32),
        "hp_fraction": torch.tensor(col("hp_fraction"), dtype=torch.float32),
        "fainted": torch.tensor(col("fainted"), dtype=torch.float32),
        "status_ids": torch.tensor(col("status_ids"), dtype=torch.long),
        "item_ids": torch.tensor(col("item_ids"), dtype=torch.long),
        "ability_ids": torch.tensor(col("ability_ids"), dtype=torch.long),
        "terastallized": torch.tensor(col("terastallized"), dtype=torch.float32),
        "tera_type_ids": torch.tensor(col("tera_type_ids"), dtype=torch.long),
        "boosts": torch.tensor(col("boosts"), dtype=torch.float32),
        "turn_norm": torch.tensor([s["state"]["turn_norm"] for s in samples], dtype=torch.float32),
        "weather_id": torch.tensor([s["state"]["weather_id"] for s in samples], dtype=torch.long),
        "field_effects": torch.tensor([s["state"]["field_effects"] for s in samples], dtype=torch.float32),
        "my_hazards": torch.tensor([s["state"]["my_hazards"] for s in samples], dtype=torch.float32),
        "opp_hazards": torch.tensor([s["state"]["opp_hazards"] for s in samples], dtype=torch.float32),
        "my_screens": torch.tensor([s["state"]["my_screens"] for s in samples], dtype=torch.float32),
        "opp_screens": torch.tensor([s["state"]["opp_screens"] for s in samples], dtype=torch.float32),
        "action_type": torch.tensor([s["action_type"] for s in samples], dtype=torch.long),
        "action_move_id": torch.tensor([s["action_move_id"] for s in samples], dtype=torch.long),
        "action_switch_slot": torch.tensor([s["action_switch_slot"] for s in samples], dtype=torch.long),
    }


def build_feature_schema(vocabs: Vocabs) -> dict:
    return {
        "slots_per_sample": 2 * MAX_TEAM_SIZE,
        "slot_order": "[0:6]=acting side (active first, then bench alphabetical), "
                      "[6:12]=opponent side (same ordering)",
        "vocab_sizes": {
            "species": len(vocabs.species), "moves": len(vocabs.moves),
            "items": len(vocabs.items), "abilities": len(vocabs.abilities),
            "weathers": len(vocabs.weathers), "statuses": len(STATUS_LIST),
            "tera_types": len(TERA_TYPES) + 1,
        },
        "fields": {
            "species_ids": "[N,12] long -- embedding index, 0=pad/unrevealed",
            "is_active": "[N,12] float -- 1.0 if this slot is the current active mon",
            "hp_fraction": "[N,12] float -- 0..1, 0.0 for fainted/pad slots",
            "fainted": "[N,12] float -- 1.0 if fainted or pad slot",
            "status_ids": "[N,12] long -- index into statuses vocab (0=none)",
            "item_ids": "[N,12] long -- embedding index, 0=unrevealed/none",
            "ability_ids": "[N,12] long -- embedding index, 0=unrevealed/none",
            "terastallized": "[N,12] float -- 1.0 if currently terastallized",
            "tera_type_ids": "[N,12] long -- index into tera_types (0=none/unrevealed)",
            "boosts": "[N,12,7] float -- atk/def/spa/spd/spe/accuracy/evasion, each /6 in [-1,1]",
            "turn_norm": "[N] float -- turn number / 100, clipped to 1.0",
            "weather_id": "[N] long -- embedding index, 0=no weather",
            "field_effects": f"[N,{len(FIELD_EFFECTS)}] float multi-hot, order={FIELD_EFFECTS}",
            "my_hazards": "[N,4] float -- [stealth_rock, spikes/3, toxic_spikes/2, sticky_web]",
            "opp_hazards": "[N,4] float -- same layout, opponent's side",
            "my_screens": "[N,4] float -- [reflect, light_screen, aurora_veil, tailwind]",
            "opp_screens": "[N,4] float -- same layout, opponent's side",
            "action_type": "[N] long -- 0=move, 1=switch (prediction target)",
            "action_move_id": "[N] long -- valid iff action_type==0, else -1. Flat vocab "
                               "index, NOT masked to the acting mon's actual legal moves "
                               "(see Step 6C for legal-action masking).",
            "action_switch_slot": "[N] long -- valid iff action_type==1, else -1. Index "
                                   "0-5 into the acting side's OWN 6 slots (pointer-style "
                                   "target, not a global species id).",
        },
        "notes": [
            "Singles formats only; non-singles replays are skipped during parsing.",
            "State is always canonicalized to the acting player's perspective.",
            "Item/ability reveals only come from explicit protocol reveal lines.",
        ],
    }


def parse_corpus(
    replay_files: List[Path],
    vocabs: Vocabs,
    winner_only: bool,
    min_turns: int,
    stats: CorpusStats,
) -> Dict[str, List[dict]]:
    """Returns {replay_id: [sample, ...]} for replays that produced samples."""
    per_replay: Dict[str, List[dict]] = {}
    stats.replays_total = len(replay_files)

    for path in replay_files:
        loaded = load_replay_log(path)
        if loaded is None:
            stats.replays_skipped_load_error += 1
            continue
        replay_id, log_text = loaded

        result = simulate_replay(log_text, replay_id, vocabs, min_turns)
        stats.parse_errors += result.parse_errors

        if result.skip_reason == "too_short":
            stats.replays_skipped_too_short += 1
            continue
        if result.skip_reason and result.skip_reason.startswith("non_singles_format"):
            stats.replays_skipped_non_singles += 1
            continue
        if result.skip_reason == "no_win_line":
            stats.replays_skipped_no_winner += 1
            if winner_only:
                continue

        samples = result.samples
        stats.samples_total += len(samples)
        if winner_only:
            samples = [s for s in samples if s.get("is_winner")]

        if not samples:
            continue

        stats.samples_kept += len(samples)
        stats.replays_used += 1
        per_replay[replay_id] = samples

    return per_replay


def split_train_val(replay_ids: List[str], val_split: float, seed: int) -> Tuple[Set[str], Set[str]]:
    """Split by REPLAY, not by sample, so no battle's states appear in both
    train and val (that would leak near-identical states across the split)."""
    rng = random.Random(seed)
    shuffled = list(replay_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if shuffled and val_split > 0 else 0
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6B: Parse Showdown replay JSONs into a tensor state-action dataset.",
    )
    parser.add_argument("--format", default="gen9ou", help="Format id / subfolder under --input-dir (default: %(default)s)")
    parser.add_argument("--input-dir", default="data/replays", help="Root folder containing {format}/*.json replays (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Output .pt path (default: data/dataset_{format}.pt)")
    parser.add_argument("--vocab-out", default=None, help="Output vocab .json path (default: data/vocab_{format}.json)")
    parser.add_argument("--schema-out", default=None, help="Output feature schema .json path (default: data/feature_schema_{format}.json)")
    parser.add_argument("--winner-only", dest="winner_only", action="store_true", default=True,
                         help="Keep only the winning side's decisions (default: on)")
    parser.add_argument("--no-winner-only", dest="winner_only", action="store_false",
                         help="Keep both players' decisions")
    parser.add_argument("--min-turns", type=int, default=5, help="Skip replays shorter than this many turns (default: %(default)s)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of REPLAYS held out for validation (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split (default: %(default)s)")
    parser.add_argument("--max-replays", type=int, default=None, help="Only process the first N replay files (useful for a quick test run)")
    parser.add_argument("--dry-run", action="store_true", help="Parse a single replay verbosely and print sample stats without writing output.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run_dry_run(args: argparse.Namespace, replay_files: List[Path]) -> None:
    if not replay_files:
        logger.error("[DRY RUN] No replay files found under %s/%s -- nothing to verify.",
                      args.input_dir, args.format)
        return
    sample_file = replay_files[0]
    logger.info("[DRY RUN] Building vocab from %d file(s)...", min(len(replay_files), 20))
    vocabs = build_vocabs(replay_files[:20])
    loaded = load_replay_log(sample_file)
    if loaded is None:
        logger.error("[DRY RUN] Could not load %s", sample_file)
        return
    replay_id, log_text = loaded
    result = simulate_replay(log_text, replay_id, vocabs, args.min_turns)
    logger.info(
        "[DRY RUN] Parsed %s | samples=%d skip_reason=%s parse_errors=%d",
        sample_file.name, len(result.samples), result.skip_reason, result.parse_errors,
    )
    for s in result.samples[:5]:
        action_desc = f"move={s['raw_action']}" if s["action_type"] == 0 else f"switch->{s['raw_action']}"
        logger.info("  turn=%2d side=%s is_winner=%s %s", s["turn"], s["side"], s.get("is_winner"), action_desc)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    replay_dir = Path(args.input_dir) / args.format
    replay_files = sorted(replay_dir.glob("*.json"))
    if args.max_replays:
        replay_files = replay_files[: args.max_replays]

    logger.info("Found %d replay file(s) under %s", len(replay_files), replay_dir)

    if args.dry_run:
        run_dry_run(args, replay_files)
        return 0

    if not replay_files:
        logger.error("No replay files found under %s -- run scrape_replays.py first.", replay_dir)
        return 1

    vocabs = build_vocabs(replay_files)
    stats = CorpusStats()
    per_replay = parse_corpus(replay_files, vocabs, args.winner_only, args.min_turns, stats)
    stats.log_summary()

    if not per_replay:
        logger.error("No usable samples were produced from this corpus. Nothing to save.")
        return 1

    train_ids, val_ids = split_train_val(list(per_replay.keys()), args.val_split, args.seed)
    train_samples = [s for rid in train_ids for s in per_replay[rid]]
    val_samples = [s for rid in val_ids for s in per_replay[rid]]
    logger.info(
        "Split | train_replays=%d train_samples=%d | val_replays=%d val_samples=%d",
        len(train_ids), len(train_samples), len(val_ids), len(val_samples),
    )

    dataset = {
        "train": stack_samples(train_samples),
        "val": stack_samples(val_samples),
        "meta": {
            "train": [{k: s[k] for k in ("replay_id", "turn", "side", "raw_action")} for s in train_samples],
            "val": [{k: s[k] for k in ("replay_id", "turn", "side", "raw_action")} for s in val_samples],
        },
    }

    output_path = Path(args.output) if args.output else Path(f"data/dataset_{args.format}.pt")
    vocab_path = Path(args.vocab_out) if args.vocab_out else Path(f"data/vocab_{args.format}.json")
    schema_path = Path(args.schema_out) if args.schema_out else Path(f"data/feature_schema_{args.format}.json")
    for p in (output_path, vocab_path, schema_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    torch.save(dataset, output_path)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocabs.to_json_dict(), f, indent=2)
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(build_feature_schema(vocabs), f, indent=2)

    logger.info("Saved dataset -> %s", output_path)
    logger.info("Saved vocab -> %s", vocab_path)
    logger.info("Saved feature schema -> %s", schema_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(main(["--dry-run"]))
    sys.exit(main())
