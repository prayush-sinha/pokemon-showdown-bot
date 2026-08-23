"""
evaluator.py -- Phase 4A: State Evaluator & Payoff Matrix Generator

Heuristic evaluation function for Pokemon battle states and a payoff
matrix generator for simultaneous-move game theory.

The evaluator scores a battle position from the bot's perspective:
  +10,000  = guaranteed win  (opponent has 0 living Pokemon)
  -10,000  = guaranteed loss (we have 0 living Pokemon)
       0   = perfectly even

The payoff matrix generator combines the evaluator with forward damage
calculations (from inverse_damage_calc.py) and Smogon priors (from
smogon_priors.py) to produce a 2D matrix:

    rows    = our available actions (moves + switches)
    columns = opponent's likely actions (from priors / revealed moves)
    cells   = expected evaluation score after each (our_action, opp_action) pair

This matrix is consumed by the Phase 4B expectiminimax search to choose
the Nash-optimal action each turn.

Usage
-----
    python evaluator.py          # run built-in tests with mock battle objects
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Any

from poke_env.data import GenData

from inverse_damage_calc import (
    calculate_damage_range,
    get_type_effectiveness,
    has_stab,
    calc_all_stats,
    _apply_stat_stage,
    FieldConditions,
    PokemonStats,
    DamageRange,
    ITEM_MODIFIERS,
    _POKEDEX,
    _MOVES_DB,
    _TYPE_CHART,
    _LEVEL,
)

logger = logging.getLogger("Evaluator")

# =============================================================================
# Scoring Constants -- tunable weights for the heuristic
# =============================================================================

# Terminal states
SCORE_WIN = 10_000.0
SCORE_LOSS = -10_000.0

# Per-Pokemon weights
WEIGHT_HP_RATIO = 500.0          # Total HP ratio (sum of living HP%) scaled
WEIGHT_ALIVE_BONUS = 150.0       # Flat bonus per living Pokemon
WEIGHT_FAINTED_PENALTY = -200.0  # Flat penalty per fainted opposing mon (from opp view)

# Active Pokemon stat stage weights (per stage)
WEIGHT_ATK_STAGE = 25.0
WEIGHT_SPA_STAGE = 25.0
WEIGHT_DEF_STAGE = 15.0
WEIGHT_SPD_STAGE = 15.0
WEIGHT_SPE_STAGE = 30.0          # Speed control is extremely valuable
WEIGHT_EVASION_STAGE = 10.0
WEIGHT_ACCURACY_STAGE = 10.0

# Status condition penalties (applied to the afflicted side)
STATUS_PENALTIES: dict[str, float] = {
    "SLP": -150.0,     # Sleep: effectively removes a Pokemon for 1-3 turns
    "FRZ": -150.0,     # Freeze: even worse, but rare
    "BRN": -80.0,      # Burn: halves phys Atk + residual damage
    "TOK": -100.0,     # Toxic: escalating damage, very threatening
    "TOX": -100.0,     # Toxic (alternative code)
    "PSN": -50.0,      # Regular poison: steady chip
    "PAR": -70.0,      # Paralysis: 25% full para + halves speed
}

# Entry hazard values (from the perspective of the side that HAS them up)
# Having hazards on the OPPONENT's side is good for us; on OUR side is bad.
HAZARD_VALUES: dict[str, float] = {
    "STEALTH_ROCK": 100.0,     # Chunks switch-ins, especially Fire/Ice/Bug/Flying
    "SPIKES": 60.0,            # Per layer (max 3): 12.5% / 16.7% / 25%
    "TOXIC_SPIKES": 70.0,      # Per layer (max 2): poison / toxic on grounded mons
    "STICKY_WEB": 80.0,        # Speed drop on switch-in, very powerful
}

# Screen values (for the side that has the screen up)
SCREEN_VALUES: dict[str, float] = {
    "REFLECT": 60.0,           # Halves physical damage for 5 turns
    "LIGHT_SCREEN": 60.0,      # Halves special damage for 5 turns
    "AURORA_VEIL": 80.0,       # Both screens in one (hail/snow only)
}

# Tailwind bonus
WEIGHT_TAILWIND = 50.0

# ─── Phase 5.2: Endgame Anti-Choke Logic ────────────────────────────────────
# When the opponent is down to their last 1-2 Pokemon, passive/greedy plays
# (laying hazards, setting up stat boosts) provide little-to-no future
# utility -- there aren't enough remaining switch-ins or turns left for them
# to pay off. Left unchecked, the search engine will happily "throw" a
# winning position by choosing Stealth Rock or Swords Dance over a lethal
# attack. These multipliers re-shape the heuristic in that situation so
# direct KO lines dominate the search.
ENDGAME_OPP_ALIVE_THRESHOLD = 2      # opponent alive count <= this triggers endgame mode
ENDGAME_HAZARD_MULTIPLIER = 0.0      # hazards provide zero value once triggered
ENDGAME_STAT_STAGE_MULTIPLIER = 0.5  # halve the value of stat boosts / setup
ENDGAME_ALIVE_BONUS_MULTIPLIER = 2.5  # amplify the material (KO) differential reward


def _is_endgame(opp_alive: int) -> bool:
    """
    True when the opponent has few enough living Pokemon that passive,
    long-horizon plays (hazards, setup) stop paying off and the engine
    should be biased hard toward direct KO lines instead.
    """
    return 0 < opp_alive <= ENDGAME_OPP_ALIVE_THRESHOLD


# =============================================================================
# Data classes
# =============================================================================
@dataclass
class ActionOutcome:
    """
    Describes a single cell in the payoff matrix.

    Attributes
    ----------
    our_action : str
        The action label (move name or "switch:species").
    opp_action : str
        The opponent's action label.
    expected_score : float
        The heuristic evaluation after this action pair resolves.
    damage_to_opp : float
        Estimated damage we deal (average of damage roll range).
    damage_to_us : float
        Estimated damage we receive (average of damage roll range).
    notes : str
        Human-readable annotation (e.g., "super effective", "OHKO").
    """
    our_action: str
    opp_action: str
    expected_score: float = 0.0
    damage_to_opp: float = 0.0
    damage_to_us: float = 0.0
    notes: str = ""


@dataclass
class PayoffMatrix:
    """
    A 2D matrix of action outcomes for one turn of simultaneous play.

    Attributes
    ----------
    our_actions : list[str]
        Row labels (our available actions).
    opp_actions : list[str]
        Column labels (opponent's likely actions).
    matrix : list[list[ActionOutcome]]
        matrix[i][j] = outcome when we play our_actions[i] and opponent
        plays opp_actions[j].
    """
    our_actions: list[str] = field(default_factory=list)
    opp_actions: list[str] = field(default_factory=list)
    matrix: list[list[ActionOutcome]] = field(default_factory=list)

    def best_action_maximin(self) -> tuple[str, float]:
        """
        Return the maximin action: the action that maximises our
        worst-case expected score (pessimistic / safe play).
        """
        if not self.our_actions or not self.matrix:
            return "", 0.0

        best_action = self.our_actions[0]
        best_worst = float("-inf")

        for i, action in enumerate(self.our_actions):
            if not self.matrix[i]:
                continue
            worst = min(cell.expected_score for cell in self.matrix[i])
            if worst > best_worst:
                best_worst = worst
                best_action = action

        return best_action, best_worst

    def best_action_expected(self, opp_probs: Optional[list[float]] = None) -> tuple[str, float]:
        """
        Return the action that maximises expected score, weighted by
        opponent action probabilities (uniform if not provided).
        """
        n_opp = len(self.opp_actions)
        if n_opp == 0 or not self.our_actions:
            return "", 0.0

        if opp_probs is None:
            opp_probs = [1.0 / n_opp] * n_opp

        best_action = self.our_actions[0]
        best_ev = float("-inf")

        for i, action in enumerate(self.our_actions):
            ev = sum(
                cell.expected_score * p
                for cell, p in zip(self.matrix[i], opp_probs)
            )
            if ev > best_ev:
                best_ev = ev
                best_action = action

        return best_action, best_ev

    def display(self) -> str:
        """Return a formatted string representation of the matrix."""
        if not self.matrix or not self.our_actions or not self.opp_actions:
            return "(empty matrix)"

        col_width = 16
        header = f"{'':>{col_width}} |"
        for opp in self.opp_actions:
            header += f" {opp[:col_width-1]:>{col_width-1}} |"
        lines = [header, "-" * len(header)]

        for i, our in enumerate(self.our_actions):
            row = f"{our[:col_width]:>{col_width}} |"
            for j in range(len(self.opp_actions)):
                score = self.matrix[i][j].expected_score
                row += f" {score:>{col_width-1}.1f} |"
            lines.append(row)

        return "\n".join(lines)


# =============================================================================
# Core evaluator
# =============================================================================
def evaluate_state(battle) -> float:
    """
    Score a battle state from the bot's perspective.

    Parameters
    ----------
    battle : poke_env.environment.Battle (or mock with same interface)
        The current battle state.

    Returns
    -------
    float
        Heuristic score. Positive = favourable for us.
        +10,000 = win, -10,000 = loss.
    """
    # ── Terminal checks ─────────────────────────────────────────────────
    if getattr(battle, "won", None) is True:
        return SCORE_WIN
    if getattr(battle, "won", None) is False:
        return SCORE_LOSS

    score = 0.0

    # ── Team HP and alive counts ────────────────────────────────────────
    our_team = getattr(battle, "team", {}) or {}
    opp_team = getattr(battle, "opponent_team", {}) or {}

    our_hp_total, our_alive, our_total = _team_hp_summary(our_team)
    opp_hp_total, opp_alive, opp_total = _team_hp_summary(opp_team)

    # In 6v6 matches, unrevealed opponent bench Pokémon are still alive at 100% HP.
    # (On Turn 1 only 1 opp mon is in opponent_team; the rest are alive and unrevealed)
    if our_total > 1 and opp_total < our_total:
        unrevealed = our_total - opp_total
        opp_alive += unrevealed
        opp_hp_total += unrevealed * 1.0
        opp_total = our_total

    # Check terminal faints if teams are populated
    if our_total > 0 and our_alive == 0:
        return SCORE_LOSS
    if opp_total > 0 and opp_alive == 0:
        return SCORE_WIN

    # HP ratio advantage: how much of our total HP% exceeds theirs
    # Scale: each side sums to max 6.0 (100% * 6 mons), difference in [-6, 6]
    hp_diff = our_hp_total - opp_hp_total
    score += hp_diff * WEIGHT_HP_RATIO

    # ── Phase 5.2: Endgame Anti-Choke -- dynamic weight shifts ───────────
    endgame = _is_endgame(opp_alive)
    alive_bonus_mult = ENDGAME_ALIVE_BONUS_MULTIPLIER if endgame else 1.0
    stat_stage_mult = ENDGAME_STAT_STAGE_MULTIPLIER if endgame else 1.0
    hazard_mult = ENDGAME_HAZARD_MULTIPLIER if endgame else 1.0

    # Alive count advantage (amplified in the endgame to force KO lines)
    alive_diff = our_alive - opp_alive
    score += alive_diff * WEIGHT_ALIVE_BONUS * alive_bonus_mult

    # ── Active Pokemon stat stages (halved in the endgame -- no more setup) ─
    score += _score_stat_stages(getattr(battle, "active_pokemon", None), multiplier=1.0 * stat_stage_mult)
    score += _score_stat_stages(getattr(battle, "opponent_active_pokemon", None), multiplier=-1.0 * stat_stage_mult)

    # ── Status conditions ───────────────────────────────────────────────
    score += _score_team_statuses(our_team, multiplier=1.0)
    score += _score_team_statuses(opp_team, multiplier=-1.0)

    # ── Entry hazards (zeroed in the endgame -- no future switch-ins to punish) ─
    # Hazards on the OPPONENT's side benefit us (positive)
    score += _score_hazards(getattr(battle, "opponent_side_conditions", {}) or {}, multiplier=1.0 * hazard_mult)
    # Hazards on OUR side hurt us (negative)
    score += _score_hazards(getattr(battle, "side_conditions", {}) or {}, multiplier=-1.0 * hazard_mult)

    # ── Screens ─────────────────────────────────────────────────────────
    score += _score_screens(getattr(battle, "side_conditions", {}) or {}, multiplier=1.0)
    score += _score_screens(getattr(battle, "opponent_side_conditions", {}) or {}, multiplier=-1.0)

    # ── Tailwind ─────────────────────────────────────────────────────────
    score += _score_tailwind(getattr(battle, "side_conditions", {}) or {}, multiplier=1.0)
    score += _score_tailwind(getattr(battle, "opponent_side_conditions", {}) or {}, multiplier=-1.0)

    return score


# =============================================================================
# Evaluation sub-components
# =============================================================================
def _team_hp_summary(team: dict) -> tuple[float, int, int]:
    """
    Summarise a team's HP state.

    Parameters
    ----------
    team : dict
        {identifier: Pokemon} mapping from poke-env.

    Returns
    -------
    (hp_total, alive_count, total_count)
        hp_total: sum of current_hp_fraction for all Pokemon (0-6 scale)
        alive_count: number of non-fainted Pokemon
        total_count: total Pokemon on team
    """
    hp_total = 0.0
    alive = 0
    total = 0

    for mon in team.values():
        total += 1
        is_fainted = getattr(mon, "fainted", False)
        hp_frac = getattr(mon, "current_hp_fraction", 0.0) or 0.0
        if not is_fainted and hp_frac > 0:
            alive += 1
            hp_total += hp_frac
    return hp_total, alive, total


def _score_stat_stages(pokemon, multiplier: float = 1.0) -> float:
    """
    Score the stat stage boosts/drops on an active Pokemon.

    Parameters
    ----------
    pokemon : Pokemon or None
        The active Pokemon (ours or opponent's).
    multiplier : float
        +1.0 for our side, -1.0 for opponent's side.

    Returns
    -------
    float
        Score contribution from stat stages.
    """
    if pokemon is None:
        return 0.0

    boosts = getattr(pokemon, "boosts", None)
    if not boosts:
        return 0.0

    stage_weights = {
        "atk": WEIGHT_ATK_STAGE,
        "spa": WEIGHT_SPA_STAGE,
        "def": WEIGHT_DEF_STAGE,
        "spd": WEIGHT_SPD_STAGE,
        "spe": WEIGHT_SPE_STAGE,
        "evasion": WEIGHT_EVASION_STAGE,
        "accuracy": WEIGHT_ACCURACY_STAGE,
    }

    score = 0.0
    for stat, weight in stage_weights.items():
        stage = boosts.get(stat, 0)
        score += stage * weight

    return score * multiplier


def _score_team_statuses(team: dict, multiplier: float = 1.0) -> float:
    """
    Penalise a team for status conditions on its members.

    Parameters
    ----------
    team : dict
        {identifier: Pokemon} mapping from poke-env.
    multiplier : float
        +1.0 for our team (penalties are negative -> hurts our score).
        -1.0 for opponent's team (their penalties become our gain).

    Returns
    -------
    float
        Score contribution from statuses.
    """
    score = 0.0
    for mon in team.values():
        if getattr(mon, "fainted", False):
            continue
        status = getattr(mon, "status", None)
        if status is not None:
            status_name = status.name if hasattr(status, "name") else str(status)
            penalty = STATUS_PENALTIES.get(status_name, 0.0)
            score += penalty

    return score * multiplier


def _score_hazards(side_conditions: dict, multiplier: float = 1.0) -> float:
    """
    Score entry hazards on a side.

    Parameters
    ----------
    side_conditions : dict
        {SideCondition: turn_count_or_layers} from poke-env.
    multiplier : float
        +1.0 if these hazards are on the OPPONENT's side (good for us).
        -1.0 if on OUR side (bad for us).

    Returns
    -------
    float
        Score contribution from hazards.
    """
    score = 0.0
    for condition, value in side_conditions.items():
        raw = condition.name if hasattr(condition, "name") else str(condition)
        cond_name = raw.upper().replace("-", "_").replace(" ", "_")
        if cond_name == "STEALTHROCK":
            cond_name = "STEALTH_ROCK"
        elif cond_name == "TOXICSPIKES":
            cond_name = "TOXIC_SPIKES"
        elif cond_name == "STICKYWEB":
            cond_name = "STICKY_WEB"

        if cond_name in HAZARD_VALUES:
            base_val = HAZARD_VALUES[cond_name]
            # Spikes and Toxic Spikes stack (value = layer count)
            if cond_name in ("SPIKES", "TOXIC_SPIKES") and isinstance(value, int):
                score += base_val * value
            else:
                score += base_val
    return score * multiplier


def _score_screens(side_conditions: dict, multiplier: float = 1.0) -> float:
    """Score screens (Reflect, Light Screen, Aurora Veil) on a side."""
    score = 0.0
    for condition in side_conditions:
        raw = condition.name if hasattr(condition, "name") else str(condition)
        cond_name = raw.upper().replace("-", "_").replace(" ", "_")
        if cond_name == "LIGHTSCREEN":
            cond_name = "LIGHT_SCREEN"
        elif cond_name == "AURORAVEIL":
            cond_name = "AURORA_VEIL"

        if cond_name in SCREEN_VALUES:
            score += SCREEN_VALUES[cond_name]
    return score * multiplier


def _score_tailwind(side_conditions: dict, multiplier: float = 1.0) -> float:
    """Score Tailwind on a side."""
    for condition in side_conditions:
        raw = condition.name if hasattr(condition, "name") else str(condition)
        cond_name = raw.upper().replace("-", "_").replace(" ", "_")
        if cond_name == "TAILWIND":
            return WEIGHT_TAILWIND * multiplier
    return 0.0


# =============================================================================
# Payoff Matrix Generator
# =============================================================================
def generate_payoff_matrix(
    battle,
    opponent_moves: Optional[list[str]] = None,
    smogon_priors=None,
) -> PayoffMatrix:
    """
    Build a 2D payoff matrix for the current turn.

    Parameters
    ----------
    battle : Battle
        The current poke-env Battle object.
    opponent_moves : list[str] or None
        Override for opponent's likely moves. If None, uses the
        opponent's revealed moves or falls back to the top 4 Smogon
        prior moves via smogon_priors.
    smogon_priors : SmogonPriors or None
        The Smogon priors engine (Phase 2).

    Returns
    -------
    PayoffMatrix
        A matrix[i][j] of ActionOutcome for each (our_action, opp_action) pair.
    """
    active = getattr(battle, "active_pokemon", None)
    opp = getattr(battle, "opponent_active_pokemon", None)

    if active is None or opp is None:
        return PayoffMatrix()

    # ── Gather our actions ──────────────────────────────────────────────
    our_actions: list[dict] = []

    # Available moves
    available_moves = getattr(battle, "available_moves", []) or []
    for move in available_moves:
        our_actions.append({
            "label": move.id,
            "type": "move",
            "move": move,
        })

    # Available switches
    available_switches = getattr(battle, "available_switches", []) or []
    for mon in available_switches:
        our_actions.append({
            "label": f"switch:{mon.species}",
            "type": "switch",
            "pokemon": mon,
        })

    if not our_actions:
        return PayoffMatrix()

    # ── Gather opponent's likely actions ─────────────────────────────────
    opp_action_list: list[dict] = []

    if opponent_moves:
        # Explicit override
        for move_id in opponent_moves:
            norm_id = move_id.lower().replace(" ", "").replace("-", "")
            move_data = _MOVES_DB.get(norm_id)
            if move_data:
                opp_action_list.append({
                    "label": move_id,
                    "type": "move",
                    "move_data": move_data,
                    "move_id": norm_id,
                })
    else:
        # Use revealed moves first
        opp_moves = getattr(opp, "moves", {}) or {}
        if opp_moves:
            for move_id, move_obj in opp_moves.items():
                norm_id = move_id.lower().replace(" ", "").replace("-", "")
                opp_action_list.append({
                    "label": move_id,
                    "type": "move",
                    "move_data": _MOVES_DB.get(norm_id),
                    "move_id": norm_id,
                })

        # Supplement with Smogon priors if we have fewer than 4 moves
        if len(opp_action_list) < 4 and smogon_priors is not None:
            build = smogon_priors.get_likely_build(opp.species)
            if build:
                known_ids = {a["move_id"] for a in opp_action_list}
                for mp in build.top_moves:
                    mid = mp.name.lower().replace(" ", "").replace("-", "")
                    if mid not in known_ids and len(opp_action_list) < 4:
                        mdata = _MOVES_DB.get(mid)
                        if mdata:
                            opp_action_list.append({
                                "label": mp.name,
                                "type": "move",
                                "move_data": mdata,
                                "move_id": mid,
                            })
                            known_ids.add(mid)

    # Fallback: if we still have no opponent moves, assume a generic attack
    if not opp_action_list:
        tackle_data = _MOVES_DB.get("tackle")
        if tackle_data:
            opp_action_list.append({
                "label": "tackle",
                "type": "move",
                "move_data": tackle_data,
                "move_id": "tackle",
            })

    # ── Resolve types ───────────────────────────────────────────────────
    our_types = _get_pokemon_types(active)
    opp_types = _get_pokemon_types(opp)

    # ── Build the matrix ────────────────────────────────────────────────
    matrix = PayoffMatrix(
        our_actions=[a["label"] for a in our_actions],
        opp_actions=[a["label"] for a in opp_action_list],
    )

    base_score = evaluate_state(battle)

    for our_action in our_actions:
        row: list[ActionOutcome] = []
        for opp_action in opp_action_list:
            outcome = _evaluate_action_pair(
                battle=battle,
                our_action=our_action,
                opp_action=opp_action,
                our_types=our_types,
                opp_types=opp_types,
                base_score=base_score,
            )
            row.append(outcome)
        matrix.matrix.append(row)

    return matrix


def _evaluate_action_pair(
    battle,
    our_action: dict,
    opp_action: dict,
    our_types: list[str],
    opp_types: list[str],
    base_score: float,
) -> ActionOutcome:
    """
    Estimate the resulting evaluation after both sides act simultaneously.

    This is a one-ply forward estimation: we approximate the score delta
    from damage dealt/received without full state simulation.

    Parameters
    ----------
    battle : Battle
        Current battle state.
    our_action : dict
        Our action descriptor.
    opp_action : dict
        Opponent's action descriptor.
    our_types : list[str]
        Our active Pokemon's types.
    opp_types : list[str]
        Opponent active Pokemon's types.
    base_score : float
        Current evaluation score (before actions).

    Returns
    -------
    ActionOutcome
        The scored outcome for this action pair.
    """
    active = getattr(battle, "active_pokemon", None)
    opp = getattr(battle, "opponent_active_pokemon", None)

    delta = 0.0
    dmg_to_opp = 0.0
    dmg_to_us = 0.0
    notes_parts: list[str] = []

    # ── Our action ──────────────────────────────────────────────────────
    if our_action["type"] == "move":
        move = our_action["move"]
        d2o = _estimate_move_damage(
            attacker=active,
            defender=opp,
            move=move,
            attacker_types=our_types,
            defender_types=opp_types,
        )
        dmg_to_opp = d2o["avg_damage"]
        delta += d2o["score_delta"]
        if d2o["notes"]:
            notes_parts.append(f"our {move.id}: {d2o['notes']}")

    elif our_action["type"] == "switch":
        # Switching costs a turn of momentum but repositions
        delta -= 30.0  # tempo cost of switching

    # ── Opponent's action ───────────────────────────────────────────────
    if opp_action["type"] == "move" and opp_action.get("move_data"):
        d2u = _estimate_opp_move_damage(
            attacker=opp,
            defender=active,
            move_data=opp_action["move_data"],
            move_id=opp_action.get("move_id", ""),
            attacker_types=opp_types,
            defender_types=our_types,
        )
        dmg_to_us = d2u["avg_damage"]
        delta -= d2u["score_delta"]
        if d2u["notes"]:
            notes_parts.append(f"opp {opp_action['label']}: {d2u['notes']}")

    return ActionOutcome(
        our_action=our_action["label"],
        opp_action=opp_action["label"],
        expected_score=base_score + delta,
        damage_to_opp=dmg_to_opp,
        damage_to_us=dmg_to_us,
        notes="; ".join(notes_parts),
    )


def _estimate_move_damage(
    attacker,
    defender,
    move,
    attacker_types: list[str],
    defender_types: list[str],
) -> dict:
    """
    Estimate damage from our move to the opponent.

    Uses the forward damage calculator with best-available stat info.

    Returns
    -------
    dict with keys:
        avg_damage : float  -- average damage (HP points)
        score_delta : float -- evaluation score delta (positive = good for us)
        notes : str         -- human-readable annotation
    """
    result = {"avg_damage": 0.0, "score_delta": 0.0, "notes": ""}

    # Status moves get a flat bonus if useful (simplified heuristic)
    category = getattr(move, "category", "Physical")
    if category == "Status":
        result["score_delta"] = _score_status_move(move, defender)
        result["notes"] = "status"
        return result

    base_power = getattr(move, "base_power", 0)
    if base_power == 0:
        return result

    move_type = move.type.name if hasattr(move.type, "name") else str(move.type)

    # Our stats (known exactly)
    our_stats = getattr(attacker, "stats", None) or {}
    if category == "Physical":
        atk_stat = our_stats.get("atk", 200)
    else:
        atk_stat = our_stats.get("spa", 200)

    # Opponent's defensive stat (estimated)
    opp_max_hp = _estimate_max_hp(defender)
    if category == "Physical":
        def_stat = _estimate_def_stat(defender)
    else:
        def_stat = _estimate_spd_stat(defender)

    # Apply boosts
    our_boosts = getattr(attacker, "boosts", {}) or {}
    opp_boosts = getattr(defender, "boosts", {}) or {}

    atk_stage = our_boosts.get("atk", 0) if category == "Physical" else our_boosts.get("spa", 0)
    def_stage = opp_boosts.get("def", 0) if category == "Physical" else opp_boosts.get("spd", 0)

    type_eff = get_type_effectiveness(move_type, defender_types)
    stab = has_stab(move_type, attacker_types)

    if type_eff == 0:
        result["notes"] = "immune"
        return result

    dmg_range = calculate_damage_range(
        attacker_stat=atk_stat,
        defender_stat=def_stat,
        base_power=base_power,
        type_effectiveness=type_eff,
        stab=stab,
        move_category=category,
        stat_stage_atk=atk_stage,
        stat_stage_def=def_stage,
    )

    avg_dmg = (dmg_range.min_damage + dmg_range.max_damage) / 2
    result["avg_damage"] = avg_dmg

    # Convert damage to score delta
    if opp_max_hp > 0:
        dmg_fraction = avg_dmg / opp_max_hp
        result["score_delta"] = dmg_fraction * WEIGHT_HP_RATIO

        # Check for KO potential
        opp_current_hp = _estimate_current_hp(defender)
        if avg_dmg >= opp_current_hp:
            result["score_delta"] += WEIGHT_ALIVE_BONUS
            result["notes"] = "potential KO"
        elif dmg_range.min_damage >= opp_current_hp:
            result["score_delta"] += WEIGHT_ALIVE_BONUS * 1.5
            result["notes"] = "guaranteed KO"

    # Annotate effectiveness
    if type_eff >= 2.0:
        result["notes"] = (result["notes"] + " SE" if result["notes"] else "SE")
    elif type_eff <= 0.5:
        result["notes"] = (result["notes"] + " NVE" if result["notes"] else "NVE")

    # Move accuracy penalty (expected value adjustment)
    accuracy = getattr(move, "accuracy", 100)
    if isinstance(accuracy, (int, float)) and accuracy < 100:
        acc_mult = accuracy / 100.0
        result["score_delta"] *= acc_mult
        result["avg_damage"] *= acc_mult

    return result


def _estimate_opp_move_damage(
    attacker,
    defender,
    move_data: dict,
    move_id: str,
    attacker_types: list[str],
    defender_types: list[str],
) -> dict:
    """
    Estimate damage from opponent's move to our Pokemon.

    Returns
    -------
    dict with keys:
        avg_damage : float
        score_delta : float (positive = damage to us, caller will negate)
        notes : str
    """
    result = {"avg_damage": 0.0, "score_delta": 0.0, "notes": ""}

    category = move_data.get("category", "Physical")
    if category == "Status":
        result["score_delta"] = 20.0  # Generic penalty for opp using a status move
        result["notes"] = "status"
        return result

    base_power = move_data.get("basePower", 0)
    if base_power == 0:
        return result

    move_type = move_data.get("type", "Normal")

    # Opponent's attack stat (estimated)
    if category == "Physical":
        atk_stat = _estimate_atk_stat(attacker)
    else:
        atk_stat = _estimate_spa_stat(attacker)

    # Our defensive stat (known exactly)
    our_stats = getattr(defender, "stats", None) or {}
    if category == "Physical":
        def_stat = our_stats.get("def", 200)
    else:
        def_stat = our_stats.get("spd", 200)

    our_max_hp = getattr(defender, "max_hp", 300) or 300

    # Apply boosts
    opp_boosts = getattr(attacker, "boosts", {}) or {}
    our_boosts = getattr(defender, "boosts", {}) or {}

    atk_stage = opp_boosts.get("atk", 0) if category == "Physical" else opp_boosts.get("spa", 0)
    def_stage = our_boosts.get("def", 0) if category == "Physical" else our_boosts.get("spd", 0)

    type_eff = get_type_effectiveness(move_type, defender_types)
    stab = has_stab(move_type, attacker_types)

    if type_eff == 0:
        result["notes"] = "we're immune"
        return result

    dmg_range = calculate_damage_range(
        attacker_stat=atk_stat,
        defender_stat=def_stat,
        base_power=base_power,
        type_effectiveness=type_eff,
        stab=stab,
        move_category=category,
        stat_stage_atk=atk_stage,
        stat_stage_def=def_stage,
    )

    avg_dmg = (dmg_range.min_damage + dmg_range.max_damage) / 2
    result["avg_damage"] = avg_dmg

    if our_max_hp > 0:
        dmg_fraction = avg_dmg / our_max_hp
        result["score_delta"] = dmg_fraction * WEIGHT_HP_RATIO

        our_current_hp = getattr(defender, "current_hp", our_max_hp)
        if our_current_hp is not None and avg_dmg >= our_current_hp:
            result["score_delta"] += WEIGHT_ALIVE_BONUS
            result["notes"] = "we may faint"
        elif our_current_hp is not None and dmg_range.min_damage >= our_current_hp:
            result["score_delta"] += WEIGHT_ALIVE_BONUS * 1.5
            result["notes"] = "we will faint"

    if type_eff >= 2.0:
        result["notes"] = (result["notes"] + " SE" if result["notes"] else "SE")
    elif type_eff <= 0.5:
        result["notes"] = (result["notes"] + " NVE" if result["notes"] else "NVE")

    # Accuracy adjustment
    accuracy = move_data.get("accuracy", 100)
    if isinstance(accuracy, (int, float)) and accuracy < 100:
        acc_mult = accuracy / 100.0
        result["score_delta"] *= acc_mult
        result["avg_damage"] *= acc_mult

    return result


# =============================================================================
# Stat estimation helpers (for opponent Pokemon with unknown stats)
# =============================================================================
def _get_pokemon_types(pokemon) -> list[str]:
    """Extract type names from a Pokemon object."""
    types = getattr(pokemon, "types", None)
    if types is None:
        return []
    return [t.name if hasattr(t, "name") else str(t)
            for t in types if t is not None]


def _estimate_max_hp(pokemon) -> int:
    """Estimate a Pokemon's max HP from available info."""
    if hasattr(pokemon, "max_hp") and pokemon.max_hp:
        return pokemon.max_hp
    return _base_stat_estimate(pokemon, "hp", default=300)


def _estimate_current_hp(pokemon) -> int:
    """Estimate a Pokemon's current HP."""
    if hasattr(pokemon, "current_hp") and pokemon.current_hp is not None:
        return pokemon.current_hp
    max_hp = _estimate_max_hp(pokemon)
    frac = getattr(pokemon, "current_hp_fraction", 1.0) or 1.0
    return int(max_hp * frac)


def _estimate_atk_stat(pokemon) -> int:
    """Estimate opponent's Attack stat."""
    return _base_stat_estimate(pokemon, "atk", default=250)


def _estimate_spa_stat(pokemon) -> int:
    """Estimate opponent's Special Attack stat."""
    return _base_stat_estimate(pokemon, "spa", default=250)


def _estimate_def_stat(pokemon) -> int:
    """Estimate opponent's Defense stat."""
    return _base_stat_estimate(pokemon, "def", default=200)


def _estimate_spd_stat(pokemon) -> int:
    """Estimate opponent's Special Defense stat."""
    return _base_stat_estimate(pokemon, "spd", default=200)


def _base_stat_estimate(pokemon, stat_key: str, default: int = 200) -> int:
    """
    Estimate a stat from base stats assuming 84 EVs and neutral nature.
    """
    species = getattr(pokemon, "species", None)
    if species is None:
        return default

    key = species.lower().replace(" ", "").replace("-", "")
    mon = _POKEDEX.get(key)
    if mon is None:
        return default

    base = mon.get("baseStats", {}).get(stat_key)
    if base is None:
        return default

    from inverse_damage_calc import calc_stat
    if stat_key == "hp":
        return calc_stat(base, ev=84, iv=31, is_hp=True)
    else:
        return calc_stat(base, ev=84, iv=31, nature_mult=1.0)


def _score_status_move(move, defender) -> float:
    """
    Rough heuristic value for using a status move.
    """
    move_id = getattr(move, "id", "").lower().replace(" ", "").replace("-", "")

    # Hazard-setting moves
    if move_id in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
        return 60.0

    # Recovery moves
    if move_id in ("recover", "roost", "softboiled", "slackoff", "moonlight",
                   "morningsun", "synthesis", "shoreup", "milkdrink"):
        return 50.0

    # Status-inflicting moves
    if move_id in ("thunderwave", "willowisp", "toxic", "spore", "sleeppowder",
                   "hypnosis", "stunspore", "glare", "yawn"):
        opp_status = getattr(defender, "status", None)
        if opp_status is not None:
            return -10.0  # Wasted turn
        return 70.0

    # Boosting moves
    if move_id in ("swordsdance", "nastyplot", "dragondance", "quiverdance",
                   "calmmind", "bulkup", "irondefense", "amnesia",
                   "agility", "shellsmash", "tailglow", "coil",
                   "victorydance"):
        return 55.0

    # Screens
    if move_id in ("reflect", "lightscreen", "auroraveil"):
        return 45.0

    # Defog / Rapid Spin
    if move_id in ("defog", "rapidspin", "courtchange", "tidyup",
                   "mortalspin"):
        return 40.0

    # Pivot moves
    if move_id in ("teleport", "partingshot", "batonpass"):
        return 30.0

    return 15.0


# =============================================================================
# Standalone tests with mock battle objects
# =============================================================================
class _MockType:
    """Minimal mock for poke-env PokemonType."""
    def __init__(self, name: str):
        self.name = name


class _MockStatus:
    """Minimal mock for poke-env Status."""
    def __init__(self, name: str):
        self.name = name


from poke_env.battle.move import Move, MoveSet
from poke_env.battle.pokemon import Pokemon


class _MockMove(Move):
    """Minimal mock for poke-env Move."""
    def __init__(self, move_id: str):
        super().__init__(move_id.lower().replace(" ", "").replace("-", ""), gen=9)


class _MockPokemon(Pokemon):
    """Minimal mock for poke-env Pokemon."""
    def __init__(
        self,
        species: str,
        hp_fraction: float = 1.0,
        stats: Optional[dict] = None,
        types: Optional[list[str]] = None,
        boosts: Optional[dict] = None,
        status: Optional[str] = None,
        fainted: bool = False,
        max_hp: int = 300,
        moves: Optional[dict] = None,
    ):
        super().__init__(species=species, gen=9)
        self._current_hp_fraction = hp_fraction
        self._current_hp = int(max_hp * hp_fraction) if not fainted else 0
        self._max_hp = max_hp
        if stats:
            self._stats = stats
        if boosts:
            self._boosts = boosts
        if status:
            self._status = _MockStatus(status)
        self._fainted = fainted
        if moves:
            # Pokemon.moves (poke-env >=0.9) reads self._moves.moves, i.e.
            # self._moves must be a MoveSet wrapper, NOT a raw dict -- mirror
            # exactly how the real Pokemon class populates it internally
            # (see Pokemon._add_move: `self._moves[move.id] = move`).
            self._moves = MoveSet({})
            for move_id, move_obj in moves.items():
                self._moves[move_id] = move_obj


class _MockSideCondition:
    """Minimal mock for poke-env SideCondition."""
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hasattr(other, "name") and self.name == other.name


class _MockBattle:
    """Minimal mock for poke-env Battle."""
    def __init__(
        self,
        our_team: dict,
        opp_team: dict,
        active_pokemon=None,
        opponent_active_pokemon=None,
        side_conditions: Optional[dict] = None,
        opponent_side_conditions: Optional[dict] = None,
        available_moves: Optional[list] = None,
        available_switches: Optional[list] = None,
        won: Optional[bool] = None,
        turn: int = 1,
        weather: Optional[dict] = None,
    ):
        self.team = our_team
        self.opponent_team = opp_team
        self.active_pokemon = active_pokemon
        self.opponent_active_pokemon = opponent_active_pokemon
        self.side_conditions = side_conditions or {}
        self.opponent_side_conditions = opponent_side_conditions or {}
        self.available_moves = available_moves or []
        self.available_switches = available_switches or []
        self.won = won
        self.turn = turn
        self.weather = weather or {}
        self.battle_tag = "test-battle"


def _run_tests():
    """Comprehensive test suite for the evaluator and payoff matrix."""

    print("=" * 72)
    print("  Phase 4A: State Evaluator & Payoff Matrix -- Test Suite")
    print("=" * 72)

    # ── Test 1: Terminal states ──────────────────────────────────────────
    print("\n--- Test 1: Terminal States ---")

    win_battle = _MockBattle(
        our_team={}, opp_team={}, won=True,
    )
    loss_battle = _MockBattle(
        our_team={}, opp_team={}, won=False,
    )
    print(f"  Win  -> {evaluate_state(win_battle):>+10,.1f}  (expected +10,000)")
    print(f"  Loss -> {evaluate_state(loss_battle):>+10,.1f}  (expected -10,000)")
    assert evaluate_state(win_battle) == SCORE_WIN
    assert evaluate_state(loss_battle) == SCORE_LOSS

    # ── Test 2: Even matchup ────────────────────────────────────────────
    print("\n--- Test 2: Perfectly Even Matchup ---")

    garchomp = _MockPokemon(
        "garchomp", hp_fraction=1.0, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
    )
    dragapult = _MockPokemon(
        "dragapult", hp_fraction=1.0, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
    )

    even_battle = _MockBattle(
        our_team={"p1: Garchomp": garchomp},
        opp_team={"p2: Dragapult": dragapult},
        active_pokemon=garchomp,
        opponent_active_pokemon=dragapult,
    )
    even_score = evaluate_state(even_battle)
    print(f"  1v1 full HP -> {even_score:>+.1f}  (expected ~0.0)")

    # ── Test 3: HP advantage ────────────────────────────────────────────
    print("\n--- Test 3: HP Advantage ---")

    garchomp_hurt = _MockPokemon(
        "garchomp", hp_fraction=0.5, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
    )
    advantage_battle = _MockBattle(
        our_team={"p1: Garchomp": garchomp},
        opp_team={"p2: Dragapult": garchomp_hurt},
        active_pokemon=garchomp,
        opponent_active_pokemon=garchomp_hurt,
    )
    adv_score = evaluate_state(advantage_battle)
    print(f"  Us 100% vs Opp 50% -> {adv_score:>+.1f}  (expected positive)")
    assert adv_score > 0, f"Expected positive, got {adv_score}"

    # ── Test 4: Stat boosts ─────────────────────────────────────────────
    print("\n--- Test 4: Stat Boosts ---")

    boosted_chomp = _MockPokemon(
        "garchomp", hp_fraction=1.0, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
        boosts={"atk": 2, "spe": 1},
    )
    boosted_battle = _MockBattle(
        our_team={"p1: Garchomp": boosted_chomp},
        opp_team={"p2: Dragapult": dragapult},
        active_pokemon=boosted_chomp,
        opponent_active_pokemon=dragapult,
    )
    boost_score = evaluate_state(boosted_battle)
    print(f"  +2 Atk / +1 Spe on our side -> {boost_score:>+.1f}  (expected > even)")
    assert boost_score > even_score

    # ── Test 5: Status conditions ───────────────────────────────────────
    print("\n--- Test 5: Status Conditions ---")

    burned_opp = _MockPokemon(
        "dragapult", hp_fraction=1.0, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
        status="BRN",
    )
    burn_battle = _MockBattle(
        our_team={"p1: Garchomp": garchomp},
        opp_team={"p2: Dragapult": burned_opp},
        active_pokemon=garchomp,
        opponent_active_pokemon=burned_opp,
    )
    burn_score = evaluate_state(burn_battle)
    print(f"  Opponent burned -> {burn_score:>+.1f}  (expected positive)")
    assert burn_score > even_score

    # ── Test 6: Entry hazards ───────────────────────────────────────────
    print("\n--- Test 6: Entry Hazards ---")

    sr_on_opp = _MockSideCondition("STEALTH_ROCK")
    spikes_on_opp = _MockSideCondition("SPIKES")

    # NOTE: opponent needs > ENDGAME_OPP_ALIVE_THRESHOLD mons alive here, or
    # the Phase 5.2 endgame anti-choke logic correctly zeroes hazard value
    # (see TestEndgameHeuristic in test_suite.py for that behavior).
    opp_filler_1 = _MockPokemon("magnezone", hp_fraction=1.0, max_hp=250,
                                 stats={"atk": 150, "def": 200, "spa": 300, "spd": 200, "spe": 180},
                                 types=["Electric", "Steel"])
    opp_filler_2 = _MockPokemon("clefable", hp_fraction=1.0, max_hp=280,
                                 stats={"atk": 150, "def": 220, "spa": 220, "spd": 260, "spe": 170},
                                 types=["Fairy"])
    our_filler_1 = _MockPokemon("corviknight", hp_fraction=1.0, max_hp=341,
                                 stats={"atk": 262, "def": 309, "spa": 137, "spd": 206, "spe": 170},
                                 types=["Flying", "Steel"])
    our_filler_2 = _MockPokemon("toxapex", hp_fraction=1.0, max_hp=304,
                                 stats={"atk": 152, "def": 353, "spa": 137, "spd": 293, "spe": 96},
                                 types=["Poison", "Water"])
    hazard_battle = _MockBattle(
        our_team={"p1: Garchomp": garchomp, "p1: Corviknight": our_filler_1, "p1: Toxapex": our_filler_2},
        opp_team={"p2: Dragapult": dragapult, "p2: Magnezone": opp_filler_1, "p2: Clefable": opp_filler_2},
        active_pokemon=garchomp,
        opponent_active_pokemon=dragapult,
        opponent_side_conditions={sr_on_opp: 1, spikes_on_opp: 2},
    )
    hazard_score = evaluate_state(hazard_battle)
    print(f"  SR + 2 layers Spikes on opp -> {hazard_score:>+.1f}  (expected positive)")
    assert hazard_score > even_score

    # ── Test 7: Multi-Pokemon teams ─────────────────────────────────────
    print("\n--- Test 7: Multi-Pokemon Teams (3v2 alive advantage) ---")

    mon1 = _MockPokemon("garchomp", hp_fraction=1.0, max_hp=357,
                         stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
                         types=["Dragon", "Ground"])
    mon2 = _MockPokemon("toxapex", hp_fraction=0.8, max_hp=304,
                         stats={"atk": 152, "def": 353, "spa": 137, "spd": 293, "spe": 96},
                         types=["Poison", "Water"])
    mon3 = _MockPokemon("corviknight", hp_fraction=0.6, max_hp=341,
                         stats={"atk": 262, "def": 309, "spa": 137, "spd": 206, "spe": 170},
                         types=["Flying", "Steel"])

    opp1 = _MockPokemon("dragapult", hp_fraction=0.9, max_hp=291,
                          stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
                          types=["Dragon", "Ghost"])
    opp2 = _MockPokemon("ironvaliant", hp_fraction=0.4, max_hp=291,
                          stats={"atk": 394, "def": 226, "spa": 339, "spd": 226, "spe": 333},
                          types=["Fairy", "Fighting"])

    team_battle = _MockBattle(
        our_team={"p1a": mon1, "p1b": mon2, "p1c": mon3},
        opp_team={"p2a": opp1, "p2b": opp2},
        active_pokemon=mon1,
        opponent_active_pokemon=opp1,
    )
    team_score = evaluate_state(team_battle)
    print(f"  3v2 with HP advantage -> {team_score:>+.1f}  (expected strongly positive)")
    assert team_score > 0

    # ── Test 8: Payoff matrix generation ────────────────────────────────
    print("\n--- Test 8: Payoff Matrix ---")

    eq_move = _MockMove("earthquake")
    sd_move = _MockMove("swordsdance")
    dc_move = _MockMove("dragonclaw")
    sr_move = _MockMove("stealthrock")

    opp_db_move = _MockMove("dracometeor")
    opp_sb_move = _MockMove("shadowball")

    matrix_chomp = _MockPokemon(
        "garchomp", hp_fraction=1.0, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
    )
    matrix_draga = _MockPokemon(
        "dragapult", hp_fraction=1.0, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
        moves={
            "dracometeor": opp_db_move,
            "shadowball": opp_sb_move,
        },
    )

    matrix_battle = _MockBattle(
        our_team={"p1: Garchomp": matrix_chomp},
        opp_team={"p2: Dragapult": matrix_draga},
        active_pokemon=matrix_chomp,
        opponent_active_pokemon=matrix_draga,
        available_moves=[eq_move, dc_move, sd_move, sr_move],
        available_switches=[],
    )

    payoff = generate_payoff_matrix(matrix_battle)

    print(f"\n  Our actions:  {payoff.our_actions}")
    print(f"  Opp actions:  {payoff.opp_actions}")
    print(f"\n{payoff.display()}\n")

    # Maximin
    maxi_action, maxi_score = payoff.best_action_maximin()
    print(f"  Maximin action: {maxi_action} (worst-case score: {maxi_score:.1f})")

    # Expected value (uniform opponent)
    ev_action, ev_score = payoff.best_action_expected()
    print(f"  EV-optimal action (uniform): {ev_action} (EV: {ev_score:.1f})")

    # ── Test 9: Switch included in matrix ───────────────────────────────
    print("\n--- Test 9: Matrix with Switches ---")

    switch_tox = _MockPokemon(
        "toxapex", hp_fraction=1.0, max_hp=304,
        stats={"atk": 152, "def": 353, "spa": 137, "spd": 293, "spe": 96},
        types=["Poison", "Water"],
    )
    switch_battle = _MockBattle(
        our_team={"p1: Garchomp": matrix_chomp, "p1: Toxapex": switch_tox},
        opp_team={"p2: Dragapult": matrix_draga},
        active_pokemon=matrix_chomp,
        opponent_active_pokemon=matrix_draga,
        available_moves=[eq_move, dc_move],
        available_switches=[switch_tox],
    )
    switch_payoff = generate_payoff_matrix(switch_battle)
    print(f"  Our actions:  {switch_payoff.our_actions}")
    print(f"  Opp actions:  {switch_payoff.opp_actions}")
    print(f"\n{switch_payoff.display()}\n")

    sw_action, sw_score = switch_payoff.best_action_maximin()
    print(f"  Maximin: {sw_action} ({sw_score:.1f})")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  All Phase 4A tests passed [OK]")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    _run_tests()
