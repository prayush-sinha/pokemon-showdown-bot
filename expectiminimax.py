"""
expectiminimax.py -- Phase 4B: Simultaneous Expectiminimax Search Engine

A game-theoretic decision engine for Pokemon Showdown that accounts for:
  1. Simultaneous move selection (Nash / EV over opponent move distribution).
  2. Incomplete information (Smogon priors + inverse damage calc profiles).
  3. Stochastic execution (accuracy hit/miss chance, speed tiers, damage rolls).
  4. Lookahead tree search (depth-limited forward simulation).

Usage
-----
    from expectiminimax import ExpectiminimaxEngine
    engine = ExpectiminimaxEngine(depth=2)
    best_obj, ranked = engine.get_best_action(battle)
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Union

from evaluator import (
    evaluate_state,
    _get_pokemon_types,
    _estimate_max_hp,
    _estimate_current_hp,
    _estimate_atk_stat,
    _estimate_spa_stat,
    _estimate_def_stat,
    _estimate_spd_stat,
    _score_status_move,
    SCORE_WIN,
    SCORE_LOSS,
    HAZARD_VALUES,
)
from inverse_damage_calc import (
    calculate_damage_range,
    get_type_effectiveness,
    has_stab,
    calc_all_stats,
    _apply_stat_stage,
    ITEM_MODIFIERS,
    _POKEDEX,
    _MOVES_DB,
    _TYPE_CHART,
    _LEVEL,
)
from smogon_priors import SmogonPriors

logger = logging.getLogger("Expectiminimax")


# =============================================================================
# Lightweight State Representation for Forward Tree Search
# =============================================================================
@dataclass
class SimPokemon:
    """
    Lightweight, cloneable snapshot of a Pokemon for tree search.
    Implements the same attribute interface expected by evaluator.py.
    """
    species: str
    current_hp: int
    max_hp: int
    current_hp_fraction: float
    stats: dict[str, int] = field(default_factory=dict)
    types: tuple[Any, ...] = field(default_factory=tuple)
    boosts: dict[str, int] = field(default_factory=dict)
    status: Optional[Any] = None
    fainted: bool = False
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> SimPokemon:
        return SimPokemon(
            species=self.species,
            current_hp=self.current_hp,
            max_hp=self.max_hp,
            current_hp_fraction=self.current_hp_fraction,
            stats=dict(self.stats),
            types=self.types,
            boosts=dict(self.boosts),
            status=self.status,
            fainted=self.fainted,
            item=self.item,
            ability=self.ability,
            moves=dict(self.moves),
        )


@dataclass
class SimState:
    """
    Lightweight, cloneable battle state for forward simulation.
    Compatible with evaluate_state(sim_state).
    """
    team: dict[str, SimPokemon]
    opponent_team: dict[str, SimPokemon]
    active_pokemon: Optional[SimPokemon]
    opponent_active_pokemon: Optional[SimPokemon]
    side_conditions: dict[Any, Any] = field(default_factory=dict)
    opponent_side_conditions: dict[Any, Any] = field(default_factory=dict)
    weather: dict[Any, Any] = field(default_factory=dict)
    turn: int = 1
    won: Optional[bool] = None

    def clone(self) -> SimState:
        # Clone pokemon mappings
        new_team = {k: v.clone() for k, v in self.team.items()}
        new_opp_team = {k: v.clone() for k, v in self.opponent_team.items()}

        # Active references point to the cloned objects in the team dicts
        new_active = None
        if self.active_pokemon:
            for v in new_team.values():
                if v.species == self.active_pokemon.species:
                    new_active = v
                    break
            if new_active is None:
                new_active = self.active_pokemon.clone()

        new_opp_active = None
        if self.opponent_active_pokemon:
            for v in new_opp_team.values():
                if v.species == self.opponent_active_pokemon.species:
                    new_opp_active = v
                    break
            if new_opp_active is None:
                new_opp_active = self.opponent_active_pokemon.clone()

        return SimState(
            team=new_team,
            opponent_team=new_opp_team,
            active_pokemon=new_active,
            opponent_active_pokemon=new_opp_active,
            side_conditions=dict(self.side_conditions),
            opponent_side_conditions=dict(self.opponent_side_conditions),
            weather=dict(self.weather),
            turn=self.turn,
            won=self.won,
        )


# =============================================================================
# Action Representations
# =============================================================================
@dataclass
class ActionCandidate:
    """An action available to our bot."""
    action_type: str                  # "move" or "switch"
    label: str                        # e.g., "earthquake", "switch:toxapex"
    poke_env_object: Any              # The Move or Pokemon instance from poke-env
    move_id: Optional[str] = None
    switch_species: Optional[str] = None


@dataclass
class OpponentActionCandidate:
    """A projected action for the opponent with prior probability."""
    action_type: str                  # "move"
    label: str                        # e.g., "dracometeor"
    move_id: str
    move_data: dict
    probability: float = 0.25


# =============================================================================
# State Conversion Helpers
# =============================================================================
def _type_name(t: Any) -> str:
    """Extract string type name from a poke-env PokemonType or string."""
    if t is None:
        return "Normal"
    return t.name if hasattr(t, "name") else str(t)


class _NamedMockType:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


def pokemon_to_sim(mon, is_opponent: bool = False, profile: Optional[dict] = None) -> SimPokemon:
    """Convert a poke-env Pokemon object to a SimPokemon."""
    if mon is None:
        return SimPokemon(species="unknown", current_hp=0, max_hp=100, current_hp_fraction=0.0, fainted=True)

    species = getattr(mon, "species", "unknown")
    current_hp = getattr(mon, "current_hp", None)
    max_hp = getattr(mon, "max_hp", None)
    hp_frac = getattr(mon, "current_hp_fraction", 1.0)
    if hp_frac is None:
        hp_frac = 1.0

    if max_hp is None or max_hp <= 0:
        max_hp = _estimate_max_hp(mon)
    if current_hp is None:
        current_hp = int(max_hp * hp_frac)

    # Stats
    stats = dict(getattr(mon, "stats", None) or {})
    if not stats or not stats.get("atk"):
        try:
            resolved = calc_all_stats(species)
            stats = {
                "hp": resolved.hp,
                "atk": resolved.atk,
                "def": resolved.defense,
                "spa": resolved.spa,
                "spd": resolved.spd,
                "spe": resolved.spe,
            }
        except Exception:
            stats = {"hp": max_hp, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}

    # If inverse calc provided estimated attack stat or item, apply it
    item = getattr(mon, "item", None)
    if profile:
        if profile.get("item") and not item:
            item = profile.get("item")
        if profile.get("est_atk"):
            stats["atk"] = profile.get("est_atk")

    # Types
    types_raw = getattr(mon, "types", ()) or ()
    types_wrapped = tuple(_NamedMockType(_type_name(t)) for t in types_raw)

    # Boosts
    boosts = dict(getattr(mon, "boosts", None) or {})

    # Status
    status = getattr(mon, "status", None)

    # Fainted
    fainted = getattr(mon, "fainted", False) or current_hp <= 0

    return SimPokemon(
        species=species,
        current_hp=current_hp,
        max_hp=max_hp,
        current_hp_fraction=hp_frac if not fainted else 0.0,
        stats=stats,
        types=types_wrapped,
        boosts=boosts,
        status=status,
        fainted=fainted,
        item=item,
        ability=getattr(mon, "ability", None),
        moves=dict(getattr(mon, "moves", None) or {}),
    )


def battle_to_sim_state(battle, opponent_profiles: Optional[dict] = None) -> SimState:
    """Convert a poke-env Battle object into a cloneable SimState."""
    opponent_profiles = opponent_profiles or {}

    our_team: dict[str, SimPokemon] = {}
    for k, mon in (getattr(battle, "team", {}) or {}).items():
        our_team[k] = pokemon_to_sim(mon, is_opponent=False)

    opp_team: dict[str, SimPokemon] = {}
    for k, mon in (getattr(battle, "opponent_team", {}) or {}).items():
        species = getattr(mon, "species", "")
        prof = opponent_profiles.get(species)
        opp_team[k] = pokemon_to_sim(mon, is_opponent=True, profile=prof)

    active = pokemon_to_sim(
        getattr(battle, "active_pokemon", None),
        is_opponent=False
    )
    opp_active_species = getattr(getattr(battle, "opponent_active_pokemon", None), "species", "")
    opp_active = pokemon_to_sim(
        getattr(battle, "opponent_active_pokemon", None),
        is_opponent=True,
        profile=opponent_profiles.get(opp_active_species),
    )

    return SimState(
        team=our_team,
        opponent_team=opp_team,
        active_pokemon=active,
        opponent_active_pokemon=opp_active,
        side_conditions=dict(getattr(battle, "side_conditions", {}) or {}),
        opponent_side_conditions=dict(getattr(battle, "opponent_side_conditions", {}) or {}),
        weather=dict(getattr(battle, "weather", {}) or {}),
        turn=getattr(battle, "turn", 1),
        won=getattr(battle, "won", None),
    )


# =============================================================================
# Speed & Turn Order Logic
# =============================================================================
def _get_effective_speed(pokemon: Optional[SimPokemon], side_conditions: dict) -> int:
    """Compute effective speed considering stat stages, paralysis, and tailwind."""
    if pokemon is None:
        return 100

    base_spe = pokemon.stats.get("spe", 200)
    spe_stage = pokemon.boosts.get("spe", 0)
    spe = _apply_stat_stage(base_spe, spe_stage)

    # Paralysis cuts speed by 50%
    if pokemon.status:
        stat_name = pokemon.status.name if hasattr(pokemon.status, "name") else str(pokemon.status)
        if stat_name == "PAR":
            spe = math.floor(spe * 0.5)

    # Tailwind doubles speed
    for cond in side_conditions:
        cond_name = cond.name if hasattr(cond, "name") else str(cond)
        if cond_name == "TAILWIND":
            spe = spe * 2
            break

    return spe


def _get_move_priority(move_id: str, move_data: Optional[dict] = None) -> int:
    """Look up move priority from GenData moves database."""
    if move_data is None:
        norm_id = move_id.lower().replace(" ", "").replace("-", "")
        move_data = _MOVES_DB.get(norm_id, {})
    return move_data.get("priority", 0)


# =============================================================================
# Forward Simulation & Chance Node Transition
# =============================================================================
def _apply_stealth_rock(pokemon: SimPokemon) -> int:
    """Compute Stealth Rock entry damage on a switch-in."""
    rock_chart = _TYPE_CHART.get("ROCK", {})
    mult = 1.0
    for t in pokemon.types:
        t_name = _type_name(t).upper()
        mult *= rock_chart.get(t_name, 1.0)
    # Base is 12.5% (1/8)
    dmg = math.floor(pokemon.max_hp * (0.125 * mult))
    return max(1, dmg) if mult > 0 else 0


def _apply_spikes(pokemon: SimPokemon, layers: int) -> int:
    """Compute Spikes damage on a grounded switch-in."""
    # Flying types immune to spikes
    for t in pokemon.types:
        if _type_name(t).upper() == "FLYING":
            return 0
    fractions = {1: 0.125, 2: 0.1666, 3: 0.25}
    frac = fractions.get(min(3, max(1, layers)), 0.125)
    return max(1, math.floor(pokemon.max_hp * frac))


def _calculate_action_damage(
    attacker: SimPokemon,
    defender: SimPokemon,
    move_id: str,
    move_data: dict,
    is_attacker_bot: bool,
) -> tuple[int, int, str]:
    """
    Calculate (min_dmg, max_dmg, notes) for a move using inverse_damage_calc.
    """
    category = move_data.get("category", "Physical")
    base_power = move_data.get("basePower", 0)
    move_type = move_data.get("type", "Normal")

    if base_power == 0 or category == "Status":
        return 0, 0, "status"

    # Attacking stat
    if category == "Physical":
        atk_stat = attacker.stats.get("atk", 200)
        atk_stage = attacker.boosts.get("atk", 0)
    else:
        atk_stat = attacker.stats.get("spa", 200)
        atk_stage = attacker.boosts.get("spa", 0)

    # Defending stat
    if category == "Physical":
        def_stat = defender.stats.get("def", 200)
        def_stage = defender.boosts.get("def", 0)
    else:
        def_stat = defender.stats.get("spd", 200)
        def_stage = defender.boosts.get("spd", 0)

    atk_types = [_type_name(t) for t in attacker.types]
    def_types = [_type_name(t) for t in defender.types]

    type_eff = get_type_effectiveness(move_type, def_types)
    if type_eff == 0:
        return 0, 0, "immune"

    stab = has_stab(move_type, atk_types)

    # Item modifier
    item_mod = 1.0
    if attacker.item and attacker.item in ITEM_MODIFIERS:
        item_mod = ITEM_MODIFIERS[attacker.item]
        if attacker.item == "Expert Belt" and type_eff <= 1.0:
            item_mod = 1.0

    dmg_range = calculate_damage_range(
        attacker_stat=atk_stat,
        defender_stat=def_stat,
        base_power=base_power,
        type_effectiveness=type_eff,
        stab=stab,
        item_modifier=item_mod,
        move_category=category,
        stat_stage_atk=atk_stage,
        stat_stage_def=def_stage,
    )

    return dmg_range.min_damage, dmg_range.max_damage, "damage"


def _simulate_ordered_outcomes(
    our_prio: int,
    opp_prio: int,
    our_spe: int,
    opp_spe: int,
) -> list[tuple[bool, float]]:
    """
    Resolve turn order into one or more (bot_goes_first, probability) chance
    branches.

    Priority always breaks ties deterministically. Effective speed only
    breaks ties deterministically when the two sides are NOT exactly equal.
    A genuine speed tie (identical priority AND identical effective speed --
    e.g. two max-Speed Jolly base-100s, or a full-team speed-tied mirror
    match) is a real 50/50 coin flip on live Showdown, so we branch into two
    chance nodes instead of arbitrarily awarding the tie to one side. This
    also prevents the search from ever "hanging" on a degenerate tie state
    by always making forward progress down both halves of the branch.
    """
    if our_prio > opp_prio:
        return [(True, 1.0)]
    if opp_prio > our_prio:
        return [(False, 1.0)]
    if our_spe == opp_spe:
        return [(True, 0.5), (False, 0.5)]
    return [(our_spe > opp_spe, 1.0)]


def simulate_turn_outcomes(
    state: SimState,
    our_action: ActionCandidate,
    opp_action: OpponentActionCandidate,
) -> list[tuple[SimState, float]]:
    """
    Transition function with chance branches for accuracy and turn resolution.
    Returns list of (resulting_sim_state, probability).
    """
    # ── 1. Handle Switches First ─────────────────────────────────────────
    s = state.clone()

    if our_action.action_type == "switch" and our_action.switch_species:
        # Find pokemon in team
        for mon in s.team.values():
            if mon.species.lower() == our_action.switch_species.lower() and not mon.fainted:
                s.active_pokemon = mon
                break

        # Apply hazard damage on switch-in
        if s.active_pokemon:
            for cond, val in s.side_conditions.items():
                cname = cond.name if hasattr(cond, "name") else str(cond)
                if cname == "STEALTH_ROCK":
                    sr_dmg = _apply_stealth_rock(s.active_pokemon)
                    s.active_pokemon.current_hp = max(0, s.active_pokemon.current_hp - sr_dmg)
                    s.active_pokemon.current_hp_fraction = s.active_pokemon.current_hp / s.active_pokemon.max_hp
                    if s.active_pokemon.current_hp == 0:
                        s.active_pokemon.fainted = True
                elif cname == "SPIKES":
                    sp_dmg = _apply_spikes(s.active_pokemon, layers=val if isinstance(val, int) else 1)
                    s.active_pokemon.current_hp = max(0, s.active_pokemon.current_hp - sp_dmg)
                    s.active_pokemon.current_hp_fraction = s.active_pokemon.current_hp / s.active_pokemon.max_hp
                    if s.active_pokemon.current_hp == 0:
                        s.active_pokemon.fainted = True

    # ── 2. Determine Action Priority & Speed ──────────────────────────────
    # Switch actions have priority +6
    our_prio = 6 if our_action.action_type == "switch" else _get_move_priority(our_action.move_id or "")
    opp_prio = _get_move_priority(opp_action.move_id, opp_action.move_data)

    our_spe = _get_effective_speed(s.active_pokemon, s.side_conditions)
    opp_spe = _get_effective_speed(s.opponent_active_pokemon, s.opponent_side_conditions)

    # Phase 5.1: Speed-Tie Chance Nodes. When priority AND effective speed
    # are exactly equal, real Showdown coin-flips who moves first. Modeling
    # this as a deterministic ">=" silently biases the whole search tree
    # toward "we always win ties" and can hang/mislead on true 50/50s, so we
    # branch into two chance nodes instead of picking one order outright.
    order_branches = _simulate_ordered_outcomes(our_prio, opp_prio, our_spe, opp_spe)

    # ── 3. Execute Moves with Hit/Miss Chance Branches ───────────────────
    outcomes: list[tuple[SimState, float]] = []

    # Case A: Bot is switching, opponent attacks
    if our_action.action_type == "switch":
        if s.active_pokemon and not s.active_pokemon.fainted and s.opponent_active_pokemon and not s.opponent_active_pokemon.fainted:
            opp_acc = opp_action.move_data.get("accuracy", 100)
            p_hit = (opp_acc / 100.0) if isinstance(opp_acc, (int, float)) else 1.0
            p_hit = max(0.0, min(1.0, p_hit))

            # Hit branch
            s_hit = s.clone()
            min_d, max_d, _ = _calculate_action_damage(
                s_hit.opponent_active_pokemon, s_hit.active_pokemon,
                opp_action.move_id, opp_action.move_data, is_attacker_bot=False
            )
            avg_d = (min_d + max_d) // 2
            s_hit.active_pokemon.current_hp = max(0, s_hit.active_pokemon.current_hp - avg_d)
            s_hit.active_pokemon.current_hp_fraction = s_hit.active_pokemon.current_hp / s_hit.active_pokemon.max_hp
            if s_hit.active_pokemon.current_hp == 0:
                s_hit.active_pokemon.fainted = True

            outcomes.append((s_hit, p_hit))
            if p_hit < 1.0:
                outcomes.append((s.clone(), 1.0 - p_hit))
        else:
            outcomes.append((s, 1.0))
        return outcomes

    # Case B: Both Attack
    our_move_id = our_action.move_id or ""
    our_norm_id = our_move_id.lower().replace(" ", "").replace("-", "")
    our_move_data = _MOVES_DB.get(our_norm_id, {})
    our_acc = our_move_data.get("accuracy", 100)
    p_our_hit = (our_acc / 100.0) if isinstance(our_acc, (int, float)) else 1.0
    p_our_hit = max(0.0, min(1.0, p_our_hit))

    opp_acc = opp_action.move_data.get("accuracy", 100)
    p_opp_hit = (opp_acc / 100.0) if isinstance(opp_acc, (int, float)) else 1.0
    p_opp_hit = max(0.0, min(1.0, p_opp_hit))

    # Fan out over each turn-order branch (normally one; two on a speed tie),
    # scaling every resulting outcome's probability by that branch's weight.
    for bot_goes_first, order_prob in order_branches:
        if bot_goes_first:
            # Branch 1: Bot Hits
            s1 = s.clone()
            min_d1, max_d1, _ = _calculate_action_damage(
                s1.active_pokemon, s1.opponent_active_pokemon,
                our_move_id, our_move_data, is_attacker_bot=True
            )
            avg_d1 = (min_d1 + max_d1) // 2
            s1.opponent_active_pokemon.current_hp = max(0, s1.opponent_active_pokemon.current_hp - avg_d1)
            s1.opponent_active_pokemon.current_hp_fraction = s1.opponent_active_pokemon.current_hp / s1.opponent_active_pokemon.max_hp

            # If opponent fainted, opponent DOES NOT counterattack
            if s1.opponent_active_pokemon.current_hp == 0:
                s1.opponent_active_pokemon.fainted = True
                outcomes.append((s1, order_prob * p_our_hit))
            else:
                # Opponent counterattacks
                s1_hit2 = s1.clone()
                min_d2, max_d2, _ = _calculate_action_damage(
                    s1_hit2.opponent_active_pokemon, s1_hit2.active_pokemon,
                    opp_action.move_id, opp_action.move_data, is_attacker_bot=False
                )
                avg_d2 = (min_d2 + max_d2) // 2
                s1_hit2.active_pokemon.current_hp = max(0, s1_hit2.active_pokemon.current_hp - avg_d2)
                s1_hit2.active_pokemon.current_hp_fraction = s1_hit2.active_pokemon.current_hp / s1_hit2.active_pokemon.max_hp
                if s1_hit2.active_pokemon.current_hp == 0:
                    s1_hit2.active_pokemon.fainted = True

                outcomes.append((s1_hit2, order_prob * p_our_hit * p_opp_hit))
                if p_opp_hit < 1.0:
                    outcomes.append((s1, order_prob * p_our_hit * (1.0 - p_opp_hit)))

            # Branch 2: Bot Misses
            if p_our_hit < 1.0:
                s2 = s.clone()
                s2_hit2 = s2.clone()
                min_d2, max_d2, _ = _calculate_action_damage(
                    s2_hit2.opponent_active_pokemon, s2_hit2.active_pokemon,
                    opp_action.move_id, opp_action.move_data, is_attacker_bot=False
                )
                avg_d2 = (min_d2 + max_d2) // 2
                s2_hit2.active_pokemon.current_hp = max(0, s2_hit2.active_pokemon.current_hp - avg_d2)
                s2_hit2.active_pokemon.current_hp_fraction = s2_hit2.active_pokemon.current_hp / s2_hit2.active_pokemon.max_hp
                if s2_hit2.active_pokemon.current_hp == 0:
                    s2_hit2.active_pokemon.fainted = True

                outcomes.append((s2_hit2, order_prob * (1.0 - p_our_hit) * p_opp_hit))
                if p_opp_hit < 1.0:
                    outcomes.append((s2, order_prob * (1.0 - p_our_hit) * (1.0 - p_opp_hit)))

        else:
            # Opponent Goes First
            # Branch 1: Opponent Hits
            s1 = s.clone()
            min_d2, max_d2, _ = _calculate_action_damage(
                s1.opponent_active_pokemon, s1.active_pokemon,
                opp_action.move_id, opp_action.move_data, is_attacker_bot=False
            )
            avg_d2 = (min_d2 + max_d2) // 2
            s1.active_pokemon.current_hp = max(0, s1.active_pokemon.current_hp - avg_d2)
            s1.active_pokemon.current_hp_fraction = s1.active_pokemon.current_hp / s1.active_pokemon.max_hp

            # If Bot fainted, Bot does NOT counterattack
            if s1.active_pokemon.current_hp == 0:
                s1.active_pokemon.fainted = True
                outcomes.append((s1, order_prob * p_opp_hit))
            else:
                # Bot counterattacks
                s1_hit1 = s1.clone()
                min_d1, max_d1, _ = _calculate_action_damage(
                    s1_hit1.active_pokemon, s1_hit1.opponent_active_pokemon,
                    our_move_id, our_move_data, is_attacker_bot=True
                )
                avg_d1 = (min_d1 + max_d1) // 2
                s1_hit1.opponent_active_pokemon.current_hp = max(0, s1_hit1.opponent_active_pokemon.current_hp - avg_d1)
                s1_hit1.opponent_active_pokemon.current_hp_fraction = s1_hit1.opponent_active_pokemon.current_hp / s1_hit1.opponent_active_pokemon.max_hp
                if s1_hit1.opponent_active_pokemon.current_hp == 0:
                    s1_hit1.opponent_active_pokemon.fainted = True

                outcomes.append((s1_hit1, order_prob * p_opp_hit * p_our_hit))
                if p_our_hit < 1.0:
                    outcomes.append((s1, order_prob * p_opp_hit * (1.0 - p_our_hit)))

            # Branch 2: Opponent Misses
            if p_opp_hit < 1.0:
                s2 = s.clone()
                s2_hit1 = s2.clone()
                min_d1, max_d1, _ = _calculate_action_damage(
                    s2_hit1.active_pokemon, s2_hit1.opponent_active_pokemon,
                    our_move_id, our_move_data, is_attacker_bot=True
                )
                avg_d1 = (min_d1 + max_d1) // 2
                s2_hit1.opponent_active_pokemon.current_hp = max(0, s2_hit1.opponent_active_pokemon.current_hp - avg_d1)
                s2_hit1.opponent_active_pokemon.current_hp_fraction = s2_hit1.opponent_active_pokemon.current_hp / s2_hit1.opponent_active_pokemon.max_hp
                if s2_hit1.opponent_active_pokemon.current_hp == 0:
                    s2_hit1.opponent_active_pokemon.fainted = True

                outcomes.append((s2_hit1, order_prob * (1.0 - p_opp_hit) * p_our_hit))
                if p_our_hit < 1.0:
                    outcomes.append((s2, order_prob * (1.0 - p_opp_hit) * (1.0 - p_our_hit)))

    return outcomes


# =============================================================================
# State-Only Action Gathering (for recursive lookahead beyond depth 1)
# =============================================================================
def _fallback_tackle_action() -> list[OpponentActionCandidate]:
    """Degenerate fallback opponent action when nothing else is known."""
    tackle_data = _MOVES_DB.get("tackle", {})
    return [OpponentActionCandidate(action_type="move", label="tackle", move_id="tackle", move_data=tackle_data, probability=1.0)]


def _gather_state_bot_actions(state: SimState) -> list[ActionCandidate]:
    """
    Derive our own legal-ish actions purely from a simulated SimState.

    The real `get_best_action` root ply gets its action list straight from
    poke-env's `battle.available_moves` / `battle.available_switches`
    (which correctly account for PP, choice-lock, trapping, etc). Once the
    search recurses past the root into a simulated future state, those
    poke-env objects no longer exist -- so this reconstructs a reasonable
    action list (known moves + alive teammates) directly from the
    SimState instead. This is what lets Iterative Deepening actually reach
    Depth 3-4 rather than stopping after one ply.
    """
    candidates: list[ActionCandidate] = []
    active = state.active_pokemon
    if active is None or active.fainted:
        return candidates

    for move_id in active.moves.keys():
        norm_id = move_id.lower().replace(" ", "").replace("-", "")
        if norm_id in _MOVES_DB:
            candidates.append(ActionCandidate(
                action_type="move",
                label=f"move: {norm_id}",
                poke_env_object=None,
                move_id=norm_id,
            ))

    for mon in state.team.values():
        if mon is active or mon.fainted:
            continue
        candidates.append(ActionCandidate(
            action_type="switch",
            label=f"switch: {mon.species}",
            poke_env_object=None,
            switch_species=mon.species,
        ))

    return candidates


# =============================================================================
# The Expectiminimax Search Engine
# =============================================================================
class ExpectiminimaxEngine:
    """
    Phase 4B Expectiminimax Engine.

    Solves the simultaneous-move turn tree by calculating the Expected Value
    (EV) across opponent action distributions and RNG chance nodes.
    """

    def __init__(
        self,
        depth: int = 2,
        smogon_priors: Optional[SmogonPriors] = None,
        max_time_ms: float = 500.0,
    ):
        self.depth = depth
        self.priors = smogon_priors
        self.max_time_ms = max_time_ms

    def get_best_action(
        self,
        battle,
        opponent_profiles: Optional[dict] = None,
    ) -> tuple[Optional[Any], list[tuple[str, float]]]:
        """
        Evaluate all legal options and return the best action object for poke-env.

        Parameters
        ----------
        battle : Battle
            Current poke-env Battle state.
        opponent_profiles : dict or None
            Inferred stats/items from Phase 3 inverse damage calculation.

        Returns
        -------
        (best_action_object, ranked_candidates)
            best_action_object: Move or Pokemon instance to pass to create_order()
            ranked_candidates: list of (label, expected_value) sorted descending
        """
        # Phase 5.1: monotonic timer -- perf_counter() is immune to wall-clock
        # adjustments (NTP jumps, DST, etc.) and gives microsecond
        # resolution, which time.time() does not guarantee.
        start_time = time.perf_counter()
        deadline_s = self.max_time_ms / 1000.0

        # Convert battle to root SimState
        root_state = battle_to_sim_state(battle, opponent_profiles)

        # ── 1. Gather Legal Bot Actions ───────────────────────────────────
        bot_actions: list[ActionCandidate] = []

        # Available moves
        for move in getattr(battle, "available_moves", []) or []:
            bot_actions.append(ActionCandidate(
                action_type="move",
                label=f"move: {move.id}",
                poke_env_object=move,
                move_id=move.id,
            ))

        # Available switches
        for mon in getattr(battle, "available_switches", []) or []:
            species = getattr(mon, "species", "")
            bot_actions.append(ActionCandidate(
                action_type="switch",
                label=f"switch: {species}",
                poke_env_object=mon,
                switch_species=species,
            ))

        if not bot_actions:
            return None, []

        # If only 1 action available (e.g. trapped or recharge), return it immediately
        if len(bot_actions) == 1:
            return bot_actions[0].poke_env_object, [(bot_actions[0].label, 0.0)]

        # ── 2. Gather Opponent Projected Actions ─────────────────────────
        opp_actions = self._gather_opponent_actions(battle)

        # ── 3. Iterative Deepening Search (Phase 5.1) ─────────────────────
        # Depth 1 always runs to completion regardless of the timer, so we
        # always have a legal action to return even under extreme time
        # pressure (e.g. a slow first turn on a loaded server).
        best_candidate, ranked_list = self._evaluate_all_actions(
            root_state, bot_actions, opp_actions, depth=1,
            start_time=start_time, deadline_s=deadline_s,
        )
        completed_depth = 1

        for current_depth in range(2, self.depth + 1):
            if (time.perf_counter() - start_time) >= deadline_s:
                logger.debug(
                    "Iterative deepening: time exhausted before starting depth %d; "
                    "keeping depth %d results.", current_depth, completed_depth,
                )
                break

            ply_candidate, ply_ranked, aborted = self._evaluate_all_actions_guarded(
                root_state, bot_actions, opp_actions, depth=current_depth,
                start_time=start_time, deadline_s=deadline_s,
            )

            if aborted:
                # Safety Fallback: a depth search aborted mid-evaluation due
                # to the timer budget -- discard the partial ply entirely
                # and keep the best action from the last FULLY completed
                # depth instead of trusting a half-evaluated ranking.
                logger.debug(
                    "Iterative deepening: depth %d aborted mid-evaluation (timeout); "
                    "discarding partial ply, keeping depth %d results.",
                    current_depth, completed_depth,
                )
                break

            best_candidate, ranked_list = ply_candidate, ply_ranked
            completed_depth = current_depth

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "Expectiminimax search (target depth=%d, completed depth=%d) evaluated "
            "%d actions in %.1fms. Best: %s (EV=%.1f)",
            self.depth, completed_depth, len(bot_actions), elapsed_ms,
            best_candidate.label, ranked_list[0][1],
        )

        return best_candidate.poke_env_object, ranked_list

    def _evaluate_all_actions(
        self,
        state: SimState,
        bot_actions: list[ActionCandidate],
        opp_actions: list[OpponentActionCandidate],
        depth: int,
        start_time: float,
        deadline_s: float,
    ) -> tuple[ActionCandidate, list[tuple[str, float]]]:
        """Evaluate every root action at a given depth, unconditionally (no abort)."""
        action_scores = [
            (action, self._evaluate_action_ev(state, action, opp_actions, depth, start_time, deadline_s))
            for action in bot_actions
        ]
        action_scores.sort(key=lambda item: item[1], reverse=True)
        ranked_list = [(cand.label, score) for cand, score in action_scores]
        return action_scores[0][0], ranked_list

    def _evaluate_all_actions_guarded(
        self,
        state: SimState,
        bot_actions: list[ActionCandidate],
        opp_actions: list[OpponentActionCandidate],
        depth: int,
        start_time: float,
        deadline_s: float,
    ) -> tuple[Optional[ActionCandidate], list[tuple[str, float]], bool]:
        """
        Evaluate every root action at a given depth, aborting (and
        signalling `aborted=True`) the moment the timer budget is blown
        mid-ply, so the caller can discard the partial results.
        """
        action_scores: list[tuple[ActionCandidate, float]] = []
        for action in bot_actions:
            if (time.perf_counter() - start_time) >= deadline_s:
                return None, [], True
            ev = self._evaluate_action_ev(state, action, opp_actions, depth, start_time, deadline_s)
            action_scores.append((action, ev))

        action_scores.sort(key=lambda item: item[1], reverse=True)
        ranked_list = [(cand.label, score) for cand, score in action_scores]
        return action_scores[0][0], ranked_list, False

    def _evaluate_action_ev(
        self,
        state: SimState,
        our_action: ActionCandidate,
        opp_actions: list[OpponentActionCandidate],
        depth_remaining: int,
        start_time: float,
        deadline_s: float,
    ) -> float:
        """
        Compute EV of a specific bot action across the opponent's projected
        move distribution, recursing `depth_remaining - 1` further plies
        deep (Phase 5.1) instead of always flattening to a 1-ply heuristic.
        """
        total_ev = 0.0

        for opp_act in opp_actions:
            # Chance node outcomes for (our_action, opp_act)
            outcomes = simulate_turn_outcomes(state, our_action, opp_act)

            opp_ev = 0.0
            for next_state, chance_prob in outcomes:
                if chance_prob <= 0.0:
                    continue
                if depth_remaining <= 1 or (time.perf_counter() - start_time) >= deadline_s:
                    leaf_score = evaluate_state(next_state)
                else:
                    # Recurse: assume we play our own best response from
                    # here, then fold the opponent's distribution back in.
                    leaf_score = self._search_best_response(
                        next_state, depth_remaining - 1, start_time, deadline_s
                    )

                opp_ev += leaf_score * chance_prob

            total_ev += opp_ev * opp_act.probability

        return total_ev

    def _search_best_response(
        self,
        state: SimState,
        depth_remaining: int,
        start_time: float,
        deadline_s: float,
    ) -> float:
        """
        Recursive interior ply (Phase 5.1): at a simulated future state,
        assume our side plays whichever of ITS available actions maximizes
        EV, then folds the opponent's projected distribution back in for
        that ply. This is what actually lets Iterative Deepening reach
        Depth 3-4, versus the prior implementation which always flattened
        every branch to a single heuristic evaluation after 1 ply.
        """
        # Terminal checks
        won = getattr(state, "won", None)
        if won is True:
            return SCORE_WIN
        if won is False:
            return SCORE_LOSS

        active = state.active_pokemon
        opp_active = state.opponent_active_pokemon
        if active is None or opp_active is None or active.fainted or opp_active.fainted:
            # A faint mid-line means a forced switch is coming that we
            # can't project cleanly -- fall back to the heuristic here
            # rather than guessing at the replacement.
            return evaluate_state(state)

        if (time.perf_counter() - start_time) >= deadline_s:
            return evaluate_state(state)

        our_candidates = _gather_state_bot_actions(state)
        if not our_candidates:
            return evaluate_state(state)

        opp_candidates = self._gather_state_opponent_actions(state)

        best_ev = float("-inf")
        for our_cand in our_candidates:
            if (time.perf_counter() - start_time) >= deadline_s:
                break
            ev = self._evaluate_action_ev(
                state, our_cand, opp_candidates, depth_remaining, start_time, deadline_s
            )
            if ev > best_ev:
                best_ev = ev

        return best_ev if best_ev != float("-inf") else evaluate_state(state)

    def _gather_opponent_actions(self, battle) -> list[OpponentActionCandidate]:
        """Gather likely opponent moves with prior probabilities (root ply)."""
        opp = getattr(battle, "opponent_active_pokemon", None)
        if opp is None:
            return _fallback_tackle_action()
        revealed_moves = getattr(opp, "moves", {}) or {}
        return self._gather_opponent_actions_for(opp.species, revealed_moves.keys())

    def _gather_state_opponent_actions(self, state: SimState) -> list[OpponentActionCandidate]:
        """
        Gather likely opponent moves with prior probabilities from a
        simulated SimState (used by recursive lookahead beyond depth 1,
        where we no longer have the original poke-env Battle object).
        """
        opp = state.opponent_active_pokemon
        if opp is None or opp.fainted:
            return _fallback_tackle_action()
        return self._gather_opponent_actions_for(opp.species, opp.moves.keys())

    def _gather_opponent_actions_for(self, species: str, revealed_move_ids) -> list[OpponentActionCandidate]:
        """
        Shared core: build the opponent's projected move distribution from
        (a) any already-revealed moves, filled out with (b) Smogon prior
        probabilities up to 4 total moves.
        """
        opp_actions: list[OpponentActionCandidate] = []

        # 1. Revealed moves
        for mid in revealed_move_ids:
            norm_id = mid.lower().replace(" ", "").replace("-", "")
            mdata = _MOVES_DB.get(norm_id)
            if mdata:
                opp_actions.append(OpponentActionCandidate(
                    action_type="move",
                    label=mid,
                    move_id=norm_id,
                    move_data=mdata,
                    probability=1.0,
                ))

        # 2. Smogon Priors if we have fewer than 4 moves
        if len(opp_actions) < 4 and self.priors is not None:
            build = self.priors.get_likely_build(species)
            if build:
                known_ids = {a.move_id for a in opp_actions}
                for mp in build.top_moves:
                    norm_id = mp.name.lower().replace(" ", "").replace("-", "")
                    if norm_id not in known_ids and len(opp_actions) < 4:
                        mdata = _MOVES_DB.get(norm_id)
                        if mdata:
                            opp_actions.append(OpponentActionCandidate(
                                action_type="move",
                                label=mp.name,
                                move_id=norm_id,
                                move_data=mdata,
                                probability=mp.probability,
                            ))
                            known_ids.add(norm_id)

        # Fallback if no moves known
        if not opp_actions:
            return _fallback_tackle_action()

        # Normalize probabilities to sum to 1.0
        total_p = sum(a.probability for a in opp_actions)
        if total_p > 0:
            for a in opp_actions:
                a.probability = a.probability / total_p
        else:
            uniform = 1.0 / len(opp_actions)
            for a in opp_actions:
                a.probability = uniform

        return opp_actions


# =============================================================================
# Standalone Unit Test Suite
# =============================================================================
def _run_tests():
    """Unit tests for ExpectiminimaxEngine."""
    print("=" * 72)
    print("  Phase 4B: Expectiminimax Engine -- Standalone Test Suite")
    print("=" * 72)

    from evaluator import _MockPokemon, _MockMove, _MockBattle

    # Scenario: Garchomp vs Dragapult
    # Garchomp has: Earthquake, Dragon Claw, Swords Dance, Stealth Rock
    # Dragapult is Dragon/Ghost and weak to Dragon Claw
    # Dragapult knows: Draco Meteor, Shadow Ball
    eq = _MockMove("earthquake")
    dc = _MockMove("dragonclaw")
    sd = _MockMove("swordsdance")
    sr = _MockMove("stealthrock")

    opp_dm = _MockMove("dracometeor")
    opp_sb = _MockMove("shadowball")

    garchomp = _MockPokemon(
        "garchomp", hp_fraction=1.0, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
    )
    toxapex = _MockPokemon(
        "toxapex", hp_fraction=1.0, max_hp=304,
        stats={"atk": 152, "def": 353, "spa": 137, "spd": 293, "spe": 96},
        types=["Poison", "Water"],
    )
    dragapult = _MockPokemon(
        "dragapult", hp_fraction=1.0, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
        moves={"dracometeor": opp_dm, "shadowball": opp_sb},
    )

    battle = _MockBattle(
        our_team={"p1: Garchomp": garchomp, "p1: Toxapex": toxapex},
        opp_team={"p2: Dragapult": dragapult},
        active_pokemon=garchomp,
        opponent_active_pokemon=dragapult,
        available_moves=[eq, dc, sd, sr],
        available_switches=[toxapex],
    )

    priors = SmogonPriors(format_id="gen9ou")
    engine = ExpectiminimaxEngine(depth=2, smogon_priors=priors)

    print("\n--- Test 1: Garchomp vs Dragapult Move Selection ---")
    best_obj, ranked = engine.get_best_action(battle)

    print("  Ranked Candidates (EV):")
    for label, ev in ranked:
        print(f"    {label:<25} -> EV: {ev:>+8.1f}")

    print(f"\n  Chosen Best Action: {ranked[0][0]} (EV: {ranked[0][1]:.1f})")
    assert best_obj is not None
    assert len(ranked) == 5  # 4 moves + 1 switch

    print("\n--- Test 2: KO Opportunity Recognition ---")
    # Dragapult is low HP (15%) -> Dragon Claw or Earthquake should easily finish it
    dragapult_low = _MockPokemon(
        "dragapult", hp_fraction=0.15, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
        moves={"dracometeor": opp_dm},
    )
    battle_low = _MockBattle(
        our_team={"p1: Garchomp": garchomp},
        opp_team={"p2: Dragapult": dragapult_low},
        active_pokemon=garchomp,
        opponent_active_pokemon=dragapult_low,
        available_moves=[eq, dc, sd, sr],
        available_switches=[],
    )
    best_low, ranked_low = engine.get_best_action(battle_low)
    print("  Ranked Candidates for Low HP Opponent:")
    for label, ev in ranked_low:
        print(f"    {label:<25} -> EV: {ev:>+8.1f}")
    assert "dragonclaw" in ranked_low[0][0] or "earthquake" in ranked_low[0][0]

    print("\n" + "=" * 72)
    print("  All Phase 4B tests passed [OK]")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    _run_tests()
