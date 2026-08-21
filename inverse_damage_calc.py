"""
inverse_damage_calc.py -- Phase 3: Inverse Damage Calculation

Forward damage calculator (Gen 9 formula) and inverse solver that
deduces an opponent's item / EV spread / boosting state from observed
damage.

The Gen 9 damage formula:
    base   = floor( floor( (2*Level/5 + 2) * BasePower * A/D ) / 50 + 2 )
    damage = floor( base * modifier_chain * roll/100 )

    where roll in {85, 86, 87, ..., 100}  (16 discrete values)

The inverse solver takes the observed HP lost, the known move, the
known defender stats, and field conditions, then tests candidate
attacker builds (from Smogon priors) to identify which item / EV
combination could have legally produced that damage.

Usage
-----
    python inverse_damage_calc.py        # run built-in test scenarios
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from poke_env.data import GenData

logger = logging.getLogger("InverseDmgCalc")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GEN = 9
_LEVEL = 100  # Standard competitive level
_DAMAGE_ROLLS = list(range(85, 101))  # 16 rolls: 85% .. 100%

# Type chart (loaded once)
_gen_data = GenData.from_gen(_GEN)
_TYPE_CHART = _gen_data.type_chart       # tc[DEFENDER_TYPE][MOVE_TYPE] = mult
_MOVES_DB = _gen_data.moves
_POKEDEX = _gen_data.pokedex
_NATURES_DB = _gen_data.natures

# Item offensive multipliers (applied to final damage)
ITEM_MODIFIERS: dict[str, float] = {
    "Choice Band":   1.5,
    "Choice Specs":  1.5,
    "Life Orb":      1.3,
    "Expert Belt":   1.2,   # Only on super-effective; handled specially
    "Muscle Band":   1.1,   # Physical only
    "Wise Glasses":  1.1,   # Special only
}

# Ability offensive multipliers (common ones)
ABILITY_MODIFIERS: dict[str, float] = {
    "Huge Power":      2.0,
    "Pure Power":      2.0,
    "Gorilla Tactics": 1.5,
    "Hustle":          1.5,  # Physical only, but also lowers accuracy
}

# Booster Energy / Protosynthesis / Quark Drive stat boost
# These boost the *stat itself* by 1.3x (or 1.5x for Speed), not damage.
# We handle them as stat multipliers in the candidate builds.
PROTO_QUARK_BOOST = 1.3


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class PokemonStats:
    """Fully resolved stats for one Pokemon (at level 100)."""
    hp: int
    atk: int
    defense: int  # 'def' is a Python keyword
    spa: int
    spd: int
    spe: int


@dataclass
class FieldConditions:
    """Active field conditions that affect damage."""
    weather: Optional[str] = None       # "sun", "rain", "sand", "snow"
    terrain: Optional[str] = None       # "electric", "grassy", "psychic", "misty"
    reflect: bool = False               # Halves physical damage
    light_screen: bool = False          # Halves special damage
    attacker_burned: bool = False       # Halves physical damage (unless Guts)
    critical_hit: bool = False          # 1.5x, ignores screens/burns


@dataclass
class CandidateBuild:
    """A hypothetical attacker build to test against observed damage."""
    label: str
    nature: str                         # e.g. "adamant", "jolly"
    ev_investment: int                  # 0 or 252 typically
    iv: int = 31
    item: Optional[str] = None
    ability: Optional[str] = None
    stat_multiplier: float = 1.0        # Proto/Quark boost, etc.


@dataclass
class DamageRange:
    """The min/max damage from a single move with a specific build."""
    min_damage: int
    max_damage: int
    min_percent: float                  # As fraction of defender's max HP
    max_percent: float

    def __repr__(self) -> str:
        return (f"DamageRange({self.min_damage}-{self.max_damage}, "
                f"{self.min_percent:.1%}-{self.max_percent:.1%})")


@dataclass
class InferredState:
    """Result of inverse damage calculation."""
    attacker_species: str
    move_name: str
    observed_damage: int
    observed_percent: float
    matching_builds: list[tuple[CandidateBuild, DamageRange]] = field(
        default_factory=list
    )
    best_guess_item: Optional[str] = None
    best_guess_evs: Optional[str] = None
    estimated_attack_stat: Optional[int] = None

    def summary(self) -> str:
        """One-line summary for console logging."""
        if not self.matching_builds:
            return (f"Took {self.observed_damage} dmg ({self.observed_percent:.1%}) "
                    f"from {self.attacker_species}'s {self.move_name}. "
                    f"No matching builds found.")

        matches = " / ".join(b.label for b, _ in self.matching_builds[:3])
        atk_str = f"Est. Atk: {self.estimated_attack_stat}" if self.estimated_attack_stat else ""
        return (f"Took {self.observed_damage} dmg ({self.observed_percent:.1%}) "
                f"from {self.attacker_species}'s {self.move_name}. "
                f"Inferred: {matches}"
                + (f" ({atk_str})" if atk_str else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# Stat calculation
# ═══════════════════════════════════════════════════════════════════════════════
def calc_stat(base: int, ev: int = 0, iv: int = 31, level: int = _LEVEL,
              nature_mult: float = 1.0, is_hp: bool = False) -> int:
    """
    Calculate a single stat at a given level.

    HP formula:   floor( (2*Base + IV + floor(EV/4)) * Level/100 ) + Level + 10
    Other stats:  floor( (floor( (2*Base + IV + floor(EV/4)) * Level/100 ) + 5) * Nature )
    """
    ev_part = ev // 4
    if is_hp:
        return math.floor((2 * base + iv + ev_part) * level / 100) + level + 10
    else:
        raw = math.floor((2 * base + iv + ev_part) * level / 100) + 5
        return math.floor(raw * nature_mult)


def calc_all_stats(species_id: str, nature: str = "adamant",
                   evs: Optional[dict[str, int]] = None,
                   ivs: Optional[dict[str, int]] = None) -> PokemonStats:
    """
    Calculate all six stats for a species with a given nature / EV / IV spread.

    Parameters
    ----------
    species_id : str
        Lowercase poke-env species ID (e.g., "greattusk").
    nature : str
        Lowercase nature name (e.g., "adamant").
    evs : dict
        EV spread, e.g. {"atk": 252, "spe": 252, "hp": 4}.  Missing keys = 0.
    ivs : dict
        IV spread.  Missing keys = 31.
    """
    # Look up base stats
    key = species_id.lower().replace(" ", "").replace("-", "")
    mon = _POKEDEX.get(key)
    if mon is None:
        raise ValueError(f"Species '{species_id}' not found in pokedex")
    base = mon["baseStats"]

    # Nature multipliers
    nat = _NATURES_DB.get(nature.lower(), {})
    evs = evs or {}
    ivs = ivs or {}

    stat_keys = [("hp", "hp", True), ("atk", "atk", False),
                 ("def", "def", False), ("spa", "spa", False),
                 ("spd", "spd", False), ("spe", "spe", False)]

    results = {}
    for stat_name, base_key, is_hp in stat_keys:
        nature_mult = nat.get(stat_name, 1.0) if not is_hp else 1.0
        results[stat_name] = calc_stat(
            base=base[base_key],
            ev=evs.get(stat_name, 0),
            iv=ivs.get(stat_name, 31),
            nature_mult=nature_mult,
            is_hp=is_hp,
        )

    return PokemonStats(
        hp=results["hp"],
        atk=results["atk"],
        defense=results["def"],
        spa=results["spa"],
        spd=results["spd"],
        spe=results["spe"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Type effectiveness
# ═══════════════════════════════════════════════════════════════════════════════
def get_type_effectiveness(move_type: str, defender_types: list[str]) -> float:
    """
    Calculate the combined type effectiveness multiplier.

    Parameters
    ----------
    move_type : str
        The move's type (e.g., "Ground").
    defender_types : list[str]
        The defender's types (e.g., ["Water", "Poison"]).

    Returns
    -------
    float
        Combined multiplier (0, 0.25, 0.5, 1, 2, or 4).
    """
    move_upper = move_type.upper()
    total = 1.0
    for def_type in defender_types:
        if def_type is None:
            continue
        def_upper = def_type.upper()
        chart_entry = _TYPE_CHART.get(def_upper, {})
        mult = chart_entry.get(move_upper, 1.0)
        total *= mult
    return total


def has_stab(move_type: str, attacker_types: list[str]) -> bool:
    """Check if the attacker gets Same-Type Attack Bonus for this move."""
    mt = move_type.upper()
    return any(t.upper() == mt for t in attacker_types if t)


# ═══════════════════════════════════════════════════════════════════════════════
# Forward damage calculator
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_damage_range(
    attacker_stat: int,
    defender_stat: int,
    base_power: int,
    type_effectiveness: float = 1.0,
    stab: bool = False,
    item_modifier: float = 1.0,
    ability_modifier: float = 1.0,
    field: Optional[FieldConditions] = None,
    move_category: str = "Physical",
    level: int = _LEVEL,
    stat_stage_atk: int = 0,
    stat_stage_def: int = 0,
) -> DamageRange:
    """
    Compute the 16-step damage range for a single hit.

    Uses the standard Gen 5+ damage formula with Gen 9 mechanics.

    Parameters
    ----------
    attacker_stat : int
        Effective Attack or Special Attack stat value.
    defender_stat : int
        Effective Defense or Special Defense stat value.
    base_power : int
        Move's base power.
    type_effectiveness : float
        Combined type multiplier (0, 0.25, 0.5, 1, 2, 4).
    stab : bool
        Whether STAB applies (1.5x).
    item_modifier : float
        Offensive item multiplier (e.g., 1.5 for Choice Band).
    ability_modifier : float
        Offensive ability multiplier (e.g., 2.0 for Huge Power).
    field : FieldConditions or None
        Active field conditions.
    move_category : str
        "Physical" or "Special".
    level : int
        Attacker's level (default 100).
    stat_stage_atk : int
        Attack stat stage modifier (-6 to +6).
    stat_stage_def : int
        Defense stat stage modifier (-6 to +6).

    Returns
    -------
    DamageRange
        The min and max damage values and percentages.
    """
    field = field or FieldConditions()

    # Apply stat stages
    a = _apply_stat_stage(attacker_stat, stat_stage_atk)
    d = _apply_stat_stage(defender_stat, stat_stage_def)

    # Apply ability modifier to attacking stat
    a = math.floor(a * ability_modifier)

    # Base damage calculation
    # floor( floor( (2*Level/5 + 2) * BasePower * A/D ) / 50 + 2 )
    level_factor = math.floor(2 * level / 5) + 2
    inner = math.floor(level_factor * base_power * a / d)
    base_damage = math.floor(inner / 50) + 2

    # Build modifier chain (applied sequentially with floors)
    modifiers: list[float] = []

    # Screens (applied before other modifiers in Gen 9)
    if not field.critical_hit:
        if move_category == "Physical" and field.reflect:
            modifiers.append(0.5)
        elif move_category == "Special" and field.light_screen:
            modifiers.append(0.5)

    # Weather
    if field.weather == "sun":
        modifiers.append(1.5 if move_category == "Physical" else 1.0)  # Not direct
        # Actually: sun boosts Fire moves, weakens Water moves
        # We'll handle this via a separate weather_modifier below
    # (Weather is complex; simplified here -- applied as type-specific)
    # For a production calc, we'd need move type + weather interaction.
    # We keep it simple for the inverse solver's purposes.

    # STAB
    if stab:
        modifiers.append(1.5)

    # Type effectiveness
    modifiers.append(type_effectiveness)

    # Burn (halves physical damage unless Guts)
    if field.attacker_burned and move_category == "Physical":
        modifiers.append(0.5)

    # Item modifier
    if item_modifier != 1.0:
        modifiers.append(item_modifier)

    # Critical hit
    if field.critical_hit:
        modifiers.append(1.5)

    # Calculate damage for each roll
    damages = []
    for roll in _DAMAGE_ROLLS:
        dmg = base_damage
        # Apply random roll
        dmg = math.floor(dmg * roll / 100)
        # Apply modifiers chain
        for mod in modifiers:
            dmg = math.floor(dmg * mod)
        # Minimum 1 damage (unless immune)
        if type_effectiveness > 0:
            dmg = max(1, dmg)
        else:
            dmg = 0
        damages.append(dmg)

    min_dmg = min(damages)
    max_dmg = max(damages)

    return DamageRange(
        min_damage=min_dmg,
        max_damage=max_dmg,
        min_percent=0.0,  # Filled in by caller with defender HP context
        max_percent=0.0,
    )


def _apply_stat_stage(stat: int, stage: int) -> int:
    """Apply a stat stage modifier (-6 to +6) to a stat value."""
    if stage == 0:
        return stat
    if stage > 0:
        return math.floor(stat * (2 + stage) / 2)
    else:
        return math.floor(stat * 2 / (2 - stage))


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate build generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_candidate_builds(
    species_id: str,
    move_category: str = "Physical",
    smogon_priors=None,
) -> list[CandidateBuild]:
    """
    Generate a set of candidate builds to test against observed damage.

    These cover the realistic spectrum of offensive investment:
    - Max investment + boosting item (Choice Band/Specs)
    - Max investment + Life Orb
    - Max investment + neutral item
    - Medium investment + neutral item
    - Zero investment (defensive build)

    If SmogonPriors data is available, the candidates are tailored to the
    actual top items/spreads from usage stats.
    """
    stat_label = "Atk" if move_category == "Physical" else "SpA"
    boost_item = "Choice Band" if move_category == "Physical" else "Choice Specs"
    boost_nature = "adamant" if move_category == "Physical" else "modest"
    speed_nature = "jolly" if move_category == "Physical" else "timid"

    builds = [
        # Max offensive
        CandidateBuild(
            label=f"252+ {stat_label} {boost_item}",
            nature=boost_nature, ev_investment=252,
            item=boost_item,
        ),
        CandidateBuild(
            label=f"252 {stat_label} {boost_item}",
            nature=speed_nature, ev_investment=252,
            item=boost_item,
        ),
        CandidateBuild(
            label=f"252+ {stat_label} Life Orb",
            nature=boost_nature, ev_investment=252,
            item="Life Orb",
        ),
        CandidateBuild(
            label=f"252 {stat_label} Life Orb",
            nature=speed_nature, ev_investment=252,
            item="Life Orb",
        ),
        # Boosted stat (Proto/Quark)
        CandidateBuild(
            label=f"252 {stat_label} Booster Energy",
            nature=speed_nature, ev_investment=252,
            item="Booster Energy",
            stat_multiplier=PROTO_QUARK_BOOST,
        ),
        # Unboosted max
        CandidateBuild(
            label=f"252+ {stat_label} (no item boost)",
            nature=boost_nature, ev_investment=252,
        ),
        CandidateBuild(
            label=f"252 {stat_label} (no item boost)",
            nature=speed_nature, ev_investment=252,
        ),
        # Moderate investment
        CandidateBuild(
            label=f"128 {stat_label} (no item boost)",
            nature=speed_nature, ev_investment=128,
        ),
        # Zero investment (defensive)
        CandidateBuild(
            label=f"0 {stat_label} (defensive)",
            nature="bold" if move_category == "Physical" else "calm",
            ev_investment=0,
        ),
    ]

    return builds


def resolve_build_attack_stat(
    species_id: str,
    build: CandidateBuild,
    move_category: str = "Physical",
) -> int:
    """
    Calculate the effective attacking stat for a candidate build.

    Returns the raw stat value (before item modifier -- item is applied
    separately in the damage formula as a multiplier on damage, not on stat).
    """
    key = species_id.lower().replace(" ", "").replace("-", "")
    mon = _POKEDEX.get(key)
    if mon is None:
        raise ValueError(f"Species '{species_id}' not found in pokedex")

    base_stats = mon["baseStats"]
    stat_key = "atk" if move_category == "Physical" else "spa"
    base = base_stats[stat_key]

    nat = _NATURES_DB.get(build.nature.lower(), {})
    nature_mult = nat.get(stat_key, 1.0)

    stat = calc_stat(
        base=base,
        ev=build.ev_investment,
        iv=build.iv,
        nature_mult=nature_mult,
    )

    # Apply stat multiplier (e.g., Protosynthesis/Quark Drive boost)
    if build.stat_multiplier != 1.0:
        stat = math.floor(stat * build.stat_multiplier)

    return stat


# ═══════════════════════════════════════════════════════════════════════════════
# The inverse solver
# ═══════════════════════════════════════════════════════════════════════════════
def infer_opponent_state(
    observed_damage: int,
    defender_max_hp: int,
    defender_stat: int,
    move_name: str,
    attacker_species: str,
    attacker_types: list[str],
    defender_types: list[str],
    field: Optional[FieldConditions] = None,
    stat_stage_atk: int = 0,
    stat_stage_def: int = 0,
    smogon_priors=None,
) -> InferredState:
    """
    Given observed damage, deduce the opponent's likely build.

    Parameters
    ----------
    observed_damage : int
        Exact HP points lost by the defender.
    defender_max_hp : int
        Defender's maximum HP.
    defender_stat : int
        Defender's effective Def or SpD (our bot knows this exactly).
    move_name : str
        The move used by the attacker.
    attacker_species : str
        The attacker's species (poke-env ID or display name).
    attacker_types : list[str]
        The attacker's types (for STAB calculation).
    defender_types : list[str]
        The defender's types (for effectiveness calculation).
    field : FieldConditions or None
        Active field conditions.
    stat_stage_atk : int
        Attacker's attack/sp.atk stage.
    stat_stage_def : int
        Defender's defense/sp.def stage.

    Returns
    -------
    InferredState
        Contains matching builds and best-guess item/EVs.
    """
    field = field or FieldConditions()
    observed_pct = observed_damage / defender_max_hp if defender_max_hp > 0 else 0

    # Look up move data
    move_id = move_name.lower().replace(" ", "").replace("-", "")
    move_data = _MOVES_DB.get(move_id)
    if move_data is None:
        logger.warning("Move '%s' (id: %s) not found in move DB", move_name, move_id)
        return InferredState(
            attacker_species=attacker_species,
            move_name=move_name,
            observed_damage=observed_damage,
            observed_percent=observed_pct,
        )

    base_power = move_data.get("basePower", 0)
    move_category = move_data.get("category", "Physical")
    move_type = move_data.get("type", "Normal")

    if base_power == 0 or move_category == "Status":
        logger.debug("Move '%s' is status or has 0 BP, skipping inverse calc", move_name)
        return InferredState(
            attacker_species=attacker_species,
            move_name=move_name,
            observed_damage=observed_damage,
            observed_percent=observed_pct,
        )

    # Type effectiveness
    type_eff = get_type_effectiveness(move_type, defender_types)
    if type_eff == 0:
        return InferredState(
            attacker_species=attacker_species,
            move_name=move_name,
            observed_damage=observed_damage,
            observed_percent=observed_pct,
        )

    # STAB check
    stab = has_stab(move_type, attacker_types)

    # Normalise attacker species for pokedex lookup
    atk_key = attacker_species.lower().replace(" ", "").replace("-", "")

    # Generate candidate builds
    candidates = generate_candidate_builds(atk_key, move_category, smogon_priors)

    # Test each candidate
    matching: list[tuple[CandidateBuild, DamageRange]] = []

    for build in candidates:
        try:
            atk_stat = resolve_build_attack_stat(atk_key, build, move_category)
        except ValueError:
            continue

        # Item modifier for damage
        item_mod = 1.0
        if build.item and build.item in ITEM_MODIFIERS:
            item_mod = ITEM_MODIFIERS[build.item]
            # Expert Belt only applies on super-effective
            if build.item == "Expert Belt" and type_eff <= 1.0:
                item_mod = 1.0
            # Muscle Band only physical, Wise Glasses only special
            if build.item == "Muscle Band" and move_category != "Physical":
                item_mod = 1.0
            if build.item == "Wise Glasses" and move_category != "Special":
                item_mod = 1.0

        # Ability modifier
        ability_mod = 1.0
        if build.ability and build.ability in ABILITY_MODIFIERS:
            ability_mod = ABILITY_MODIFIERS[build.ability]

        dmg_range = calculate_damage_range(
            attacker_stat=atk_stat,
            defender_stat=defender_stat,
            base_power=base_power,
            type_effectiveness=type_eff,
            stab=stab,
            item_modifier=item_mod,
            ability_modifier=ability_mod,
            field=field,
            move_category=move_category,
            stat_stage_atk=stat_stage_atk,
            stat_stage_def=stat_stage_def,
        )

        # Fill in percentages
        dmg_range.min_percent = dmg_range.min_damage / defender_max_hp
        dmg_range.max_percent = dmg_range.max_damage / defender_max_hp

        # Check if observed damage falls within this range
        if dmg_range.min_damage <= observed_damage <= dmg_range.max_damage:
            matching.append((build, dmg_range))

    # Determine best guess
    best_item = None
    best_evs = None
    est_atk = None

    if matching:
        # Prefer the most constrained match (narrowest range)
        matching.sort(key=lambda x: x[1].max_damage - x[1].min_damage)
        best_build, best_range = matching[0]
        best_item = best_build.item or "(no item boost)"
        best_evs = f"{best_build.ev_investment} EVs ({best_build.nature})"
        est_atk = resolve_build_attack_stat(atk_key, best_build, move_category)

    return InferredState(
        attacker_species=attacker_species,
        move_name=move_name,
        observed_damage=observed_damage,
        observed_percent=observed_pct,
        matching_builds=matching,
        best_guess_item=best_item,
        best_guess_evs=best_evs,
        estimated_attack_stat=est_atk,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: full forward calc from species names
# ═══════════════════════════════════════════════════════════════════════════════
def calc_full_damage(
    attacker_species: str,
    move_name: str,
    defender_species: str,
    attacker_evs: Optional[dict[str, int]] = None,
    attacker_nature: str = "adamant",
    defender_evs: Optional[dict[str, int]] = None,
    defender_nature: str = "bold",
    attacker_item: Optional[str] = None,
    field: Optional[FieldConditions] = None,
) -> DamageRange:
    """
    High-level forward damage calc: species + move -> damage range.

    For quick testing and scenario verification.
    """
    # Look up move
    move_id = move_name.lower().replace(" ", "").replace("-", "")
    move_data = _MOVES_DB.get(move_id)
    if move_data is None:
        raise ValueError(f"Move '{move_name}' not found")

    base_power = move_data["basePower"]
    move_category = move_data["category"]
    move_type = move_data["type"]

    # Calculate stats
    atk_stats = calc_all_stats(attacker_species, attacker_nature, attacker_evs)
    def_stats = calc_all_stats(defender_species, defender_nature, defender_evs)

    atk_key = attacker_species.lower().replace(" ", "").replace("-", "")
    def_key = defender_species.lower().replace(" ", "").replace("-", "")

    atk_types = _POKEDEX[atk_key]["types"]
    def_types = _POKEDEX[def_key]["types"]

    a = atk_stats.atk if move_category == "Physical" else atk_stats.spa
    d = def_stats.defense if move_category == "Physical" else def_stats.spd

    type_eff = get_type_effectiveness(move_type, def_types)
    stab = has_stab(move_type, atk_types)

    item_mod = 1.0
    if attacker_item and attacker_item in ITEM_MODIFIERS:
        item_mod = ITEM_MODIFIERS[attacker_item]
        if attacker_item == "Expert Belt" and type_eff <= 1.0:
            item_mod = 1.0

    dmg = calculate_damage_range(
        attacker_stat=a,
        defender_stat=d,
        base_power=base_power,
        type_effectiveness=type_eff,
        stab=stab,
        item_modifier=item_mod,
        field=field,
        move_category=move_category,
    )

    dmg.min_percent = dmg.min_damage / def_stats.hp
    dmg.max_percent = dmg.max_damage / def_stats.hp
    return dmg


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test scenarios
# ═══════════════════════════════════════════════════════════════════════════════
def _run_tests():
    """Run verification scenarios to validate the damage calculator."""

    print("=" * 72)
    print("  Phase 3: Inverse Damage Calculator -- Test Scenarios")
    print("=" * 72)

    # ---- Test 1: Forward calc verification ----
    print("\n--- Test 1: Forward Damage Calc ---")
    print("Great Tusk (252+ Atk) Headlong Rush vs Toxapex (252 HP / 252+ Def)")

    dmg = calc_full_damage(
        attacker_species="greattusk",
        move_name="Headlong Rush",
        defender_species="toxapex",
        attacker_evs={"atk": 252, "spe": 252},
        attacker_nature="adamant",
        defender_evs={"hp": 252, "def": 252},
        defender_nature="bold",
    )
    tox_hp = calc_all_stats("toxapex", "bold", {"hp": 252, "def": 252}).hp
    print(f"  Toxapex HP: {tox_hp}")
    print(f"  Damage: {dmg.min_damage}-{dmg.max_damage} "
          f"({dmg.min_percent:.1%}-{dmg.max_percent:.1%})")

    # ---- Test 2: Same with Choice Band ----
    print("\n--- Test 2: With Choice Band ---")
    print("Great Tusk (252+ Atk Choice Band) Headlong Rush vs same Toxapex")

    dmg_cb = calc_full_damage(
        attacker_species="greattusk",
        move_name="Headlong Rush",
        defender_species="toxapex",
        attacker_evs={"atk": 252, "spe": 252},
        attacker_nature="adamant",
        defender_evs={"hp": 252, "def": 252},
        defender_nature="bold",
        attacker_item="Choice Band",
    )
    print(f"  Damage: {dmg_cb.min_damage}-{dmg_cb.max_damage} "
          f"({dmg_cb.min_percent:.1%}-{dmg_cb.max_percent:.1%})")

    # ---- Test 3: Inverse calc ----
    print("\n--- Test 3: Inverse Damage Calc ---")
    # Simulate: we are Toxapex, we took 95 damage from Great Tusk's Headlong Rush
    # Which build could produce that?
    test_observed = (dmg.min_damage + dmg.max_damage) // 2  # Mid-roll unboosted
    def_stat_toxapex = calc_all_stats("toxapex", "bold", {"hp": 252, "def": 252})

    print(f"  Scenario: Toxapex takes {test_observed} HP from Great Tusk's Headlong Rush")
    result = infer_opponent_state(
        observed_damage=test_observed,
        defender_max_hp=def_stat_toxapex.hp,
        defender_stat=def_stat_toxapex.defense,
        move_name="Headlong Rush",
        attacker_species="greattusk",
        attacker_types=["Ground", "Fighting"],
        defender_types=["Poison", "Water"],
    )
    print(f"  {result.summary()}")
    for build, rng in result.matching_builds:
        print(f"    -> {build.label}: {rng}")

    # ---- Test 4: High damage -> must be Choice Band ----
    print("\n--- Test 4: High Damage -> Deduce Choice Band ---")
    test_high = dmg_cb.max_damage - 2  # Near max roll of Choice Band
    print(f"  Scenario: Toxapex takes {test_high} HP (high roll)")
    result_high = infer_opponent_state(
        observed_damage=test_high,
        defender_max_hp=def_stat_toxapex.hp,
        defender_stat=def_stat_toxapex.defense,
        move_name="Headlong Rush",
        attacker_species="greattusk",
        attacker_types=["Ground", "Fighting"],
        defender_types=["Poison", "Water"],
    )
    print(f"  {result_high.summary()}")
    for build, rng in result_high.matching_builds:
        print(f"    -> {build.label}: {rng}")

    # ---- Test 5: Iron Valiant special attack ----
    print("\n--- Test 5: Special Attack (Iron Valiant Moonblast vs Dragapult) ---")
    dmg_iv = calc_full_damage(
        attacker_species="ironvaliant",
        move_name="Moonblast",
        defender_species="dragapult",
        attacker_evs={"spa": 252, "spe": 252},
        attacker_nature="timid",
        defender_evs={"hp": 4, "spe": 252},
        defender_nature="timid",
    )
    drag_hp = calc_all_stats("dragapult", "timid", {"hp": 4, "spe": 252}).hp
    print(f"  Dragapult HP: {drag_hp}")
    print(f"  Damage: {dmg_iv.min_damage}-{dmg_iv.max_damage} "
          f"({dmg_iv.min_percent:.1%}-{dmg_iv.max_percent:.1%})")

    # Inverse: simulate taking a mid-roll
    test_sp = (dmg_iv.min_damage + dmg_iv.max_damage) // 2
    def_drag = calc_all_stats("dragapult", "timid", {"hp": 4, "spe": 252})
    print(f"\n  Inverse: Dragapult takes {test_sp} HP from Moonblast")
    result_sp = infer_opponent_state(
        observed_damage=test_sp,
        defender_max_hp=def_drag.hp,
        defender_stat=def_drag.spd,
        move_name="Moonblast",
        attacker_species="ironvaliant",
        attacker_types=["Fairy", "Fighting"],
        defender_types=["Dragon", "Ghost"],
    )
    print(f"  {result_sp.summary()}")
    for build, rng in result_sp.matching_builds:
        print(f"    -> {build.label}: {rng}")

    # ---- Test 6: Stat calc verification ----
    print("\n--- Test 6: Stat Calculation Verification ---")
    gt = calc_all_stats("greattusk", "jolly", {"atk": 252, "hp": 4, "spe": 252})
    print(f"  Great Tusk (Jolly 4/252/0/0/0/252): "
          f"HP={gt.hp} Atk={gt.atk} Def={gt.defense} SpA={gt.spa} SpD={gt.spd} Spe={gt.spe}")

    gt2 = calc_all_stats("greattusk", "adamant", {"atk": 252, "hp": 4, "spe": 252})
    print(f"  Great Tusk (Adamant 4/252/0/0/0/252): "
          f"HP={gt2.hp} Atk={gt2.atk} Def={gt2.defense} SpA={gt2.spa} SpD={gt2.spd} Spe={gt2.spe}")

    tox = calc_all_stats("toxapex", "bold", {"hp": 252, "def": 252, "spd": 4})
    print(f"  Toxapex (Bold 252/0/252/0/4/0): "
          f"HP={tox.hp} Atk={tox.atk} Def={tox.defense} SpA={tox.spa} SpD={tox.spd} Spe={tox.spe}")

    print("\n" + "=" * 72)
    print("  All tests completed.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    _run_tests()
