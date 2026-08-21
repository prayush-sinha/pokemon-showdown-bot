"""
bot.py -- Phase 1+2+3+4: Server Bridge + Priors + Inverse Calc + Expectiminimax

A Pokemon Showdown bot powered by:
  - Phase 1: poke-env server bridge & event loop.
  - Phase 2: Smogon statistical usage priors for opponent builds.
  - Phase 3: Inverse damage calculation to deduce hidden stats/items.
  - Phase 4: Simultaneous Expectiminimax search engine for turn decisions.

Usage
-----
  # 1. Start a local Pokemon Showdown server (optional for live play)
  # 2. Run:
  python bot.py                         # accept challenges from anyone
  python bot.py --challenge <username>  # challenge a specific user
  python bot.py --ladder 5              # play 5 ladder games
  python bot.py --self-test 3           # local bot-vs-bot smoke test
  python bot.py --dry-run               # mock battle smoke test (no server needed)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import Player, RandomPlayer

import config
from smogon_priors import SmogonPriors
from inverse_damage_calc import (
    infer_opponent_state,
    calc_all_stats,
    FieldConditions,
    PokemonStats,
    _POKEDEX,
)
from expectiminimax import ExpectiminimaxEngine

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("FutureSightBot")


# =============================================================================
# FutureSightBot -- Complete Integrated AI
# =============================================================================
class FutureSightBot(Player):
    """
    Phase 4 Integrated Bot:
      - Inverse damage calculations reverse-engineer hidden opponent builds.
      - Smogon usage priors scout likely unknown moves, items, and abilities.
      - Expectiminimax search engine calculates Expected Values across
        simultaneous turns, chance/RNG nodes, and speed tiers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Phase 2: Smogon priors engine (shared across all battles)
        self._priors = SmogonPriors(
            format_id=config.PRIORS_FORMAT,
            cache_max_age=config.PRIORS_CACHE_MAX_AGE,
        )

        # Phase 4: Expectiminimax Search Engine
        self.search_engine = ExpectiminimaxEngine(
            depth=2,
            smogon_priors=self._priors,
            max_time_ms=500.0,
        )

        # Per-battle tracking: which opponent species we've already scouted.
        # Keyed by battle.battle_tag -> set of species strings.
        self._scouted: dict[str, set[str]] = {}

        # Phase 3: HP snapshots for detecting incoming damage.
        # Keyed by battle.battle_tag -> {pokemon_species: last_known_hp}
        self._hp_snapshots: dict[str, dict[str, int]] = {}

        # Phase 3: Confirmed opponent info from inverse calc.
        # Keyed by battle.battle_tag -> {species: {"item": ..., "ev_level": ...}}
        self._opponent_profiles: dict[str, dict[str, dict]] = {}

    # ── Core decision hook ──────────────────────────────────────────────────
    def choose_move(self, battle):
        """
        Called every turn by poke-env with the current Battle state.

        Turn Pipeline:
          1. Log turn snapshot.
          2. Detect & analyse incoming damage (Phase 3 inverse calc).
          3. Scout new opponent Pokemon (Phase 2 Smogon priors).
          4. Snapshot our active Pokemon's HP for next turn.
          5. Run Expectiminimax search engine to find optimal action (Phase 4B).
          6. Return poke-env order (or safe fallback).
        """
        # 1. Log turn summary
        _log_turn_state(battle)

        # 2. Phase 3: detect and analyse incoming damage
        self._analyse_incoming_damage(battle)

        # 3. Phase 2: scout the opponent if we haven't yet this battle
        self._scout_opponent(battle)

        # 4. Phase 3: snapshot our active Pokemon's HP for next turn
        self._snapshot_hp(battle)

        # 5. Phase 4: Expectiminimax Search
        try:
            tag = getattr(battle, "battle_tag", "")
            profiles = self._opponent_profiles.get(tag, {})
            best_action, ranked = self.search_engine.get_best_action(
                battle,
                opponent_profiles=profiles,
            )

            if ranked:
                top_3 = ", ".join(f"{label} ({ev:+.1f} EV)" for label, ev in ranked[:3])
                logger.info(
                    "[BRAIN] Turn %s candidates: [%s] -> Chosen: %s",
                    getattr(battle, "turn", "?"), top_3, ranked[0][0]
                )

            if best_action is not None:
                return self.create_order(best_action)
        except Exception as exc:
            logger.warning("[BRAIN] Search failed with error (%s), falling back to random move", exc, exc_info=True)

        # 6. Fallback
        return self.choose_random_move(battle)

    # ── Phase 2: Smogon prior scouting ──────────────────────────────────────
    def _scout_opponent(self, battle) -> None:
        """
        Check if the opponent's active Pokemon is new to us this battle.
        If so, query Smogon priors and log the predicted build.
        """
        opp = getattr(battle, "opponent_active_pokemon", None)
        if opp is None:
            return

        tag = getattr(battle, "battle_tag", "")
        if tag not in self._scouted:
            self._scouted[tag] = set()

        species = getattr(opp, "species", None)
        if not species or species in self._scouted[tag]:
            return

        self._scouted[tag].add(species)

        build = self._priors.get_likely_build(species)
        if build is None:
            logger.info(
                "[PRIOR] Opponent sent out %s -- no data in %s",
                species, config.PRIORS_FORMAT,
            )
            return

        logger.info(
            "[PRIOR] Opponent sent out %s. Assuming: %s",
            build.species, build.summary(),
        )

        known_moves = [m.id for m in opp.moves.values()] if getattr(opp, "moves", None) else []
        known_item = getattr(opp, "item", None)
        known_ability = getattr(opp, "ability", None)
        if known_moves:
            logger.info("[PRIOR]   Already revealed moves: %s", known_moves)
        if known_item:
            logger.info("[PRIOR]   Already revealed item: %s", known_item)
        if known_ability:
            logger.info("[PRIOR]   Already revealed ability: %s", known_ability)

    # ── Phase 3: HP snapshot ────────────────────────────────────────────────
    def _snapshot_hp(self, battle) -> None:
        """Record our active Pokemon's current HP for damage detection next turn."""
        active = getattr(battle, "active_pokemon", None)
        if active is None:
            return

        tag = getattr(battle, "battle_tag", "")
        if tag not in self._hp_snapshots:
            self._hp_snapshots[tag] = {}

        species = getattr(active, "species", None)
        if species:
            self._hp_snapshots[tag][species] = getattr(active, "current_hp", 0)

    # ── Phase 3: Inverse damage analysis ────────────────────────────────────
    def _analyse_incoming_damage(self, battle) -> None:
        """
        Compare our active Pokemon's HP to last turn's snapshot.
        If it dropped, and the opponent used a damaging move, run the
        inverse damage calculator to deduce their build.
        """
        active = getattr(battle, "active_pokemon", None)
        opp = getattr(battle, "opponent_active_pokemon", None)
        tag = getattr(battle, "battle_tag", "")

        if active is None or opp is None:
            return

        # Need a previous snapshot to compare against
        prev_hp_map = self._hp_snapshots.get(tag, {})
        prev_hp = prev_hp_map.get(active.species)
        if prev_hp is None:
            return

        current_hp = getattr(active, "current_hp", 0)
        damage_taken = prev_hp - current_hp

        if damage_taken <= 0:
            return

        opp_last_move = getattr(opp, "last_move", None)
        if opp_last_move is None:
            return

        # Skip status moves
        if getattr(opp_last_move, "category", "") == "Status" or getattr(opp_last_move, "base_power", 0) == 0:
            return

        move_category = getattr(opp_last_move, "category", "Physical")
        our_stats = getattr(active, "stats", None)
        if our_stats is None:
            return

        if move_category == "Physical":
            defender_stat = our_stats.get("def")
        else:
            defender_stat = our_stats.get("spd")

        if defender_stat is None:
            return

        defender_max_hp = getattr(active, "max_hp", 0)
        if defender_max_hp <= 0:
            return

        # Resolve types
        atk_types = [t.name if hasattr(t, "name") else str(t)
                     for t in (getattr(opp, "types", ()) or ()) if t is not None]
        def_types = [t.name if hasattr(t, "name") else str(t)
                     for t in (getattr(active, "types", ()) or ()) if t is not None]

        field = _build_field_conditions(battle)

        opp_boosts = getattr(opp, "boosts", {}) or {}
        atk_stage = opp_boosts.get("atk", 0) if move_category == "Physical" else opp_boosts.get("spa", 0)

        our_boosts = getattr(active, "boosts", {}) or {}
        def_stage = our_boosts.get("def", 0) if move_category == "Physical" else our_boosts.get("spd", 0)

        try:
            result = infer_opponent_state(
                observed_damage=damage_taken,
                defender_max_hp=defender_max_hp,
                defender_stat=defender_stat,
                move_name=opp_last_move.id,
                attacker_species=opp.species,
                attacker_types=atk_types,
                defender_types=def_types,
                field=field,
                stat_stage_atk=atk_stage,
                stat_stage_def=def_stage,
            )
        except Exception as exc:
            logger.debug("[INVERSE CALC] Error: %s", exc)
            return

        if result.matching_builds:
            logger.info("[INVERSE CALC] %s", result.summary())

            if tag not in self._opponent_profiles:
                self._opponent_profiles[tag] = {}
            if opp.species not in self._opponent_profiles[tag]:
                self._opponent_profiles[tag][opp.species] = {}

            profile = self._opponent_profiles[tag][opp.species]
            if result.best_guess_item and not profile.get("item"):
                profile["item"] = result.best_guess_item
                profile["ev_level"] = result.best_guess_evs
                profile["est_atk"] = result.estimated_attack_stat
                logger.info(
                    "[INVERSE CALC]   Updated profile for %s: item=%s, %s",
                    opp.species, profile["item"], profile["ev_level"],
                )

    # ── Team-preview hook ───────────────────────────────────────────────────
    def teampreview(self, battle):
        """Called once at start of formats with team preview."""
        n_pokemon = len(getattr(battle, "team", {}) or {})
        order = "".join(str(i + 1) for i in range(n_pokemon))
        return f"/team {order}"


# =============================================================================
# Diagnostic logging helper
# =============================================================================
def _log_turn_state(battle) -> None:
    """Print a human-readable snapshot of the turn for debugging."""
    active = getattr(battle, "active_pokemon", None)
    opp = getattr(battle, "opponent_active_pokemon", None)

    avail_moves = getattr(battle, "available_moves", []) or []
    avail_switches = getattr(battle, "available_switches", []) or []

    moves = [m.id for m in avail_moves]
    switches = [getattr(p, "species", "?") for p in avail_switches]

    logger.info(
        "Turn %s | %s (%.0f%% HP) vs %s (%.0f%% HP) | Moves: %s | Switches: %s",
        getattr(battle, "turn", "?"),
        getattr(active, "species", "???") if active else "???",
        (active.current_hp_fraction * 100) if (active and active.current_hp_fraction) else 0,
        getattr(opp, "species", "???") if opp else "???",
        (opp.current_hp_fraction * 100) if (opp and opp.current_hp_fraction) else 0,
        moves or "--",
        switches or "--",
    )


# =============================================================================
# Phase 3: Field condition builder
# =============================================================================
def _build_field_conditions(battle) -> FieldConditions:
    """Translate poke-env's battle state into a FieldConditions object."""
    from poke_env.battle.weather import Weather
    from poke_env.battle.side_condition import SideCondition

    field = FieldConditions()

    weather_map = {
        Weather.SUNNYDAY: "sun",
        Weather.DESOLATELAND: "sun",
        Weather.RAINDANCE: "rain",
        Weather.PRIMORDIALSEA: "rain",
        Weather.SANDSTORM: "sand",
        Weather.SNOWSCAPE: "snow",
        Weather.HAIL: "snow",
    }
    for w in getattr(battle, "weather", {}) or {}:
        if w in weather_map:
            field.weather = weather_map[w]
            break

    our_sides = getattr(battle, "side_conditions", {}) or {}
    if SideCondition.REFLECT in our_sides:
        field.reflect = True
    if SideCondition.LIGHT_SCREEN in our_sides:
        field.light_screen = True

    opp = getattr(battle, "opponent_active_pokemon", None)
    if opp and getattr(opp, "status", None) and getattr(opp.status, "name", "") == "BRN":
        field.attacker_burned = True

    return field


# =============================================================================
# Factory -- build a properly-configured FutureSightBot
# =============================================================================
def make_bot(
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_showdown: bool = False,
) -> FutureSightBot:
    """Construct a FutureSightBot wired to the correct server."""
    username = username or config.BOT_USERNAME
    password = password or config.BOT_PASSWORD

    kwargs = dict(
        battle_format=config.BATTLE_FORMAT,
        max_concurrent_battles=config.MAX_CONCURRENT_BATTLES,
        log_level=config.LOG_LEVEL,
    )

    if use_showdown or config.SERVER_MODE == "showdown":
        kwargs["account_configuration"] = AccountConfiguration(username, password)
        kwargs["server_configuration"] = ShowdownServerConfiguration
        logger.info("Connecting to play.pokemonshowdown.com as '%s'", username)
    else:
        kwargs["account_configuration"] = AccountConfiguration(username, None)
        logger.info("Connecting to local server as '%s'", username)

    return FutureSightBot(**kwargs)


# =============================================================================
# Run modes
# =============================================================================
async def mode_accept_challenges(bot: FutureSightBot, n: int = 1) -> None:
    """Sit in the lobby and accept challenges from anyone."""
    logger.info("Waiting for %d challenge(s) from any user ...", n)
    await bot.accept_challenges(None, n)
    logger.info("All challenges completed.")


async def mode_challenge(bot: FutureSightBot, opponent: str, n: int = 1) -> None:
    """Send a challenge to a specific user."""
    logger.info("Challenging '%s' (%d game(s)) ...", opponent, n)
    await bot.send_challenges(opponent, n_challenges=n)
    logger.info("Challenge session finished.")


async def mode_ladder(bot: FutureSightBot, n: int = 5) -> None:
    """Queue on the ranked ladder."""
    logger.info("Entering ladder for %d game(s) ...", n)
    await bot.ladder(n)
    for tag, battle in bot.battles.items():
        logger.info(
            "  %s  ->  %s  (rating: %s vs %s)",
            tag,
            "WIN" if battle.won else "LOSS",
            battle.rating,
            battle.opponent_rating,
        )


async def mode_self_test(n_battles: int = 3) -> None:
    """
    Local smoke test: spin up TWO bots on localhost and have them
    fight each other. No human interaction required.
    """
    logger.info("=" * 60)
    logger.info("  SELF-TEST MODE -- %d bot-vs-bot battles on localhost", n_battles)
    logger.info("=" * 60)

    bot_a = FutureSightBot(
        account_configuration=AccountConfiguration("FutureSight-A", None),
        battle_format=config.BATTLE_FORMAT,
        max_concurrent_battles=config.MAX_CONCURRENT_BATTLES,
        log_level=config.LOG_LEVEL,
    )
    bot_b = RandomPlayer(
        account_configuration=AccountConfiguration("RandomBaseline", None),
        battle_format=config.BATTLE_FORMAT,
        max_concurrent_battles=config.MAX_CONCURRENT_BATTLES,
        log_level=config.LOG_LEVEL,
    )

    try:
        await bot_a.battle_against(bot_b, n_battles=n_battles)
    finally:
        await bot_a.ps_client.stop_listening()
        await bot_b.ps_client.stop_listening()

    wins = bot_a.n_won_battles
    total = bot_a.n_finished_battles
    logger.info("=" * 60)
    logger.info("  RESULTS: FutureSight-A won %d / %d", wins, total)
    logger.info("=" * 60)


def mode_dry_run() -> None:
    """
    Dry-run simulation test: runs choose_move on mock Battle objects
    to verify end-to-end integration without needing a server.
    """
    print("=" * 72)
    print("  FutureSightBot Dry-Run Simulation Test")
    print("=" * 72)

    from evaluator import _MockPokemon, _MockMove, _MockBattle

    bot = FutureSightBot(
        account_configuration=AccountConfiguration("TestBot", None),
        battle_format="gen9ou",
    )

    # Setup mock battle
    chomp = _MockPokemon(
        "garchomp", hp_fraction=1.0, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
    )
    tox = _MockPokemon(
        "toxapex", hp_fraction=1.0, max_hp=304,
        stats={"atk": 152, "def": 353, "spa": 137, "spd": 293, "spe": 96},
        types=["Poison", "Water"],
    )
    opp_draga = _MockPokemon(
        "dragapult", hp_fraction=1.0, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
    )

    eq = _MockMove("earthquake")
    dc = _MockMove("dragonclaw")
    sd = _MockMove("swordsdance")

    mock_battle = _MockBattle(
        our_team={"p1a": chomp, "p1b": tox},
        opp_team={"p2a": opp_draga},
        active_pokemon=chomp,
        opponent_active_pokemon=opp_draga,
        available_moves=[eq, dc, sd],
        available_switches=[tox],
        turn=1,
    )

    print("\n--- Testing bot.choose_move(mock_battle) ---")
    order = bot.choose_move(mock_battle)
    print(f"  Resulting Order: {order}")
    print("\n" + "=" * 72)
    print("  Dry-run verification completed successfully [OK]")
    print("=" * 72)


# =============================================================================
# CLI Entry Point
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FutureSight Bot -- Phase 4 Expectiminimax Battle AI",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--accept",
        metavar="N",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help="Accept N challenges from any user (default: 1).",
    )
    group.add_argument(
        "--challenge",
        metavar="USERNAME",
        type=str,
        default=None,
        help="Challenge a specific user.",
    )
    group.add_argument(
        "--ladder",
        metavar="N",
        type=int,
        nargs="?",
        const=5,
        default=None,
        help="Play N ladder games (default: 5).",
    )
    group.add_argument(
        "--self-test",
        metavar="N",
        type=int,
        nargs="?",
        const=3,
        default=None,
        dest="self_test",
        help="Run N bot-vs-bot games locally (default: 3).",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run an in-memory simulation test without connecting to a server.",
    )

    parser.add_argument(
        "--showdown",
        action="store_true",
        help="Connect to play.pokemonshowdown.com instead of localhost.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Override the bot username from config.py.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Override the bot password from config.py.",
    )

    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Dry-run mode ───────────────────────────────────────────────────────
    if args.dry_run:
        mode_dry_run()
        return

    # ── Self-test mode ─────────────────────────────────────────────────────
    if args.self_test is not None:
        await mode_self_test(args.self_test)
        return

    # ── Live play ──────────────────────────────────────────────────────────
    bot = make_bot(
        username=args.username,
        password=args.password,
        use_showdown=args.showdown,
    )

    try:
        if args.challenge:
            await mode_challenge(bot, args.challenge)
        elif args.ladder is not None:
            await mode_ladder(bot, args.ladder)
        else:
            n = args.accept if args.accept is not None else 1
            await mode_accept_challenges(bot, n)
    finally:
        await bot.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
