"""
test_suite.py -- Phase 5 Battle-Hardening Automated Regression Suite

Ten targeted regression tests, one class per Phase 5 fix/feature, covering
exactly what the upgrade roadmap specified:

  TestIterativeDeepening   (5.1: expectiminimax.py)
    - test_timeout_graceful_abort
    - test_speed_tie_branch_normalization
  TestZoroarkGuard          (5.3: bot.py)
    - test_disguised_damage_bypass
    - test_break_reactivation
  TestDittoGuard            (5.3: bot.py / inverse_damage_calc.py)
    - test_ditto_species_bypass
    - test_transform_damage_profile_isolation
  TestEndgameHeuristic      (5.2: evaluator.py)
    - test_endgame_detection
    - test_hazard_zeroing_and_ko_bonus
  TestRangeIntersection     (5.3: inverse_damage_calc.py)
    - test_successive_narrowing
    - test_empty_intersection_fallback

Run with:
    python3 test_suite.py
    python3 -m unittest test_suite -v
"""
import time
import unittest

from evaluator import (
    evaluate_state,
    _is_endgame,
    ENDGAME_OPP_ALIVE_THRESHOLD,
    ENDGAME_ALIVE_BONUS_MULTIPLIER,
    ENDGAME_HAZARD_MULTIPLIER,
    _MockPokemon,
    _MockMove,
    _MockBattle,
    _MockSideCondition,
)
from expectiminimax import (
    ExpectiminimaxEngine,
    _simulate_ordered_outcomes,
)
from inverse_damage_calc import infer_opponent_state
from bot import FutureSightBot


# ─────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────
def _garchomp(**overrides):
    kwargs = dict(
        species="garchomp", hp_fraction=1.0, max_hp=357,
        stats={"atk": 394, "def": 226, "spa": 196, "spd": 206, "spe": 333},
        types=["Dragon", "Ground"],
    )
    kwargs.update(overrides)
    return _MockPokemon(**kwargs)


def _dragapult(**overrides):
    kwargs = dict(
        species="dragapult", hp_fraction=1.0, max_hp=291,
        stats={"atk": 339, "def": 186, "spa": 299, "spd": 186, "spe": 421},
        types=["Dragon", "Ghost"],
    )
    kwargs.update(overrides)
    return _MockPokemon(**kwargs)


def _toxapex(**overrides):
    kwargs = dict(
        species="toxapex", hp_fraction=1.0, max_hp=304,
        stats={"atk": 152, "def": 353, "spa": 137, "spd": 293, "spe": 96},
        types=["Poison", "Water"],
    )
    kwargs.update(overrides)
    return _MockPokemon(**kwargs)


# =============================================================================
class TestIterativeDeepening(unittest.TestCase):
    """Phase 5.1: iterative deepening loop, monotonic timer, speed ties."""

    def test_timeout_graceful_abort(self):
        """
        Under an almost-zero time budget, the engine must still return a
        legal action immediately (Depth 1 always completes unconditionally
        as a safety floor) rather than hanging or returning None -- this is
        the "discard the partial ply, keep the last full depth" fallback
        exercised at its most extreme.
        """
        eq, dc, toxic = _MockMove("earthquake"), _MockMove("dragonclaw"), _MockMove("toxic")
        opp_dm = _MockMove("dracometeor")

        garchomp = _garchomp()
        toxapex = _toxapex()
        dragapult = _dragapult(moves={"dracometeor": opp_dm})

        battle = _MockBattle(
            our_team={"p1: Garchomp": garchomp, "p1: Toxapex": toxapex},
            opp_team={"p2: Dragapult": dragapult},
            active_pokemon=garchomp,
            opponent_active_pokemon=dragapult,
            available_moves=[eq, dc, toxic],
            available_switches=[toxapex],
        )

        engine = ExpectiminimaxEngine(depth=4, smogon_priors=None, max_time_ms=0.001)
        start = time.perf_counter()
        best_action, ranked = engine.get_best_action(battle)
        elapsed_s = time.perf_counter() - start

        self.assertIsNotNone(best_action, "must return a legal action even under a near-zero time budget")
        self.assertGreater(len(ranked), 0)
        self.assertLess(elapsed_s, 2.0, "search must never hang, regardless of budget")

    def test_speed_tie_branch_normalization(self):
        """
        Identical priority AND identical effective speed must branch into
        exactly two 50/50 chance nodes (real Showdown coin-flips this)
        rather than deterministically awarding the tie to one side. A
        genuine speed edge, or a priority edge, must NOT branch.
        """
        tied = _simulate_ordered_outcomes(our_prio=0, opp_prio=0, our_spe=333, opp_spe=333)
        self.assertEqual(len(tied), 2)
        probs = sorted(p for _, p in tied)
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)
        self.assertEqual({b for b, _ in tied}, {True, False})
        self.assertAlmostEqual(sum(p for _, p in tied), 1.0)

        faster = _simulate_ordered_outcomes(our_prio=0, opp_prio=0, our_spe=400, opp_spe=100)
        self.assertEqual(faster, [(True, 1.0)])

        priority_override = _simulate_ordered_outcomes(our_prio=1, opp_prio=0, our_spe=1, opp_spe=999)
        self.assertEqual(priority_override, [(True, 1.0)])


# =============================================================================
class TestZoroarkGuard(unittest.TestCase):
    """Phase 5.3: Zoroark Illusion identity protection (bot.py)."""

    def test_disguised_damage_bypass(self):
        """
        If Zoroark/Zoroark-Hisui is anywhere on the opponent's revealed
        team and the displayed active species hasn't taken direct damage
        yet this battle, inverse calc must be bypassed for it -- its
        identity is unconfirmed and it could BE the disguise.
        """
        b = FutureSightBot(start_listening=False)

        zoroark = _MockPokemon("zoroark", hp_fraction=1.0, max_hp=280,
                                stats={"atk": 260, "def": 150, "spa": 260, "spd": 150, "spe": 250},
                                types=["Dark"])
        disguised = _garchomp()  # displayed identity, possibly fake

        battle = _MockBattle(
            our_team={}, opp_team={"p2: Zoroark": zoroark, "p2: Garchomp": disguised},
            opponent_active_pokemon=disguised,
        )

        should_bypass, reason = b._should_bypass_inverse_calc(battle, disguised)
        self.assertTrue(should_bypass)
        self.assertIn("Illusion", reason)

    def test_break_reactivation(self):
        """
        Once the displayed active Pokemon takes direct observable damage
        (an HP drop between two _track_opponent_identity snapshots), the
        Illusion guard must release and inverse calc must run normally
        again on subsequent hits.
        """
        b = FutureSightBot(start_listening=False)

        zoroark = _MockPokemon("zoroark", hp_fraction=1.0, max_hp=280,
                                stats={"atk": 260, "def": 150, "spa": 260, "spd": 150, "spe": 250},
                                types=["Dark"])
        disguised = _garchomp()

        battle = _MockBattle(
            our_team={}, opp_team={"p2: Zoroark": zoroark, "p2: Garchomp": disguised},
            opponent_active_pokemon=disguised,
        )

        # Turn N: first snapshot, still unconfirmed -> bypass active
        b._track_opponent_identity(battle)
        still_unconfirmed, _ = b._should_bypass_inverse_calc(battle, disguised)
        self.assertTrue(still_unconfirmed)

        # Turn N+1: it took damage -- HP dropped since the last snapshot
        disguised._current_hp = int(disguised.max_hp * 0.7)
        b._track_opponent_identity(battle)

        now_confirmed, _ = b._should_bypass_inverse_calc(battle, disguised)
        self.assertFalse(now_confirmed, "guard must release once direct damage confirms identity")


# =============================================================================
class TestDittoGuard(unittest.TestCase):
    """Phase 5.3: Ditto / Transform / Imposter profile isolation (bot.py)."""

    def test_ditto_species_bypass(self):
        """A bare Ditto (species alone) must always be bypassed."""
        b = FutureSightBot(start_listening=False)
        ditto = _MockPokemon("ditto", hp_fraction=1.0, max_hp=250,
                              stats={"atk": 88, "def": 90, "spa": 88, "spd": 90, "spe": 90},
                              types=["Normal"])
        battle = _MockBattle(our_team={}, opp_team={"p2: Ditto": ditto}, opponent_active_pokemon=ditto)

        should_bypass, reason = b._should_bypass_inverse_calc(battle, ditto)
        self.assertTrue(should_bypass)
        self.assertIn("Ditto", reason)

    def test_transform_damage_profile_isolation(self):
        """
        Any Pokemon holding the Imposter ability, or that has revealed
        Transform as a used move, has COPIED stats from whatever it
        transformed into. Inverse calc must be bypassed so those borrowed
        stats never overwrite that species' real profile.
        """
        b = FutureSightBot(start_listening=False)

        imposter_mon = _MockPokemon("mrmime", hp_fraction=1.0, max_hp=280,
                                     stats={"atk": 260, "def": 300, "spa": 260, "spd": 300, "spe": 260},
                                     types=["Psychic", "Fairy"])
        imposter_mon.ability = "imposter"  # setter writes _ability since it starts None

        battle = _MockBattle(our_team={}, opp_team={"p2: MrMime": imposter_mon}, opponent_active_pokemon=imposter_mon)
        should_bypass, _ = b._should_bypass_inverse_calc(battle, imposter_mon)
        self.assertTrue(should_bypass)

        transform_move = _MockMove("transform")
        transformed_mon = _MockPokemon("smeargle", hp_fraction=1.0, max_hp=250,
                                        stats={"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
                                        types=["Normal"], moves={"transform": transform_move})
        battle2 = _MockBattle(our_team={}, opp_team={"p2: Smeargle": transformed_mon},
                               opponent_active_pokemon=transformed_mon)
        should_bypass2, _ = b._should_bypass_inverse_calc(battle2, transformed_mon)
        self.assertTrue(should_bypass2)


# =============================================================================
class TestEndgameHeuristic(unittest.TestCase):
    """Phase 5.2: Endgame Anti-Choke Logic (evaluator.py)."""

    def test_endgame_detection(self):
        self.assertFalse(_is_endgame(0), "0 alive is terminal, not an 'endgame setup' state")
        self.assertTrue(_is_endgame(1))
        self.assertTrue(_is_endgame(ENDGAME_OPP_ALIVE_THRESHOLD))
        self.assertFalse(_is_endgame(ENDGAME_OPP_ALIVE_THRESHOLD + 1))
        self.assertFalse(_is_endgame(6))

    def test_hazard_zeroing_and_ko_bonus(self):
        """
        With the opponent down to their last Pokemon: hazard value must
        collapse to exactly 0 (vs. clearly positive outside the endgame),
        and the alive/KO differential multiplier the evaluator applies
        must be the specified 2.5x.
        """
        garchomp = _garchomp()
        dragapult = _dragapult()
        filler_a = _dragapult(species="hydreigon")
        filler_b = _dragapult(species="salamence")
        sr = _MockSideCondition("STEALTH_ROCK")

        # Non-endgame baseline (opponent has 3 mons alive): hazards score positive.
        baseline_with = _MockBattle(
            our_team={"p1: Garchomp": garchomp},
            opp_team={"p2: Dragapult": dragapult, "p2: Hydreigon": filler_a, "p2: Salamence": filler_b},
            active_pokemon=garchomp, opponent_active_pokemon=dragapult,
            opponent_side_conditions={sr: 1},
        )
        baseline_without = _MockBattle(
            our_team={"p1: Garchomp": garchomp},
            opp_team={"p2: Dragapult": dragapult, "p2: Hydreigon": filler_a, "p2: Salamence": filler_b},
            active_pokemon=garchomp, opponent_active_pokemon=dragapult,
        )
        baseline_hazard_value = evaluate_state(baseline_with) - evaluate_state(baseline_without)
        self.assertGreater(baseline_hazard_value, 0)

        # Endgame (opponent down to their last Pokemon): hazard value must be exactly 0.
        endgame_with = _MockBattle(
            our_team={"p1: Garchomp": garchomp}, opp_team={"p2: Dragapult": dragapult},
            active_pokemon=garchomp, opponent_active_pokemon=dragapult,
            opponent_side_conditions={sr: 1},
        )
        endgame_without = _MockBattle(
            our_team={"p1: Garchomp": garchomp}, opp_team={"p2: Dragapult": dragapult},
            active_pokemon=garchomp, opponent_active_pokemon=dragapult,
        )
        endgame_hazard_value = evaluate_state(endgame_with) - evaluate_state(endgame_without)
        self.assertEqual(endgame_hazard_value, 0.0)

        # The multiplier constants are exactly what the roadmap specified,
        # and are what evaluate_state actually reads at runtime.
        self.assertEqual(ENDGAME_ALIVE_BONUS_MULTIPLIER, 2.5)
        self.assertEqual(ENDGAME_HAZARD_MULTIPLIER, 0.0)


# =============================================================================
class TestRangeIntersection(unittest.TestCase):
    """Phase 5.3: Multi-Hit Damage Range Intersection (inverse_damage_calc.py)."""

    _COMMON = dict(
        defender_max_hp=357,
        defender_stat=226,
        move_name="tackle",
        attacker_species="garchomp",
        attacker_types=["Dragon", "Ground"],
        defender_types=["Water"],
    )

    def test_successive_narrowing(self):
        """
        A second hit's window, intersected with the first hit's window,
        must land inside it and be no wider -- repeated hits monotonically
        narrow (or hold), never widen, the estimated Attack/Sp.Atk range.
        """
        first = infer_opponent_state(observed_damage=60, **self._COMMON)
        if first.estimated_attack_range is None:
            self.skipTest("No matching builds for this synthetic scenario (dex data unavailable)")

        second = infer_opponent_state(
            observed_damage=72, existing_range=first.estimated_attack_range, **self._COMMON
        )
        self.assertIsNotNone(second.estimated_attack_range)

        f_lo, f_hi = first.estimated_attack_range
        s_lo, s_hi = second.estimated_attack_range

        self.assertLessEqual(s_hi - s_lo, f_hi - f_lo, "window must not widen after a second hit")
        self.assertGreaterEqual(s_lo, f_lo, "narrowed window must stay inside the original")
        self.assertLessEqual(s_hi, f_hi, "narrowed window must stay inside the original")

    def test_empty_intersection_fallback(self):
        """
        If a prior window and the current hit's window are mutually
        exclusive (e.g. a misattributed hit or a mid-battle stat change),
        the intersection must NOT collapse into an inverted/empty range --
        it must fall back to trusting the latest observation instead.
        """
        result = infer_opponent_state(
            observed_damage=50,
            existing_range=(999_999, 1_000_000),  # deliberately inconsistent prior
            **self._COMMON,
        )
        if result.estimated_attack_range is None:
            self.skipTest("No matching builds for this synthetic scenario (dex data unavailable)")

        lo, hi = result.estimated_attack_range
        self.assertLessEqual(lo, hi, "range must never be inverted or empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
