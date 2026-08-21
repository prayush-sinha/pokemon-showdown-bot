# FutureSightBot
## An Autonomous, Game-Theoretic Architecture for Competitive Pokémon Battle Simulation

**Technical Whitepaper & 60-Day Engineering Development Log**
Version 1.0 · Format Target: Generation 9 OverUsed (gen9ou) · Platform: Pokémon Showdown

---

## Section 1 — Executive Summary & Theoretical Framework

### 1.1 Abstract

FutureSightBot is an autonomous agent for competitive Pokémon battling on the Pokémon Showdown simulator, built on `poke-env` as the WebSocket/protocol bridge. Unlike reinforcement-learning agents that approximate a value function via self-play (e.g., AlphaZero-style architectures), FutureSightBot is a **deterministic, search-and-inference system**: it combines (1) an empirical Bayesian prior model of the metagame derived from Smogon usage statistics, (2) an inverse-damage-calculation engine that deduces hidden opponent state (EVs, nature, held item) from observed battle events, and (3) a depth-limited **Expectiminimax** search over a **simultaneous-move normal-form game** at each turn, scored by a hand-engineered heuristic evaluator. This document formalizes the theoretical basis for that design, and reconstructs the 60-day engineering process by which it was built.

```text
=========================================================
          SYSTEM ARCHITECTURE & TURN PIPELINE
=========================================================
              [ Pokémon Showdown Server ]
                         | (WebSocket Event)
                         v
                    [ bot.py ] 
              (poke-env BattleState)
                         |
                         v
               [ smogon_priors.py ]
         (Filters 800+ moves down to top 4)
                         |
                         v
           [ inverse_damage_calc.py ]
      (Deduces hidden stats and Choice items)
                         |
                         v
               [ expectiminimax.py ]
    (Builds the game tree and chance node matrices)
                         |
                         v
                 [ evaluator.py ]
     (Scores board states using dynamic heuristics)
                         |
                         v
               [ Execute Action ]
          (Sends final move to server)
=========================================================
```

### 1.2 Pokémon as a Formal Game Environment

Formally, a single Pokémon battle turn can be classified as:

$$
\Gamma = \langle N, \{A_i\}_{i \in N}, \{u_i\}_{i \in N}, \Omega, P \rangle
$$

where $N = \{1, 2\}$ (two players), $A_i$ is the action set available to player $i$ (moves ∪ available switches), $u_i$ is player $i$'s utility function, $\Omega$ is a space of chance outcomes (RNG events), and $P$ is a probability distribution over $\Omega$. This makes competitive Pokémon:

- **Two-player** — exactly one opponent per battle (singles format).
- **Zero-sum** — under the win-probability utility convention used here, $u_1(s) = -u_2(s)$ for any terminal state $s$; one player's win-probability gain is the other's loss.
- **Simultaneous-move** — both players submit their turn's action (move or switch) without observing the opponent's choice; action resolution order is determined *after* both commit, based on priority tier and Speed.
- **Non-deterministic** — damage rolls, critical hits, secondary-effect procs, accuracy checks, and speed ties are stochastic.
- **Imperfect information** — a player's exact opponent state (EVs, IVs, nature, held item, remaining PP, ability in ambiguous cases) is hidden and must be inferred from revealed actions.

This combination — simultaneous + stochastic + imperfect-information — is precisely the combination that breaks the two classical game-tree algorithms most engineers reach for first.

### 1.3 Why Standard Minimax and MCTS Fail Here

**Minimax** assumes an alternating-move, perfect-information tree: player 1 acts, player 2 *observes* that action and best-responds. If FutureSightBot's search were implemented naively as "assume I move first in the tree, then let the opponent respond," the opponent branch is granted illegitimate information it would never have during a real turn. This produces a **non-Nash, exploitable policy** — the classical illustration is Rock-Paper-Scissors: a minimax tree that lets player 2 see player 1's committed action before choosing its own will always find a pure counter, which is impossible in the real simultaneous game (whose equilibrium is a *mixed* strategy, uniform 1/3 each). Pokémon has structurally identical "mind-game" subgames (e.g., Protect vs. attack, switch vs. attack into a predicted Choice-locked move).

**Vanilla MCTS with UCB1** has the same defect at the implementation level: each edge in the search tree represents a single actor's decision, so unless the simultaneous-move variant is used (e.g., Smooth-UCT, Exploitability-Descent, or regret-matching self-play), the tree still serializes the two players' choices and converges toward a pure strategy in a domain whose equilibria are frequently mixed. There is also a combinatorial cost: the *effective* branching factor of a simultaneous-move node is the **joint action space**,

$$
b_{\text{eff}} = |A_1| \times |A_2|
$$

rather than $|A_1|$ alone, since every bot action must be evaluated against every plausible opponent action, not just one.

FutureSightBot instead solves each turn as a **one-shot normal-form matrix game** nested inside a stochastic (chance-node) expectation — Expectiminimax rather than Minimax, described next.

### 1.4 The Normal-Form Payoff Matrix

At turn $t$, define the bot's action set $A_1 = \{a_1, a_2, \dots, a_m\}$ (legal moves ∪ legal switches) and the opponent's *estimated* action set $A_2 = \{b_1, \dots, b_n\}$, weighted by the empirical prior model (§1.5). The turn is represented as a payoff matrix $M$:

$$
M =
\begin{bmatrix}
u(a_1, b_1) & u(a_1, b_2) & \cdots & u(a_1, b_n) \\
u(a_2, b_1) & u(a_2, b_2) & \cdots & u(a_2, b_n) \\
\vdots & \vdots & \ddots & \vdots \\
u(a_m, b_1) & u(a_m, b_2) & \cdots & u(a_m, b_n)
\end{bmatrix}
$$

where each cell $u(a_i, b_j)$ is itself not a scalar but an **expectation over the chance node** for that joint action (damage rolls, crit chance, accuracy, speed order) — see §1.5's Expectiminimax formulation. In principle $M$ should be solved for its Nash equilibrium via linear programming over mixed strategies. FutureSightBot uses the pragmatic engineering approximation of **weighting each opponent column $b_j$ by its prior probability $P(b_j)$** (rather than solving a full LP each turn under a hard timer budget) and selecting the bot row that maximizes prior-weighted expected utility — a best-response-to-belief approximation rather than a full equilibrium solve. This tradeoff, and its failure modes, is revisited in Section 3.

### 1.5 Expectiminimax: Decision Nodes vs. Chance Nodes

The search tree alternates three node types:

```text
=========================================================
             SIMULTANEOUS EXPECTIMINIMAX TREE
=========================================================

 [MAX NODE] Bot Decision (e.g., Attack vs. Switch)
   │
   ├── [Bot Action 1: Attack]
   │    │
   │    └── [MIN NODE] Opponent Decision (Simultaneous)
   │         │
   │         ├── [Opponent Action A] (Prior Weight: 65%)
   │         │    │
   │         │    └── [CHANCE NODE] RNG Resolution
   │         │         ├── Roll 100% (Prob: 6%) -> Leaf: V(s1)
   │         │         ├── Roll  85% (Prob: 6%) -> Leaf: V(s2)
   │         │         └── Miss/Crit (Prob: X%) -> Leaf: V(s3)
   │         │
   │         └── [Opponent Action B] (Prior Weight: 35%)
   │              │
   │              └── [CHANCE NODE] ... -> Leaf: V(s4)
   │
   └── [Bot Action 2: Switch] ...

     EV(Node) = Σ [ P(Opponent_Action) * P(RNG_Roll) * V(s) ]
=========================================================
```

Formally, for a MAX node:

$$
\text{EV}_{\text{MAX}}(s) = \max_{a_i \in A_1} \sum_{b_j \in A_2} P(b_j)\; \text{EV}_{\text{CHANCE}}(s, a_i, b_j)
$$

and for a CHANCE node, given the discrete outcome space $\Omega$ of an action pair (16 damage rolls × crit/no-crit × accuracy-hit/miss, collapsed into a manageable discretization):

$$
\text{EV}_{\text{CHANCE}}(s, a_i, b_j) = \sum_{\omega \in \Omega} P(\omega)\; V\big(\text{apply}(s, a_i, b_j, \omega)\big)
$$

At the horizon depth $d$ reached under the iterative-deepening timer budget (§ Phase 5), $V(\cdot)$ is not a recursive Expectiminimax call but a direct call into `evaluator.py`'s static heuristic. Turn-candidate logs from the running system illustrate the resulting output format:

```
[BRAIN] Turn 1 candidates:
  move: flareblitz  (+10000.0 EV)
  move: knockoff     (+10000.0 EV)
  move: wildcharge   (+3598.8 EV)
  -> Chosen: move: flareblitz
```

(EV values here are heuristic-scaled, not literal win-probabilities; a capped +10000.0 denotes a heuristically-detected guaranteed-KO line.)

### 1.6 Empirical Bayesian Prior Model (Opponent Movepool / Item Prediction)

Because the opponent's exact set (moves, item, ability, EV spread) is hidden, FutureSightBot treats it as a latent variable with a prior derived from Smogon's published Chaos usage statistics for the target metagame (gen9ou):

$$
P(\text{set} \mid \text{species}) = P(\text{moves}, \text{item}, \text{ability}, \text{spread} \mid \text{species})
$$

sourced empirically as usage-weighted frequency tables ($P(\text{move}_k \mid \text{species})$, $P(\text{item} \mid \text{species})$, etc.) rather than assumed uniform or hand-authored — an **empirical Bayes** approach, since the "prior" is itself estimated from a large observed corpus (the aggregated Chaos ladder logs) rather than derived analytically.

As the battle proceeds and the opponent reveals actions, the prior is updated into a posterior via Bayes' rule:

$$
P(\text{set} \mid \text{observed\_action}) = \frac{P(\text{observed\_action} \mid \text{set}) \cdot P(\text{set})}{P(\text{observed\_action})}
$$

For example, observing a Choice-locked move eliminates all sets in the support that lack a Choice item; observing a super-effective coverage move outside the species' "expected" STAB moves shifts probability mass toward sets that include tech coverage. Species absent from the current usage snapshot (untracked or below the statistical cutoff) fall back to a uniform prior over base-stat-consistent movepool — a logged, explicit degradation path rather than a silent failure:

```
WARNING | Species 'gothitelle' (normalised: 'Gothitelle') not found in gen9ou stats
INFO    | [PRIOR] Opponent sent out gothitelle -- no data in gen9ou
```

### 1.7 Inverse Damage Calculation & Multi-Roll Range Intersection

The canonical (forward) damage formula used by the core series — and by Showdown's simulator — is, at its heart:

$$
\text{Damage} = \left\lfloor \left\lfloor \frac{\left\lfloor \frac{2 \cdot \text{Level}}{5} \right\rfloor + 2 \cdot \text{Power} \cdot \frac{A}{D}}{50} \right\rfloor + 2 \right\rfloor \times \text{Modifier}
$$

$$
\text{Modifier} = \text{STAB} \times \text{TypeEff} \times \text{Random} \times \text{Burn} \times \text{Weather} \times \text{Item} \times \dots
$$

where $\text{Random} \in \{0.85, 0.86, \dots, 1.00\}$ (16 discrete rolls), and $A$/$D$ are the attacker's Attack/Sp.Atk and defender's Defense/Sp.Def *after* stage boosts.

FutureSightBot needs the **inverse**: given an *observed* HP-percent delta (Showdown reports HP rounded to whole-percent granularity, itself an additional uncertainty band) resulting from a known move (power, type, category) against a partially-known defender, solve for the unknown attacking stat $A$ (or defending stat $D$, depending on which side is unknown):

$$
A \in \left[\; \frac{(\text{Damage}_{\min}) \cdot 50 \cdot D}{(\lfloor 2L/5\rfloor + 2)\cdot \text{Power} \cdot \text{Modifier}_{\max}} ,\;\; \frac{(\text{Damage}_{\max}) \cdot 50 \cdot D}{(\lfloor 2L/5\rfloor + 2)\cdot \text{Power} \cdot \text{Modifier}_{\min}} \;\right]
$$

Because a single observation only bounds the stat within a wide interval (the 15% roll spread, HP-rounding slack, and modifier ambiguity all stack), each *additional* damaging hit against the same target produces an independent bounding interval. The engine narrows the true stat value by **intersecting all observed intervals**:

$$
\text{Stat}_{\text{final}} = \bigcap_{i=1}^{n} \left[\text{Stat}_{\min}^{(i)}, \text{Stat}_{\max}^{(i)}\right]
$$

As $n$ grows, this interval provably tightens (intersection is monotonically non-expanding), eventually converging on a small set of stat values consistent with legal EV/IV/nature combinations under the game's base-stat table (`_POKEDEX`). A sudden narrow interval sitting *above* the maximum value achievable without a boosting item is itself a signal — the mechanism used for Choice item detection in Phase 3 of the engineering log.

---

## Section 2 — 60-Day Chronological Engineering Log

The log below reconstructs the development timeline as five 12-day phases. Each phase table lists per-day focus, the concrete deliverable, and engineering notes; narrative call-outs follow each phase for the days with the most architecturally significant decisions.

### Phase 1 — Days 1–12: Environment & Bridge

| Day | Focus | Deliverable | Notes |
|----|----|----|----|
| 1 | Project scaffolding | Repo init, venv, `requirements.txt` (`poke-env`, `asyncio`, `websockets`) | Established directory layout: `bot.py`, `config.py`, test fixtures folder |
| 2 | `poke-env` architecture study | Notes on `Player` base class contract | Identified `choose_move(battle)` as the core abstract method to override |
| 3 | Server bridge scaffolding | `AccountConfiguration` / `ShowdownServerConfiguration` wiring | Configured for local Showdown server target first, ladder later |
| 4 | Dependency resolution | Fixed `ModuleNotFoundError: poke_env` | Root cause: package installed outside active venv; resolved via venv activation + `pip install poke-env` |
| 5 | First live connection | Bot connects and completes a trivial random-move battle | Confirmed WebSocket handshake and auth flow end-to-end |
| 6 | State serialization | Internal `BattleState` representation | Converts `poke-env` `Battle` object into a stable internal schema (active mon, bench, hazards, weather, terrain, turn count) |
| 7 | `config.py` v1 | Format constant (`gen9ou`), logging setup | Established per-module loggers (`FutureSightBot`, `SmogonPriors`, etc.) |
| 8 | Mock battle fixtures | Hand-authored JSON battle states | Enabled fast unit testing without a live server dependency |
| 9 | Async loop hardening | Timeout/disconnect/forfeit handling | Prevented event-loop deadlocks on dropped connections |
| 10 | Self-test harness | `bot.py --self-test N` CLI flag | Runs N local bot-vs-bot battles for rapid iteration outside the ladder queue |
| 11 | Turn logging pipeline | Structured per-turn log lines (HP%, legal moves/switches) | Basis for all later offline log analysis |
| 12 | Phase 1 integration test | 100 self-test battles, zero crashes | Baseline control: "always pick first legal move" bot |

**Spotlight — Day 4:** The `ModuleNotFoundError: No module named 'poke_env'` failure, while trivial in isolation, motivated locking the dependency-management workflow (explicit venv activation documented in `README`) before any further engineering — a decision that paid off across all 60 days by eliminating an entire class of environment-drift bugs.

### Phase 2 — Days 13–24: Empirical Priors Engine

| Day | Focus | Deliverable | Notes |
|----|----|----|----|
| 13 | Smogon Chaos schema research | Field-mapping notes (usage%, moves%, items%, spreads%, teammates%) | Target: raw JSON stats endpoint for the current gen9ou snapshot |
| 14 | `smogon_priors.py` skeleton | `SmogonPriors` class stub | Constructor takes format string, exposes `get_likely_moves`, `get_likely_item` |
| 15 | HTTP client + disk cache | TTL-based local cache layer | Avoids re-fetching the full stats JSON every battle; logs `Using cached stats data for gen9ou` |
| 16 | Species normalization | ID/display-name/Chaos-key mapping | Handles edge cases like regional forms and hyphenated names |
| 17 | Missing-species fallback | Uniform-prior degradation path | Logged explicitly rather than silently defaulting (see §1.6 log excerpt) |
| 18 | Movepool pruning | Usage-threshold cutoff ($\epsilon$) | Moves below threshold usage% excluded from the candidate opponent action set to keep $b_{\text{eff}}$ tractable |
| 19 | Item probability model | $P(\text{item}\mid\text{species})$ table, Choice-flag extraction | Feeds directly into inverse-damage Choice detection in Phase 3 |
| 20 | Ability inference | $P(\text{ability}\mid\text{species})$ prior | Narrowed post-hoc once ability-triggered effects are observed |
| 21 | Spread clustering | Discrete EV/nature "stat-tier" hypotheses | Collapses the continuous EV space into a small number of representative spreads (e.g., max-Attack Jolly vs. bulky Adamant) |
| 22 | Posterior update logic | Bayes-rule set narrowing on observed action | Implements §1.6's posterior formula |
| 23 | Unit test suite | `SmogonPriors` tests against known snapshots | Regression protection against future stats-schema drift |
| 24 | Phase 2 integration | Priors wired into `bot.py` decision context | First end-to-end run logging live candidate opponent movesets per switch-in |

**Spotlight — Day 17:** Explicitly logging the missing-species fallback (rather than crashing or silently guessing) was a deliberate reliability decision — the metagame usage snapshot is never fully complete, and low-usage or newly-viable Pokémon must degrade gracefully to a base-stat-only prior.

### Phase 3 — Days 25–36: Inverse Damage Deduction

| Day | Focus | Deliverable | Notes |
|----|----|----|----|
| 25 | Damage formula formalization | Full modifier stack documented | STAB, type chart, burn, weather, screens, item, ability modifiers enumerated |
| 26 | `inverse_damage_calc.py` skeleton | `PokemonStats`, `FieldConditions` dataclasses | Typed containers for known/unknown stat state and battlefield modifiers |
| 27 | HP-rounding uncertainty | Additional slack band on observed damage% | Showdown reports HP in rounded percent, not exact fractions |
| 28 | Forward roll-range computation | 16-bucket damage roll enumeration (85–100%) | Baseline used to validate the inversion in Day 29 |
| 29 | Core inversion routine | Solve stat interval from observed damage | Implements the closed-form inversion in §1.7 |
| 30 | `calc_all_stats()` | Full candidate stat spread generation | Cross-references `_POKEDEX` base stats against legal IV/EV/nature combinations |
| 31 | Multi-roll intersection algebra | `infer_opponent_state()` interval narrowing | Implements $\text{Stat}_{\text{final}} = \bigcap_i [\text{Stat}_{\min}^{(i)}, \text{Stat}_{\max}^{(i)}]$ |
| 32 | Choice-item detection | Stat-ceiling anomaly heuristic | Interval sitting above max non-boosted ceiling flags Choice Band/Specs/Scarf |
| 33 | Residual-damage filtering | Leftovers/hazard/weather-chip exclusion | Prevents non-attack HP deltas from contaminating the inversion pipeline |
| 34 | Crit-vs-high-roll disambiguation | Modifier-ambiguity handling | A crit and a high non-crit roll can be observationally similar; both hypotheses retained until resolved |
| 35 | Validation harness | Replay-based accuracy scoring | Logged battles replayed against post-battle revealed EV spreads to score inference accuracy |
| 36 | Phase 3 integration | Inverse-damage output feeding `evaluator.py` | Opponent-model confidence weighting now informed by deduced stat tiers, not priors alone |

**Spotlight — Day 32:** Choice-item detection is the clearest example of the inverse-damage engine paying for itself strategically — a Pokémon whose deduced Attack stat exceeds what is achievable under *any* legal non-Choice EV spread is, with high confidence, locked into its revealed move, directly informing whether a risky switch-in is safe.

### Phase 4 — Days 37–48: Expectiminimax Search & Heuristic Evaluation

| Day | Focus | Deliverable | Notes |
|----|----|----|----|
| 37 | `evaluator.py` skeleton | Feature extraction layer | HP% (both sides), status, stat boosts, hazard layers, weather/terrain turns remaining |
| 38 | Heuristic scoring v1 | Weighted linear feature combination | Weights hand-tuned and centralized in `config.py` |
| 39 | Terminal-state detection | Faint / forced-switch / game-end conditions | Required for correct EV backpropagation termination |
| 40 | `expectiminimax.py` skeleton | MAX / MIN / CHANCE node types | Recursive tree structure per §1.5 |
| 41 | Payoff matrix construction | Bot action set × prior-weighted opponent action set | Implements the normal-form matrix of §1.4 |
| 42 | Chance-node expansion | Damage-roll buckets, crit branch, accuracy branch | Collapsed into expectation per §1.5's CHANCE formula |
| 43 | EV backpropagation + logging | `[BRAIN]` candidate logging format | Matches the live log excerpt shown in §1.5 |
| 44 | **Neural evaluator deprecation** | Reverted to deterministic heuristic scoring | Root-caused in Section 3.1 — static supervised weights failed to generalize to novel tactical lines |
| 45 | Depth-limited horizon cutoff | Heuristic call at search horizon | Full-game rollout infeasible under per-turn timer budget |
| 46 | Switch-in EV branch | Separate switch vs. attack evaluation path | Entry-hazard and ability-on-switch-in effects accounted for |
| 47 | Dominance pruning | Strictly-dominated opponent response elimination | Shrinks effective matrix before EV solve, reducing $b_{\text{eff}}$ |
| 48 | Phase 4 integration | End-to-end search pipeline live | First self-test battles using real search rather than heuristic-only fallback |

**Spotlight — Day 44:** This is the single most consequential engineering decision in the project's history; see Section 3.1 for full post-mortem.

### Phase 5 — Days 49–60: Battle-Hardening & Optimization

| Day | Focus | Deliverable | Notes |
|----|----|----|----|
| 49 | Search profiling | Per-turn latency outlier identification | Deep switch-trees and large action sets flagged as dominant cost centers |
| 50 | Iterative deepening | Progressive depth increase until budget exhausted | Guarantees a legal move is always available regardless of position complexity |
| 51 | Hard timer cutoff | Showdown per-turn timer compliance | Falls back to best-so-far candidate if cutoff is reached mid-search |
| 52 | Speed-tie handling | 50/50 branch-doubling chance node | See Section 3.3 for branching-factor cost analysis |
| 53 | Ditto Transform edge case | Mid-battle identity-swap invalidation | Cached opponent model explicitly invalidated on Transform, not silently reused |
| 54 | Zoroark Illusion edge case | Retroactive prior/state correction on reveal | Apparent species is wrong until the illusion breaks; bookkeeping patched post-hoc |
| 55 | **Anti-Choke multiplier** | Dynamic stage-weighted evaluator re-scoring | See Section 3.2 — resolves systematic 2v1/3v1 endgame throws |
| 56 | Regression suite | 500 self-test battles vs. RandomPlayer + heuristic-only baselines | Win-rate delta used as the primary release gate |
| 57 | Ladder dry run | Live `ShowdownServerConfiguration` deployment | First real-opponent stress test outside self-test harness |
| 58 | Observability polish | Structured turn/EV-candidate log dumps | Matches production log format used for all post-battle analysis |
| 59 | Bug-bounty pass | Fixes for ladder-discovered state corruption | Form-changer and Terastallization interaction edge cases |
| 60 | Release freeze | Documentation pass, `config.py` threshold lock | Tagged v1.0 |

---

## Section 3 — Post-Mortem & Algorithmic Failure Analysis

### 3.1 The "Neural Network Trap"

An early Phase 4 prototype (superseded on Day 44) replaced the deterministic heuristic in `evaluator.py` with a small supervised value network trained on logged self-test and ladder battle outcomes. It failed for reasons that are, in retrospect, structural rather than incidental:

1. **Distributional shift.** The network's weights encoded regularities of the *training* metagame snapshot. As soon as the search explored a tactical line under-represented in that snapshot — the canonical example being **laying Spikes against a bench with multiple Flying-types or Levitate users** — the network had no mechanism to recognize that the hazard's expected value collapses to near-zero against that specific composition. A hand-written heuristic term, by contrast, can encode the rule explicitly ("credit hazard-setting EV proportional to the count of *grounded* opponent-bench targets") and generalizes zero-shot to any bench composition, seen in training or not.
2. **Non-stationarity of the domain.** The competitive metagame shifts continuously (usage trends, tier shifts, new sets); a static set of learned weights is a snapshot of a moving target, whereas the empirical-prior engine (§1.6) is designed to be *refreshed* against current Smogon data rather than baked into frozen weights.
3. **Compute cost inside search.** A neural forward pass evaluated at every leaf of an Expectiminimax tree (potentially thousands of nodes per turn under iterative deepening) is orders of magnitude more expensive than a closed-form weighted-feature sum, directly competing with the hard per-turn timer budget established in Phase 5.

The deterministic heuristic is not claimed to be more *accurate* in the abstract — it is more **robust to distribution shift** and **cheap enough to evaluate at search scale**, which matters more inside a real-time, timer-constrained search than raw representational power.

### 3.2 The "Endgame Choke"

Early heuristic evaluators (Phase 4, pre-Day 55) scored macro-state balance features — aggregate HP%, hazard count, stat boosts — with fixed weights regardless of game phase. This produced a specific, reproducible failure pattern: when leading 3-Pokémon-to-1 or 3-to-2, the bot would continue making "generally optimal" plays (setting up additional hazards, greedy stat boosting, speculative switches) instead of "closing" plays (direct KO lines, risk-minimizing conservative attacks) — occasionally throwing a mathematically winning position.

**Root cause:** the marginal utility of a feature (e.g., one additional hazard layer, one HP percentage point) is **not constant across game phase**. Early-game, tempo and board-state investment has high expected value because the game is long. Late-game with a decisive material lead, the same investment is dominated by simply minimizing variance until the game ends — a well-known asymmetry in sequential zero-sum games with a "clock" (fewer remaining decision points inflates the value of certainty relative to expectation).

```text
=========================================================
       DYNAMIC HEURISTIC WEIGHT SHIFT (ANTI-CHOKE)
=========================================================
Feature                | Mid-Game Weight | End-Game Weight 
                       | (Opponent >= 3) | (Opponent <= 2)
---------------------------------------------------------
Set Hazards (Spikes)   |      +60        |       0.0
Stat Boost (Setup)     |      +80        |      +16.0  (0.2x)
Immediate Guaranteed KO|     +400        |    +1000.0  (2.5x)
Preserve Own HP        |     +100        |     +150.0  (1.5x)
---------------------------------------------------------
* Logic: As material lead increases, the search requires 
  variance reduction. Setup is penalized; KOs are forced.
=========================================================
```

**Fix — the Anti-Choke multiplier (Day 55):** the evaluator's feature weights are re-scaled by a multiplier keyed on the bot's remaining-Pokémon differential relative to the opponent. As the differential increases, risk-tolerant/exploratory terms (setup, speculative hazard value) are deflated and risk-averse/closing terms (guaranteed-KO detection, chip-damage avoidance, safe-switch preference) are inflated — a dynamic, stage-dependent re-weighting rather than a single static weight vector.

### 3.3 Competitive Edge Cases

**Speed ties (branch doubling).** When both active Pokémon share an identical post-modifier Speed stat, move-order resolution becomes an explicit 50/50 chance node inserted at that ply. Because every downstream subtree must now be evaluated under *both* orderings, this doubles the effective branching factor at that node:

$$
b_{\text{eff}}^{\text{tie}} = 2 \times b_{\text{eff}}
$$

Left unmitigated across a full search depth, this compounds multiplicatively per tied ply. The mitigation implemented in Phase 5 special-cases detection of ties whose resolution *does not change the bot's optimal action* (i.e., both orderings yield the same argmax), short-circuiting the duplicate subtree evaluation.

**Form-changer state corruption.** Ditto's Transform and Zoroark's Illusion both violate a tacit assumption baked into early `BattleState` serialization — that a Pokémon's identity (species, stats, ability, movepool) is stable for the duration of its time on the field. Transform swaps essentially the *entire* identity mid-battle; Illusion reports an incorrect apparent species until broken by direct damage. Both required explicit invalidation/patch hooks (Days 53–54) rather than being handled implicitly, since any cached prior or inverse-damage-deduced stat range keyed to the *pre-reveal* identity is not just stale but actively wrong, and silently trusting it produces confidently incorrect decisions rather than merely suboptimal ones.

---

## Section 4 — Architecture & Class Reference

| Module | Responsibility | Key Interfaces |
|---|---|---|
| **`config.py`** | Centralized configuration: target format (`gen9ou`), per-turn timer budget, heuristic feature weights, prior-usage thresholds, logging setup. | Module-level constants and logger factory consumed by every other module — the single source of truth for tunables, avoiding scattered magic numbers. |
| **`smogon_priors.py`** | Empirical prior model over opponent sets, sourced from Smogon Chaos usage statistics. | `SmogonPriors` class — `get_likely_moves(species)`, `get_likely_item(species)`, `get_likely_ability(species)`, internal disk-cache and species-normalization layer. Degrades explicitly (logged) to base-stat-only priors for species absent from the current snapshot. |
| **`inverse_damage_calc.py`** | Deduces hidden opponent stats/items from observed damage events. | `infer_opponent_state(...)`, `calc_all_stats(...)`, `PokemonStats` / `FieldConditions` dataclasses, `_POKEDEX` base-stat lookup table. Implements the forward formula, its inversion, and multi-roll interval intersection (§1.7). |
| **`evaluator.py`** | Static heuristic scoring of a (possibly non-terminal) battle state — the $V(s)$ used at Expectiminimax search horizons. | Feature-extraction + weighted-sum scoring function; terminal-state detection; hosts the Phase-5 Anti-Choke dynamic re-weighting logic (§3.2). |
| **`expectiminimax.py`** | Per-turn simultaneous-move search: constructs the payoff matrix, expands chance nodes, backpropagates expected value, applies iterative deepening under the timer budget. | Tree builder over MAX/MIN/CHANCE node types (§1.5); emits the `[BRAIN]` turn-candidate log; dominance pruning; speed-tie short-circuiting; hard-cutoff fallback-to-best-so-far. |
| **`bot.py`** | Entry point and `poke-env` integration layer. | Subclasses `poke_env.player.Player`; wires `AccountConfiguration` / `ShowdownServerConfiguration`; implements `choose_move(battle)`; exposes the `--self-test N` CLI harness for local bot-vs-bot iteration; owns the top-level turn logging pipeline. |

**Data flow per turn:**

```
 Showdown WebSocket
        |
        v
 +--------------+        +----------------------+
 |   bot.py     |------->|  BattleState          |
 | (poke-env    |        |  serialization        |
 |  Player)     |        +----------------------+
 +--------------+                  |
        |                          v
        |                +----------------------+
        |--------------->|  smogon_priors.py     |
        |                |  (opponent set prior) |
        |                +----------------------+
        |                          |
        |                          v
        |                +----------------------+
        |--------------->| inverse_damage_calc.py|
        |                | (hidden stat dedux.)  |
        |                +----------------------+
        |                          |
        v                          v
 +-------------------------------------------+
 |            expectiminimax.py               |
 |  MAX/MIN/CHANCE search, EV backprop,        |
 |  iterative deepening, timer cutoff          |
 +-------------------------------------------+
                    |
                    v
 +-------------------------------------------+
 |               evaluator.py                 |
 |   heuristic scoring at search horizon       |
 +-------------------------------------------+
                    |
                    v
          Chosen action -> Showdown
```

---

*End of document.*
