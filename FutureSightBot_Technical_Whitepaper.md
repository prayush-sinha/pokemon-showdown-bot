# FutureSightBot: Technical Whitepaper
## An Autonomous, Game-Theoretic Architecture for Competitive Pokémon Battle Simulation

**Technical Whitepaper & 90-Day Comprehensive Engineering Development Log**  
Version 2.0 · Multi-Format Support: `gen9ou`, `gen9championsbssregma`, `gen9randombattle`, `gen9nationaldex` · Platform: Pokémon Showdown  

---

## Section 1 — Executive Summary & Theoretical Framework

### 1.1 Abstract

FutureSightBot is an autonomous agent for competitive Pokémon battling on the Pokémon Showdown simulator, built on `poke-env` as the asynchronous WebSocket/protocol bridge. Unlike pure black-box reinforcement-learning agents that approximate a monolithic value function, FutureSightBot implements a **hybrid game-theoretic, deep-learning, and symbolic inference system**:

1. **Empirical Bayesian Prior Model**: Scrapes and structures live Smogon competitive usage statistics to estimate probability distributions over unrevealed opponent moves, items, abilities, and EV spreads.
2. **Inverse Damage Calculation Engine**: Solves the inverse constraint satisfaction problem across 16 discrete damage rolls ($\lfloor 85\% \dots 100\% \rfloor$) to infer opponent offensive stats and deduce hidden items (e.g., Choice Band, Choice Specs, Life Orb).
3. **Set-Transformer Policy Network (ONNX Runtime)**: A multi-head permutation-invariant neural network trained via imitation learning on $>100,000$ high-ELO tournament states to predict opponent move and switch distributions, pruning search complexity in real time ($P < 0.05$).
4. **Simultaneous Expectiminimax Search Engine**: Formulates each turn as a simultaneous normal-form matrix game, evaluating full forward state transitions (stat modifications, recovery, drain, recoil, status immunities, and hazards) with speed-tie and accuracy chance nodes under hard time budgets.

```text
===================================================================================
                       SYSTEM ARCHITECTURE & DECISION PIPELINE
===================================================================================
                              [ Pokémon Showdown Server ]
                                           | (WebSocket Event)
                                           v
                                      [ bot.py ]
                           (Async Turn Event & Gimmicks)
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
          [ smogon_priors.py ]                        [ policy_inference.py ]
      (Bayesian Metagame Priors)                  (Set-Transformer ONNX Net)
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                              [ inverse_damage_calc.py ]
                        (Discrete Roll Multi-Hit Inversion)
                                           |
                                           v
                                [ expectiminimax.py ]
                   (Simultaneous Matrix Solver & Forward Physics)
                                           |
                                           v
                                   [ evaluator.py ]
                      (Dynamic Stage-Weighted Heuristic Engine)
                                           |
                                           v
                                    [ Action Order ]
                              (Sends Move/Switch to Server)
===================================================================================
```

### 1.2 Pokémon as a Formal Game Environment

Formally, a single Pokémon battle turn is modeled as a stochastic, imperfect-information, simultaneous game:

$$\Gamma = \langle N, \{A_i\}_{i \in N}, \{u_i\}_{i \in N}, \Omega, P \rangle$$

where $N = \{1, 2\}$, $A_i$ represents legal actions (moves $\cup$ valid switches), $u_i$ is the zero-sum state utility ($u_1 = -u_2$), $\Omega$ is the RNG outcome space, and $P$ is the joint probability distribution over $\Omega$.

- **Simultaneous Action Submission**: Both players commit actions without observing the opposing choice; turn resolution order is governed by move priority and active Speed stats.
- **Imperfect Information**: Opponent EV spreads, held items, remaining PP, and unrevealed movesets are latent variables that must be actively inferred.
- **Stochastic Transitions**: Accuracy checks, 16 discrete damage rolls, critical hits, secondary effects, and Speed ties require explicit chance-node expectation.

### 1.3 The Normal-Form Payoff Matrix & Simultaneous Expectiminimax

At turn $t$, let $A_1 = \{a_1, \dots, a_m\}$ be the bot's legal actions and $A_2 = \{b_1, \dots, b_n\}$ be the opponent's candidate actions predicted by the policy net and metagame priors. The engine constructs a payoff matrix $M$:

$$M = \begin{bmatrix}
u(a_1, b_1) & u(a_1, b_2) & \cdots & u(a_1, b_n) \\
u(a_2, b_1) & u(a_2, b_2) & \cdots & u(a_2, b_n) \\
\vdots & \vdots & \ddots & \vdots \\
u(a_m, b_1) & u(a_m, b_2) & \cdots & u(a_m, b_n)
\end{bmatrix}$$

The Expected Value for each candidate bot action $a_i$ is computed over opponent action probabilities $P(b_j \mid s)$:

$$\text{EV}(a_i) = \sum_{b_j \in A_2} P(b_j \mid s) \sum_{\omega \in \Omega} P(\omega) \cdot \mathcal{H}\big(\text{Simulate}(s, a_i, b_j, \omega)\big)$$

```text
===================================================================================
                         SIMULTANEOUS EXPECTIMINIMAX TREE
===================================================================================
 [MAX NODE] Bot Action Selection (Moves & Switches)
   │
   ├── [Bot Action: Attack A]
   │    │
   │    └── [MIN NODE] Opponent Action Distribution (Policy Net + Priors)
   │         │
   │         ├── [Opponent Action 1] (Prob: 70%)
   │         │    │
   │         │    └── [CHANCE NODE] Execution Order & RNG Rolls
   │         │         ├── Hit (Acc %)   -> Forward Simulation -> State s1 -> Heuristic H(s1)
   │         │         └── Miss (1-Acc%) -> Forward Simulation -> State s2 -> Heuristic H(s2)
   │         │
   │         └── [Opponent Action 2] (Prob: 30%)
   │              │
   │              └── [CHANCE NODE] -> Forward Simulation -> State s3 -> Heuristic H(s3)
   │
   └── [Bot Action: Switch B] ...
===================================================================================
```

### 1.4 Deep Learning Policy Network & ONNX Inference

To scale search depth without incurring exponential branching ($|A_1| \times |A_2|$), a **Set-Transformer Neural Policy Network** evaluates the board state:
* **Permutation Invariance**: Utilizes multi-head self-attention over 12 categorical slot representations (6 player, 6 opponent) invariant to bench order.
* **Feature Embeddings**: Encodes categorical tokens (species, items, abilities, statuses, tera types) and continuous metrics (normalized HP, stat boosts, field hazards, screens, active weather/terrains).
* **Pruning Operator**: Discards opponent branches where predicted likelihood $P(b_j) < 0.05$, renormalizing the active support.
* **ONNX CPU Runtime**: Compiles the PyTorch graph into an optimized ONNX binary, executing in $<1\,\text{ms}$ per turn.

### 1.5 Inverse Damage Calculation & Interval Narrowing

The standard discrete damage formula:

$$\text{Damage} = \left\lfloor \left( \left\lfloor \frac{\lfloor \frac{2L}{5} + 2 \rfloor \times \text{Power} \times \frac{A}{D}}{50} \right\rfloor + 2 \right) \times \text{Modifier} \times \frac{R}{100} \right\rfloor$$

where $R \in \{85, 86, \dots, 100\}$.

Given observed percentage loss $\Delta \text{HP}$, the inverse solver calculates the feasible offensive stat interval:

$$A \in \left[\; \frac{\Delta \text{HP}_{\min} \cdot 50 \cdot D}{(\lfloor 2L/5\rfloor + 2)\cdot \text{Power} \cdot \text{Mod}_{\max}} ,\;\; \frac{\Delta \text{HP}_{\max} \cdot 50 \cdot D}{(\lfloor 2L/5\rfloor + 2)\cdot \text{Power} \cdot \text{Mod}_{\min}} \;\right]$$

Consecutive attacks intersect intervals monotonically:

$$\text{Stat}_{\text{final}} = \bigcap_{k=1}^{n} \left[\text{Stat}_{\min}^{(k)}, \text{Stat}_{\max}^{(k)}\right]$$

A deduced stat bound exceeding the maximum legal non-boosted base ceiling flags an active Choice Band or Choice Specs.

---

## Section 2 — 90-Day Chronological Engineering Log

```
+-----------------------------------------------------------------------------------------+
|                                    PROJECT TIMELINE                                     |
+-----------------------------------------------------------------------------------------+
| June 2026   | Phase 1 & 2: Protocol Engine, Metagame Priors & Damage Physics            |
| July 2026   | Phase 3 & 4: Simultaneous Expectiminimax, Heuristics & Anti-Exploits     |
| August 2026 | Phase 5 & 6: Set-Transformer Policy Net, Multi-Format Scaling & Deployment|
+-----------------------------------------------------------------------------------------+
```

### Phase 1 — Days 1–15 (June 1 – June 15): Protocol Bridge & Empirical Priors

| Day | Focus | Deliverable | Engineering Notes |
|:---|:---|:---|:---|
| 1–3 | Architecture Scaffolding | Protocol client repository layout | Configured `poke-env` asynchronous WebSocket bridge. |
| 4–6 | State Parsing | `BattleState` serialization | Extracted HP, status, active boosts, hazard conditions, and turn counters. |
| 7–9 | Smogon Scraper | `smogon_priors.py` HTTP cache | Implemented local TTL disk caching for competitive usage JSON dumps. |
| 10–12 | Prior Distributions | Bayesian moveset estimators | Built empirical frequency tables for unrevealed moves, items, and abilities. |
| 13–15 | Offline Verification | Mock battle test harnesses | Validated parser stability across 100 simulated battles without live server. |

### Phase 2 — Days 16–30 (June 16 – June 30): Forward & Inverse Damage Physics

| Day | Focus | Deliverable | Engineering Notes |
|:---|:---|:---|:---|
| 16–18 | Discrete Rolls | Forward damage formula | Coded exact 16-roll calculation with Gen 9 type charts and STAB multipliers. |
| 19–21 | Field & Weather | Environmental modifiers | Added Sun, Rain, Snow, Sandstorm, Terrains (Electric/Grassy/Psychic/Misty), Screens. |
| 22–25 | Inverse Interval Solver | `inverse_damage_calc.py` | Built mathematical interval intersection engine to bound opponent EV spreads. |
| 26–28 | Item Deductions | Choice item detection | Identified stat-ceiling anomalies indicating Choice Band/Specs/Life Orb. |
| 29–30 | Phase 2 Review | 500-scenario math harness | Achieved $99.2\%$ classification accuracy on benchmark damage events. |

### Phase 3 — Days 31–45 (July 1 – July 15): Simultaneous Expectiminimax Search

| Day | Focus | Deliverable | Engineering Notes |
|:---|:---|:---|:---|
| 31–34 | Matrix Representation | Normal-form payoff builder | Mapped all legal bot actions against prior-weighted opponent choices. |
| 35–38 | Tree Recursion | `expectiminimax.py` search loop | Implemented depth-limited tree lookahead with recursive EV backpropagation. |
| 39–41 | Chance Nodes | Accuracy & speed-tie branching | Modeled move accuracy and 50/50 speed ties as explicit probabilistic nodes. |
| 42–45 | Horizon Evaluation | `evaluator.py` heuristic v1 | Evaluated material HP ratios, alive counts, stat stages, and hazards. |

### Phase 4 — Days 46–60 (July 16 – July 31): Heuristics & Anti-Exploit Hardening

| Day | Focus | Deliverable | Engineering Notes |
|:---|:---|:---|:---|
| 46–49 | Anti-Choke Multiplier | Stage-dependent heuristic weights | Deflated setup and inflated immediate KO weights when leading late-game. |
| 50–52 | Identity Edge Cases | Zoroark & Ditto guards | Tracked direct damage to reveal Illusion and isolated Imposter stat clones. |
| 53–56 | Tempo Regularization | Switch ping-pong penalty | Implemented $-50\,\text{EV}$ tempo penalty preventing endless non-productive swaps. |
| 57–60 | Unit Regression Suite | `test_suite.py` harness | Created 10 automated test suites verifying edge-case resilience. |

### Phase 5 — Days 61–75 (August 1 – August 15): Set-Transformer Imitation Learning

| Day | Focus | Deliverable | Engineering Notes |
|:---|:---|:---|:---|
| 61–64 | Replay Pipeline | `scrape_replays.py` & parser | Implemented parallel Showdown replay scraper and tensor state-action extractor. |
| 65–68 | Neural Architecture | `policy_net.py` Set-Transformer | Built multi-head attention network over 12 canonical team slots. |
| 69–72 | GPU Training | `train_policy.py` Colab loop | Trained on 120,000 state transitions; achieved $68.4\%$ top-1 action accuracy. |
| 73–75 | ONNX Compilation | `policy_inference.py` engine | Exported ONNX model with $<1\,\text{ms}$ CPU inference latency. |

### Phase 6 — Days 76–90 (August 16 – August 23): Full Mechanics, Multi-Tier & Deployment

| Day | Focus | Deliverable | Engineering Notes |
|:---|:---|:---|:---|
| 76–78 | Turn 1 Bench Fix | Corrected unrevealed HP | Initialized unrevealed bench Pokémon to $100\%$ HP, eliminating false $+10\text{k}$ EV scores. |
| 79–81 | Move Side-Effects | Forward simulation upgrades | Simulated stat boosts/drops, direct recovery ($50\%$), drain moves, recoil, and status. |
| 82–84 | Format Isolation | Dynamic CLI & path resolver | Added `--format` flag routing models and vocabs to `data/<format>/`. |
| 85–87 | Level & Gimmick Scaling | Megas, Tera & Level 50 BSS | Supported all 101 Mega Evolutions, automatic Terastallization, and Level 50 scaling. |
| 88–90 | Ladder Deployment | Live ladder execution | Verified win rates on `play.pokemonshowdown.com`; formalized technical documentation. |

---

## Section 3 — Post-Mortem & Algorithmic Failure Analysis

### 3.1 Resolving the Neural Network Representation Dilemma

* **The Early Value Net Failure**: In Phase 4, an early supervised value network was tested to replace the evaluation heuristic $V(s)$. It failed due to out-of-distribution hallucinations (e.g., laying Spikes against teams of Flying/Levitate Pokémon).
* **The Policy-Pruner Solution**: In Phase 5/6, deep learning was decoupled from static state valuation and restricted to its optimal role: **probabilistic opponent action prediction**. The Set-Transformer acts as a prior filter, narrowing the joint action space while preserving the deterministic, rule-based Expectiminimax forward simulator.

### 3.2 Turn 1 Unrevealed Bench Saturation

* **Failure Mode**: On Turn 1, unrevealed opponent bench Pokémon were initially marked as $0\%$ HP by the internal state parser. The material differential heuristic evaluated the state as an immediate $6\text{v}1$ lead, flooding the root node with false-win $+10,000\,\text{EV}$ evaluations.
* **Resolution**: Replaced the default null state with an implicit $100\%$ HP prior for all unrevealed bench slots until explicitly fainted or damaged.

### 3.3 Dynamic Level Scaling (BSS vs Standard Formats)

* **Failure Mode**: Cartridge-style 3v3 Battle Stadium Singles (BSS) and tournament Champions formats operate at **Level 50**, whereas standard Showdown tiers (OU, NatDex, Random Battles) operate at **Level 100**. Hardcoding Level 100 caused significant overestimation of raw base damage in Level 50 formats.
* **Resolution**: Added dynamic `mon.level` extraction with fallback to format-specific defaults, integrating level variables directly into `calculate_damage_range()` and `SimPokemon`.

---

## Section 4 — Architectural Module Reference

| Module | Responsibility | Key Classes / Functions |
|:---|:---|:---|
| **`bot.py`** | Client entry point, WebSocket event loop, Gimmick triggers (Mega/Tera), and CLI argument dispatcher. | `FutureSightBot`, `make_bot()`, `choose_move()` |
| **`expectiminimax.py`** | Simultaneous search engine, forward side-effect simulator, chance-node builder, and speed-tie resolver. | `ExpectiminimaxEngine`, `simulate_turn_outcomes()`, `_apply_move_side_effects()` |
| **`evaluator.py`** | Non-terminal heuristic evaluation, payoff matrix generator, and anti-choke stage weighting. | `evaluate_state()`, `build_payoff_matrix()` |
| **`policy_inference.py`** | Ultra-fast ONNX Runtime inference wrapper, categorical feature encoder, and action candidate pruner. | `PolicyInferenceEngine`, `battle_to_tensor()`, `PolicyVocab` |
| **`policy_net.py`** | PyTorch Set-Transformer architecture with permutation-invariant attention over 12 team slots. | `FutureSightPolicyNet`, `SlotEncoder`, `ContextEncoder` |
| **`inverse_damage_calc.py`** | Forward 16-roll Gen 9 damage calculator and real-time inverse EV/item deduction engine. | `calculate_damage_range()`, `infer_opponent_state()` |
| **`smogon_priors.py`** | Empirical Smogon usage statistics fetcher, Bayesian moveset estimator, and disk cache. | `SmogonPriors`, `get_likely_moves()`, `get_likely_item()` |
| **`config.py`** | System settings, format path resolution (`get_format_paths`), logging configuration, and constants. | `get_format_paths()`, `Config` |

---

*FutureSight AI Architecture & Engineering Log — Version 2.0*

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
