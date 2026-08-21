# FutureSightBot — Engineering Upgrade Roadmap
## Ordered Implementation Guide: Phase 5 Battle-Hardening & Phase 6 Contextual Policy Neural Network

**Target Environment:** Pokémon Showdown (Gen 9 OverUsed / Random Battles)  
**System Architecture:** Hybrid Search-and-Inference Engine (Bayesian Priors + Inverse Damage Calc + Expectiminimax + Policy Neural Network)  
**Hardware Profile:** Standard Multi-Core CPU, 16GB RAM, Integrated Graphics (No dedicated GPU required for deployment)

---

## 1. Roadmap Overview & Phased Progression

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       FUTURE SIGHT BOT UPGRADE PIPELINE                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  [STEP 1: Phase 5 Battle-Hardening]  (Immediate / Zero-Training)           │
│    ├── 1.1 Iterative Deepening & Precision Time Guards (expectiminimax.py)  │
│    ├── 1.2 Endgame Anti-Choke Dynamic Heuristic (evaluator.py)            │
│    ├── 1.3 Zoroark Illusion & Ditto/Transform Isolation (bot.py)          │
│    ├── 1.4 Multi-Hit Damage Range Intersection (inverse_damage_calc.py)    │
│    └── 1.5 Automated 10-Test Regression Suite (test_suite.py)              │
│                                                                            │
│                                      ▼                                     │
│                                                                            │
│  [STEP 2: Phase 6 Contextual Policy Neural Network] (Deep Learning Upgrade)│
│    ├── 2.1 Replay Data Pipeline (50k High-Elo Replays, Elo > 1650)         │
│    ├── 2.2 Permutation-Invariant Set-Transformer Architecture              │
│    ├── 2.3 Supervised Behavioral Cloning Training (Free Colab / Local CPU) │
│    ├── 2.4 ONNX Runtime CPU Inference (<1ms execution latency)            │
│    └── 2.5 Search Space Pruning (Branching 16 -> 5, Search Depth 2 -> 4)   │
│                                                                            │
│                                      ▼                                     │
│                                                                            │
│  [STEP 3: Phase 7 Advanced Game-Theoretic Optimizations] (Long-Term)       │
│    ├── 3.1 Linear Programming Nash Equilibrium Mixed-Strategy Solver      │
│    └── 3.2 NNUE Positional Leaf Value Function                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. STEP 1 — Phase 5: Battle-Hardening & Core Engine Optimization
*(Immediate priority — 100% deterministic, 0 seconds training time, fixes all critical game-play flaws)*

### 2.1 Iterative Deepening & Precision Timing Guards
* **Target File:** `expectiminimax.py`
* **Objective:** Replace static depth searches with an incremental progressive search loop bounded by a monotonic wall-clock timer.
* **Technical Details:**
  1. **Monotonic Timing:** Replace standard timestamps with `time.perf_counter()` for microsecond resolution.
  2. **Iterative Deepening Loop:**
     ```python
     for current_depth in range(1, self.depth + 1):
         if (time.perf_counter() - start_time) >= deadline_s:
             break
         # Evaluate full ply at current_depth
         # Only update best_action if the full depth completed without timing out
     ```
  3. **Safety Fallback:** If a depth search aborts mid-evaluation due to the 500ms timer budget, discard the partial ply and return the best action from the last fully completed depth.
  4. **Speed-Tie Chance Nodes:** When active speeds and priorities are identical, branch into two separate chance nodes ($0.5$ probability each for bot-first vs. opp-first) via `_simulate_ordered_outcomes`, preventing tree search hangs.

### 2.2 Endgame Anti-Choke Logic
* **Target File:** `evaluator.py`
* **Objective:** Eliminate late-game throws (e.g. laying hazards or setting up boosts when leading 3v1 or 2v1) by dynamically altering heuristic weights based on remaining material.
* **Technical Details:**
  1. **Endgame Detection:** Trigger when opponent living Pokémon $\le 2$ (`_is_endgame(opp_alive)`).
  2. **Dynamic Weight Shifts:**
     * **Entry Hazards (Stealth Rock / Spikes):** Valuation drops to **`0.0`** (laying hazards when the opponent has only 1–2 Pokémon left provides zero future utility).
     * **Stat Boosts / Passive Setup:** Valuation halved ($0.5\times$ weight) to discourage greedy setup.
     * **Lethal KO Bonus:** Material differential reward is multiplied by **$2.5\times$** to mathematically force the search engine to choose direct KO lines over all other moves.

### 2.3 State & Identity Edge-Case Protections
* **Target Files:** `bot.py` & `inverse_damage_calc.py`
* **Objective:** Prevent false deductions and corrupted opponent profiles caused by identity-shifting mechanics.
* **Technical Details:**
  1. **Zoroark Illusion Guard:**
     * Check if `zoroark` or `zoroarkhisui` is present anywhere in `battle.opponent_team`.
     * If present, bypass inverse damage calculations for unconfirmed species until that Pokémon takes direct observable damage (confirming Illusion is broken).
  2. **Ditto / Transform Guard:**
     * If the opponent species is `ditto` or has the `transform` / `imposter` ability, bypass inverse damage calculation to prevent copied stats from overwriting the underlying profile.
  3. **Multi-Hit Range Intersection:**
     * Pass `existing_range` (the previous $[low, high]$ Attack stat window) into `infer_opponent_state`.
     * Calculate the mathematical intersection:
       $$\text{Range}_{\text{new}} = [\max(low_{\text{prev}}, low_{\text{curr}}), \; \min(high_{\text{prev}}, high_{\text{curr}})]$$
     * Continually narrows the stat window on repeated hits, accurately flagging Choice Band / Choice Specs.

### 2.4 Unit Test Suite
* **Target File:** `test_suite.py`
* **Coverage (10 Automated Tests):**
  - `TestIterativeDeepening`: Timeout graceful abort + 50/50 speed tie branch normalization.
  - `TestZoroarkGuard`: Disguised damage bypass + break reactivation.
  - `TestDittoGuard`: Transform damage profile isolation.
  - `TestEndgameHeuristic`: Endgame detection + hazard zeroing + $2.5\times$ KO payoff.
  - `TestRangeIntersection`: Successive damage window narrowing.

---

## 3. STEP 2 — Phase 6: Contextual Policy Neural Network
*(Deep tactical upgrade — learns high-Elo human decision patterns without the pitfalls of DQN)*

```
[WebSocket Game State]
          │
          ▼
[smogon_priors.py]  ──────► Filters 800+ moves down to 4 plausible moves
          │
          ▼
[★ Policy Transformer ★] ──► Analyzes 6v6 Matchup & Board State:
          │                  P(Draco Meteor | Garchomp @ 35%) = 90%
          │                  P(Stealth Rock | Garchomp @ 35%) = 1% (PRUNED)
          │                  P(Switch to Corviknight)         = 9%
          ▼
[expectiminimax.py] ──────► Focuses search tree ONLY on high-probability branches
          │                  (Search branching drops from 16 -> 5; Depth increases 2 -> 4)
          ▼
[evaluator.py]     ──────► Exact heuristic evaluation at horizon leaves
```

### 3.1 Why a Policy Network Beats Model-Free DQN
* **No Reinforcement Learning instability:** Does not rely on noisy reward shaping or slow browser simulation loops.
* **Supervised Learning on High-Elo Replays:** Learns directly from expert human games ($\text{Elo} > 1650$).
* **Safe & Deterministic:** The neural network only predicts move probabilities for the search tree. All damage math, type checks, and rule boundaries remain 100% deterministic.

### 3.2 Network Architecture & Input Representation
* **Architecture:** Permutation-Invariant Set-Transformer / Multi-Head Cross-Attention MLP.
* **Model Size:** $\approx 250\text{k}$ parameters ($\approx 1.5\,\text{MB}$ file size).
* **Input Tensor ($\approx 384$ Dimensions):**
  1. **Active Matchup (64 dims):** Active bot species, active opp species, current HP%, stat stages, status, revealed moves.
  2. **Team Benches (256 dims):** Unordered 6v6 team embeddings (species ID, HP%, fainted mask, item category).
  3. **Field Conditions (64 dims):** Weather, terrain, Trick Room, screens, hazards (Stealth Rock, Spikes layers).
* **Output:** Softmax distribution over legal actions (4 moves + 5 switches).

### 3.3 Training Pipeline (Free & Fast)
* **Dataset:** 50,000 Gen 9 OU public replay JSONs from Pokémon Showdown database.
* **Loss Function:** Cross-Entropy Loss:
  $$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$
* **Training Platform:** Free Google Colab GPU (NVIDIA T4, takes **35 minutes**) or multi-threaded local CPU (**20 minutes**).

### 3.4 Integration & Deployment via ONNX Runtime
* **Deployment Format:** Export PyTorch model to `policy_model.onnx`.
* **Inference Engine:** `onnxruntime` executing on CPU.
* **Latency:** **$0.8\text{--}1.2\,\text{ms}$ per turn** (negligible CPU/RAM overhead on 16GB systems).
* **Search Acceleration:**
  - Prunes low-probability candidate actions ($P < 0.05$).
  - Effective branching factor drops from $b \approx 16 \rightarrow 5$.
  - Allows Iterative Deepening in Phase 5 to reach **Depth 3 and Depth 4** within the standard 500ms turn timer.

---

## 4. STEP 3 — Phase 7: Advanced Game-Theoretic Refinements
*(Long-Term Future Optimizations)*

### 4.1 Linear Programming Nash Mixed-Strategy Solver
* **Current Approximation:** Expected value weighted against opponent prior distribution.
* **Phase 7 Upgrade:** Solve the 2D Payoff Matrix $M$ for its exact Minimax / Nash Mixed Strategy equilibrium via Linear Programming (`scipy.optimize.linprog`):
  $$\max v \quad \text{s.t.} \quad M^T p \ge v \mathbf{1}, \quad \sum p_i = 1, \quad p \ge 0$$
* **Impact:** Plays unexploitable mixed strategies in simultaneous mind-game situations (e.g. 50/50 Sucker Punch vs. attack).

### 4.2 NNUE (Efficiently Updatable Neural Network) Leaf Evaluator
* **Concept:** Adapt Stockfish's NNUE architecture for Pokémon state evaluation.
* **Role:** Evaluates deep positional advantage (tempo, win-condition preservation, long-term sacrifice value) at search leaf nodes.
* **Speed:** Quantized integer arithmetic running in $< 5\,\mu\text{s}$ per leaf.

---

## 5. Summary of Expected Performance Trajectory

| Milestone | Architecture Highlights | Search Depth | Estimated Rating |
|---|---|---|---|
| **Phase 4 Baseline (Current)** | Fixed 2-Ply Expectiminimax + Static Smogon Priors | Depth 2 | $\approx 1250\text{ Elo}$ |
| **Phase 5 Upgrade (Step 1)** | Iterative Deepening + Anti-Choke + Range Intersection + Edge Guards | Depth 2–3 | $\approx 1500\text{ Elo}$ |
| **Phase 6 Policy NN (Step 2)** | Contextual Transformer Prior + $70\%$ Branch Pruning + High-Elo Switch Reads | Depth 3–4 | $\approx 1700+\text{ Elo}$ |
| **Phase 7 Full Nash Engine (Step 3)** | Simultaneous LP Matrix Solver + Quantized NNUE Positional Evaluator | Depth 4–5 | $\approx 1850+\text{ Elo}$ (Grandmaster) |

---

*This document serves as the implementation specification for all subsequent development phases.*
