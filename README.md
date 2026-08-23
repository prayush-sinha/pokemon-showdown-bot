# FutureSight AI: Game-Theoretic Pokemon Showdown Battle Engine

FutureSight AI is an autonomous, high-performance competitive Pokemon Showdown battle engine. It combines simultaneous-move game theory, Set-Transformer neural policy networks, reverse-engineered state estimation (inverse damage calculation), and empirical metagame priors to solve imperfect-information decisions in real time.

---

## 1. System Architecture

The decision engine operates through a multi-stage pipeline designed to handle simultaneous actions, hidden information, and probabilistic execution:

```
+-----------------------------------------------------------------------------------+
|                                  FutureSightBot                                   |
+-----------------------------------------------------------------------------------+
     |                                                                         ^
     v                                                                         |
[Protocol Bridge & State Extraction]                                  [Selected Action]
  - Asynchronous WebSocket protocol handler (`poke-env`)                       |
  - Real-time battle state tracking and gimmick management (Mega / Tera)       |
     |                                                                         |
     v                                                                         |
[Metagame Prior Engine]                                                        |
  - Empirical usage distributions sourced from Smogon competitive datasets     |
  - Bayesian priors for unrevealed moves, items, abilities, and EV spreads     |
     |                                                                         |
     v                                                                         |
[Inverse Damage Calculator]                                                    |
  - Exact Generation 9 discrete forward damage model (16 rolls, weather, field)|
  - Reverse-engineers opponent EV investment, items, and defensive tiers       |
     |                                                                         |
     v                                                                         |
[Set-Transformer Policy Network (ONNX)]                                        |
  - Multi-head self-attention over active and bench Pokemon slots              |
  - Predicts opponent action probabilities and prunes low-likelihood branches  |
     |                                                                         |
     v                                                                         |
[Simultaneous Expectiminimax Tree Search] -------------------------------------+
  - Nash Expected Value optimization over simultaneous action matrices
  - Full forward simulation of stat modifications, recovery, drain, and hazards
  - Speed-tie chance nodes and tempo penalty regularization
  - Adaptive level scaling (Level 50 for BSS/VGC; Level 100 for standard tiers)
```

---

## 2. Core Engine Components

### 2.1 Simultaneous Expectiminimax Search
Unlike sequential games (e.g., Chess), Pokemon battles feature simultaneous turn execution. The search engine constructs a payoff matrix for each turn pair $(a_{\text{bot}}, a_{\text{opp}})$ and evaluates leaves using an Expected Value heuristic over opponent action probabilities:

$$\text{EV}(a_{\text{bot}}) = \sum_{a_{\text{opp}}} P(a_{\text{opp}} \mid s) \cdot \mathbb{E}[\mathcal{H}(s')]$$

* **Forward Simulation**: Accurately simulates stat stage modifiers ($\pm 6$), primary/secondary stat drops (e.g., Close Combat, Draco Meteor), direct healing (e.g., Recover, Roost), damage-proportional draining (e.g., Drain Punch, Draining Kiss), recoil recoil damage, status afflictions with type immunities, and persistent hazards.
* **Speed-Tie Chance Nodes**: Models 50/50 priority and speed ties explicitly via probabilistic branching.
* **Switch Tempo Regularization**: Penalizes non-productive switching cycles when aggressive active lines exist.

### 2.2 Neural Policy Network (ONNX Runtime)
* **Architecture**: Set-Transformer with multi-head self-attention invariant to bench ordering. Encodes 12 Pokemon slots (6 friendly, 6 opponent) with categorical embeddings (species, items, abilities, status, tera types) and continuous state attributes.
* **Action Pruning**: Opponent action branches with predicted probability below the threshold ($P < 0.05$) are pruned, with the remaining distribution normalized.
* **Inference**: Exported to ONNX and executed via ONNX Runtime CPU execution provider with sub-millisecond latency.

### 2.3 Inverse Damage Calculation
* Intersects observed damage intervals against the 16 discrete damage rolls ($\lfloor 85\% \dots 100\% \rfloor$) to infer opponent offensive stats and identify hidden items (e.g., Choice Band, Choice Specs, Life Orb).
* Supports complete Generation 9 modifiers: Harsh Sunlight, Heavy Rain, Snow Defense boost (Ice), Sandstorm Sp. Def boost (Rock), Electric/Grassy/Psychic/Misty terrains, and Reflect/Light Screen.

---

## 3. Supported Formats

The bot supports dynamic data and model loading across multiple competitive tiers:

* **Gen 9 Random Battles**: `gen9randombattle`
* **Gen 9 OverUsed (OU)**: `gen9ou`
* **Gen 9 Champions Singles BSS (with Mega Evolutions)**: `gen9championsbssregma`, `gen9championsbssregmb`
* **Gen 9 National Dex**: `gen9nationaldex`

---

## 4. Repository Structure

```text
pokemon-showdown-bot/
├── bot.py                     # Main client, game loop, and CLI runner
├── expectiminimax.py          # Simultaneous Expectiminimax search engine
├── evaluator.py               # Heuristic evaluation functions & payoff matrix constructor
├── inverse_damage_calc.py     # Forward & inverse damage calculation models
├── smogon_priors.py           # Empirical Smogon usage statistics provider
├── policy_net.py              # Set-Transformer PyTorch neural network definition
├── policy_inference.py        # ONNX Runtime inference wrapper
├── scrape_replays.py          # Showdown replay scraping and ingestion pipeline
├── dataset_parser.py          # Replay log parser & tensor dataset generator
├── train_policy.py            # Neural network training loop with ONNX exporter
├── test_suite.py              # Automated regression and edge-case unit test suite
├── config.py                  # System configuration and path management
├── requirements.txt           # Python package dependencies
└── data/                      # Local format-specific artifacts (gitignored)
    ├── gen9ou/
    ├── gen9championsbssregma/
    └── gen9randombattle/
```

---

## 5. Getting Started

### 5.1 Prerequisites and Installation

Clone the repository and install the runtime dependencies:

```bash
git clone https://github.com/prayush-sinha/pokemon-showdown-bot.git
cd pokemon-showdown-bot
pip install -r requirements.txt
```

### 5.2 Verification

Execute the offline unit tests and simulation dry-run:

```bash
# Verify the complete decision pipeline offline
python bot.py --dry-run

# Run full unit and regression test suite
python test_suite.py
```

### 5.3 Live Battle Execution

Run the bot against the live Pokémon Showdown ladder:

```bash
# Play on the standard Gen 9 Random Battle ladder
python bot.py --ladder 5 --format gen9randombattle

# Play on the Gen 9 OU ladder
python bot.py --ladder 1 --format gen9ou

# Play Champions Singles BSS (with Mega Evolution support)
python bot.py --ladder 1 --format gen9championsbssregma

# Challenge a specific user directly
python bot.py --challenge "OpponentUsername" --format gen9ou

# Accept incoming challenges from the lobby
python bot.py --accept 1
```

---

## 6. Training Pipeline (Google Colab / GPU)

To train custom policy models on high-ELO replay datasets:

### Step 1: Scrape Replays
```bash
python scrape_replays.py --format gen9ou --count 5000 --min-rating 1500
```

### Step 2: Parse Dataset and Generate Vocabularies
```bash
python dataset_parser.py --format gen9ou
```

### Step 3: Train Policy Network and Export to ONNX
```bash
python train_policy.py \
    --dataset data/dataset_gen9ou.pt \
    --feature-schema data/feature_schema_gen9ou.json \
    --output data/gen9ou/policy_net.pth \
    --export-onnx \
    --onnx-output data/gen9ou/policy_net.onnx \
    --epochs 100 \
    --batch-size 64
```

Place the generated `policy_net.onnx`, `policy_net.pth`, `vocab_<format>.json`, and `feature_schema_<format>.json` into the corresponding `data/<format>/` directory.

---

## 7. Configuration Reference

System options can be configured via environment variables or modified directly in `config.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `SERVER_MODE` | `showdown` | Server target: `"showdown"` (play.pokemonshowdown.com) or `"local"` (localhost) |
| `BOT_USERNAME` | — | Account username on Pokémon Showdown |
| `BOT_PASSWORD` | — | Account password on Pokémon Showdown |
| `BATTLE_FORMAT` | `gen9randombattle` | Default battle format |
| `PRIORS_FORMAT` | `gen9ou` | Smogon stats endpoint for empirical priors |
| `POLICY_NET_ENABLED` | `true` | Enables policy-network opponent action pruning |
| `POLICY_PRUNE_THRESHOLD` | `0.05` | Probability cutoff threshold for action pruning |
| `LOG_LEVEL` | `25` | Logging verbosity level |

---

## 8. License

This project is licensed under the MIT License.
