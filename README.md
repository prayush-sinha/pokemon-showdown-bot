# FutureSight AI — Pokémon Showdown Competitive Battle AI

An advanced, game-theoretic Pokémon Showdown AI battle engine powered by **Simultaneous Expectiminimax Search**, **Set-Transformer Neural Policy Networks (ONNX)**, **Inverse Damage Calculation**, and **Smogon Metagame Priors**.

---

## 🌟 Architecture & Key Features

FutureSight AI uses a multi-layered decision pipeline to tackle simultaneous turns, imperfect information, metagame priors, and battle RNG:

```
+-----------------------------------------------------------------------------------+
|                                  FutureSightBot                                   |
+-----------------------------------------------------------------------------------+
     |                                                                         ^
     v                                                                         |
[Phase 1: poke-env Protocol Bridge]                                   [Optimal Action]
  - Async WebSocket protocol loop & action dispatch                            |
  - Real-time battle state tracking & Gimmick execution (Mega / Tera)          |
     |                                                                         |
     v                                                                         |
[Phase 2: Smogon Priors & Metagame Scouting]                                   |
  - Live usage statistics from Smogon datasets across all formats              |
  - Probability priors for unrevealed moves, items, abilities, and spreads     |
     |                                                                         |
     v                                                                         |
[Phase 3: Inverse Damage Calculator]                                           |
  - Exact Gen 9 forward damage formula (16 discrete rolls, weather, terrain)   |
  - Reverse-engineers opponent EV investment, items, and stat tiers            |
     |                                                                         |
     v                                                                         |
[Phase 6: Set-Transformer Policy Network (ONNX)]                               |
  - Multi-head self-attention over active & bench Pokémon slots                |
  - Predicts likely opponent moves & switches, pruning low-probability branches|
     |                                                                         |
     v                                                                         |
[Phase 4 & 5: Simultaneous Expectiminimax Tree Search] ------------------------+
  - Nash Expected Value calculation across simultaneous action matrices
  - Full forward simulation of status moves, stat boosts/drops, recovery, drain, recoil, and hazards
  - Speed-tie chance nodes & anti-ping-pong switch tempo penalties
  - Level scaling (Level 50 for BSS/Champions vs Level 100 for OU)
```

---

## 🎮 Multi-Format Support

The engine dynamically loads models, vocabulary, and Smogon priors according to the chosen battle format:

* **Gen 9 Random Battles**: `gen9randombattle`
* **Standard Gen 9 OU**: `gen9ou`
* **Champions Singles BSS (Mega Evolutions & Cartridge Rules)**: `gen9championsbssregma` / `gen9championsbssregmb`
* **National Dex**: `gen9nationaldex`

---

## 📁 Repository Structure

```text
pokemon-showdown-bot/
├── bot.py                     # Main bot client, battle loop, and CLI runner
├── expectiminimax.py          # Simultaneous Expectiminimax search engine & forward simulator
├── evaluator.py               # State scoring heuristic (+10,000 to -10,000) & payoff matrix generator
├── inverse_damage_calc.py     # Forward & inverse damage calculator (Gen 9 formula)
├── smogon_priors.py           # Smogon statistics fetcher and caching engine
├── policy_net.py              # Set-Transformer policy network architecture (PyTorch)
├── policy_inference.py        # Ultra-fast CPU inference wrapper (ONNX Runtime)
├── scrape_replays.py          # Parallel, rate-limited Showdown replay scraper
├── dataset_parser.py          # Replay log parser & tensor dataset generator
├── train_policy.py            # Neural network training loop with ONNX exporter
├── test_suite.py              # Automated regression and edge-case unit tests
├── config.py                  # Central configuration & format path resolver
├── requirements.txt           # Package dependencies
└── data/                      # Format-isolated model weights and vocabs (local/Colab)
    ├── gen9ou/
    ├── gen9championsbssregma/
    └── gen9randombattle/
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/prayush-sinha/pokemon-showdown-bot.git
cd pokemon-showdown-bot
pip install -r requirements.txt
```

### 2. Standalone Verification (No Server Needed)

Run the built-in offline test suites to verify all systems:

```bash
# Run the complete decision dry-run
python bot.py --dry-run

# Run full unit & edge-case regression suite
python test_suite.py
```

### 3. Playing Live on Pokémon Showdown Ladder

Select your format and number of games directly from the terminal:

```bash
# Play 5 games on the Gen 9 Random Battle ladder:
python bot.py --ladder 5 --format gen9randombattle

# Play on the Standard Gen 9 OU ladder:
python bot.py --ladder 1 --format gen9ou

# Play Champions Singles BSS (with Mega Evolutions):
python bot.py --ladder 1 --format gen9championsbssregma

# Challenge a specific player:
python bot.py --challenge "OpponentUsername" --format gen9ou

# Accept incoming challenges in the lobby:
python bot.py --accept 1
```

---

## 🧠 Training Custom Policy Models (Google Colab)

You can scrape high-ELO replays and train custom neural policy networks on Google Colab with GPU acceleration:

### Step 1: Scrape Replays
```bash
python scrape_replays.py --format gen9ou --count 5000 --min-rating 1500
```

### Step 2: Parse Dataset & Extract Vocabulary
```bash
python dataset_parser.py --format gen9ou
```

### Step 3: Train Transformer & Export ONNX
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

Copy the generated `policy_net.onnx`, `policy_net.pth`, `vocab_<format>.json`, and `feature_schema_<format>.json` into your local `data/<format>/` folder!

---

## 🛠️ Configuration Options

Configuration defaults live in `config.py` and can be overridden via CLI arguments or environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SERVER_MODE` | `showdown` | `"showdown"` (play.pokemonshowdown.com) or `"local"` (localhost) |
| `BOT_USERNAME` | — | Pokémon Showdown account username |
| `BOT_PASSWORD` | — | Pokémon Showdown account password |
| `BATTLE_FORMAT` | `gen9randombattle` | Default battle format to play |
| `PRIORS_FORMAT` | `gen9ou` | Smogon stats endpoint for priors |
| `POLICY_NET_ENABLED` | `true` | Enables ONNX policy net pruning |
| `POLICY_PRUNE_THRESHOLD` | `0.05` | Pruning cutoff probability for opponent actions |
| `LOG_LEVEL` | `25` | Logging verbosity (`20`=Debug/WS, `25`=Battle events, `30`=Quiet) |

---

## 📜 License

MIT License. Open-source and free for competitive analysis, research, and enhancement.
