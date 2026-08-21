# FutureSight AI — Pokémon Showdown Competitive Battle AI

An advanced, game-theoretic Pokémon Showdown AI battle bot built with Python and `poke-env`.

---

## 🌟 Architecture & Key Features

FutureSight AI uses a 4-phase decision pipeline to tackle simultaneous turns, imperfect information, and RNG in Pokémon battles:

```
+-----------------------------------------------------------------------+
|                           FutureSightBot                              |
+-----------------------------------------------------------------------+
     |                                                      ^
     v                                                      |
[Phase 1: poke-env Protocol Bridge]                    [Move Selection]
  - Async WebSocket protocol loop                           |
  - Turn state parsing & action dispatch                    |
     |                                                      |
     v                                                      |
[Phase 2: Smogon Priors]                                    |
  - Live usage statistics from Smogon datasets              |
  - Probability priors for unrevealed moves & items         |
     |                                                      |
     v                                                      |
[Phase 3: Inverse Damage Calc]                              |
  - Gen 9 forward damage formula (16 rolls, STAB, items)    |
  - Reverse-engineers opponent EV investment & items        |
     |                                                      |
     v                                                      |
[Phase 4: The Brain (Simultaneous Expectiminimax)] --------+
  - Simultaneous turn resolution (Nash EV over opponent moves)
  - Speed priority & KO suppression modeling
  - Accuracy chance nodes & hazard chip projections
  - Fast execution (<20ms per turn)
```

---

## 📁 Repository Structure

- `bot.py`: Main bot client, game loop, and CLI runner.
- `expectiminimax.py`: Phase 4B simultaneous expectiminimax tree search engine.
- `evaluator.py`: Phase 4A heuristic state scoring (+10,000 to -10,000) & payoff matrix generator.
- `inverse_damage_calc.py`: Phase 3 forward & inverse damage calculator.
- `smogon_priors.py`: Phase 2 Smogon statistics fetcher and caching engine.
- `config.py`: Central settings and environment variable bindings.
- `requirements.txt`: Python package dependencies.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-username>/pokemon-showdown-battle-simulator.git
cd pokemon-showdown-battle-simulator
pip install -r requirements.txt
```

### 2. Standalone Verification (No Server Needed)

You can run individual module tests or dry-run the complete decision loop:

```bash
# Test the complete bot decision loop
python bot.py --dry-run

# Test individual AI components
python expectiminimax.py
python evaluator.py
python inverse_damage_calc.py
python smogon_priors.py
```

### 3. Playing Live on Pokémon Showdown Ladder

Configure your credentials via environment variables or CLI flags:

```bash
# Run on the live ranked ladder
python bot.py --showdown --username "YourUsername" --password "YourPassword" --ladder 5

# Challenge a specific opponent
python bot.py --showdown --username "YourUsername" --password "YourPassword" --challenge "OpponentName"

# Accept incoming challenges in the lobby
python bot.py --showdown --username "YourUsername" --password "YourPassword" --accept 1
```

---

## 🛠️ Configuration

You can customize the bot by setting environment variables or editing `config.py`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SERVER_MODE` | `showdown` | `"showdown"` (public server) or `"local"` (localhost) |
| `BOT_USERNAME` | — | Pokémon Showdown username |
| `BOT_PASSWORD` | — | Pokémon Showdown password |
| `BATTLE_FORMAT` | `gen9randombattle` | Battle format (`gen9randombattle`, `gen9ou`, etc.) |
| `PRIORS_FORMAT` | `gen9ou` | Smogon stats endpoint to pull build probabilities |
| `LOG_LEVEL` | `25` | Logging verbosity level |

---

## 📜 License

MIT License.
