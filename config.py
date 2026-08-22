"""
config.py — Central configuration for the Pokémon Showdown Battle Bot.

All connection parameters, credentials, and tuning knobs live here.
Can be overridden via environment variables or CLI arguments.
"""

import os

# ─── Server Mode ───────────────────────────────────────────────────────────────
# "local"    → connect to a local Pokémon Showdown server (no auth needed)
# "showdown" → connect to play.pokemonshowdown.com (requires credentials)
SERVER_MODE = os.getenv("SERVER_MODE", "showdown")

# ─── Account Credentials (only needed for SERVER_MODE="showdown") ──────────────
BOT_USERNAME = os.getenv("BOT_USERNAME", "YourBotUsername")
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "YourBotPassword")

# ─── Battle Settings ───────────────────────────────────────────────────────────
# The format the bot will play. "gen9randombattle" requires no team.
BATTLE_FORMAT = os.getenv("BATTLE_FORMAT", "gen9randombattle")

# Maximum number of battles the bot may play simultaneously.
MAX_CONCURRENT_BATTLES = int(os.getenv("MAX_CONCURRENT_BATTLES", "1"))

# ─── Logging ───────────────────────────────────────────────────────────────────
# 20 = logging.INFO  (shows every WS message — great for debugging)
# 25 = poke-env custom level (shows only key events)
# 30 = logging.WARNING (quiet)
LOG_LEVEL = int(os.getenv("LOG_LEVEL", "25"))

# ─── Smogon Priors (Phase 2) ──────────────────────────────────────────────────
# The Smogon format to pull usage statistics from.
# Must match a valid pkmn.github.io/smogon endpoint (e.g. "gen9ou", "gen9uu").
PRIORS_FORMAT = os.getenv("PRIORS_FORMAT", "gen9ou")

# How long (in seconds) before re-fetching stats from the server.
# Default: 24 hours. Set to 0 to always fetch fresh data.
PRIORS_CACHE_MAX_AGE = int(os.getenv("PRIORS_CACHE_MAX_AGE", str(24 * 60 * 60)))
