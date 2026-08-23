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
BOT_USERNAME = os.getenv("BOT_USERNAME", "uourbotsname")
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "password")

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

# ─── Policy Net (Phase 6, Step 6E) ──────────────────────────────────────────────
# If True, the search engine tries to load a trained policy net (ONNX
# preferred, PyTorch checkpoint as fallback) and uses it to weight/prune the
# opponent's projected moves and switches. If the files below don't exist,
# or the required runtime (onnxruntime / torch) isn't installed, the engine
# transparently falls back to Smogon priors only -- this flag does not need
# to be turned off just because you haven't trained a model yet.
POLICY_NET_ENABLED = os.getenv("POLICY_NET_ENABLED", "true").lower() not in ("0", "false", "no")

POLICY_ONNX_PATH = os.getenv("POLICY_ONNX_PATH", "data/policy_net.onnx")
POLICY_PTH_PATH = os.getenv("POLICY_PTH_PATH", "data/policy_net.pth")
POLICY_FEATURE_SCHEMA_PATH = os.getenv("POLICY_FEATURE_SCHEMA_PATH", "data/feature_schema_gen9ou.json")
POLICY_VOCAB_PATH = os.getenv("POLICY_VOCAB_PATH", "data/vocab_gen9ou.json")

from pathlib import Path

# Opponent actions predicted with probability below this are pruned from the
# search tree (and the remaining candidates renormalized) so search time
# isn't spent on moves/switches the policy net considers unrealistic.
POLICY_PRUNE_THRESHOLD = float(os.getenv("POLICY_PRUNE_THRESHOLD", "0.05"))


def get_format_paths(format_id: str) -> dict[str, str]:
    """
    Resolve model and schema filepaths for a specific format.
    Checks data/<format_id>/ first, then falls back to root data/.
    """
    clean_id = format_id.lower().replace("-", "").replace(" ", "").replace("[", "").replace("]", "")
    format_dir = Path("data") / clean_id

    onnx_path = str(format_dir / "policy_net.onnx") if (format_dir / "policy_net.onnx").exists() else POLICY_ONNX_PATH
    pth_path = str(format_dir / "policy_net.pth") if (format_dir / "policy_net.pth").exists() else POLICY_PTH_PATH
    schema_path = (
        str(format_dir / f"feature_schema_{clean_id}.json")
        if (format_dir / f"feature_schema_{clean_id}.json").exists()
        else (str(format_dir / "feature_schema.json") if (format_dir / "feature_schema.json").exists() else POLICY_FEATURE_SCHEMA_PATH)
    )
    vocab_path = (
        str(format_dir / f"vocab_{clean_id}.json")
        if (format_dir / f"vocab_{clean_id}.json").exists()
        else (str(format_dir / "vocab.json") if (format_dir / "vocab.json").exists() else POLICY_VOCAB_PATH)
    )

    return {
        "onnx": onnx_path,
        "pth": pth_path,
        "schema": schema_path,
        "vocab": vocab_path,
    }
