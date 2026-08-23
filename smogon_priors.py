"""
smogon_priors.py — Phase 2: The Data Layer

Fetches Smogon competitive usage statistics and provides probabilistic
"priors" for unknown opponent builds.  When an opponent reveals a Pokémon
species, this module returns the most likely moves, items, abilities,
spreads, and tera-types based on real ladder data.

Data source:  https://pkmn.github.io/smogon/data/stats/<format>.json
Cached locally in  .cache/  to avoid redundant HTTP requests.

Usage
─────
    from smogon_priors import SmogonPriors

    priors = SmogonPriors(format_id="gen9ou")
    build  = priors.get_likely_build("Great Tusk")
    print(build)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from poke_env.data import GenData

logger = logging.getLogger("SmogonPriors")

# ─── Constants ─────────────────────────────────────────────────────────────────
_STATS_URL_TEMPLATE = "https://pkmn.github.io/smogon/data/stats/{format_id}.json"
_SETS_URL_TEMPLATE = "https://pkmn.github.io/smogon/data/sets/{format_id}.json"

_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes — structured outputs
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MoveProb:
    """A single move with its usage probability (0-1)."""
    name: str
    probability: float

    def __repr__(self) -> str:
        return f"{self.name} ({self.probability:.0%})"


@dataclass
class ItemProb:
    """A single item with its usage probability (0-1)."""
    name: str
    probability: float

    def __repr__(self) -> str:
        return f"{self.name} ({self.probability:.0%})"


@dataclass
class LikelyBuild:
    """
    The statistically most probable build for a Pokémon species.

    Attributes
    ----------
    species       : Display name of the species.
    top_moves     : Top N moves sorted by usage probability.
    top_items     : Top N items sorted by usage probability.
    top_ability   : Most used ability and its probability.
    top_spreads   : Top N EV spreads (as "Nature:HP/Atk/Def/SpA/SpD/Spe")
                    with their probabilities.
    top_tera      : Top N tera-types with their probabilities.
    usage_rate    : Overall usage rate of this species in the format (0-1).
    """
    species: str
    top_moves: list[MoveProb] = field(default_factory=list)
    top_items: list[ItemProb] = field(default_factory=list)
    top_ability: tuple[str, float] = ("", 0.0)
    top_spreads: list[tuple[str, float]] = field(default_factory=list)
    top_tera: list[tuple[str, float]] = field(default_factory=list)
    usage_rate: float = 0.0

    def summary(self) -> str:
        """One-line human-readable summary for console logging."""
        moves_str = ", ".join(str(m) for m in self.top_moves)
        items_str = " or ".join(str(i) for i in self.top_items)
        ability_str = f"{self.top_ability[0]} ({self.top_ability[1]:.0%})"
        return (
            f"Moves: [{moves_str}] | Items: [{items_str}] "
            f"| Ability: {ability_str}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Name normalisation — bridge between poke-env IDs and Smogon display names
# ═══════════════════════════════════════════════════════════════════════════════
def _build_name_lookup(gen: int = 9) -> dict[str, str]:
    """
    Build a bidirectional lookup:  lowered-no-spaces  →  Smogon display name.

    poke-env uses IDs like 'greattusk', 'ironvaliant'.
    Smogon stats JSON uses display names like 'Great Tusk', 'Iron Valiant'.
    The poke-env GenData pokedex maps  id → {"name": "Display Name", ...}.
    """
    lookup: dict[str, str] = {}
    try:
        gd = GenData.from_gen(gen)
        for poke_id, info in gd.pokedex.items():
            display_name = info.get("name", "")
            if display_name:
                # Map the lowered-no-spaces ID to display name
                lookup[poke_id] = display_name
                # Also map the lowered display name for fuzzy matching
                lookup[display_name.lower().replace(" ", "").replace("-", "")] = display_name
    except Exception:
        logger.warning("Could not build name lookup from GenData", exc_info=True)
    return lookup


def _normalise_species(raw: str, lookup: dict[str, str]) -> str:
    """
    Convert any species identifier to the Smogon display name.

    Handles:
      - poke-env IDs:  "greattusk"  → "Great Tusk"
      - Already correct: "Great Tusk" → "Great Tusk"
      - Partial/lower:   "great tusk" → "Great Tusk"
    """
    # First try: direct lookup of the raw string
    if raw in lookup:
        return lookup[raw]

    # Second try: strip spaces/hyphens, lower
    key = raw.lower().replace(" ", "").replace("-", "")
    if key in lookup:
        return lookup[key]

    # Fallback: title-case the raw input (best effort)
    logger.debug("Species '%s' not found in lookup, using title-case fallback", raw)
    return raw.replace("-", " ").title()


# ═══════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _cache_path(format_id: str, data_type: str) -> Path:
    """Return the filesystem path for a cached JSON file."""
    return _CACHE_DIR / f"{format_id}_{data_type}.json"


def _is_cache_fresh(path: Path, max_age: int = _CACHE_MAX_AGE_SECONDS) -> bool:
    """Check if a cached file exists and is younger than max_age seconds."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age


def _read_cache(path: Path) -> dict:
    """Read and parse a cached JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_cache(path: Path, data: dict) -> None:
    """Write data to a cache JSON file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("Cached data written to %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# The SmogonPriors class
# ═══════════════════════════════════════════════════════════════════════════════
class SmogonPriors:
    """
    Fetches and queries Smogon competitive usage statistics.

    Parameters
    ----------
    format_id : str
        The Smogon format to fetch stats for (e.g., "gen9ou", "gen9uu").
    gen : int
        The generation number, used for name lookup (default: auto-detected
        from format_id).
    cache_max_age : int
        Maximum age of cached data in seconds before re-fetching.

    Example
    -------
    >>> priors = SmogonPriors("gen9ou")
    >>> build = priors.get_likely_build("Garchomp")
    >>> print(build.summary())
    """

    def __init__(
        self,
        format_id: str = "gen9ou",
        gen: Optional[int] = None,
        cache_max_age: int = _CACHE_MAX_AGE_SECONDS,
    ):
        self.format_id = format_id
        self.gen = gen or self._detect_gen(format_id)
        self.cache_max_age = cache_max_age

        # Lazy-loaded data stores
        self._stats_data: Optional[dict] = None
        self._sets_data: Optional[dict] = None
        self._name_lookup: Optional[dict[str, str]] = None
        self._logged_missing: set[str] = set()

    # ── Generation detection ────────────────────────────────────────────────
    @staticmethod
    def _detect_gen(format_id: str) -> int:
        """Extract generation number from format string like 'gen9ou'."""
        match = re.match(r"gen(\d+)", format_id)
        return int(match.group(1)) if match else 9

    # ── Name lookup (lazy init) ─────────────────────────────────────────────
    @property
    def name_lookup(self) -> dict[str, str]:
        if self._name_lookup is None:
            self._name_lookup = _build_name_lookup(self.gen)
        return self._name_lookup

    # ── Data fetching with cache ────────────────────────────────────────────
    def _fetch_json(self, url: str, data_type: str) -> dict:
        """
        Fetch JSON from a URL, using the local cache if fresh.

        Parameters
        ----------
        url : str
            The remote URL to fetch from.
        data_type : str
            Identifier for cache filename (e.g., "stats", "sets").

        Returns
        -------
        dict
            The parsed JSON data.

        Raises
        ------
        ConnectionError
            If the HTTP request fails and no cache is available.
        """
        cache = _cache_path(self.format_id, data_type)

        # Try cache first
        if _is_cache_fresh(cache, self.cache_max_age):
            logger.info("Using cached %s data for %s", data_type, self.format_id)
            return _read_cache(cache)

        # Fetch from remote
        logger.info("Fetching %s data from %s …", data_type, url)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            _write_cache(cache, data)
            logger.info(
                "Fetched %s data for %s (%d bytes)",
                data_type,
                self.format_id,
                len(response.content),
            )
            return data
        except requests.RequestException as exc:
            # If fetch fails but we have stale cache, use it
            if cache.exists():
                logger.warning(
                    "Fetch failed (%s), using stale cache for %s",
                    exc,
                    self.format_id,
                )
                return _read_cache(cache)
            raise ConnectionError(
                f"Failed to fetch {data_type} for {self.format_id} "
                f"and no cache available: {exc}"
            ) from exc

    @property
    def stats_data(self) -> dict:
        """Lazy-load the usage statistics JSON."""
        if self._stats_data is None:
            url = _STATS_URL_TEMPLATE.format(format_id=self.format_id)
            self._stats_data = self._fetch_json(url, "stats")
        return self._stats_data

    @property
    def sets_data(self) -> dict:
        """Lazy-load the curated sets JSON."""
        if self._sets_data is None:
            url = _SETS_URL_TEMPLATE.format(format_id=self.format_id)
            self._sets_data = self._fetch_json(url, "sets")
        return self._sets_data

    # ── The main query method ───────────────────────────────────────────────
    def get_likely_build(
        self,
        species_name: str,
        n_moves: int = 4,
        n_items: int = 2,
        n_spreads: int = 3,
        n_tera: int = 3,
    ) -> Optional[LikelyBuild]:
        """
        Return the statistically most likely build for a species.

        Parameters
        ----------
        species_name : str
            Any form of the species name — poke-env ID ("greattusk"),
            display name ("Great Tusk"), or lowered ("great tusk").
        n_moves : int
            Number of top moves to return.
        n_items : int
            Number of top items to return.
        n_spreads : int
            Number of top EV spreads to return.
        n_tera : int
            Number of top tera-types to return.

        Returns
        -------
        LikelyBuild or None
            The build data, or None if the species isn't found in this format.
        """
        display_name = _normalise_species(species_name, self.name_lookup)

        pokemon_data = self.stats_data.get("pokemon", {})

        # Try exact match first, then case-insensitive search
        mon = pokemon_data.get(display_name)
        if mon is None:
            # Fallback: search case-insensitively
            lower_target = display_name.lower()
            for key, value in pokemon_data.items():
                if key.lower() == lower_target:
                    mon = value
                    display_name = key  # Use the canonical key
                    break

        if mon is None:
            if display_name not in self._logged_missing:
                self._logged_missing.add(display_name)
                logger.info(
                    "Species '%s' not found in %s stats -- using base-stat prior fallback",
                    display_name,
                    self.format_id,
                )
            return None

        # ── Parse moves ─────────────────────────────────────────────────
        raw_moves = mon.get("moves", {})
        top_moves = sorted(raw_moves.items(), key=lambda kv: kv[1], reverse=True)
        top_moves = [MoveProb(name=m, probability=p) for m, p in top_moves[:n_moves]]

        # ── Parse items ─────────────────────────────────────────────────
        raw_items = mon.get("items", {})
        top_items = sorted(raw_items.items(), key=lambda kv: kv[1], reverse=True)
        top_items = [ItemProb(name=i, probability=p) for i, p in top_items[:n_items]]

        # ── Parse abilities ─────────────────────────────────────────────
        raw_abilities = mon.get("abilities", {})
        if raw_abilities:
            best_ability = max(raw_abilities.items(), key=lambda kv: kv[1])
        else:
            best_ability = ("Unknown", 0.0)

        # ── Parse spreads ───────────────────────────────────────────────
        raw_spreads = mon.get("spreads", {})
        top_spreads = sorted(raw_spreads.items(), key=lambda kv: kv[1], reverse=True)
        top_spreads = [(s, p) for s, p in top_spreads[:n_spreads]]

        # ── Parse tera types ────────────────────────────────────────────
        raw_tera = mon.get("teraTypes", {})
        top_tera = sorted(raw_tera.items(), key=lambda kv: kv[1], reverse=True)
        top_tera = [(t, p) for t, p in top_tera[:n_tera]]

        # ── Usage rate ──────────────────────────────────────────────────
        usage = mon.get("usage", {})
        usage_rate = usage.get("weighted", usage.get("raw", 0.0))

        return LikelyBuild(
            species=display_name,
            top_moves=top_moves,
            top_items=top_items,
            top_ability=best_ability,
            top_spreads=top_spreads,
            top_tera=top_tera,
            usage_rate=usage_rate,
        )

    # ── Convenience: list all available species ─────────────────────────────
    def available_species(self) -> list[str]:
        """Return a sorted list of all species in the loaded stats."""
        return sorted(self.stats_data.get("pokemon", {}).keys())

    # ── Convenience: total battles in the dataset ───────────────────────────
    def total_battles(self) -> int:
        """Return the total number of battles in the dataset."""
        return self.stats_data.get("battles", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    priors = SmogonPriors("gen9ou")

    print(f"\n{'=' * 70}")
    print(f"  Smogon Usage Statistics -- {priors.format_id}")
    print(f"  Total battles in dataset: {priors.total_battles():,}")
    print(f"{'=' * 70}\n")

    test_species = [
        "Great Tusk",       # Display name
        "greattusk",        # poke-env ID
        "Garchomp",         # Standard name
        "Slowking-Galar",   # Regional form
        "Iron Valiant",     # Paradox mon
        "NonexistentMon",   # Should return None gracefully
    ]

    for species in test_species:
        build = priors.get_likely_build(species)
        if build:
            print(f"  [OK] {build.species} (usage: {build.usage_rate:.1%})")
            print(f"    {build.summary()}")
            if build.top_spreads:
                print(f"    Top spread: {build.top_spreads[0][0]} ({build.top_spreads[0][1]:.0%})")
            if build.top_tera:
                tera_str = ", ".join(f"{t} ({p:.0%})" for t, p in build.top_tera)
                print(f"    Tera: {tera_str}")
        else:
            print(f"  [--] {species} -- not found in {priors.format_id}")
        print()
