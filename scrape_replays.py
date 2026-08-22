#!/usr/bin/env python3
"""
scrape_replays.py
==================
Phase 6, Step 6A — Showdown Replay Scraper & Ingestion Pipeline.

Downloads high-Elo Pokemon Showdown replays to disk for use as raw training
data in Step 6B (dataset_parser.py). Talks only to Showdown's public,
unauthenticated replay endpoints:

    Search API : https://replay.pokemonshowdown.com/search.json?format={format}&page={page}
    Replay JSON: https://replay.pokemonshowdown.com/{replay_id}.json

Design goals for this step:
  * Be a polite citizen of Showdown's servers (throttled requests, real
    exponential backoff on 429/5xx/timeouts, a real User-Agent).
  * Never re-download a replay we already have on disk.
  * Only persist replays that are actually useful training signal (high
    enough rating, long enough to reflect real decision-making rather than
    an early forfeit/disconnect).
  * Fail loudly on genuine bugs, but degrade gracefully on flaky network
    conditions (which are the norm when scraping thousands of replays).

Dependencies: Python 3.9+ stdlib + `requests`. No torch/onnx in this step —
that arrives in 6C/6D once we have clean data to train on.

Usage:
    python scrape_replays.py --format gen9ou --count 500 --min-rating 1600
    python scrape_replays.py --dry-run
    python scrape_replays.py                      # no args -> self-verification dry run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"

DEFAULT_FORMAT = "gen9ou"
DEFAULT_MIN_RATING = 1600
DEFAULT_MIN_TURNS = 5
DEFAULT_OUTPUT_DIR = "data/replays"
DEFAULT_MAX_PAGES = 200

MIN_SLEEP_SECONDS = 0.5
MAX_SLEEP_SECONDS = 1.0

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.5

USER_AGENT = "PokemonAI-ReplayScraper/1.0 (research/training data collection)"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

TURN_RE = re.compile(r"^\|turn\|(\d+)", re.MULTILINE)
FORFEIT_RE = re.compile(r"forfeited", re.IGNORECASE)

logger = logging.getLogger("scrape_replays")


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

@dataclass
class ScrapeStats:
    pages_scanned: int = 0
    candidates_seen: int = 0
    filtered_by_rating: int = 0
    filtered_duplicate: int = 0
    filtered_by_content: int = 0
    downloaded: int = 0
    errors: int = 0

    def log_summary(self) -> None:
        logger.info(
            "SUMMARY | pages=%d candidates=%d rating_filtered=%d "
            "duplicates_skipped=%d content_filtered=%d downloaded=%d errors=%d",
            self.pages_scanned,
            self.candidates_seen,
            self.filtered_by_rating,
            self.filtered_duplicate,
            self.filtered_by_content,
            self.downloaded,
            self.errors,
        )


# ---------------------------------------------------------------------------
# HTTP layer: session, throttling, retry/backoff
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def polite_sleep(min_s: float = MIN_SLEEP_SECONDS, max_s: float = MAX_SLEEP_SECONDS) -> None:
    """Randomized delay between requests so we don't hammer Showdown's
    servers with a fixed, easily-detected cadence."""
    time.sleep(random.uniform(min_s, max_s))


def fetch_json(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    max_retries: int = MAX_RETRIES,
) -> Optional[dict]:
    """GET a URL and parse it as JSON, retrying transient failures with
    exponential backoff + jitter.

    Returns None if the resource genuinely doesn't exist (404), the response
    isn't valid JSON, or retries are exhausted -- callers treat None as
    "skip this one, don't crash the whole run".
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            resp = session.get(url, params=params, timeout=15)
        except (requests.ConnectionError, requests.Timeout) as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error("Network error on %s after %d retries: %s", url, max_retries, exc)
                return None
            backoff = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Network error (%s). Backing off %.1fs (attempt %d/%d)",
                exc, backoff, attempt, max_retries,
            )
            time.sleep(backoff)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                logger.error("Malformed JSON response from %s", url)
                return None

        if resp.status_code == 404:
            logger.debug("404 Not Found: %s", url)
            return None

        if resp.status_code in RETRYABLE_STATUS_CODES:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Giving up on %s after %d retries (last status %d)",
                    url, max_retries, resp.status_code,
                )
                return None
            backoff = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "HTTP %d from %s. Backing off %.1fs (attempt %d/%d)",
                resp.status_code, url, backoff, attempt, max_retries,
            )
            time.sleep(backoff)
            continue

        logger.error("Unexpected HTTP %d from %s", resp.status_code, url)
        return None

    return None


# ---------------------------------------------------------------------------
# Search API + candidate iteration
# ---------------------------------------------------------------------------

def fetch_search_page(session: requests.Session, format_id: str, page: int) -> List[dict]:
    """Fetch one page of the replay search results. Handles both the "bare
    list" and "{'replays': [...]}" response shapes defensively, since
    third-party API docs for Showdown's replay search are informal and have
    drifted before."""
    params = {"format": format_id, "page": page}
    data = fetch_json(session, SEARCH_URL, params=params)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("replays", []) or []
    logger.warning("Unexpected search.json shape: %r", type(data))
    return []


def iter_candidates(
    session: requests.Session,
    format_id: str,
    max_pages: int,
    stats: ScrapeStats,
) -> Iterator[dict]:
    """Yield replay metadata dicts across search result pages until the API
    stops returning results or we hit the safety page cap."""
    page = 1
    while page <= max_pages:
        results = fetch_search_page(session, format_id, page)
        stats.pages_scanned += 1
        if not results:
            logger.info("No more results at page %d; stopping pagination.", page)
            return
        for candidate in results:
            stats.candidates_seen += 1
            yield candidate
        page += 1
        polite_sleep()


def passes_rating_filter(candidate: dict, min_rating: int) -> bool:
    """Showdown's search results include a 'rating' field that is null for
    unrated/unranked games. --min-rating 0 disables this filter entirely."""
    if min_rating <= 0:
        return True
    rating = candidate.get("rating")
    if rating is None:
        return False
    try:
        return int(rating) >= min_rating
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Replay content parsing + quality filters
# ---------------------------------------------------------------------------

def count_turns(log_text: str) -> int:
    """Battle logs contain '|turn|N' markers; the highest N is the turn
    count the game reached before ending."""
    matches = TURN_RE.findall(log_text or "")
    if not matches:
        return 0
    return max(int(m) for m in matches)


def is_forfeit(log_text: str) -> bool:
    return bool(FORFEIT_RE.search(log_text or ""))


def passes_content_filter(replay_data: dict, min_turns: int) -> bool:
    """Reject games that are too short to contain meaningful decision
    sequences. This subsumes "ignore forfeits under N turns" -- a forfeit at
    turn 40 still has 40 turns of real play worth keeping; a forfeit (or any
    other early termination) under the threshold does not."""
    log_text = replay_data.get("log", "")
    turns = count_turns(log_text)
    return turns >= min_turns


# ---------------------------------------------------------------------------
# Replay fetch + disk I/O
# ---------------------------------------------------------------------------

def fetch_replay(session: requests.Session, replay_id: str) -> Optional[dict]:
    url = REPLAY_URL.format(replay_id=replay_id)
    return fetch_json(session, url)


def replay_output_path(output_dir: Path, format_id: str, replay_id: str) -> Path:
    return output_dir / format_id / f"{replay_id}.json"


def already_downloaded(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def save_replay(output_dir: Path, format_id: str, replay_id: str, data: dict) -> Path:
    """Write via a temp file + atomic rename so a crash mid-write never
    leaves a corrupt, half-written JSON file that would silently poison
    Step 6B's parser."""
    path = replay_output_path(output_dir, format_id, replay_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp_path.replace(path)
    return path


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def run_scrape(args: argparse.Namespace, session: requests.Session, stats: ScrapeStats) -> None:
    output_dir = Path(args.output_dir)
    collected = 0

    for candidate in iter_candidates(session, args.format, args.max_pages, stats):
        if collected >= args.count:
            break

        if not passes_rating_filter(candidate, args.min_rating):
            stats.filtered_by_rating += 1
            continue

        replay_id = candidate.get("id")
        if not replay_id:
            logger.debug("Candidate missing 'id' field, skipping: %r", candidate)
            continue

        out_path = replay_output_path(output_dir, args.format, replay_id)
        if already_downloaded(out_path):
            stats.filtered_duplicate += 1
            continue

        replay_data = fetch_replay(session, replay_id)
        polite_sleep()

        if replay_data is None:
            stats.errors += 1
            continue

        if not passes_content_filter(replay_data, args.min_turns):
            stats.filtered_by_content += 1
            continue

        save_replay(output_dir, args.format, replay_id, replay_data)
        stats.downloaded += 1
        collected += 1

        if collected % 10 == 0 or collected == args.count:
            logger.info("Progress: %d/%d replays downloaded", collected, args.count)

    if collected < args.count:
        logger.warning(
            "Stopped at %d/%d target replays -- search pages exhausted (or --max-pages "
            "cap of %d hit). Try lowering --min-rating/--min-turns, or raising --max-pages.",
            collected, args.count, args.max_pages,
        )


# ---------------------------------------------------------------------------
# Dry run / self-verification
# ---------------------------------------------------------------------------

def run_dry_run(args: argparse.Namespace, session: requests.Session, stats: ScrapeStats) -> None:
    """Fetch page 1 of search results, print matching replay IDs, then pull
    and parse ONE full replay JSON as an end-to-end connectivity + parsing
    sanity check. Downloads nothing to disk."""
    logger.info("[DRY RUN] Verifying connectivity to %s ...", SEARCH_URL)
    results = fetch_search_page(session, args.format, page=1)
    stats.pages_scanned += 1

    if not results:
        logger.error(
            "[DRY RUN] Got zero results from page 1. Either the network is unreachable, "
            "'%s' isn't a valid/active format id, or Showdown's API shape has changed.",
            args.format,
        )
        return

    logger.info("[DRY RUN] Retrieved %d candidates from page 1.", len(results))

    matching = []
    for candidate in results:
        stats.candidates_seen += 1
        if passes_rating_filter(candidate, args.min_rating):
            matching.append(candidate)
        else:
            stats.filtered_by_rating += 1

    logger.info(
        "[DRY RUN] %d/%d candidates meet --min-rating %d",
        len(matching), len(results), args.min_rating,
    )
    for candidate in matching[:20]:
        logger.info(
            "  id=%-24s rating=%-6s players=%s",
            candidate.get("id"), candidate.get("rating"), candidate.get("players"),
        )

    sample_pool = matching or results
    sample = random.choice(sample_pool)
    sample_id = sample.get("id")
    if not sample_id:
        logger.warning("[DRY RUN] Sampled candidate had no 'id' field: %r", sample)
        return

    logger.info("[DRY RUN] Fetching full replay JSON for sample id=%s ...", sample_id)
    replay_data = fetch_replay(session, sample_id)
    if replay_data is None:
        logger.error("[DRY RUN] Failed to fetch/parse sample replay %s", sample_id)
        stats.errors += 1
        return

    log_text = replay_data.get("log", "")
    turns = count_turns(log_text)
    forfeited = is_forfeit(log_text)
    would_pass = passes_content_filter(replay_data, args.min_turns)

    logger.info(
        "[DRY RUN] Parsed sample OK | id=%s format=%s turns=%d forfeit=%s rating=%s players=%s",
        sample_id,
        replay_data.get("formatid", replay_data.get("format")),
        turns,
        forfeited,
        replay_data.get("rating"),
        replay_data.get("players"),
    )
    logger.info(
        "[DRY RUN] Sample replay would %s the content filter (min_turns=%d)",
        "PASS" if would_pass else "FAIL",
        args.min_turns,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6A: Download high-Elo Pokemon Showdown replays for AI training.",
    )
    parser.add_argument("--format", default=DEFAULT_FORMAT,
                         help="Showdown format id, e.g. gen9ou (default: %(default)s)")
    parser.add_argument("--count", type=int, default=500,
                         help="Target number of NEW replays to collect this run (default: %(default)s)")
    parser.add_argument("--min-rating", type=int, default=DEFAULT_MIN_RATING,
                         help="Minimum rating/Elo to accept a replay; 0 disables the filter (default: %(default)s)")
    parser.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS,
                         help="Minimum turn count required to keep a replay (default: %(default)s)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help="Local destination folder (default: %(default)s)")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                         help="Safety cap on search-result pages to paginate through (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Fetch the first page and print matching replay IDs without downloading the full set.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    session = create_session()
    stats = ScrapeStats()

    logger.info(
        "Starting scrape_replays.py | format=%s min_rating=%d min_turns=%d "
        "output_dir=%s dry_run=%s",
        args.format, args.min_rating, args.min_turns, args.output_dir, args.dry_run,
    )

    try:
        if args.dry_run:
            run_dry_run(args, session, stats)
        else:
            run_scrape(args, session, stats)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; partial progress on disk is preserved.")
    finally:
        stats.log_summary()

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No CLI arguments at all -> self-verification mode. This is the
        # "runnable dry run" required by spec 5: it proves the endpoint is
        # reachable and that a real replay JSON parses correctly, without
        # requiring the user to know the --dry-run flag exists yet.
        sys.exit(main(["--dry-run"]))
    sys.exit(main())
