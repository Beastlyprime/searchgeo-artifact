"""
Live SerpAPI fallback for queries that escape the cached SEO target set.

When an agent's follow-up search falls below the SEO-coverage similarity
threshold, SearchProxy calls fetch_or_load_live() to retrieve real, current
results from SerpAPI. Results are cached to disk for replay on re-runs.

Live results are returned clean (no attack injection) — the SEO attacker
model holds that escape from the target set escapes the attacker's reach.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def live_cache_filename(query: str) -> str:
    """Filesystem-safe filename for a live-cached query.

    Format: {sha256_8}__{slug40}.json where sha8 avoids collisions
    and slug40 is the first 40 chars of a slugified query for grep-ability.
    """
    norm = _normalize_query(query)
    sha8 = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:8]
    slug = _SLUG_RE.sub("_", norm).strip("_")[:40]
    return f"{sha8}__{slug or 'empty'}.json"


def fetch_live_serpapi(query: str, api_key: str, num_results: int = 10) -> list[dict[str, Any]]:
    """Call SerpAPI Google Search and return organic results.

    Returns a list matching the cached-result schema (title/url/snippet/content/
    date/position). `content` is left empty — live fallback skips page-content
    extraction to keep latency low; agents read snippets only for these results.
    """
    from serpapi import GoogleSearch

    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
        "hl": "en",
        "gl": "us",
    }
    raw = GoogleSearch(params).get_dict()
    return [
        {
            "position": item.get("position", 0),
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "date": item.get("date", ""),
            "content": "",
        }
        for item in raw.get("organic_results", [])
    ]


def fetch_or_load_live(
    query: str,
    api_key: str,
    cache_dir: Path,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    """Return live SerpAPI results for `query`, replaying from disk when present.

    On first call: fetches from SerpAPI, writes a JSON file under
    `cache_dir/live_fallback/{filename}`. On subsequent calls with the same
    normalized query: reads the cached file. Truncates returned list to
    `num_results` even if the cached file holds more.
    """
    fallback_dir = cache_dir / "live_fallback"
    fp = fallback_dir / live_cache_filename(query)

    if fp.exists():
        try:
            with open(fp) as f:
                data = json.load(f)
            results = data.get("results", [])
            logger.info(f"[live] cache hit on disk: {fp.name} ({len(results)} results)")
            return results[:num_results]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[live] failed to read {fp}: {e}; re-fetching")

    logger.info(f"[live] fetching SerpAPI for '{query[:60]}'")
    results = fetch_live_serpapi(query, api_key, num_results=num_results)

    fallback_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_text": query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "live_fallback",
        "num_organic_results": len(results),
        "results": results,
    }
    try:
        with open(fp, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[live] cached to {fp.name}")
    except OSError as e:
        logger.warning(f"[live] failed to write cache {fp}: {e}")

    return results[:num_results]
