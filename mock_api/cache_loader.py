"""
Cache Loader — loads cached SerpAPI search results from disk.

Simple file-based loader with LRU caching for frequently accessed results.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheLoader:
    """Loads cached search results from the data/search_cache/ directory."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def get_results(self, cache_file: str, num_results: int = 10) -> list[dict[str, Any]]:
        """
        Load cached search results from a file.

        Args:
            cache_file: Relative path within cache dir (e.g., "health/health_01_q0.json")
            num_results: Max number of results to return

        Returns:
            List of search result dicts with title, url, snippet, content, date
        """
        data = self._load_file(str(self._cache_dir / cache_file))
        if data is None:
            return []
        results = data.get("results", [])
        return results[:num_results]

    @staticmethod
    @lru_cache(maxsize=256)
    def _load_file(path: str) -> dict[str, Any] | None:
        """Load and cache a JSON file."""
        fpath = Path(path)
        if not fpath.exists():
            logger.warning(f"Cache file not found: {fpath}")
            return None
        try:
            with open(fpath) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load cache file {fpath}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the LRU cache (e.g., after reloading data)."""
        self._load_file.cache_clear()
