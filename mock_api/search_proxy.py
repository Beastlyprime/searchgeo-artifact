"""
Search Proxy — In-process search with cached results and attack injection.

Replaces the HTTP mock API server for experiments. Encapsulates the full
pipeline: query matching → cache loading → attack injection, with a live
SerpAPI fallback for queries that escape the SEO target set.

Usage:
    proxy = SearchProxy(project_root)
    proxy.configure(task_id="health_01", attack_mode="2A")
    results = proxy.search("best supplements for diabetes")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from mock_api.cache_loader import CacheLoader
from mock_api.injection_rules import InjectionRuleLoader
from mock_api.live_search import fetch_or_load_live
from mock_api.query_matcher import QueryMatcher
from mock_api.result_injector import ResultInjector

logger = logging.getLogger(__name__)


class SearchProxy:
    """
    In-process search proxy with cached real results and attack injection.

    Manages per-task state (task_id, attack_mode, search_count). Routes:
      - First search of each task → task's primary cached query + inject
      - Follow-up search, similarity ≥ followup_threshold → cached + inject
      - Follow-up search, similarity < followup_threshold → live SerpAPI (clean)

    The SEO attacker model: cached queries (primary + related) are the
    attacker's pre-poisoned ranking targets. Live fallback represents the agent
    searching outside that coverage; those results are never injected.
    """

    def __init__(
        self,
        project_root: Path,
        injection_config: str = "exp1",
        followup_threshold: float = 0.80,
        enable_live_fallback: bool = True,
        serpapi_key: str | None = None,
        attack_content_dir: Path | None = None,
    ) -> None:
        cache_dir = project_root / "data" / "search_cache"
        attack_dir = attack_content_dir or (project_root / "data" / "attack_content")
        rules_dir = project_root / "config" / "injection_rules"

        self._cache_dir = cache_dir
        self._query_matcher = QueryMatcher(cache_dir)
        self._cache_loader = CacheLoader(cache_dir)
        self._result_injector = ResultInjector(attack_dir)
        self._injection_loader = InjectionRuleLoader(rules_dir)

        self._injection_config = injection_config
        self._followup_threshold = followup_threshold
        self._enable_live_fallback = enable_live_fallback
        self._serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")

        if self._enable_live_fallback and not self._serpapi_key:
            logger.warning(
                "Live fallback enabled but no SERPAPI_KEY/SERPAPI_API_KEY found; "
                "cache misses will return [] until a key is provided."
            )

        # Per-task state
        self._task_id: str | None = None
        self._attack_mode: str | None = None
        self._injection_variant: str | None = None
        self._search_count: int = 0

        # Per-search audit state (read by HTTP server for /search _match debug field)
        self._last_match: Any | None = None
        self._last_injected: bool = False
        self._last_path: str = ""  # "cached" | "live" | "none"

        logger.info(
            f"SearchProxy initialized: {self._query_matcher.num_queries} cached queries, "
            f"injection_config={injection_config}, "
            f"followup_threshold={followup_threshold}, "
            f"live_fallback={'on' if self._enable_live_fallback else 'off'}"
        )

    def configure(
        self,
        task_id: str,
        attack_mode: str | None = None,
        injection_variant: str | None = None,
        injection_config: str | None = None,
    ) -> None:
        """Configure proxy for a new task. Resets search count.

        `injection_config` is optional and overrides the constructor default
        for this task only; subsequent configure() calls revert unless re-set.
        """
        self._task_id = task_id
        self._attack_mode = attack_mode
        self._injection_variant = injection_variant
        if injection_config is not None:
            self._injection_config = injection_config
        self._search_count = 0
        logger.info(
            f"[Proxy] Configured: task={task_id}, mode={attack_mode}, "
            f"variant={injection_variant}, config={self._injection_config}"
        )

    def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Execute a search query and return ranked results.

        Routes through three paths depending on task state and similarity:
        forced-primary (first search of a task), cached-match-with-injection
        (follow-up within SEO coverage), and live-SerpAPI-clean (follow-up
        outside SEO coverage).
        """
        is_first_search = self._search_count == 0
        self._search_count += 1

        # No task context: fall back to the global matcher
        if not self._task_id:
            return self._search_no_task(query, num_results)

        # First search: force the task's primary cached query for clean/attack parity
        if is_first_search:
            match = self._query_matcher.get_primary_for_task(self._task_id)
            if match is None:
                # Task has no primary entry — fall through to follow-up logic
                logger.warning(f"[Proxy] No primary cached query for task {self._task_id}")
            else:
                return self._serve_cached(query, match, num_results, force_inject=True, phase="first")

        # Follow-up: find best match within the task's cached query set
        match = self._query_matcher.best_match_for_task(query, self._task_id)
        if match is not None and match.similarity >= self._followup_threshold:
            return self._serve_cached(query, match, num_results, force_inject=False, phase="follow-up-hit")

        # Outside SEO coverage → live SerpAPI, no injection
        best_sim = match.similarity if match else 0.0
        return self._serve_live(query, num_results, best_sim)

    def _serve_cached(
        self,
        query: str,
        match: Any,
        num_results: int,
        force_inject: bool,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Return cached results, with attack injection if attack_mode is set."""
        cached = self._cache_loader.get_results(match.cache_file, num_results=num_results)

        should_inject = bool(self._attack_mode) and (
            force_inject or match.similarity >= self._followup_threshold
        )
        if should_inject:
            rule = self._injection_loader.get_rule(
                config_name=self._injection_config,
                attack_mode=self._attack_mode,
                variant=self._injection_variant,
            )
            results = self._result_injector.inject(cached, match, rule)
        else:
            results = cached

        self._last_match = match
        self._last_injected = should_inject
        self._last_path = "cached"

        logger.info(
            f"[Proxy] {phase} | task={self._task_id} | mode={self._attack_mode} | "
            f"q='{query[:50]}' | match={match.query_type}({match.similarity:.3f}) | "
            f"path=cached | injected={'yes' if should_inject else 'no'} | n={len(results)}"
        )
        return results

    def _serve_live(self, query: str, num_results: int, best_sim: float) -> list[dict[str, Any]]:
        """Return live SerpAPI results (clean, never injected)."""
        self._last_match = None
        self._last_injected = False

        if not self._enable_live_fallback or not self._serpapi_key:
            self._last_path = "none"
            logger.info(
                f"[Proxy] follow-up-miss | task={self._task_id} | q='{query[:50]}' | "
                f"best_sim={best_sim:.3f} < {self._followup_threshold} | path=none | n=0"
            )
            return []

        try:
            results = fetch_or_load_live(
                query,
                api_key=self._serpapi_key,
                cache_dir=self._cache_dir,
                num_results=num_results,
            )
        except Exception as e:
            logger.error(f"[Proxy] live fallback error for '{query[:50]}': {e}")
            self._last_path = "live_error"
            return []

        self._last_path = "live"
        logger.info(
            f"[Proxy] follow-up-miss | task={self._task_id} | mode={self._attack_mode} | "
            f"q='{query[:50]}' | best_sim={best_sim:.3f} < {self._followup_threshold} | "
            f"path=live | injected=no | n={len(results)}"
        )
        return results

    def _search_no_task(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Search without a configured task — used for dev / ad-hoc calls."""
        match = self._query_matcher.match(query)
        if match is None:
            return []
        results = self._cache_loader.get_results(match.cache_file, num_results=num_results)
        logger.info(
            f"[Proxy] no-task | q='{query[:50]}' | "
            f"match={match.query_type}({match.similarity:.3f}) | n={len(results)}"
        )
        return results

    @property
    def num_queries(self) -> int:
        return self._query_matcher.num_queries

    @property
    def search_count(self) -> int:
        return self._search_count

    @property
    def last_match(self) -> Any | None:
        """Most recent MatchResult; None if last search went through live fallback."""
        return self._last_match

    @property
    def last_injected(self) -> bool:
        return self._last_injected

    @property
    def last_path(self) -> str:
        """One of 'cached', 'live', 'none', or 'live_error'."""
        return self._last_path

    @property
    def current_config(self) -> dict[str, Any]:
        """Snapshot of current per-task configuration (used by HTTP /proxy/status)."""
        return {
            "task_id": self._task_id,
            "attack_mode": self._attack_mode,
            "injection_variant": self._injection_variant,
            "injection_config": self._injection_config,
        }
