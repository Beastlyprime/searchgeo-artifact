"""
Result Injector — injects attack content into cached real search results.

Takes real cached results + injection rules + attack content and produces
the final mixed result list that agents see.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from mock_api.injection_rules import InjectionRule
from mock_api.query_matcher import MatchResult

logger = logging.getLogger(__name__)


class ResultInjector:
    """
    Injects attack content into real search results based on injection rules.

    Attack content is loaded from data/attack_content/{query_id}/mode_{X}.yaml.
    """

    def __init__(self, attack_content_dir: Path) -> None:
        self._attack_dir = attack_content_dir
        self._content_cache: dict[str, dict] = {}

    def inject(
        self,
        cached_results: list[dict[str, Any]],
        match_result: MatchResult,
        injection_rule: InjectionRule,
    ) -> list[dict[str, Any]]:
        """
        Inject attack content into cached real results.

        If match is attack-eligible and injection rule has positions:
          - Replace results at specified positions with attack content
          - Keep remaining positions as cached real results
        If not attack-eligible or no injection:
          - Return cached results as-is

        Args:
            cached_results: Real search results from cache
            match_result: Query match info (attack eligibility, query_id)
            injection_rule: Where to inject (positions, mode)

        Returns:
            Final result list with attacks injected at specified positions
        """
        # No injection if not attack-eligible or no attack positions
        if not match_result.is_attack_eligible or not injection_rule.attack_positions:
            return list(cached_results)

        # Load attack content for this task + mode. Some experiment conditions
        # are aliases over a shared content file, selecting a specific source.
        content_mode = injection_rule.content_mode or injection_rule.attack_mode
        attack_sources = self._load_attack_content(
            match_result.query_id,
            content_mode,
        )
        if injection_rule.source_indices is not None:
            attack_sources = [
                attack_sources[i]
                for i in injection_rule.source_indices
                if 0 <= i < len(attack_sources)
            ]

        if not attack_sources:
            logger.warning(
                f"No attack content for {match_result.query_id} mode {content_mode}, "
                "returning clean results"
            )
            return list(cached_results)

        # Build result list with injections
        results = list(cached_results)
        attack_idx = 0

        for pos in sorted(injection_rule.attack_positions):
            if attack_idx >= len(attack_sources):
                # Cycle through the selected attack sources when there are more
                # injection positions than sources. This applies whether or not
                # source_indices was specified — e.g. dose-response ablations on
                # a single-source mode want the same source replicated across
                # positions, not silently dropped.
                attack_idx = attack_idx % len(attack_sources)

            attack = attack_sources[attack_idx]
            pos_0 = pos - 1  # Convert 1-indexed to 0-indexed

            # Build clean result dict — only fields the agent should see
            injected_result = {
                "position": pos,
                "title": attack.get("title", ""),
                "url": attack.get("url", ""),
                "snippet": attack.get("snippet", ""),
                "content": attack.get("content", ""),
                "date": attack.get("date", ""),
                "injected": True,
                "attack_mode": injection_rule.attack_mode,
                "content_mode": content_mode,
            }

            if pos_0 < len(results):
                results[pos_0] = injected_result
            else:
                results.append(injected_result)

            attack_idx += 1

        logger.info(
            f"Injected {len(injection_rule.attack_positions)} attack results "
            f"for {match_result.query_id} mode {injection_rule.attack_mode} "
            f"at positions {injection_rule.attack_positions}"
        )
        return results

    def _load_attack_content(self, query_id: str, attack_mode: str) -> list[dict]:
        """Load attack content for a task + mode from YAML files."""
        cache_key = f"{query_id}_{attack_mode}"
        if cache_key in self._content_cache:
            return self._content_cache[cache_key].get("sources", [])

        # Try data/attack_content/{query_id}/mode_{X}.yaml
        content_file = self._attack_dir / query_id / f"mode_{attack_mode}.yaml"
        if not content_file.exists():
            logger.debug(f"Attack content not found: {content_file}")
            return []

        try:
            with open(content_file) as f:
                data = yaml.safe_load(f) or {}
            self._content_cache[cache_key] = data
            sources = data.get("sources", [])
            logger.info(f"Loaded {len(sources)} attack sources from {content_file.name}")
            return sources
        except Exception as e:
            logger.error(f"Failed to load attack content {content_file}: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear the attack content cache."""
        self._content_cache.clear()
