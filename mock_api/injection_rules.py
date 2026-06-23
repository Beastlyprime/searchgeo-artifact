"""
Injection Rules — defines where attack content gets inserted into real results.

Loads injection rule configs from YAML files and provides a clean interface
for the result injector to query positions and modes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class InjectionRule:
    """Defines how attack content is injected for a specific mode/variant."""
    attack_mode: str
    attack_positions: list[int]  # 1-indexed positions in result list
    content_mode: str | None = None  # Optional attack-content file alias
    source_indices: list[int] | None = None  # Optional 0-indexed source subset

    @property
    def num_attacks(self) -> int:
        return len(self.attack_positions)


class InjectionRuleLoader:
    """Loads and manages injection rule configurations."""

    def __init__(self, rules_dir: Path | None = None) -> None:
        self._rules: dict[str, dict[str, Any]] = {}  # config_name -> parsed YAML
        if rules_dir and rules_dir.exists():
            self._load_all(rules_dir)

    def _load_all(self, rules_dir: Path) -> None:
        """Load all YAML rule files from directory."""
        for yaml_file in sorted(rules_dir.glob("*.yaml")):
            name = yaml_file.stem
            with open(yaml_file) as f:
                self._rules[name] = yaml.safe_load(f) or {}
            logger.info(f"Loaded injection rules: {name}")

    def get_rule(self, config_name: str, attack_mode: str, variant: str | None = None) -> InjectionRule:
        """
        Get injection rule for a specific config + mode combination.

        Args:
            config_name: Name of the config (e.g., "exp1")
            attack_mode: Attack mode (e.g., "2A", "baseline")
            variant: Optional variant name for ablation configs (e.g., "n3_top")

        Returns:
            InjectionRule with positions and mode info
        """
        config = self._rules.get(config_name, {})

        # Check mode defaults first; variants may override only positions.
        modes = config.get("modes", {})
        mode_data = modes.get(attack_mode, {})

        # Check variants first for position/rank ablations.
        if variant and "variants" in config:
            variant_data = config["variants"].get(variant, {})
            positions = variant_data.get("attack_positions", mode_data.get("attack_positions", []))
            return InjectionRule(
                attack_mode=attack_mode,
                attack_positions=positions,
                content_mode=variant_data.get("content_mode", mode_data.get("content_mode")),
                source_indices=variant_data.get("source_indices", mode_data.get("source_indices")),
            )

        # Check modes
        positions = mode_data.get("attack_positions", [])

        return InjectionRule(
            attack_mode=attack_mode,
            attack_positions=positions,
            content_mode=mode_data.get("content_mode"),
            source_indices=mode_data.get("source_indices"),
        )

    def list_configs(self) -> list[str]:
        """List all loaded config names."""
        return sorted(self._rules.keys())
