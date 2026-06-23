"""Load defense system prompts for the public SearchGEO artifact."""

from pathlib import Path
from typing import Dict, List


UNIFIED_DEFENSE = "unified_defense"
ALL_DEFENSES = [UNIFIED_DEFENSE]


def _get_prompts_dir() -> Path:
    """Get the path to the prompts directory."""
    return Path(__file__).parent / "prompts" / "v1"


def load_defense(defense_id: str = UNIFIED_DEFENSE) -> str:
    """Load a public defense system prompt by ID."""
    if defense_id not in ALL_DEFENSES:
        raise ValueError(
            f"Unknown defense: {defense_id}. "
            f"Valid options: {', '.join(ALL_DEFENSES)}"
        )

    prompt_file = _get_prompts_dir() / f"{defense_id}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Defense prompt file not found: {prompt_file}")
    return prompt_file.read_text()


def load_all_defenses() -> Dict[str, str]:
    """Load all public defense prompts."""
    return {defense_id: load_defense(defense_id) for defense_id in ALL_DEFENSES}


def load_combined_defense(defense_ids: List[str]) -> str:
    """Combine selected defense prompts into a single system prompt addition."""
    separator = "\n\n" + "=" * 70 + "\n\n"
    return separator.join(load_defense(defense_id) for defense_id in defense_ids)


def get_defense_description(defense_id: str) -> str:
    """Get a human-readable description of a public defense prompt."""
    if defense_id == UNIFIED_DEFENSE:
        return (
            "Unified defense prompt covering source scrutiny, cross-validation, "
            "and injection-pattern detection."
        )
    return f"Unknown defense: {defense_id}"


def list_available_defenses() -> Dict[str, str]:
    """List all public defense prompts with descriptions."""
    return {defense_id: get_defense_description(defense_id) for defense_id in ALL_DEFENSES}
