"""Utility functions for score parsing, result formatting, and JSON serialization."""

import json
import re
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path


def parse_llm_json(response: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response.

    Robust parsing that handles responses with surrounding text or explanation,
    including ```json ... ``` markdown code fences (commonly emitted by Gemini).

    Args:
        response: Raw LLM response that may contain JSON and other text

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found in response
    """
    # Strip markdown code fences first (```json ... ``` or ``` ... ```)
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
    if fence_match:
        response = fence_match.group(1).strip()

    # Find JSON object boundaries
    first_brace = response.find("{")
    last_brace = response.rfind("}")

    if first_brace == -1 or last_brace == -1:
        raise ValueError(f"No JSON object found in response: {response[:200]}")

    json_str = response[first_brace : last_brace + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}") from e


def format_result(
    query: str,
    agent_output: str,
    scores: Dict[str, float],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Format evaluation results into standardized output structure.

    Args:
        query: Original user query
        agent_output: Agent's response
        scores: Dictionary of judge names to scores
        metadata: Evaluation context metadata

    Returns:
        Formatted result dictionary with all relevant information
    """
    return {
        "query": query,
        "agent_output": agent_output,
        "metadata": metadata,
        "scores": scores,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score_summary": {
            "asr": scores.get("asr"),
            "oss": scores.get("oss"),
            "ss": scores.get("ss"),
            "average": sum(scores.values()) / len(scores) if scores else None,
        },
    }


def save_results(
    results: list[Dict[str, Any]],
    output_dir: Path,
    experiment_name: str = "evaluation",
) -> Path:
    """Save evaluation results to JSON file with metadata.

    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save results to
        experiment_name: Name for this evaluation run

    Returns:
        Path to saved results file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = output_dir / filename

    output = {
        "experiment": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_evaluations": len(results),
        "results": results,
        "statistics": compute_statistics(results),
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def compute_statistics(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across evaluation results.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not results:
        return {}

    metrics = ["asr", "oss", "ss"]
    stats = {}

    for metric in metrics:
        scores = [r["scores"].get(metric) for r in results if metric in r["scores"]]
        scores = [s for s in scores if s is not None]

        if scores:
            stats[metric] = {
                "mean": sum(scores) / len(scores),
                "count": len(scores),
                "min": min(scores),
                "max": max(scores),
            }

            # Compute standard deviation
            if len(scores) > 1:
                mean = stats[metric]["mean"]
                variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
                stats[metric]["std"] = variance**0.5

    return stats


def extract_score_from_response(
    response: str, score_type: str = "float"
) -> Optional[float]:
    """Extract a numeric score from an LLM response.

    Handles multiple formats: JSON, plain numbers, yes/no strings.

    Args:
        response: Raw LLM response text
        score_type: One of "binary", "float", or "json"

    Returns:
        Extracted score as float, or None if extraction fails
    """
    try:
        if score_type == "json":
            parsed = parse_llm_json(response)
            # Look for common score field names
            for key in ["score", "scores", "value", "judgment", "rating"]:
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif isinstance(val, str):
                        return float(val)
            return None

        elif score_type == "binary":
            lower = response.lower()
            if "1" in response or "yes" in lower or "true" in lower:
                return 1.0
            elif "0" in response or "no" in lower or "false" in lower:
                return 0.0
            return None

        elif score_type == "float":
            # Try JSON first
            try:
                parsed = parse_llm_json(response)
                for v in parsed.values():
                    if isinstance(v, (int, float)):
                        return float(v)
            except ValueError:
                pass

            # Look for float pattern
            matches = re.findall(r"(\d+\.?\d*)", response)
            if matches:
                score = float(matches[0])
                return max(0.0, min(1.0, score))

        return None

    except (ValueError, TypeError):
        return None


def format_score_report(scores: Dict[str, float]) -> str:
    """Format evaluation scores into human-readable report.

    Args:
        scores: Dictionary of metric names to scores

    Returns:
        Formatted score report string
    """
    lines = ["Evaluation Scores:", "-" * 40]

    metric_names = {
        "asr": "Attack Success Rate (binary)",
        "oss": "Output Shift Score [0-1]",
        "ss": "Stealth Score [0-1]",
    }

    for metric, name in metric_names.items():
        if metric in scores:
            score = scores[metric]
            lines.append(f"{name:35s}: {score:.4f}")

    if scores:
        avg = sum(scores.values()) / len(scores)
        lines.append("-" * 40)
        lines.append(f"{'Average Score':35s}: {avg:.4f}")

    return "\n".join(lines)
