#!/usr/bin/env python3
"""
Evaluation Runner — Runs the LLM-as-Judge evaluation pipeline on experiment results.

Reads raw agent outputs from results/ and scores them with ASR, OSS, and SS judges.
Produces per-case scores and aggregated statistics.

Usage:
    python scripts/run_eval.py                          # Evaluate all results
    python scripts/run_eval.py --exp exp1               # Evaluate Experiment 1 only
    python scripts/run_eval.py --exp exp1 --dry-run     # Use mock judge (no API calls)
    python scripts/run_eval.py --exp exp1 --judges asr oss  # Run specific judges
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ─── Project Root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API keys
load_dotenv(PROJECT_ROOT / "api_keys.env")
load_dotenv(Path.home() / ".openclaw" / ".env")
load_dotenv(PROJECT_ROOT / ".env")

from eval_pipeline.pipeline import EvaluationPipeline, EvaluationCase, EvaluationResult
from eval_pipeline.base_judge import JudgeConfig, LLMClient

# ─── Logging ──────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs" / "eval"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("run_eval")


# ─── Mock LLM Client (for dry-run / testing) ─────────────────────────

class MockLLMClient:
    """Deterministic mock LLM client for testing evaluation pipeline without API calls.

    Returns responses whose JSON keys match the real judge prompts exactly, so
    dry-runs exercise the same parser paths as production.
    """

    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Return mock judge responses based on the judge type detected in the prompt."""
        system_msg = ""
        user_msg = ""
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            elif m.get("role") == "user":
                user_msg = m.get("content", "")

        prompt_text = (system_msg + " " + user_msg).lower()

        # ASR judge — real prompt asks for {"judgment": "yes"|"no", "reasoning": "..."}
        if "target claim to evaluate" in prompt_text or '"judgment"' in prompt_text:
            return json.dumps({
                "judgment": "no",
                "reasoning": "Mock ASR: no clear endorsement of the target claim in agent output.",
            })

        # OSS judge — real prompt asks for {"shift_score": 0.x, "analysis": "..."}
        if "attacker's objective" in prompt_text or '"shift_score"' in prompt_text:
            return json.dumps({
                "shift_score": 0.15,
                "analysis": "Mock OSS: minimal shift from gold baseline.",
            })

        # SS judge — real prompt asks for {"stealth_score": 0.x, "red_flags": [...], "overall_assessment": "..."}
        if '"stealth_score"' in prompt_text or "credibility auditor" in prompt_text:
            return json.dumps({
                "stealth_score": 0.75,
                "red_flags": [],
                "overall_assessment": "Mock SS: output appears reasonably credible.",
            })

        # Default — unknown prompt; emit a generic numeric score
        return json.dumps({"score": 0.5, "reasoning": "Mock: unknown prompt type."})


# ─── Real LLM Client ─────────────────────────────────────────────────

class MultiProviderLLMClient:
    """LLM client that routes to appropriate provider based on model config."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Route chat completion to the appropriate provider based on model."""
        judge_cfg = self.config.get("models", {}).get("judge", {})
        primary_cfg = judge_cfg.get("primary", {})
        ss_cfg = judge_cfg.get("ss_auditor", {})

        # Route to provider based on model ID
        if model and model == ss_cfg.get("model_id"):
            provider = ss_cfg.get("provider", "anthropic")
            model_id = model
        else:
            provider = primary_cfg.get("provider", "google")
            model_id = model or primary_cfg.get("model_id", "gemini-3-flash-preview")

        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 2048),
            )
            return resp.choices[0].message.content or ""

        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic()
            system_msg = ""
            filtered_messages = []
            for m in messages:
                if m.get("role") == "system":
                    system_msg = m.get("content", "")
                else:
                    filtered_messages.append(m)
            resp = client.messages.create(
                model=model_id,
                system=system_msg,
                messages=filtered_messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 2048),
            )
            return resp.content[0].text

        elif provider == "google":
            from google import genai
            from google.genai import types
            import os
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

            system_instruction = None
            genai_contents = []
            for m in messages:
                if m.get("role") == "system":
                    system_instruction = m.get("content", "")
                elif m.get("role") == "user":
                    genai_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.get("content", ""))]))
                elif m.get("role") == "assistant":
                    text = m.get("content", "")
                    if text:
                        genai_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.0),
                max_output_tokens=kwargs.get("max_tokens", 2048),
                system_instruction=system_instruction,
            )
            resp = client.models.generate_content(
                model=model_id,
                contents=genai_contents,
                config=config,
            )
            return resp.text or ""

        elif provider == "openrouter":
            from openai import OpenAI
            import os
            client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 2048),
            )
            return resp.choices[0].message.content or ""

        else:
            raise ValueError(f"Unsupported provider for judge: {provider}")


# ─── Result Collection ────────────────────────────────────────────────

def collect_result_files(results_dir: Path, exp_filter: str | None = None) -> list[Path]:
    """Collect all result JSON files, optionally filtered by experiment."""
    def is_candidate(path: Path) -> bool:
        rel_parts = path.relative_to(results_dir).parts
        # Evaluation outputs are derived artifacts. Including them in an
        # unfiltered run creates empty/invalid cases and can recursively
        # evaluate previous evaluations.
        top_level = rel_parts[0]
        return not (top_level.startswith("eval") or top_level.startswith("_"))

    if exp_filter:
        target_dir = results_dir / exp_filter
        if not target_dir.exists():
            logger.warning(f"Results directory not found: {target_dir}")
            return []
        return sorted(p for p in target_dir.rglob("*.json") if is_candidate(p))
    else:
        return sorted(p for p in results_dir.rglob("*.json") if is_candidate(p))


def load_result_file(path: Path) -> dict[str, Any] | None:
    """Load a single result JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


# ─── Task Metadata Cache ─────────────────────────────────────────────

_task_cache: dict[str, dict[str, Any]] = {}


def _load_task_metadata(domain: str, query_id: str) -> dict[str, Any]:
    """Load task metadata from tasks/{domain}.yaml, cached."""
    cache_key = f"{domain}_{query_id}"
    if cache_key in _task_cache:
        return _task_cache[cache_key]

    tasks_file = PROJECT_ROOT / "tasks" / f"{domain}.yaml"
    if tasks_file.exists():
        with open(tasks_file) as f:
            tasks = yaml.safe_load(f)
        for t in tasks:
            tk = f"{t.get('domain', domain)}_{t.get('query_id', '')}"
            _task_cache[tk] = t

    return _task_cache.get(cache_key, {})


def _build_case_id(result: dict[str, Any], mode: str, domain: str, query_id: str) -> str:
    """Build a unique case_id that includes defense and architecture info."""
    parts = [result.get("backend", "unknown")]
    defense = result.get("defense", "")
    if defense:
        parts.append(defense)
    arch = result.get("architecture", "")
    if arch and arch != "tool_calling":
        parts.append(arch)
    variant = result.get("ablation_variant", "")
    if variant:
        parts.append(variant)
    parts.extend([mode or "none", domain, query_id])
    return "_".join(parts)


def result_to_eval_case(result: dict[str, Any], result_path: Path) -> EvaluationCase | None:
    """Convert a raw experiment result to an EvaluationCase for judging.

    Enriches result data with task metadata (key_facts, attacker_target_claim)
    from tasks/{domain}.yaml when not present in the result file.
    """
    try:
        domain = result.get("domain", "")
        query_id = result.get("query_id", "")
        query = result.get("query", "")
        output = result.get("output", "")
        mode = result.get("mode") or result.get("attack_mode", "")

        if not domain or not query_id or not query or not output:
            logger.warning(
                f"Skipping non-result or incomplete result file {result_path}: "
                f"domain={bool(domain)} query_id={bool(query_id)} query={bool(query)} output={bool(output)}"
            )
            return None

        # Load task file to get key_facts and attacker_target_claim
        task_meta = _load_task_metadata(domain, query_id)
        result_task_meta = result.get("task_metadata", {}) or {}

        # Resolve fields: result snapshot > root fields > current task file > empty
        gold_answer = (
            result_task_meta.get("gold_answer")
            or result.get("gold_answer")
            or task_meta.get("gold_answer", "")
        )
        attacker_objective = (
            result_task_meta.get("attacker_objective")
            or result.get("attacker_objective")
            or task_meta.get("attacker_objective", "")
        )
        target_claim = (
            result_task_meta.get("attacker_target_claim")
            or result.get("attacker_target_claim")
            or task_meta.get("attacker_target_claim")
            or attacker_objective
        )

        # key_facts from result snapshot, then root field, then task file
        raw_facts = (
            result_task_meta.get("key_facts")
            or result.get("key_facts")
            or task_meta.get("key_facts", [])
        )
        key_facts = [
            f.get("fact", f) if isinstance(f, dict) else str(f)
            for f in raw_facts
        ]

        return EvaluationCase(
            query=query,
            agent_output=output,
            metadata={
                "target_claim": target_claim,
                "domain": domain,
                "gold_answer": gold_answer,
                "attacker_objective": attacker_objective,
                "key_facts": key_facts,
                "experiment": result.get("experiment", ""),
                "attack_mode": mode,
                "backend": result.get("backend", ""),
                "defense": result.get("defense", ""),
                "query_id": query_id,
                "scene_id": result.get("scene_id", ""),
                "exp_type": result.get("exp_type", ""),
            },
            case_id=result.get("scene_id") or _build_case_id(result, mode, domain, query_id),
        )
    except Exception as e:
        logger.error(f"Failed to create eval case from {result_path}: {e}")
        return None


# ─── Score Aggregation ────────────────────────────────────────────────

def aggregate_scores(
    eval_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics from evaluation results."""
    import numpy as np

    if not eval_results:
        return {}

    metrics = ["asr", "oss", "ss"]
    stats: dict[str, Any] = {"total_cases": len(eval_results)}

    for metric in metrics:
        values = [r["scores"].get(metric) for r in eval_results if metric in r.get("scores", {})]
        if values:
            arr = np.array(values, dtype=float)
            stats[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "n": len(values),
            }

            # Bootstrap 95% CI
            bootstrap_means = []
            rng = np.random.default_rng(42)
            for _ in range(1000):
                sample = rng.choice(arr, size=len(arr), replace=True)
                bootstrap_means.append(float(np.mean(sample)))
            bootstrap_means.sort()
            stats[metric]["ci_lower"] = bootstrap_means[25]
            stats[metric]["ci_upper"] = bootstrap_means[974]

    return stats


def aggregate_by_group(
    eval_results: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate scores grouped by a metadata key (e.g., attack_mode, domain, backend)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in eval_results:
        key = r.get("metadata", {}).get(group_key, "unknown")
        groups.setdefault(key, []).append(r)

    return {k: aggregate_scores(v) for k, v in sorted(groups.items())}


# ─── Main Evaluation Flow ────────────────────────────────────────────

def run_evaluation(
    results_dir: Path,
    output_dir: Path,
    exp_filter: str | None = None,
    dry_run: bool = False,
    judges_filter: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run evaluation pipeline on experiment results."""

    # Collect result files
    result_files = collect_result_files(results_dir, exp_filter)
    logger.info(f"Found {len(result_files)} result files to evaluate")

    if not result_files:
        logger.warning("No result files found. Run experiments first.")
        return {"error": "No results to evaluate"}

    # Build evaluation cases
    cases: list[EvaluationCase] = []
    for path in result_files:
        data = load_result_file(path)
        if data:
            case = result_to_eval_case(data, path)
            if case:
                cases.append(case)

    logger.info(f"Created {len(cases)} evaluation cases")

    # Initialize pipeline
    if dry_run:
        llm_client = MockLLMClient()
    else:
        llm_client = MultiProviderLLMClient(config or {})

    # Get judge model IDs from config
    judge_cfg = (config or {}).get("models", {}).get("judge", {})
    primary_model = judge_cfg.get("primary", {}).get("model_id", "gemini-3-flash-preview")
    ss_model = judge_cfg.get("ss_auditor", {}).get("model_id", "claude-sonnet-4-20250514")

    pipeline = EvaluationPipeline(
        llm_client=llm_client,
        primary_model=primary_model,
        ss_model=ss_model,
        output_dir=output_dir,
        judges_filter=judges_filter,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    eval_results = pipeline.evaluate_batch(cases)

    # Convert to dicts for aggregation
    results_dicts = []
    for er in eval_results:
        results_dicts.append({
            "case_id": er.case_id,
            "query": er.query,
            "scores": er.scores,
            "errors": er.errors,
            "metadata": er.metadata,
        })

    # Save per-case results
    exp_name = exp_filter or "all"
    output_file = output_dir / f"eval_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "experiment": exp_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_cases": len(results_dicts),
            "results": results_dicts,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Per-case results saved to {output_file}")

    # Compute and save aggregate statistics
    overall_stats = aggregate_scores(results_dicts)
    by_attack = aggregate_by_group(results_dicts, "attack_mode")
    by_domain = aggregate_by_group(results_dicts, "domain")
    by_backend = aggregate_by_group(results_dicts, "backend")

    stats_file = output_dir / f"stats_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump({
            "experiment": exp_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall": overall_stats,
            "by_attack_mode": by_attack,
            "by_domain": by_domain,
            "by_backend": by_backend,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Aggregate statistics saved to {stats_file}")

    # Print summary
    pipeline.print_summary(eval_results)

    return {
        "total_cases": len(results_dicts),
        "overall_stats": overall_stats,
        "output_file": str(output_file),
        "stats_file": str(stats_file),
    }


# ─── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on experiment results")
    parser.add_argument("--exp", help="Experiment to evaluate: exp1, exp2, exp3, exp4, or omit for all")
    parser.add_argument("--dry-run", action="store_true", help="Use mock judge (no API calls)")
    parser.add_argument("--judges", nargs="*", help="Run specific judges (asr, oss, ss)")
    parser.add_argument("--results-dir", default="results", help="Results directory (default: results/)")
    parser.add_argument("--output-dir", default="results/eval", help="Output directory for eval results")
    args = parser.parse_args()

    # Load config
    config: dict[str, Any] = {}
    config_path = PROJECT_ROOT / "config" / "models.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config["models"] = yaml.safe_load(f)

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = PROJECT_ROOT / args.output_dir

    summary = run_evaluation(
        results_dir=results_dir,
        output_dir=output_dir,
        exp_filter=args.exp,
        dry_run=args.dry_run,
        judges_filter=args.judges,
        config=config,
    )

    logger.info(f"\nEvaluation complete: {summary.get('total_cases', 0)} cases evaluated")


if __name__ == "__main__":
    main()
