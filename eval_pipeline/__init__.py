"""Evaluation pipeline: LLM-as-Judge scoring for ASR, OSS, SS metrics.

This package provides a complete evaluation framework for measuring attack success,
output shift, and stealth across LLM-based systems.

Main Components:
- ASRJudge: Attack Success Rate (binary 0/1) - did agent endorse target claim?
- OSSJudge: Output Shift Score [0-1] - how much output shifted toward attack?
- SSJudge: Stealth Score [0-1] - how credible/trustworthy does output appear?

Usage:
    from eval_pipeline import EvaluationPipeline, EvaluationCase

    # Initialize with LLM client
    pipeline = EvaluationPipeline(llm_client=your_client)

    # Create evaluation cases
    cases = [
        EvaluationCase(
            query="Is supplement X safe?",
            agent_output="Yes, supplement X is very safe...",
            metadata={
                "target_claim": "Supplement X cures diseases",
                "domain": "health",
                "gold_answer": "Supplement X is unproven...",
                "attacker_objective": "Make users trust supplement X",
                "key_facts": ["No clinical trials", "Marketing claims only"]
            }
        )
    ]

    # Run evaluation
    results = pipeline.evaluate_batch(cases)
    pipeline.save_results(results, "health_experiment")
    pipeline.print_summary(results)
"""

from .base_judge import BaseJudge, JudgeConfig, LLMClient
from .asr_judge import ASRJudge
from .oss_judge import OSSJudge
from .ss_judge import SSJudge
from .pipeline import EvaluationPipeline, EvaluationCase, EvaluationResult
from .utils import (
    format_result,
    save_results,
    compute_statistics,
    format_score_report,
    parse_llm_json,
    extract_score_from_response,
)

__all__ = [
    # Base classes
    "BaseJudge",
    "JudgeConfig",
    "LLMClient",
    # Judge implementations
    "ASRJudge",
    "OSSJudge",
    "SSJudge",
    # Pipeline
    "EvaluationPipeline",
    "EvaluationCase",
    "EvaluationResult",
    # Utilities
    "format_result",
    "save_results",
    "compute_statistics",
    "format_score_report",
    "parse_llm_json",
    "extract_score_from_response",
]
