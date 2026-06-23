"""Pipeline orchestrator for running the paper metrics on evaluation batches.

Coordinates ASR, OSS, and SS judges, collecting scores and producing
consolidated results with proper error handling and logging.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from .base_judge import BaseJudge, JudgeConfig, LLMClient
from .asr_judge import ASRJudge
from .oss_judge import OSSJudge
from .ss_judge import SSJudge
from .utils import format_result, save_results, compute_statistics, format_score_report


logger = logging.getLogger(__name__)


@dataclass
class EvaluationCase:
    """Single evaluation case with query, agent output, and metadata."""

    query: str
    agent_output: str
    metadata: Dict[str, Any]
    case_id: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result from evaluating a single case."""

    case_id: str
    query: str
    agent_output: str
    scores: Dict[str, float]
    errors: Dict[str, str]
    metadata: Dict[str, Any]


class EvaluationPipeline:
    """Orchestrates evaluation across all paper-metric judges.

    Manages judge instantiation, batch processing, error handling, and result
    aggregation. Can save results to structured JSON format.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        primary_model: str = "openai/gpt-5.4-mini",
        ss_model: str = "openai/gpt-5.4-mini",
        output_dir: Path = Path("results"),
        enable_ss_judge: bool = True,
        judges_filter: Optional[List[str]] = None,
    ):
        """Initialize evaluation pipeline with judges.

        Args:
            llm_client: Callable LLM client for making API calls
            primary_model: Model ID for ASR and OSS judges
            ss_model: Model ID for SS (Stealth Score) judge
            output_dir: Directory for saving evaluation results
            enable_ss_judge: Whether to run SS judge
            judges_filter: If set, only instantiate judges whose lowercase name
                appears in this list (e.g. ["asr", "oss"]). None = all judges.
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.enable_ss_judge = enable_ss_judge

        # Initialize judges. `provider` left unset — the LLM client routes by
        # model_id (see scripts/run_eval.py:MultiProviderLLMClient).
        primary_config = JudgeConfig(
            model=primary_model,
            temperature=0.0,
            max_tokens=2048,
        )

        ss_config = JudgeConfig(
            model=ss_model,
            temperature=0.0,
            max_tokens=2048,
        )

        all_judges: Dict[str, BaseJudge] = {
            "asr": ASRJudge(primary_config, llm_client),
            "oss": OSSJudge(primary_config, llm_client),
        }

        if enable_ss_judge:
            all_judges["ss"] = SSJudge(ss_config, llm_client)

        if judges_filter:
            allowed = {j.lower() for j in judges_filter}
            unknown = allowed - set(all_judges.keys())
            if unknown:
                raise ValueError(
                    f"Unknown judges in filter: {sorted(unknown)}. "
                    f"Valid: {sorted(all_judges.keys())}"
                )
            self.judges = {k: v for k, v in all_judges.items() if k in allowed}
        else:
            self.judges = all_judges

        logger.info(
            f"Pipeline initialized with {len(self.judges)} judges "
            f"({sorted(self.judges.keys())}). "
            f"Primary: {primary_model}, SS: {ss_model if 'ss' in self.judges else 'disabled'}"
        )

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate a single case with all judges.

        Args:
            case: Evaluation case with query, output, and metadata

        Returns:
            EvaluationResult with scores for each judge and any errors

        Raises:
            ValueError: If case metadata missing required fields for a judge
        """
        case_id = case.case_id or f"case_{hash(case.query) % 10000}"
        scores: Dict[str, float] = {}
        errors: Dict[str, str] = {}

        # Run each judge
        for judge_name, judge in self.judges.items():
            try:
                logger.debug(f"Running {judge_name} judge on {case_id}")
                score = judge.judge(case.query, case.agent_output, case.metadata)
                scores[judge_name] = float(score)
                logger.debug(f"{judge_name} score for {case_id}: {score:.4f}")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                errors[judge_name] = error_msg
                logger.error(f"{judge_name} judge failed on {case_id}: {error_msg}")

        return EvaluationResult(
            case_id=case_id,
            query=case.query,
            agent_output=case.agent_output,
            scores=scores,
            errors=errors,
            metadata=case.metadata,
        )

    def evaluate_batch(
        self, cases: List[EvaluationCase], verbose: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate a batch of cases.

        Args:
            cases: List of evaluation cases
            verbose: Whether to log progress

        Returns:
            List of EvaluationResult objects

        Raises:
            ValueError: If cases list is empty
        """
        if not cases:
            raise ValueError("Cannot evaluate empty batch")

        results = []
        for i, case in enumerate(cases):
            if verbose:
                logger.info(
                    f"Evaluating case {i + 1}/{len(cases)}: {case.case_id or case.query[:50]}..."
                )

            result = self.evaluate(case)
            results.append(result)

        if verbose:
            logger.info(
                f"Batch evaluation complete. {len(results)} cases processed, "
                f"{sum(1 for r in results if not r.errors)} successful"
            )

        return results

    def save_results(
        self,
        results: List[EvaluationResult],
        experiment_name: str = "evaluation",
    ) -> Path:
        """Save evaluation results to JSON file.

        Args:
            results: List of evaluation results
            experiment_name: Name for this evaluation run

        Returns:
            Path to saved results file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to dict format
        results_dict = [
            format_result(r.query, r.agent_output, r.scores, r.metadata)
            for r in results
        ]

        filepath = save_results(results_dict, self.output_dir, experiment_name)
        logger.info(f"Results saved to {filepath}")

        return filepath

    def print_summary(self, results: List[EvaluationResult]) -> None:
        """Print summary statistics from evaluation results.

        Args:
            results: List of evaluation results
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nCases evaluated: {len(results)}")
        successful = sum(1 for r in results if not r.errors)
        print(f"Successful: {successful}/{len(results)}")

        # Print score statistics
        if results and results[0].scores:
            results_dict = [
                format_result(r.query, r.agent_output, r.scores, r.metadata)
                for r in results
            ]
            stats = compute_statistics(results_dict)

            print("\nScore Statistics (across all cases):")
            print("-" * 60)

            for metric in ["asr", "oss", "ss"]:
                if metric in stats:
                    s = stats[metric]
                    print(f"\n{metric.upper()}:")
                    print(f"  Mean:  {s['mean']:.4f}")
                    print(f"  Std:   {s.get('std', 'N/A')}")
                    print(f"  Range: {s['min']:.4f} - {s['max']:.4f}")

        # Print error summary
        if any(r.errors for r in results):
            print("\n" + "-" * 60)
            print("Judge Failures:")
            for judge_name in self.judges:
                failed = sum(1 for r in results if judge_name in r.errors)
                if failed > 0:
                    print(f"  {judge_name}: {failed} failures")

        print("\n" + "=" * 60 + "\n")

    def get_judge(self, judge_name: str) -> Optional[BaseJudge]:
        """Get a specific judge instance.

        Args:
            judge_name: One of "asr", "oss", "ss"

        Returns:
            Judge instance or None if not found
        """
        return self.judges.get(judge_name.lower())
