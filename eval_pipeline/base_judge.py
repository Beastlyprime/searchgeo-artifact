"""Base judge class providing abstract interface and LLM integration for evaluation metrics.

This module defines the interface for all judge implementations and handles
low-level LLM communication using a configurable provider abstraction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Optional
from dataclasses import dataclass
import json


class LLMClient(Protocol):
    """Protocol for LLM client implementations (OpenAI, Anthropic, etc.)."""

    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call an LLM with messages and return text response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model identifier (e.g., "openai/gpt-5.4-mini")
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Text response from the model
        """
        ...


@dataclass
class JudgeConfig:
    """Configuration for a judge instance.

    Note: `provider` is informational only — actual routing is performed by the
    LLM client implementation (see scripts/run_eval.py:MultiProviderLLMClient),
    which dispatches based on `model` matched against config/models.yaml.
    """

    model: str
    provider: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    prompt_template_path: Optional[str] = None


class BaseJudge(ABC):
    """Abstract base class for all judge implementations.

    Judges evaluate LLM agent outputs against attack success criteria.
    Each judge measures a different aspect of the attack's effectiveness.
    """

    def __init__(
        self,
        config: JudgeConfig,
        llm_client: LLMClient,
        prompt_template: Optional[str] = None,
    ):
        """Initialize judge with configuration and LLM client.

        Args:
            config: Judge configuration with model and provider settings
            llm_client: Callable LLM client for making API calls
            prompt_template: Optional prompt template string. If not provided,
                uses template from config.prompt_template_path
        """
        self.config = config
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    @abstractmethod
    def judge(self, query: str, agent_output: str, metadata: Dict[str, Any]) -> float:
        """Judge the agent's output and return a score.

        Args:
            query: The original user query/input
            agent_output: The agent's response to evaluate
            metadata: Domain-specific metadata (domain, target_claim, key_facts, etc.)

        Returns:
            A score appropriate to the judge type (0/1 for binary, float [0,1] continuous)
        """
        ...

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_retries: int = 2,
    ) -> str:
        """Call the LLM and return response text.

        Retries on empty/whitespace-only responses, which occur stochastically with
        some providers (notably Gemini and gpt-* via OpenRouter even at temperature=0).
        After max_retries+1 total attempts the last response is returned regardless.

        Args:
            system_prompt: System prompt establishing the judge's role
            user_message: The evaluation task prompt
            max_retries: Number of retries on empty response (default 2 = up to 3 attempts)

        Returns:
            Raw response text from the LLM
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = ""
        for attempt in range(max_retries + 1):
            response = self.llm_client(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) or ""
            if response.strip():
                return response
            if attempt < max_retries:
                import logging
                logging.getLogger(__name__).warning(
                    f"Empty judge LLM response; retrying {attempt + 1}/{max_retries}"
                )

        return response

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences (```json ... ```) from LLM responses."""
        import re
        # Match ```json ... ``` or ``` ... ```
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _extract_score(
        self, response: str, score_type: str = "binary"
    ) -> float:
        """Parse score from LLM response.

        Args:
            response: Raw LLM response text
            score_type: One of "binary" (0 or 1), "float" (0.0-1.0), or "json"

        Returns:
            Parsed score value

        Raises:
            ValueError: If score cannot be extracted from response
        """
        # Strip markdown code fences that Gemini often adds
        response = self._strip_markdown_fences(response)
        response_lower = response.strip().lower()

        if score_type == "binary":
            if "1" in response_lower or "yes" in response_lower:
                return 1.0
            elif "0" in response_lower or "no" in response_lower:
                return 0.0
            else:
                raise ValueError(
                    f"Could not extract binary score from response: {response}"
                )

        elif score_type == "float":
            # Look for JSON score first
            try:
                if "{" in response:
                    json_match = response[response.find("{") : response.rfind("}") + 1]
                    parsed = json.loads(json_match)
                    score = None
                    if "score" in parsed:
                        score = float(parsed["score"])
                    else:
                        # Try first numeric value
                        for v in parsed.values():
                            if isinstance(v, (int, float)):
                                score = float(v)
                                break
                    if score is not None:
                        return max(0.0, min(1.0, score))
            except (json.JSONDecodeError, ValueError, StopIteration):
                pass

            # Extract first float-like number
            import re
            matches = re.findall(r"0\.\d+|\d+\.\d+|\b[0-1]\b", response)
            if matches:
                score = float(matches[0])
                return max(0.0, min(1.0, score))

            raise ValueError(
                f"Could not extract float score from response: {response}"
            )

        elif score_type == "json":
            import re as _re
            # Try multiple extraction strategies
            # Strategy 1: direct parse (response is already fence-stripped)
            try:
                return json.loads(response.strip())
            except (json.JSONDecodeError, ValueError):
                pass
            # Strategy 2: find outermost { ... }
            brace_start = response.find("{")
            brace_end = response.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    return json.loads(response[brace_start:brace_end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass
            # Strategy 3: regex for JSON object
            json_match = _re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, _re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except (json.JSONDecodeError, ValueError):
                    pass
            # Strategy 4: truncated JSON — extract key-value pairs with regex
            if brace_start >= 0:
                text = response[brace_start:]
                extracted = {}
                for m in _re.finditer(r'"(\w+)"\s*:\s*(-?[\d.]+)', text):
                    extracted[m.group(1)] = float(m.group(2))
                if extracted:
                    return extracted
            raise ValueError(
                f"Could not extract JSON score from response: {response[:200]}"
            )

        raise ValueError(f"Unknown score_type: {score_type}")
