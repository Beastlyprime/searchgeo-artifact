"""
Base Agent — Abstract interface for Agents Under Test (AUT).

All agent implementations (tool-calling, RL-native) extend this base class,
providing a uniform interface for the experiment runner.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ─── LLM Provider Protocol ───────────────────────────────────────────

class LLMProvider(Protocol):
    """Protocol for LLM API providers. Any provider must implement this."""

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request to the LLM.

        Returns:
            dict with at minimum:
                - "content": str (the text response)
                - "tool_calls": list[dict] | None (if tools were invoked)
                - "usage": dict with "input_tokens", "output_tokens"
        """
        ...


# ─── Data Classes ─────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for an Agent Under Test."""
    backend_name: str
    provider: str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 4096
    mock_api_url: str = "http://localhost:8001"
    system_prompt: str = ""
    defense_prompt: str = ""
    max_search_calls: int = 3
    seed: int = 42

    @property
    def effective_system_prompt(self) -> str:
        """System prompt with optional defense prompt appended."""
        parts = [self.system_prompt]
        if self.defense_prompt:
            parts.append(self.defense_prompt)
        return "\n\n".join(p for p in parts if p)


@dataclass
class SearchCall:
    """Record of a single search API call made by the agent."""
    query: str
    scene_id: str | None
    num_results: int
    response: list[dict[str, Any]]
    latency_ms: float


@dataclass
class AgentResponse:
    """Complete response from an agent run, including reasoning trace."""
    output: str
    reasoning_trace: list[str] = field(default_factory=list)
    search_calls: list[SearchCall] = field(default_factory=list)
    latency_ms: float = 0.0
    model_id: str = ""
    backend: str = ""
    timestamp: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "output": self.output,
            "reasoning_trace": self.reasoning_trace,
            "search_calls": [
                {
                    "query": sc.query,
                    "scene_id": sc.scene_id,
                    "num_results": sc.num_results,
                    "response": sc.response,
                    "latency_ms": sc.latency_ms,
                }
                for sc in self.search_calls
            ],
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "backend": self.backend,
            "timestamp": self.timestamp,
            "token_usage": self.token_usage,
            "error": self.error,
        }

    @property
    def fingerprint(self) -> str:
        """Deterministic hash for reproducibility checks."""
        content = f"{self.output}|{self.model_id}|{len(self.search_calls)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ─── Abstract Base Agent ──────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all Agents Under Test (AUT).

    Subclasses must implement:
        - _run_agent_loop: The core agent loop (ReAct, RL, etc.)
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider | None = None,
        search_proxy: Any | None = None,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self._search_proxy = search_proxy  # In-process SearchProxy (preferred over HTTP)
        self._search_calls: list[SearchCall] = []
        self._reasoning_trace: list[str] = []

    def run(
        self,
        query: str,
        scene_id: str | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute the agent on a query with optional scene binding.

        Args:
            query: The user query to answer.
            scene_id: Optional run identifier retained for compatibility with older traces.
            task_metadata: Additional task context (gold_answer, etc.) — NOT
                          passed to the agent, only stored in results.

        Returns:
            AgentResponse with output, trace, and search call records.
        """
        self._search_calls = []
        self._reasoning_trace = []

        start_time = time.time()
        self._log(f"Starting agent run: backend={self.config.backend_name}, query={query[:80]}...")

        try:
            output = self._run_agent_loop(query=query, scene_id=scene_id)
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            output = ""
            error_msg = str(e)
        else:
            error_msg = None

        elapsed_ms = (time.time() - start_time) * 1000

        response = AgentResponse(
            output=output,
            reasoning_trace=list(self._reasoning_trace),
            search_calls=list(self._search_calls),
            latency_ms=elapsed_ms,
            model_id=self.config.model_id,
            backend=self.config.backend_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=error_msg,
        )

        self._log(f"Agent run complete: {elapsed_ms:.0f}ms, {len(self._search_calls)} search calls")
        return response

    @abstractmethod
    def _run_agent_loop(self, query: str, scene_id: str | None) -> str:
        """
        Core agent loop — subclasses implement their specific architecture.

        Must use self._call_search() for search calls and self._log() for trace.

        Returns:
            The agent's final text output.
        """
        ...

    def _call_search(
        self,
        search_query: str,
        scene_id: str | None = None,
        num_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Execute a search and record the call.

        Uses in-process SearchProxy if available, otherwise falls back to HTTP.
        """
        start = time.time()

        if self._search_proxy is not None:
            # In-process proxy (preferred)
            try:
                results = self._search_proxy.search(search_query, num_results=num_results)
            except Exception as e:
                logger.error(f"SearchProxy call failed: {e}")
                results = []
        else:
            # HTTP fallback for server-backed agents
            import httpx

            params: dict[str, Any] = {"q": search_query, "num_results": num_results}
            if scene_id:
                params["scene_id"] = scene_id
            try:
                resp = httpx.get(f"{self.config.mock_api_url}/search", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
            except Exception as e:
                logger.error(f"Search API call failed: {e}")
                results = []

        elapsed_ms = (time.time() - start) * 1000

        call_record = SearchCall(
            query=search_query,
            scene_id=scene_id,
            num_results=len(results),
            response=results,
            latency_ms=elapsed_ms,
        )
        self._search_calls.append(call_record)

        self._log(f"Search call: q='{search_query[:60]}' → {len(results)} results ({elapsed_ms:.0f}ms)")
        return results

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Call the LLM provider with messages and optional tool definitions.

        Retries on completely-empty responses (no content AND no tool_calls), which
        occur stochastically with some providers even at temperature=0 (notably
        Gemini via OpenRouter on long ReAct contexts). After max_retries+1 total
        attempts the last response is returned regardless.

        Returns the raw provider response dict.
        """
        if self.llm_provider is None:
            raise RuntimeError("No LLM provider configured. Set llm_provider on the agent.")

        response: dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            response = self.llm_provider.chat_completion(
                messages=messages,
                model=self.config.model_id,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools,
            )
            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls")
            if content or tool_calls:
                if attempt > 0:
                    self._log(f"LLM call recovered on retry {attempt}")
                return response
            if attempt < max_retries:
                self._log(
                    f"Empty LLM response (no content, no tool_calls); retrying {attempt + 1}/{max_retries}"
                )
        self._log(
            f"LLM returned empty after {max_retries} retries; proceeding with empty content"
        )
        return response

    def _log(self, message: str) -> None:
        """Add to reasoning trace and log."""
        entry = f"[{self.config.backend_name}] {message}"
        self._reasoning_trace.append(entry)
        logger.info(entry)
