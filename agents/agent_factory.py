"""
Agent Factory — Creates configured agent instances from config/models.yaml.

Provides a unified factory interface for creating agents with the correct
backend configuration and LLM provider.

Architecture options:
  - "openclaw"     : Real OpenClaw via CLI subprocess (requires openclaw installed)
  - "tool_calling" : Python ReAct-style tool-calling loop used for the main public results

When architecture="openclaw" is requested but OpenClaw is not installed,
falls back to "tool_calling" with a warning.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from agents.base_agent import AgentConfig, BaseAgent, LLMProvider
from agents.tool_calling_agent import ToolCallingAgent
from agents.llm_providers import create_provider

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_models_config() -> dict[str, Any]:
    """Load models.yaml configuration."""
    config_path = PROJECT_ROOT / "config" / "models.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_backends() -> list[str]:
    """List all configured LLM backend names."""
    config = load_models_config()
    return list(config.get("backends", {}).keys())


def check_openclaw_status() -> dict[str, Any]:
    """
    Check whether real OpenClaw is available and return status info.

    Returns:
        dict with keys:
            - available: bool
            - version: str | None
            - fallback: str (architecture to use if not available)
    """
    from agents.openclaw_agent import is_openclaw_available, get_openclaw_version

    available = is_openclaw_available()
    return {
        "available": available,
        "version": get_openclaw_version() if available else None,
        "fallback": "tool_calling",
    }


def create_provider_from_config(backend_name: str) -> LLMProvider:
    """Create a real LLM provider from models.yaml backend config."""
    config = load_models_config()
    backends = config.get("backends", {})
    if backend_name not in backends:
        raise ValueError(f"Unknown backend '{backend_name}'")
    backend_cfg = backends[backend_name]
    return create_provider(
        provider_name=backend_cfg["provider"],
        api_base=backend_cfg.get("api_base"),
    )


def create_agent(
    backend_name: str,
    architecture: str = "tool_calling",
    llm_provider: LLMProvider | None = None,
    mock_api_url: str = "http://localhost:8001",
    system_prompt: str = "",
    defense_prompt: str = "",
    max_search_calls: int = 3,
    search_proxy: Any | None = None,
) -> BaseAgent:
    """
    Create a configured agent for the specified backend.

    Args:
        backend_name: Name of the LLM backend (e.g., "gpt-5.4-mini", "claude-sonnet").
        architecture: Agent architecture:
            - "openclaw": Real OpenClaw via CLI (falls back to tool_calling if unavailable)
            - "tool_calling": Python ReAct simulation (OpenClaw-style)
        llm_provider: LLM provider implementation. If None, a mock is used.
        mock_api_url: URL of the Mock Search API server.
        system_prompt: Override system prompt (default uses architecture-specific prompt).
        defense_prompt: Defense prompt to append to system prompt.
        max_search_calls: Maximum number of search API calls per run.

    Returns:
        Configured BaseAgent instance.

    Raises:
        ValueError: If backend_name or architecture is unknown.
    """
    config = load_models_config()
    backends = config.get("backends", {})

    if backend_name not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend '{backend_name}'. Available: {available}")

    backend_cfg = backends[backend_name]

    agent_config = AgentConfig(
        backend_name=backend_name,
        provider=backend_cfg["provider"],
        model_id=backend_cfg["model_id"],
        temperature=backend_cfg.get("temperature", 0.0),
        max_tokens=backend_cfg.get("max_tokens", 4096),
        mock_api_url=mock_api_url,
        system_prompt=system_prompt,
        defense_prompt=defense_prompt,
        max_search_calls=max_search_calls,
    )

    actual_architecture = architecture  # Track what we actually use

    if architecture == "openclaw":
        from agents.openclaw_agent import is_openclaw_available, OpenClawAgent

        if is_openclaw_available():
            agent = OpenClawAgent(config=agent_config)
            actual_architecture = "openclaw"
        else:
            logger.warning(
                "OpenClaw is not installed — falling back to tool_calling simulation. "
                "Install with: npm install -g openclaw"
            )
            agent = ToolCallingAgent(config=agent_config, llm_provider=llm_provider, search_proxy=search_proxy)
            actual_architecture = "tool_calling (openclaw fallback)"

    elif architecture == "tool_calling":
        agent = ToolCallingAgent(config=agent_config, llm_provider=llm_provider, search_proxy=search_proxy)

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Use 'openclaw' or 'tool_calling'."
        )

    logger.info(
        f"Created {actual_architecture} agent: backend={backend_name}, "
        f"model={backend_cfg['model_id']}, provider={backend_cfg['provider']}"
    )
    return agent


class MockLLMProvider:
    """
    Mock LLM provider for testing and dry runs.

    Generates deterministic responses that simulate tool-calling behavior
    without making actual API calls.
    """

    def __init__(self, simulate_search: bool = True):
        self._simulate_search = simulate_search
        self._call_count = 0

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Simulate an LLM response."""
        self._call_count += 1

        # Extract the user query from messages
        user_query = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # If tools are provided and this is the first call, simulate a search
        has_tool_results = any(m.get("role") == "tool" for m in messages)

        if tools and self._simulate_search and not has_tool_results:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{self._call_count}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": f'{{"query": "{user_query[:100]}"}}',
                        },
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }

        # Otherwise, produce a final answer
        return {
            "content": (
                f"Based on the search results, here is a comprehensive answer about '{user_query[:80]}'. "
                f"[Mock response from {model} — call #{self._call_count}]"
            ),
            "tool_calls": None,
            "usage": {"input_tokens": 200, "output_tokens": 150},
        }
