"""
Agent wrappers for the Web-Attack Research project.

Provides a unified interface for running different LLM agents (Agent Under Test)
with the Mock Search API.

Supported architectures:
  - "openclaw"     : Real OpenClaw agent via CLI (requires `npm install -g openclaw`)
  - "tool_calling" : Python simulation of OpenClaw's ReAct tool-calling loop
"""

from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
from agents.tool_calling_agent import ToolCallingAgent
from agents.openclaw_agent import OpenClawAgent, is_openclaw_available
from agents.agent_factory import (
    check_openclaw_status,
    create_agent,
    create_provider_from_config,
    list_backends,
    MockLLMProvider,
)
from agents.llm_providers import create_provider

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",
    "ToolCallingAgent",
    "OpenClawAgent",
    "MockLLMProvider",
    "check_openclaw_status",
    "create_agent",
    "is_openclaw_available",
    "list_backends",
    "create_provider",
    "create_provider_from_config",
]
