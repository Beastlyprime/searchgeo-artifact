"""
Real LLM Provider implementations for experiment execution.

Supports:
- OpenAI (GPT-4o etc.)
- Anthropic (Claude Sonnet etc.)
- Google (Gemini etc.)
- OpenAI-compatible endpoints (local vLLM, Together AI, etc.)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load env files (first loaded wins — dotenv won't override existing vars)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / "api_keys.env")
load_dotenv(Path.home() / ".openclaw" / ".env")
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """LLM provider using the OpenAI SDK. Works with OpenAI API and any OpenAI-compatible endpoint (vLLM, Together, etc.)."""

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        elif api_base:
            # Use OpenRouter key for openrouter.ai, otherwise OPENAI_API_KEY or dummy
            if "openrouter" in (api_base or ""):
                kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "not-needed")
            else:
                kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "not-needed")
        if api_base:
            kwargs["base_url"] = api_base

        self._client = OpenAI(**kwargs)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        last_error: Exception | None = None
        response = None
        choice = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None)
                if choices:
                    choice = choices[0]
                    break
                last_error = RuntimeError("LLM response contained no choices")
            except Exception as e:
                last_error = e
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
        if response is None or choice is None:
            raise RuntimeError(f"LLM provider returned no usable choice after retries: {last_error}")

        # Extract content — handle reasoning models (e.g. Qwen3 with reasoning parser)
        # where content may be None and the actual text is in the reasoning field
        content = choice.message.content or ""
        if not content and hasattr(choice.message, "reasoning") and choice.message.reasoning:
            content = choice.message.reasoning

        # Extract tool calls
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }


class AnthropicProvider:
    """LLM provider using the Anthropic SDK."""

    def __init__(self, api_key: str | None = None):
        from anthropic import Anthropic

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = Anthropic(**kwargs)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Anthropic separates system prompt from messages and uses different
        # tool result format than OpenAI. Convert OpenAI-style messages:
        #   {role: "assistant", tool_calls: [...]} → assistant with tool_use blocks
        #   {role: "tool", tool_call_id, content} → user with tool_result block
        system_prompt = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg.get("content", "")
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # Convert to Anthropic assistant message with tool_use blocks
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", "tool_0"),
                        "name": func.get("name", ""),
                        "input": args,
                    })
                filtered_messages.append({"role": "assistant", "content": content_blocks})
            elif msg["role"] == "tool":
                # Convert to Anthropic user message with tool_result block
                filtered_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", "tool_0"),
                        "content": msg.get("content", ""),
                    }],
                })
            else:
                filtered_messages.append({"role": msg["role"], "content": msg.get("content", "")})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        # Convert OpenAI tool format to Anthropic format
        if tools:
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = anthropic_tools

        response = self._client.messages.create(**kwargs)

        # Parse response
        content = ""
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }


class GoogleProvider:
    """LLM provider using the google-genai SDK (new unified SDK)."""

    def __init__(self, api_key: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from google.genai import types

        # Extract system instruction and convert messages
        system_instruction = None
        genai_contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg.get("content", "")
            elif msg["role"] == "user":
                genai_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.get("content", ""))]))
            elif msg["role"] == "assistant":
                text = msg.get("content") or ""
                # Skip empty assistant messages (e.g. tool-call-only responses)
                if text:
                    genai_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
            elif msg["role"] == "tool":
                genai_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Tool result: {msg.get('content', '')}")]))

        # Build config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        # Convert tools to Gemini format
        if tools:
            function_declarations = []
            for tool in tools:
                func = tool.get("function", {})
                params = func.get("parameters", {})
                function_declarations.append(types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=params,
                ))
            config.tools = [types.Tool(function_declarations=function_declarations)]

        response = self._client.models.generate_content(
            model=model,
            contents=genai_contents,
            config=config,
        )

        # Parse response
        content = response.text or ""
        tool_calls = None

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_calls.append({
                        "id": f"call_{fc.name}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args) if fc.args else {}),
                        },
                    })

        usage_meta = response.usage_metadata
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": usage_meta.prompt_token_count if usage_meta else 0,
                "output_tokens": usage_meta.candidates_token_count if usage_meta else 0,
            },
        }


def create_provider(provider_name: str, api_key: str | None = None, api_base: str | None = None) -> Any:
    """Factory function to create the right LLM provider.

    Args:
        provider_name: One of "openai", "anthropic", "google", "openai_compatible"
        api_key: Optional API key override
        api_base: Optional base URL (for openai_compatible providers like vLLM)
    """
    if provider_name in ("openai", "openai_compatible"):
        return OpenAIProvider(api_key=api_key, api_base=api_base)
    elif provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    elif provider_name == "google":
        return GoogleProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use openai, anthropic, google, or openai_compatible.")
