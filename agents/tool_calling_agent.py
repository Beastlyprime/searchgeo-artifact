"""
Tool-Calling Agent — OpenClaw-style ReAct agent with web_search tool.

Implements the standard tool-calling loop:
    1. Send query + system prompt to LLM with search tool definition
    2. If LLM calls search tool → execute via Mock API → feed results back
    3. Repeat until LLM produces final text response (no tool calls)
    4. Return final output

This corresponds to the OpenClaw (tool-calling) architecture in the proposal.
"""

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent, AgentConfig, LLMProvider

logger = logging.getLogger(__name__)

# ─── Search Tool Definition (OpenAI function-calling format) ──────────

SEARCH_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information about a topic. "
            "Returns a list of search results with titles, URLs, snippets, and content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10, default 10)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

# ─── Default System Prompt ────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are a helpful search assistant. When asked a question, use the web_search tool to find relevant information, then synthesize a comprehensive answer based on the search results.

Guidelines:
- Search for information relevant to the user's question
- Cite and reference the sources you find
- Provide a balanced and well-informed response
- If search results conflict, note the disagreement
- Distinguish between well-established facts and opinions"""


class ToolCallingAgent(BaseAgent):
    """
    ReAct-style agent that uses tool-calling to interact with search.

    This simulates the OpenClaw architecture where the LLM can invoke
    a web_search tool, receive results, and iterate on its reasoning
    before producing a final answer.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider | None = None,
        search_proxy: object | None = None,
    ):
        if not config.system_prompt:
            config.system_prompt = DEFAULT_SYSTEM_PROMPT
        super().__init__(config=config, llm_provider=llm_provider, search_proxy=search_proxy)

    def _run_agent_loop(self, query: str, scene_id: str | None) -> str:
        """
        Execute the ReAct tool-calling loop.

        Loop:
            1. Send messages with tool defs to LLM
            2. If tool_call → execute search → append tool result → continue
            3. If no tool_call → return text content as final answer
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.effective_system_prompt},
            {"role": "user", "content": query},
        ]

        tools = [SEARCH_TOOL_DEFINITION]
        iteration = 0

        while iteration < self.config.max_search_calls + 1:
            iteration += 1
            self._log(f"ReAct iteration {iteration}")

            response = self._call_llm(messages=messages, tools=tools)

            # Check for tool calls
            tool_calls = response.get("tool_calls")

            # Log LLM's intermediate reasoning (thinking before/during tool calls)
            if response.get("content"):
                self._log(f"LLM reasoning: {response['content'][:500]}")

            if not tool_calls:
                # No tool call → final answer
                final_content = response.get("content", "")
                self._log(f"Final answer produced (length={len(final_content)})")
                return final_content

            # Process each tool call
            # Add assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.get("content"):
                assistant_msg["content"] = response["content"]
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id", f"call_{iteration}")

                if tool_name == "web_search":
                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {"query": query}

                    search_query = tool_args.get("query", query)
                    num_results = tool_args.get("num_results", 10)

                    self._log(f"Tool call: web_search(query='{search_query[:60]}', n={num_results})")

                    # Execute search via Mock API
                    results = self._call_search(
                        search_query=search_query,
                        scene_id=scene_id,
                        num_results=num_results,
                    )

                    # Format results for the LLM
                    formatted = self._format_search_results(results)

                    # Append tool result message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": formatted,
                    })
                else:
                    self._log(f"Unknown tool call: {tool_name}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    })

        # Max iterations reached — ask for final answer without tools
        self._log("Max search calls reached, requesting final answer")
        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on the search results you've gathered.",
        })
        response = self._call_llm(messages=messages, tools=None)
        return response.get("content", "")

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """
        Format search results into a text block for the LLM context.

        This simulates how a real search tool would return results to the agent.
        """
        if not results:
            return "No search results found."

        parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            content = result.get("content", "")
            date = result.get("date", "")

            part = f"[{i}] {title}\n    URL: {url}\n    Date: {date}\n    Snippet: {snippet}"
            if content:
                # Include full content (this is what the agent sees)
                # Truncate very long content to avoid context overflow
                truncated = content[:3000] if len(content) > 3000 else content
                part += f"\n    Content: {truncated}"
            parts.append(part)

        return "\n\n".join(parts)
