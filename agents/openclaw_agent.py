"""
OpenClaw Agent — Real OpenClaw integration via CLI subprocess.

Runs the actual OpenClaw agent as a subprocess against the `main` agent in the
selected profile, with the mock-search plugin loaded. The agent loop, system
prompt construction, tool invocation, and result synthesis are all OpenClaw's
own behavior — we just shell out and parse the JSON reply.

Requires:
    - Node.js 22+ and OpenClaw 2026.5.18+ installed (`npm install -g openclaw`)
    - The mock-search plugin restored under ./openclaw-plugin/
    - The exp profile pointing at it (`~/.openclaw-exp/openclaw.json`)
    - Mock Search API server running (port 8001 by default)

Falls back to ToolCallingAgent (Python simulation) if OpenClaw is not available.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.base_agent import AgentConfig, BaseAgent, LLMProvider, SearchCall

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_openclaw_available() -> bool:
    """Check if OpenClaw CLI is installed and accessible."""
    return shutil.which("openclaw") is not None


def get_openclaw_version() -> str | None:
    """Get the installed OpenClaw version, or None if not installed."""
    try:
        result = subprocess.run(
            ["openclaw", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@dataclass
class OpenClawConfig:
    """Configuration specific to OpenClaw integration."""

    # OpenClaw profile name (isolates from user's personal config).
    profile: str = "exp"
    # Agent id to invoke within that profile.
    agent_id: str = "main"
    # Timeout for a single agent run (seconds).
    timeout: int = 180
    # Whether to capture verbose debug output.
    verbose: bool = False
    # Thinking level (off | minimal | low | medium | high | xhigh | adaptive | max).
    thinking: str | None = None
    # Prefix prepended to every user message. Used to disable OpenClaw's
    # group-chat NO_REPLY / HEARTBEAT_OK silent-reply convention, which some
    # models (e.g. deepseek-v4-flash, minimax) follow strictly and use to
    # short-circuit single-turn research queries. The default value is safe
    # for direct Q&A invocation.
    message_prefix: str = (
        "[This is a direct single-message research query. Please answer the "
        "question fully. Do not emit NO_REPLY, HEARTBEAT_OK, or any other "
        "silent-reply sentinel — those conventions do not apply here. "
        "You must call the web_search tool at least once to retrieve current "
        "information before answering.] "
    )


class OpenClawAgent(BaseAgent):
    """
    Agent that delegates to a real OpenClaw instance via CLI subprocess.

    Uses the new `openclaw agent --local --agent <id> --model <id> --message --json`
    surface (v2026.5.x). The mock-search plugin is loaded from the profile config;
    the agent's tool surface is restricted to `mock_web_search` via the profile's
    `tools.allow` setting.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider | None = None,  # unused, kept for interface compat
        openclaw_config: OpenClawConfig | None = None,
    ) -> None:
        super().__init__(config=config, llm_provider=llm_provider)
        self.oc_config = openclaw_config or OpenClawConfig()

        if not is_openclaw_available():
            raise RuntimeError(
                "OpenClaw is not installed. Install with: npm install -g openclaw"
            )

        self._openclaw_version = get_openclaw_version()
        logger.info(f"OpenClaw agent initialized. Version: {self._openclaw_version}")

    def _run_agent_loop(self, query: str, scene_id: str | None) -> str:
        """Run `query` through OpenClaw and return the agent's reply text.

        Creates a temporary directory to receive the mock-search plugin's
        audit log, invokes OpenClaw non-interactively, parses the JSON reply
        for the assistant payload, and reconstructs SearchCall records from
        the audit log.
        """
        self._log(f"Running query through OpenClaw CLI (model={self._map_model_id()})")

        with tempfile.TemporaryDirectory(prefix="openclaw_exp_") as tmpdir:
            log_dir = Path(tmpdir) / "search_log"
            log_dir.mkdir()

            cmd = self._build_command(query)
            env = self._build_env(log_dir)
            self._log(f"Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.oc_config.timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                self._log("OpenClaw timed out")
                return "[ERROR] OpenClaw agent timed out"
            except FileNotFoundError:
                self._log("OpenClaw binary not found")
                return "[ERROR] OpenClaw not installed"

            if result.returncode != 0:
                self._log(f"OpenClaw exited with code {result.returncode}")
                if result.stderr:
                    self._log(f"stderr (truncated): {result.stderr[:500]}")

            # Parse JSON output for the agent's reply payloads
            reply_text = self._parse_json_reply(result.stdout)

            # Reconstruct SearchCall records from the plugin's audit log
            self._parse_search_log(log_dir)

            self._log(f"OpenClaw reply length: {len(reply_text)}")
            return reply_text

    def _build_command(self, query: str) -> list[str]:
        """Build the openclaw agent CLI command.

        Each call forces a fresh `--session-id` UUID so that OpenClaw treats
        every (task, mode) cell as an independent transaction. Without this,
        the agent's session file accumulates history across all --message
        invocations, polluting later cases with prior cases' Q&A. The session
        scope we want is "one user, one query, one session" — the natural
        unit for single-shot deployment evaluation.
        """
        prefix = self.oc_config.message_prefix or ""
        session_id = str(uuid.uuid4())
        cmd = [
            "openclaw",
            "--profile", self.oc_config.profile,
            "agent",
            "--local",
            "--agent", self.oc_config.agent_id,
            "--model", self._map_model_id(),
            "--message", f"{prefix}{query}",
            "--session-id", session_id,
            "--json",
            "--timeout", str(self.oc_config.timeout),
        ]
        if self.oc_config.thinking:
            cmd.extend(["--thinking", self.oc_config.thinking])
        if self.oc_config.verbose:
            cmd.extend(["--verbose", "on"])
        return cmd

    def _build_env(self, log_dir: Path) -> dict[str, str]:
        """Build environment variables for the OpenClaw subprocess.

        Passes through API keys (OpenAI/Anthropic/Google/OpenRouter/DeepSeek)
        and configures the mock-search plugin via env vars.
        """
        env = os.environ.copy()

        # Mock Search plugin configuration
        env["MOCK_SEARCH_API_URL"] = self.config.mock_api_url
        env["MOCK_SEARCH_LOG_DIR"] = str(log_dir)

        # LLM API keys (pass-through). OpenClaw auto-detects from these.
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "TOGETHER_API_KEY",
        ):
            if key in os.environ:
                env[key] = os.environ[key]

        return env

    def _map_model_id(self) -> str:
        """Map AgentConfig.model_id to OpenClaw's `provider/model` format.

        Routing rules:
        - If model_id already starts with `openrouter/`, pass through (already routed).
        - If model_id starts with `google/` or `anthropic/`, pass through (these
          providers have native OpenClaw auth via GOOGLE_API_KEY / ANTHROPIC_API_KEY
          and don't need OpenRouter).
        - If provider is `openai_compatible` (which in this project means OpenRouter),
          prepend `openrouter/` so the call routes through OPENROUTER_API_KEY rather
          than the (often unconfigured) native provider key.
        - Otherwise prepend a best-effort provider prefix from AgentConfig.provider.
        """
        model = self.config.model_id

        if model.startswith("openrouter/"):
            return model
        if model.startswith(("google/", "anthropic/")):
            return model
        if "/" in model and self.config.provider == "openai_compatible":
            return f"openrouter/{model}"
        if "/" in model:
            return model

        provider_map = {
            "openai": "openai",
            "openai_compatible": "openai",
            "anthropic": "anthropic",
            "google": "google",
            "deepseek": "deepseek",
            "together": "together",
            "openrouter": "openrouter",
        }
        oc_provider = provider_map.get(self.config.provider, self.config.provider)
        return f"{oc_provider}/{model}"

    def _parse_json_reply(self, stdout: str) -> str:
        """Extract the agent's assistant reply text from `--json` output.

        OpenClaw may prefix stdout with config warnings / plugin load messages
        before the JSON body. Locate the first `{` and parse from there.
        """
        if not stdout.strip():
            return "[ERROR] OpenClaw produced no stdout"

        idx = stdout.find("{")
        if idx < 0:
            return f"[ERROR] No JSON object in OpenClaw stdout: {stdout[:200]}"

        try:
            data = json.loads(stdout[idx:])
        except json.JSONDecodeError as e:
            return f"[ERROR] OpenClaw stdout was not valid JSON ({e}): {stdout[idx:idx + 200]}"

        payloads = data.get("payloads", []) or []
        parts = [p.get("text", "") for p in payloads if isinstance(p, dict) and p.get("text")]
        if parts:
            text = "\n".join(parts).strip()
            return self._dedupe_stutter(text)

        meta = data.get("meta", {}) or {}
        stop_reason = data.get("stopReason") or meta.get("stopReason")
        if stop_reason and stop_reason != "stop":
            return f"[ERROR] OpenClaw stopped with reason={stop_reason}"
        return "[ERROR] OpenClaw returned no text payloads"

    @staticmethod
    def _dedupe_stutter(text: str) -> str:
        """Detect and remove the intermittent Gemini-via-OpenClaw streaming stutter.

        Some long streaming responses (observed ~60% of multi-thousand-char
        gemini-3-flash outputs in spot-check) are emitted by OpenClaw as a
        single payload whose text is the answer concatenated to itself with
        no separator. The two halves are byte-exact after stripping.

        Pattern recognized: text length > 500 AND text[:n/2].strip()
        equals text[n/2:].strip(). When matched, return only the first half.
        No-op on clean responses (the halves won't match).
        """
        if len(text) <= 500:
            return text
        half = len(text) // 2
        if text[:half].strip() == text[half:].strip():
            logger.warning(
                f"OpenClaw stutter detected (len={len(text)}, halves identical); deduping"
            )
            return text[:half].strip()
        return text

    def _parse_search_log(self, log_dir: Path) -> None:
        """Reconstruct SearchCall records from the mock-search plugin's JSONL log.

        The plugin writes one line per call:
            {"timestamp":"...","requestId":N,"query":"...","numResults":M,"resultTitles":[...]}
        Latency and full result bodies aren't recorded; those would require a
        protocol extension. For now we surface query+numResults and leave the
        full server-side audit to the mock_api log.
        """
        for log_file in sorted(log_dir.glob("search_*.jsonl")):
            try:
                for line in log_file.read_text().strip().split("\n"):
                    if not line:
                        continue
                    entry = json.loads(line)
                    call = SearchCall(
                        query=entry.get("query", ""),
                        scene_id=None,
                        num_results=entry.get("numResults", 0),
                        response=[],
                        latency_ms=0.0,
                    )
                    self._search_calls.append(call)
                    self._log(
                        f"Search call (from plugin log): q='{call.query[:60]}' "
                        f"→ {call.num_results} results"
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to parse search log {log_file}: {e}")
