"""
HTTPProxyClient — Drop-in replacement for SearchProxy.configure() that drives
the FastAPI server's `/configure` endpoint.

Used by experiment runners when an agent lives in a subprocess (e.g.,
OpenClawAgent's CLI invocation) and therefore cannot share the in-process
SearchProxy. The runner switches to this client when `--arch openclaw` is
selected; downstream code calls the same `configure(...)` method either way.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class HTTPProxyClient:
    """HTTP shim that mirrors SearchProxy's configure() interface."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 5.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def health(self) -> bool:
        """Return True iff the mock API server is up and the proxy is available."""
        try:
            r = httpx.get(f"{self._base}/health", timeout=self._timeout)
            r.raise_for_status()
            return r.json().get("proxy_mode") == "available"
        except (httpx.HTTPError, ValueError):
            return False

    def configure(
        self,
        task_id: str,
        attack_mode: str | None = None,
        injection_variant: str | None = None,
        injection_config: str = "exp1",
    ) -> None:
        """POST /configure to set the server-side proxy state for this task."""
        payload = {
            "task_id": task_id,
            "attack_mode": attack_mode,
            "injection_variant": injection_variant,
            "injection_config": injection_config,
        }
        r = httpx.post(f"{self._base}/configure", json=payload, timeout=self._timeout)
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "ok":
            raise RuntimeError(f"/configure returned non-ok: {body}")
        logger.info(
            f"[HTTPProxyClient] configured server-side: task={task_id}, "
            f"mode={attack_mode}, variant={injection_variant}"
        )
