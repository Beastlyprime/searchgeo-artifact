"""
Mock Search API Server — HTTP wrapper around the current SearchProxy.

The server exposes the same cached-result, injection, and live-fallback proxy used
by the in-process experiment runners. It intentionally exposes only the current proxy API
used by the public experiments.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

# Use project-local HF cache to avoid permission issues with system cache.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mock_api.search_proxy import SearchProxy

# Logging setup.
LOG_DIR = Path(os.getenv("MOCK_API_LOG_DIR", "logs/mock_api"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "search_cache"

# Single proxy instance shared across requests; per-task state is mutated via /configure.
search_proxy: SearchProxy | None = None


class SearchResponse(BaseModel):
    """Search API response format, compatible with Tavily/SerpAPI."""

    results: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize SearchProxy on startup."""
    global search_proxy

    if CACHE_DIR.exists() and (CACHE_DIR / "manifest.json").exists():
        search_proxy = SearchProxy(PROJECT_ROOT, injection_config="exp1")
        logger.info(f"[Proxy] SearchProxy ready: {search_proxy.num_queries} cached queries")
    else:
        logger.warning("[Proxy] No search cache found; proxy mode disabled")

    yield


app = FastAPI(
    title="Mock Search API",
    description="HTTP wrapper around SearchProxy: cached SEO targets + injection + live fallback",
    version="0.4.0",
    lifespan=lifespan,
)


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query string"),
    num_results: int = Query(10, ge=1, le=10, description="Number of results to return"),
) -> JSONResponse:
    """Search endpoint backed by the current SearchProxy."""
    return _proxy_search(q, num_results)


def _proxy_search(q: str, num_results: int) -> JSONResponse:
    """Delegate to in-process SearchProxy."""
    if search_proxy is None:
        logger.warning("Proxy mode not available (no cache). Returning empty results.")
        return JSONResponse(content={"results": []})

    results = search_proxy.search(q, num_results=num_results)
    response_data: dict[str, Any] = {"results": results[:num_results]}

    last_match = search_proxy.last_match
    if last_match is not None:
        cfg = search_proxy.current_config
        response_data["_match"] = {
            "query_id": last_match.query_id,
            "query_type": last_match.query_type,
            "similarity": round(last_match.similarity, 3),
            "attack_eligible": last_match.is_attack_eligible,
            "attack_mode": cfg["attack_mode"],
            "injected": search_proxy.last_injected,
            "path": search_proxy.last_path,
        }
    else:
        response_data["_match"] = {
            "path": search_proxy.last_path,
            "injected": False,
        }

    matched_id = last_match.query_id if last_match is not None else None
    _log_request(q, matched_id, response_data, matched=last_match is not None, mode="proxy")
    return JSONResponse(content=response_data)


class ConfigureRequest(BaseModel):
    """Request body for /configure endpoint."""

    attack_mode: str | None = None
    task_id: str | None = None
    injection_config: str = "exp1"
    injection_variant: str | None = None


@app.post("/configure")
async def configure(req: ConfigureRequest) -> dict[str, Any]:
    """Configure the proxy for a specific experiment task."""
    if search_proxy is None:
        return {"status": "error", "message": "Proxy not initialized (no cache)"}

    if req.task_id is None:
        return {"status": "error", "message": "task_id is required"}

    search_proxy.configure(
        task_id=req.task_id,
        attack_mode=req.attack_mode,
        injection_variant=req.injection_variant,
        injection_config=req.injection_config,
    )

    logger.info(
        f"[Configure] task_id={req.task_id}, attack_mode={req.attack_mode}, "
        f"injection_config={req.injection_config}, variant={req.injection_variant}"
    )
    return {
        "status": "ok",
        "attack_mode": req.attack_mode,
        "task_id": req.task_id,
        "injection_config": req.injection_config,
    }


@app.get("/proxy/status")
async def proxy_status() -> dict[str, Any]:
    """Check proxy mode status."""
    if search_proxy is None:
        return {
            "proxy_available": False,
            "cached_queries": 0,
            "current_config": {},
        }
    return {
        "proxy_available": True,
        "cached_queries": search_proxy.num_queries,
        "current_config": search_proxy.current_config,
        "search_count_since_configure": search_proxy.search_count,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "proxy_mode": "available" if search_proxy else "unavailable",
    }


def _log_request(
    query: str,
    query_id: str | None,
    response: dict[str, Any],
    matched: bool,
    mode: str = "proxy",
) -> None:
    """Log request-to-response mapping for experiment audit."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "query": query,
        "query_id": query_id,
        "matched": matched,
        "num_results": len(response.get("results", [])),
    }
    logger.info(json.dumps(log_entry))

    log_file = LOG_DIR / f"requests_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            full_entry = {**log_entry, "response_summary": {"num_results": len(response.get("results", []))}}
            f.write(json.dumps(full_entry) + "\n")
    except OSError as e:
        logger.warning(f"Failed to write request log to {log_file}: {e}")
