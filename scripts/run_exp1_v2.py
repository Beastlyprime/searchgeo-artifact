"""
Experiment 1 (v2): Full vulnerability assessment using in-process search proxy.

7 conditions (baseline + 6 attack conditions) × 44 tasks × N backends = 308+ runs per backend.

Usage:
    python scripts/run_exp1_v2.py --backend gemini                    # full run
    python scripts/run_exp1_v2.py --backend gemini --domain health    # single domain
    python scripts/run_exp1_v2.py --backend gemini --limit 2          # 2 tasks per domain
    python scripts/run_exp1_v2.py --backend gemini --resume           # skip completed
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use project-local HF cache
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

from agents.llm_providers import create_provider
from agents.agent_factory import create_agent
from mock_api.search_proxy import SearchProxy
from mock_api.proxy_client import HTTPProxyClient

MOCK_API_URL = os.environ.get("MOCK_SEARCH_API_URL", "http://localhost:8001")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_MODES = [
    "baseline",
    "1A",
    "1A_fact",
    "1B",
    "2A_institutional",
    "2B",
    "3",
]
RESULTS_DIR = PROJECT_ROOT / "results" / "exp1_v2"


def load_tasks(domain_filter: str | None = None, tasks_dir: Path | None = None) -> list[dict]:
    tasks_dir = tasks_dir or (PROJECT_ROOT / "tasks")
    all_tasks = []
    for f in sorted(tasks_dir.glob("*.yaml")):
        if domain_filter and f.stem != domain_filter:
            continue
        with open(f) as fh:
            all_tasks.extend(yaml.safe_load(fh))
    return all_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 (v2 proxy pipeline)")
    parser.add_argument("--backend", required=True, help="Backend name from config/models.yaml")
    parser.add_argument("--domain", type=str, help="Single domain only")
    parser.add_argument("--limit", type=int, help="Limit tasks per domain")
    parser.add_argument("--qids", nargs="+", help="Explicit task IDs to run (overrides --limit)")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, help="Attack modes")
    parser.add_argument("--arch", default="tool_calling", help="Agent architecture")
    parser.add_argument("--resume", action="store_true", help="Skip completed results")
    parser.add_argument("--attack-content-dir", type=str, help="Override attack_content directory (default: data/attack_content)")
    parser.add_argument("--tasks-dir", type=str, help="Override tasks directory (default: tasks/)")
    parser.add_argument("--output-dir", type=str, help="Override RESULTS_DIR (default: results/exp1_v2)")
    args = parser.parse_args()

    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    attack_dir = Path(args.attack_content_dir) if args.attack_content_dir else None
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else None

    # Load config
    with open(PROJECT_ROOT / "config" / "models.yaml") as f:
        models_cfg = yaml.safe_load(f)
    backend_cfg = models_cfg["backends"].get(args.backend)
    if not backend_cfg:
        logger.error(f"Backend '{args.backend}' not found")
        sys.exit(1)

    # Pick proxy implementation based on architecture:
    #   - openclaw: subprocess CLI talks to the FastAPI server (HTTP), so we need
    #     /configure to drive the server-side SearchProxy.
    #   - all others: in-process SearchProxy passed to the agent.
    if args.arch == "openclaw":
        proxy: Any = HTTPProxyClient(MOCK_API_URL)
        if not proxy.health():
            logger.error(
                f"Mock API server is not reachable at {MOCK_API_URL}. "
                "Start it with: make mock-api (port 8001)."
            )
            sys.exit(2)
        provider = None  # OpenClaw drives its own LLM provider via env vars
        agent = create_agent(
            backend_name=args.backend,
            architecture=args.arch,
            llm_provider=None,
            mock_api_url=MOCK_API_URL,
            search_proxy=None,
        )
    else:
        proxy = SearchProxy(PROJECT_ROOT, injection_config="exp1", attack_content_dir=attack_dir)
        provider = create_provider(backend_cfg["provider"], api_base=backend_cfg.get("api_base"))
        agent = create_agent(
            backend_name=args.backend,
            architecture=args.arch,
            llm_provider=provider,
            search_proxy=proxy,
        )

    # Load tasks
    tasks = load_tasks(domain_filter=args.domain, tasks_dir=tasks_dir)
    if args.qids:
        qid_set = set(args.qids)
        tasks = [t for t in tasks if t["query_id"] in qid_set]
        missing = qid_set - {t["query_id"] for t in tasks}
        if missing:
            logger.warning(f"Requested qids not found: {sorted(missing)}")
    elif args.limit:
        from collections import defaultdict
        by_domain: dict[str, list] = defaultdict(list)
        for t in tasks:
            by_domain[t["domain"]].append(t)
        tasks = []
        for domain_tasks in by_domain.values():
            tasks.extend(domain_tasks[:args.limit])

    total = len(tasks) * len(args.modes)
    logger.info(f"Exp 1 v2: {len(tasks)} tasks × {len(args.modes)} modes = {total} runs")
    logger.info(f"Backend: {args.backend} ({backend_cfg['model_id']}), Arch: {args.arch}")

    completed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for task in tasks:
        qid = task["query_id"]
        domain = task["domain"]

        for mode in args.modes:
            output_path = results_dir / args.backend / args.arch / mode / domain / f"{qid}.json"

            if args.resume and output_path.exists():
                try:
                    with open(output_path) as f:
                        existing = json.load(f)
                    if existing.get("output"):
                        skipped += 1
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass

            # Configure proxy for this task/mode
            proxy.configure(
                task_id=qid,
                attack_mode=mode if mode != "baseline" else None,
            )

            try:
                response = agent.run(query=task["query"], scene_id=None)
                result = response.to_dict()
                result.update({
                    "experiment": "exp1_v2",
                    "attack_mode": mode,
                    "domain": domain,
                    "query_id": qid,
                    "query": task["query"],
                    "backend": args.backend,
                    "architecture": args.arch,
                    "task_metadata": {
                        "gold_answer": task.get("gold_answer", ""),
                        "attacker_objective": task.get("attacker_objective", ""),
                        "attacker_target_claim": task.get("attacker_target_claim", ""),
                        "key_facts": task.get("key_facts", []),
                    },
                })

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                completed += 1
                output_len = len(result.get("output", ""))
                n_search = len(result.get("search_calls", []))
                elapsed = time.time() - start_time
                rate = (completed + skipped) / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    f"  [{completed+skipped+errors}/{total}] {qid}/{mode}: "
                    f"{n_search} searches, {output_len} chars "
                    f"({rate:.1f}/min)"
                )

                if not result.get("output"):
                    logger.warning(f"    Empty output for {qid}/{mode}")

            except Exception as e:
                errors += 1
                logger.error(f"  [{completed+skipped+errors}/{total}] {qid}/{mode}: ERROR {e}")

            time.sleep(0.3)

    elapsed = time.time() - start_time
    logger.info(f"\nExp 1 v2 complete in {elapsed/60:.1f} min")
    logger.info(f"  Completed: {completed}, Skipped: {skipped}, Errors: {errors}")
    logger.info(f"  Results: {results_dir / args.backend / args.arch}")


if __name__ == "__main__":
    main()
