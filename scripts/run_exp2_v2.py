"""
Experiment 2 (v2): Pipeline ablation — poisoning ratio and rank position.

Tests how the number and position of injected attack results affect ASR
across selected multi-source attack modes. 2A source types are split into
single-source modes and should be run separately with rank-position variants.

Usage:
    python scripts/run_exp2_v2.py --backend gemini
    python scripts/run_exp2_v2.py --backend gemini --modes 2B
    python scripts/run_exp2_v2.py --backend gemini --modes 2A_institutional 2B 3 --variants n1_top n2_top n3_top n3_mid n3_bot
    python scripts/run_exp2_v2.py --backend gemini --domain product --limit 2
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

from agents.llm_providers import create_provider
from agents.agent_factory import create_agent
from mock_api.search_proxy import SearchProxy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp2_v2"
VARIANTS = ["n1_top", "n2_top", "n3_top", "n3_mid", "n3_bot"]
MODES = ["2B", "3"]


def load_tasks(domain_filter: str | None = None) -> list[dict]:
    tasks_dir = PROJECT_ROOT / "tasks"
    all_tasks = []
    for f in sorted(tasks_dir.glob("*.yaml")):
        if domain_filter and f.stem != domain_filter:
            continue
        with open(f) as fh:
            all_tasks.extend(yaml.safe_load(fh))
    return all_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 ablation")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--arch", default="tool_calling")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "models.yaml") as f:
        models_cfg = yaml.safe_load(f)
    backend_cfg = models_cfg["backends"][args.backend]

    proxy = SearchProxy(PROJECT_ROOT, injection_config="exp2_ablation")
    provider = create_provider(backend_cfg["provider"], api_base=backend_cfg.get("api_base"))
    agent = create_agent(args.backend, args.arch, provider, search_proxy=proxy)

    tasks = load_tasks(domain_filter=args.domain)
    if args.limit:
        from collections import defaultdict
        by_domain: dict[str, list] = defaultdict(list)
        for t in tasks:
            by_domain[t["domain"]].append(t)
        tasks = []
        for dt in by_domain.values():
            tasks.extend(dt[:args.limit])

    total = len(tasks) * len(args.variants) * len(args.modes)
    logger.info(
        f"Exp 2: {len(tasks)} tasks × {len(args.variants)} variants × {len(args.modes)} modes = {total} runs"
    )

    completed = errors = skipped = 0
    start_time = time.time()

    for task in tasks:
        qid = task["query_id"]
        domain = task["domain"]
        for mode in args.modes:
            for variant in args.variants:
                output_path = (
                    RESULTS_DIR / args.backend / args.arch / mode / variant / domain / f"{qid}.json"
                )

                if args.resume and output_path.exists():
                    try:
                        with open(output_path) as f:
                            if json.load(f).get("output"):
                                skipped += 1
                                continue
                    except (json.JSONDecodeError, KeyError):
                        pass

                proxy.configure(task_id=qid, attack_mode=mode, injection_variant=variant)

                try:
                    response = agent.run(query=task["query"], scene_id=None)
                    result = response.to_dict()
                    result.update({
                        "experiment": "exp2_v2",
                        "attack_mode": mode,
                        "ablation_variant": variant,
                        "domain": domain,
                        "query_id": qid,
                        "query": task["query"],
                        "backend": args.backend,
                        "architecture": args.arch,
                        "task_metadata": {
                            "gold_answer": task.get("gold_answer", ""),
                            "attacker_objective": task.get("attacker_objective", ""),
                            "attacker_target_claim": task.get("attacker_target_claim", ""),
                        },
                    })
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    completed += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"  {qid}/{mode}/{variant}: {e}")

                time.sleep(0.3)

    elapsed = time.time() - start_time
    logger.info(f"Exp 2 complete in {elapsed/60:.1f} min: {completed} done, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
