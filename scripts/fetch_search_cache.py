"""
Fetch real search results from SerpAPI and cache them locally.

Reads generated_queries.yaml, calls SerpAPI for each query, extracts page content
with trafilatura, and caches everything as static JSON files for reproducibility.

Usage:
    python scripts/fetch_search_cache.py                     # all queries
    python scripts/fetch_search_cache.py --limit 2           # first 2 tasks
    python scripts/fetch_search_cache.py --domain health     # single domain
    python scripts/fetch_search_cache.py --skip-content      # skip page content extraction
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "search_cache"
QUERIES_FILE = CACHE_DIR / "generated_queries.yaml"
MANIFEST_FILE = CACHE_DIR / "manifest.json"


def fetch_serpapi(query: str, api_key: str, num_results: int = 10) -> dict:
    """Call SerpAPI Google Search and return raw response."""
    from serpapi import GoogleSearch

    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
        "hl": "en",
        "gl": "us",
    }
    search = GoogleSearch(params)
    return search.get_dict()


def extract_organic_results(raw: dict) -> list[dict]:
    """Extract organic search results from SerpAPI response."""
    results = []
    for item in raw.get("organic_results", []):
        results.append({
            "position": item.get("position", 0),
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "date": item.get("date", ""),
            "content": "",  # filled by page extraction
        })
    return results


def extract_page_content(url: str, timeout: float = 15.0) -> str:
    """Fetch a URL and extract main text content using trafilatura."""
    import trafilatura

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
        })
        if response.status_code != 200:
            return ""
        html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
        # Truncate very long content
        if len(text) > 8000:
            text = text[:8000] + "\n[... content truncated ...]"
        return text
    except Exception as e:
        logger.debug(f"Failed to extract content from {url}: {e}")
        return ""


def cache_file_path(domain: str, query_id: str, query_index: int) -> Path:
    """Get the cache file path for a query."""
    return CACHE_DIR / domain / f"{query_id}_q{query_index}.json"


def build_manifest(queries_data: dict) -> dict:
    """Build manifest.json from cached files."""
    entries = []
    for qid, data in queries_data.items():
        domain = data["domain"]
        for i, q in enumerate(data["queries"]):
            fpath = cache_file_path(domain, qid, i)
            if fpath.exists():
                entries.append({
                    "query_text": q["text"],
                    "query_id": qid,
                    "query_index": i,
                    "query_type": q["type"],
                    "file_path": str(fpath.relative_to(CACHE_DIR)),
                    "domain": domain,
                })
    return {"queries": entries, "built_at": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache SerpAPI search results")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to process")
    parser.add_argument("--domain", type=str, help="Process only one domain")
    parser.add_argument("--query-id", action="append", help="Process only the given task/query_id; may be repeated")
    parser.add_argument("--query-index", type=int, action="append", help="Process only the given query index; may be repeated")
    parser.add_argument(
        "--query",
        action="append",
        help="Process one exact query as query_id:index, for example product_03:3; may be repeated",
    )
    parser.add_argument("--refresh", action="store_true", help="Overwrite existing cached files for selected queries")
    parser.add_argument("--skip-content", action="store_true", help="Skip page content extraction")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between SerpAPI calls (seconds)")
    parser.add_argument("--num-results", type=int, default=10, help="Number of results per query")
    args = parser.parse_args()

    # Load API key
    from dotenv import load_dotenv
    import os
    load_dotenv(PROJECT_ROOT / "api_keys.env")
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        logger.error("No SERPAPI_KEY or SERPAPI_API_KEY found in environment or api_keys.env")
        sys.exit(1)

    # Load generated queries
    if not QUERIES_FILE.exists():
        logger.error(f"Queries file not found: {QUERIES_FILE}")
        logger.error("Run scripts/generate_queries.py first")
        sys.exit(1)

    with open(QUERIES_FILE) as f:
        queries_data = yaml.safe_load(f)

    # Filter by domain
    if args.domain:
        queries_data = {k: v for k, v in queries_data.items() if v["domain"] == args.domain}

    exact_queries: set[tuple[str, int]] = set()
    if args.query:
        for spec in args.query:
            try:
                qid, index = spec.rsplit(":", 1)
                exact_queries.add((qid, int(index)))
            except ValueError:
                logger.error(f"Invalid --query value {spec!r}; expected query_id:index")
                sys.exit(1)
        args.query_id = list(set(args.query_id or []) | {qid for qid, _ in exact_queries})

    if args.query_id:
        wanted_ids = set(args.query_id)
        queries_data = {k: v for k, v in queries_data.items() if k in wanted_ids}
        missing_ids = wanted_ids - set(queries_data)
        if missing_ids:
            logger.error(f"Unknown query_id(s): {', '.join(sorted(missing_ids))}")
            sys.exit(1)

    wanted_indices = set(args.query_index or [])

    # Limit tasks
    if args.limit:
        limited = dict(list(queries_data.items())[:args.limit])
        queries_data = limited

    total_queries = sum(len(v["queries"]) for v in queries_data.values())
    logger.info(f"Processing {len(queries_data)} tasks, {total_queries} queries total")

    fetched = 0
    skipped = 0

    for qid, data in queries_data.items():
        domain = data["domain"]
        (CACHE_DIR / domain).mkdir(parents=True, exist_ok=True)

        for i, q in enumerate(data["queries"]):
            if exact_queries and (qid, i) not in exact_queries:
                skipped += 1
                logger.info(f"  Skipping {qid}_q{i} (exact query not selected)")
                continue

            if wanted_indices and i not in wanted_indices:
                skipped += 1
                logger.info(f"  Skipping {qid}_q{i} (query index not selected)")
                continue

            fpath = cache_file_path(domain, qid, i)

            if fpath.exists() and not args.refresh:
                skipped += 1
                logger.info(f"  Skipping {qid}_q{i} (cached)")
                continue

            query_text = q["text"]
            logger.info(f"  Fetching {qid}_q{i}: '{query_text[:60]}'")

            try:
                raw = fetch_serpapi(query_text, api_key, num_results=args.num_results)
            except Exception as e:
                logger.error(f"  SerpAPI error for {qid}_q{i}: {e}")
                continue

            results = extract_organic_results(raw)
            logger.info(f"    Got {len(results)} organic results")

            # Extract page content
            if not args.skip_content:
                for j, r in enumerate(results):
                    if r["url"]:
                        logger.info(f"    Extracting content {j+1}/{len(results)}: {r['url'][:60]}...")
                        r["content"] = extract_page_content(r["url"])
                        time.sleep(0.3)  # polite crawling

            # Save cached result
            cache_entry = {
                "query_id": qid,
                "query_index": i,
                "query_text": query_text,
                "query_type": q["type"],
                "domain": domain,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "num_organic_results": len(results),
                "results": results,
            }

            with open(fpath, "w") as f:
                json.dump(cache_entry, f, indent=2, ensure_ascii=False)

            fetched += 1
            logger.info(f"    Cached to {fpath.relative_to(PROJECT_ROOT)}")

            # Rate limit for SerpAPI
            time.sleep(args.delay)

    # Build manifest
    logger.info("Building manifest...")
    # Reload all queries (including previously cached)
    with open(QUERIES_FILE) as f:
        all_queries = yaml.safe_load(f)
    manifest = build_manifest(all_queries)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Fetched: {fetched}, Skipped: {skipped}")
    logger.info(f"Manifest: {MANIFEST_FILE} ({len(manifest['queries'])} entries)")


if __name__ == "__main__":
    main()
