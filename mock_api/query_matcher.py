"""
Query Matcher — Semantic embedding similarity matching against cached search queries.

At server startup, loads the manifest of cached queries and encodes them with
a sentence-transformer model (bge-small-en-v1.5). At request time, encodes the
incoming query and finds the best match via cosine similarity.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class MatchResult:
    """Result of matching an agent query against the cached query set."""
    query_id: str
    query_index: int
    query_type: str  # "primary" or "related"
    query_text: str  # the cached query that matched
    similarity: float
    cache_file: str  # relative path within cache dir
    domain: str

    @property
    def is_attack_eligible(self) -> bool:
        """All cached queries are SEO targets and attack-eligible.

        The attacker is modelled as having pre-SEO'd the task's cached query set
        (primary + related). Any query that matches one of those targets returns
        the poisoned ranking; queries outside the set escape SEO coverage.
        """
        return True


class QueryMatcher:
    """
    Embedding-based query matcher for the search proxy.

    Loads cached query texts from manifest.json, encodes them with
    bge-small-en-v1.5, and provides cosine similarity matching.
    """

    def __init__(self, cache_dir: Path, threshold: float = 0.70) -> None:
        self._cache_dir = cache_dir
        self._threshold = threshold
        self._entries: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._model = None

        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load manifest and build embedding index."""
        manifest_path = self._cache_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        self._entries = manifest.get("queries", [])
        if not self._entries:
            logger.warning("No queries in manifest")
            return

        # Load embedding model and encode all cached queries
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        texts = [e["query_text"] for e in self._entries]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        logger.info(
            f"QueryMatcher initialized: {len(self._entries)} cached queries, "
            f"dim={self._embeddings.shape[1]}, threshold={self._threshold}"
        )

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to a normalized embedding vector."""
        return self._model.encode([query], normalize_embeddings=True)

    def match(self, query: str, domain: str | None = None) -> MatchResult | None:
        """
        Find the best matching cached query for an incoming agent query.

        Args:
            query: The agent's search query text
            domain: Optional domain filter (only match within this domain)

        Returns:
            MatchResult if similarity >= threshold, else None
        """
        if not self._entries or self._model is None or self._embeddings is None:
            return None

        query_vec = self._encode_query(query)
        # Dot product on normalized vectors = cosine similarity
        similarities = (self._embeddings @ query_vec.T).flatten()

        # Apply domain filter
        if domain:
            for i, entry in enumerate(self._entries):
                if entry["domain"] != domain:
                    similarities[i] = 0.0

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score < self._threshold:
            logger.info(f"No match for '{query[:60]}' (best score: {best_score:.3f} < {self._threshold})")
            return None

        entry = self._entries[best_idx]
        result = MatchResult(
            query_id=entry["query_id"],
            query_index=entry["query_index"],
            query_type=entry["query_type"],
            query_text=entry["query_text"],
            similarity=best_score,
            cache_file=entry["file_path"],
            domain=entry["domain"],
        )

        logger.info(
            f"Matched '{query[:50]}' → '{result.query_text[:50]}' "
            f"(sim={best_score:.3f}, {result.query_type}, {result.query_id})"
        )
        return result

    def match_for_task(self, query: str, task_id: str) -> MatchResult | None:
        """
        Match query but restrict to entries for a specific task_id.

        Used when the experiment runner has configured a specific task context.
        Returns the best match within that task's query set.
        """
        if not self._entries or self._model is None or self._embeddings is None:
            return None

        query_vec = self._encode_query(query)
        similarities = (self._embeddings @ query_vec.T).flatten()

        # Only consider entries for this task
        for i, entry in enumerate(self._entries):
            if entry["query_id"] != task_id:
                similarities[i] = 0.0

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score < self._threshold:
            logger.info(
                f"No task-scoped match for '{query[:60]}' in {task_id} "
                f"(best score: {best_score:.3f} < {self._threshold})"
            )
            return None

        entry = self._entries[best_idx]
        return MatchResult(
            query_id=entry["query_id"],
            query_index=entry["query_index"],
            query_type=entry["query_type"],
            query_text=entry["query_text"],
            similarity=best_score,
            cache_file=entry["file_path"],
            domain=entry["domain"],
        )

    def best_match_for_task(self, query: str, task_id: str) -> MatchResult | None:
        """Return the highest-similarity entry within task_id, without thresholding.

        Used by SearchProxy to decide between cached-injection and live-fallback
        paths: the caller compares `similarity` to its own threshold rather than
        relying on the matcher's built-in cutoff.

        Returns None only when the task has no cached queries at all.
        """
        if not self._entries or self._model is None or self._embeddings is None:
            return None

        query_vec = self._encode_query(query)
        similarities = (self._embeddings @ query_vec.T).flatten()

        # Mask entries outside the task with a sentinel below any valid cosine
        in_task_indices = [i for i, e in enumerate(self._entries) if e["query_id"] == task_id]
        if not in_task_indices:
            return None
        mask = np.full_like(similarities, -1.0)
        for i in in_task_indices:
            mask[i] = similarities[i]

        best_idx = int(mask.argmax())
        best_score = float(mask[best_idx])

        entry = self._entries[best_idx]
        return MatchResult(
            query_id=entry["query_id"],
            query_index=entry["query_index"],
            query_type=entry["query_type"],
            query_text=entry["query_text"],
            similarity=best_score,
            cache_file=entry["file_path"],
            domain=entry["domain"],
        )

    def get_primary_for_task(self, task_id: str) -> MatchResult | None:
        """Return the primary cached query for a task, ignoring agent query phrasing.

        Used on the first search of each task with an active attack mode, per
        paper §5.2: the attacker has SEO-placed the primary query, so the first
        agent search lands in attacker scope regardless of exact wording.
        """
        for entry in self._entries:
            if entry["query_id"] == task_id and entry["query_type"] == "primary":
                return MatchResult(
                    query_id=entry["query_id"],
                    query_index=entry["query_index"],
                    query_type="primary",
                    query_text=entry["query_text"],
                    similarity=1.0,  # forced match
                    cache_file=entry["file_path"],
                    domain=entry["domain"],
                )
        return None

    @property
    def num_queries(self) -> int:
        return len(self._entries)
