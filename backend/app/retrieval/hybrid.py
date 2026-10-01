"""Hybrid retrieval: lexical (BM25) and vector search fused by Reciprocal Rank Fusion.

The index on disk (`chunks.json` + `manifest.json`, written by the ingestion) is the
reference. The vector path is used only when that index was built in hybrid mode with
the embedding model and vector store configured now; otherwise, and whenever the
embedder or the store fails at query time, the search falls back to BM25 alone.
"""

import json
from functools import lru_cache

from app.core.config import get_settings, on_reset
from app.core.logging import get_logger
from app.core.text import tokenize
from app.core.types import Chunk, RetrievedChunk
from app.retrieval import embeddings, vector_store
from app.retrieval.bm25 import BM25Index

logger = get_logger(__name__)

# Usual constant of Reciprocal Rank Fusion: it flattens the gap between the first ranks
# so that no single list dominates the fused order.
RRF_K = 60

# A list of (position of the chunk in `Retriever.chunks`, score), best first.
Hits = list[tuple[int, float]]


class IndexNotReady(Exception):
    """No usable index on disk: the corpus has to be ingested first."""


def search_text(chunk: Chunk) -> str:
    """Text indexed for a chunk, lexically and as a vector.

    The document title and the section give a short chunk its context: a table row
    "Premium | 2 h" only answers a question about delays through its heading.
    """
    return f"{chunk.title}\n{chunk.section}\n{chunk.text}"


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        mode: str = "bm25",
        store: vector_store.VectorStore | None = None,
        manifest: dict | None = None,
    ) -> None:
        if mode == "hybrid" and store is None:
            raise ValueError("hybrid mode needs a vector store")
        self.chunks = chunks
        self.mode = mode  # "hybrid" | "bm25"
        self.manifest = manifest or {}
        self._store = store
        self._bm25 = BM25Index([tokenize(search_text(chunk)) for chunk in chunks])
        self._position = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

    def search(
        self, query: str, top_k: int | None = None, client_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the `top_k` chunks visible to `client_id` that best match `query`.

        `score` is the RRF score in hybrid mode and the BM25 score when only the lexical
        list is available (bm25 mode, or vector search unavailable for this call).
        """
        settings = get_settings()
        top_k = settings.top_k if top_k is None else top_k
        allowed = {
            i
            for i, chunk in enumerate(self.chunks)
            if vector_store.is_visible(chunk.client_id, client_id)
        }
        lexical = self._bm25.search(tokenize(query), settings.candidate_k, allowed)
        dense = None
        if self.mode == "hybrid":
            dense = self._dense_search(query, settings.candidate_k, allowed, client_id)
        if dense is None:
            return [
                RetrievedChunk(self.chunks[i], score=score, bm25_rank=rank, bm25_score=score)
                for rank, (i, score) in enumerate(lexical[:top_k], start=1)
            ]
        return self._fuse(lexical, dense, top_k)

    def _dense_search(
        self, query: str, top_n: int, allowed: set[int], client_id: str | None
    ) -> Hits | None:
        """Vector hits, or None when the vector path cannot answer this call."""
        try:
            vector = embeddings.get_embedder().embed([query])[0]
            found = self._store.search(vector, top_n, client_id)
        except (embeddings.EmbeddingUnavailable, vector_store.VectorStoreUnavailable) as exc:
            logger.warning(
                "dense_search_degraded",
                extra={"error": type(exc).__name__, "cause": str(exc)},
            )
            return None
        hits = []
        for chunk_id, score in found:
            i = self._position.get(chunk_id)
            # The store already filters by client. Checking again against chunks.json
            # keeps a stale or foreign collection from returning a chunk that this index
            # does not know or does not open to this client. A similarity of zero or
            # less means "nothing in common": such a hit is no candidate.
            if i is not None and i in allowed and score > 0:
                hits.append((i, score))
        return hits

    def _fuse(self, lexical: Hits, dense: Hits, top_k: int) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: score = sum over both lists of 1 / (RRF_K + rank)."""
        bm25 = {i: (rank, score) for rank, (i, score) in enumerate(lexical, start=1)}
        vect = {i: (rank, score) for rank, (i, score) in enumerate(dense, start=1)}
        fused: dict[int, float] = {}
        for ranked in (bm25, vect):
            for i, (rank, _) in ranked.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        # Ties (e.g. first of one list against first of the other) keep corpus order.
        best = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        results = []
        for i in best:
            bm25_rank, bm25_score = bm25.get(i, (None, None))
            dense_rank, dense_score = vect.get(i, (None, None))
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[i],
                    score=fused[i],
                    bm25_rank=bm25_rank,
                    dense_rank=dense_rank,
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                )
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Load the index from `settings.index_dir` (cached until `reset_retriever()`).

    Raises `IndexNotReady` when the index files are missing or unreadable; the failure
    is not cached, so the next call sees an index ingested in the meantime.
    """
    index_dir = get_settings().index_dir
    try:
        raw_chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(raw_chunks, list) or not isinstance(manifest, dict):
            raise ValueError("unexpected index file layout")
        chunks = [Chunk.from_dict(item) for item in raw_chunks]
    except (OSError, ValueError, TypeError) as exc:
        # JSON errors are ValueError; a chunk with missing or unknown fields is TypeError.
        raise IndexNotReady(type(exc).__name__) from exc
    store = _usable_vector_store(manifest)
    mode = "hybrid" if store is not None else "bm25"
    logger.info("retriever_loaded", extra={"chunks": len(chunks), "mode": mode})
    return Retriever(chunks, mode, store, manifest)


def _usable_vector_store(manifest: dict) -> vector_store.VectorStore | None:
    """Return the vector store when the index on disk matches the current settings."""
    settings = get_settings()
    if settings.retrieval_mode != "hybrid":
        return None
    if manifest.get("retrieval_mode") != "hybrid":
        # Built without vectors (embedding model unavailable, or RETRIEVAL_MODE=bm25
        # at ingestion time): lexical search only until the corpus is ingested again.
        logger.warning("index_without_vectors", extra={"index": manifest.get("retrieval_mode")})
        return None
    try:
        store = vector_store.get_vector_store()
    except vector_store.VectorStoreUnavailable:
        return None
    current = {
        "embedding_backend": settings.embedding_backend,
        "embedding_model": embeddings.get_embedder().name,
        "vector_store": store.name,
    }
    built_with = {key: manifest.get(key) for key in current}
    if built_with != current:
        # Vectors of another model (or kept in another store) cannot be compared with
        # the query vector: lexical search only until the corpus is ingested again.
        logger.warning("index_settings_mismatch", extra={"index": built_with, "current": current})
        return None
    return store


@on_reset
def reset_retriever() -> None:
    """Forget the cached retriever; the next `get_retriever()` reloads the index."""
    get_retriever.cache_clear()


def index_status() -> dict:
    """Summary of the index for readiness checks. Never raises."""
    try:
        retriever = get_retriever()
    except IndexNotReady:
        return {"ready": False, "chunks": 0, "mode": "none"}
    manifest = retriever.manifest
    return {
        "ready": True,
        "chunks": len(retriever.chunks),
        "mode": retriever.mode,
        "docs": manifest.get("docs"),
        "created_at": manifest.get("created_at"),
        "embedding_model": manifest.get("embedding_model"),
        "vector_store": manifest.get("vector_store"),
    }
