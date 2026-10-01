"""Vector stores: a NumPy file for single-node use and Qdrant for a shared server.

Both apply the same access rule as the lexical index: a chunk is visible when it
belongs to no client or to the client making the request (`is_visible`).
"""

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.types import Chunk
from app.retrieval.embeddings import l2_normalise

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, models

logger = get_logger(__name__)

_UPSERT_BATCH = 256


class VectorStoreUnavailable(Exception):
    """The store cannot be read or written. The message is an exception class name."""


class VectorStore(Protocol):
    name: str

    def replace(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Replace the whole content of the store with `chunks` and their vectors."""
        ...

    def search(
        self, vector: np.ndarray, top_n: int, client_id: str | None = None
    ) -> list[tuple[str, float]]:
        """Return `(chunk_id, cosine similarity)` of the visible chunks, best first."""
        ...


def is_visible(chunk_client_id: str | None, client_id: str | None) -> bool:
    """Access rule: public chunks for everyone, client chunks for that client only."""
    return chunk_client_id is None or chunk_client_id == client_id


def _check_shapes(chunks: list[Chunk], vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("expected one vector row per chunk")


class NumpyVectorStore:
    """Exact cosine search over a matrix kept in `vectors.npy` + `vector_ids.json`.

    Every query scans all vectors, which is fine up to tens of thousands of chunks.
    """

    name = "numpy"

    def __init__(self, index_dir: Path) -> None:
        self._vectors_path = index_dir / "vectors.npy"
        self._ids_path = index_dir / "vector_ids.json"
        self._loaded: tuple[np.ndarray, list[str], list[str | None]] | None = None

    def replace(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        _check_shapes(chunks, vectors)
        matrix = l2_normalise(vectors)
        ids = {
            "ids": [chunk.chunk_id for chunk in chunks],
            "client_ids": [chunk.client_id for chunk in chunks],
        }
        try:
            self._vectors_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self._vectors_path, matrix)
            self._ids_path.write_text(json.dumps(ids), encoding="utf-8")
        except OSError as exc:
            logger.warning("vector_files_unwritable", extra={"error": type(exc).__name__})
            raise VectorStoreUnavailable(type(exc).__name__) from exc
        self._loaded = (matrix, ids["ids"], ids["client_ids"])

    def search(
        self, vector: np.ndarray, top_n: int, client_id: str | None = None
    ) -> list[tuple[str, float]]:
        matrix, ids, client_ids = self._load()
        if matrix.shape[0] == 0:
            return []
        if matrix.shape[1] != vector.shape[0]:
            # The index was built with another embedding model than the one in use.
            raise VectorStoreUnavailable("DimensionMismatch")
        scores = matrix @ l2_normalise(vector.reshape(1, -1))[0]
        visible = [i for i, owner in enumerate(client_ids) if is_visible(owner, client_id)]
        visible.sort(key=lambda i: (-scores[i], i))
        return [(ids[i], float(scores[i])) for i in visible[:top_n]]

    def _load(self) -> tuple[np.ndarray, list[str], list[str | None]]:
        if self._loaded is None:
            try:
                matrix = np.load(self._vectors_path, allow_pickle=False)
                ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
                chunk_ids, client_ids = ids["ids"], ids["client_ids"]
                # The two files are written one after the other: an interrupted
                # ingestion can leave them out of step.
                if matrix.ndim != 2 or not len(chunk_ids) == len(client_ids) == len(matrix):
                    raise ValueError("vectors.npy and vector_ids.json do not match")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("vector_files_unreadable", extra={"error": type(exc).__name__})
                raise VectorStoreUnavailable(type(exc).__name__) from exc
            self._loaded = (matrix, chunk_ids, client_ids)
        return self._loaded


class QdrantVectorStore:
    """Cosine collection on a Qdrant server; the payload of a point is the chunk itself.

    qdrant-client is imported inside the methods: the import takes seconds and
    deployments on the NumPy store never need it.
    """

    name = "qdrant"

    def __init__(self, client: "QdrantClient", collection: str) -> None:
        self._client = client
        self._collection = collection

    def replace(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models

        _check_shapes(chunks, vectors)
        try:
            # Dropping the collection is the simplest full rebuild; searches arriving
            # during the rebuild fail and fall back to lexical search.
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(self._collection)
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=vectors.shape[1], distance=models.Distance.COSINE
                ),
            )
            for start in range(0, len(chunks), _UPSERT_BATCH):
                batch = range(start, min(start + _UPSERT_BATCH, len(chunks)))
                points = [
                    models.PointStruct(
                        id=point_id(chunks[i].chunk_id),
                        vector=vectors[i].tolist(),
                        payload=chunks[i].to_dict(),
                    )
                    for i in batch
                ]
                self._client.upsert(self._collection, points=points, wait=True)
        except _qdrant_errors() as exc:
            logger.warning("qdrant_replace_failed", extra={"error": type(exc).__name__})
            raise VectorStoreUnavailable(type(exc).__name__) from exc

    def search(
        self, vector: np.ndarray, top_n: int, client_id: str | None = None
    ) -> list[tuple[str, float]]:
        try:
            response = self._client.query_points(
                self._collection,
                query=vector.tolist(),
                query_filter=_visibility_filter(client_id),
                limit=top_n,
                with_payload=["chunk_id"],
            )
        except _qdrant_errors() as exc:
            logger.warning("qdrant_search_failed", extra={"error": type(exc).__name__})
            raise VectorStoreUnavailable(type(exc).__name__) from exc
        return [(point.payload["chunk_id"], float(point.score)) for point in response.points]


def point_id(chunk_id: str) -> str:
    """Qdrant accepts only integers or UUIDs as ids: derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _qdrant_errors() -> tuple[type[Exception], ...]:
    """Exceptions meaning that Qdrant cannot serve the call."""
    from qdrant_client.common.client_exceptions import QdrantException
    from qdrant_client.http.exceptions import ApiException

    # ApiException: server unreachable, timeout or request refused. QdrantException: rate
    # limit. ValueError: raised by the embedded (":memory:") client, for instance when
    # the collection does not exist or the vector has the wrong size.
    return (ApiException, QdrantException, ValueError)


def _visibility_filter(client_id: str | None) -> "models.Filter":
    """`is_visible` expressed as a Qdrant filter on the `client_id` payload field."""
    from qdrant_client import models

    public = models.IsNullCondition(is_null=models.PayloadField(key="client_id"))
    if client_id is None:
        return models.Filter(must=[public])
    own = models.FieldCondition(key="client_id", match=models.MatchValue(value=client_id))
    return models.Filter(should=[public, own])  # `should` alone means "at least one"


def get_vector_store() -> VectorStore:
    """Qdrant when `settings.qdrant_url` is set, else the NumPy file under `index_dir`."""
    settings = get_settings()
    if not settings.qdrant_url:
        return NumpyVectorStore(settings.index_dir)
    from qdrant_client import QdrantClient

    try:
        # The constructor does not wait for the server (its version check runs in a
        # background thread): an unreachable server shows up at the first call.
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    except ValueError as exc:  # malformed QDRANT_URL
        logger.warning("qdrant_url_invalid", extra={"error": type(exc).__name__})
        raise VectorStoreUnavailable(type(exc).__name__) from exc
    return QdrantVectorStore(client, settings.qdrant_collection)
