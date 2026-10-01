"""Text embeddings behind one small interface, with a real and an offline backend.

- `FastEmbedEmbedder` runs a multilingual sentence-embedding model locally (ONNX, CPU).
  Its weights are downloaded once into `settings.model_cache_dir`; nothing is sent to a
  third party at query time.
- `HashEmbedder` needs no model: tests and CI use it to exercise the whole vector path.

Both raise `EmbeddingUnavailable` when they cannot produce vectors, which callers treat
as "fall back to lexical search".
"""

import threading
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import get_settings, on_reset
from app.core.logging import get_logger
from app.core.text import tokenize

logger = get_logger(__name__)


class EmbeddingUnavailable(Exception):
    """The embedding model cannot be loaded or run. The message is an exception class name."""


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised float32 row per text, shape (len(texts), dim)."""
        ...


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so that a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an all-zero row stays zero instead of becoming NaN
    return (matrix / norms).astype(np.float32)


class FastEmbedEmbedder:
    """Sentence embeddings computed locally by fastembed (ONNX runtime, CPU).

    The default model reads at most 128 tokens, roughly 450 characters of French: the
    end of a longer chunk does not weigh on its vector. The lexical index covers the
    whole text, which is one reason the two are combined.
    """

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._load().embedding_size

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, model.embedding_size), dtype=np.float32)
        try:
            vectors = np.array(list(model.embed(texts)), dtype=np.float32)
        except Exception as exc:
            logger.warning("embedding_failed", extra={"error": type(exc).__name__})
            raise EmbeddingUnavailable(type(exc).__name__) from exc
        # fastembed returns the mean-pooled vectors of this model unnormalised.
        return l2_normalise(vectors)

    def _load(self):
        """Load the model on first use; the constructor stays instant and offline."""
        # The API serves requests from several threads: the lock keeps two first
        # requests from loading the model twice.
        with self._lock:
            if self._model is None:
                try:
                    # Imported here: fastembed takes seconds to import and is not
                    # needed in bm25 mode or with the hash backend.
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(self.name, cache_dir=str(self._cache_dir))
                except Exception as exc:
                    # Unknown model name, download refused, corrupt cache, missing ONNX
                    # runtime: the caller degrades the same way in every case.
                    logger.warning(
                        "embedding_model_unavailable",
                        extra={"model": self.name, "error": type(exc).__name__},
                    )
                    raise EmbeddingUnavailable(type(exc).__name__) from exc
            return self._model


class HashEmbedder:
    """Deterministic stand-in for a model: a hashed bag of stemmed tokens.

    Two texts are close when they share words; there is no notion of meaning. It is
    meant for tests and offline runs, not for production search quality.
    """

    name = "hash-256"
    dim = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                # crc32 gives the same bucket in every process, unlike the built-in
                # hash(), which is salted per run.
                matrix[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        return l2_normalise(matrix)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the embedder selected by `settings.embedding_backend`.

    The fastembed model is loaded lazily, so `EmbeddingUnavailable` is raised by the
    first `embed()` (or `dim`) call rather than here.
    """
    settings = get_settings()
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)


on_reset(get_embedder.cache_clear)
