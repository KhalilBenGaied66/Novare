"""Build the search index from the documents directory.

Every run is a full rebuild: the result depends only on the files present, so running
it twice gives the same index. Output, under `settings.index_dir`:
- `chunks.json`: every chunk (text and metadata), the reference for retrieval;
- `manifest.json`: when and how the index was built;
- the vectors, in the configured vector store, when the hybrid mode is available.
"""

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import IngestResponse
from app.core.types import Chunk
from app.ingestion import chunk, loaders
from app.retrieval import embeddings, hybrid, vector_store

logger = get_logger(__name__)

# Two ingestions writing the same files at once would mix their outputs. The lock covers
# one process only: with several API workers, run the ingestion as a separate job.
_LOCK = threading.Lock()


def ingest_docs(docs_dir: Path | None = None) -> IngestResponse:
    """Index every supported file of `docs_dir` (default: `settings.docs_dir`).

    A file that cannot be read is skipped with a warning. A `manifest.json` that cannot
    be read raises `ValueError`: see `loaders.load_manifest`.
    """
    with _LOCK:
        return _ingest(docs_dir or get_settings().docs_dir)


def _ingest(docs_dir: Path) -> IngestResponse:
    started = time.perf_counter()
    settings = get_settings()
    warnings: list[str] = []

    chunks, n_docs = _load_chunks(docs_dir, warnings)
    if not chunks:
        warnings.append("Aucun document indexable n'a été trouvé : l'index est vide.")

    index = {
        "retrieval_mode": "bm25",
        "embedding_backend": None,
        "embedding_model": None,
        "dim": None,
        "vector_store": None,
    }
    if settings.retrieval_mode == "hybrid":
        index.update(_build_vector_index(chunks, warnings))

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "docs": n_docs,
        "chunks": len(chunks),
        **index,
    }
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    _write_json(settings.index_dir / "chunks.json", [item.to_dict() for item in chunks])
    _write_json(settings.index_dir / "manifest.json", manifest)
    hybrid.reset_retriever()

    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "ingest_done",
        extra={
            "docs": n_docs,
            "chunks": len(chunks),
            "retrieval_mode": index["retrieval_mode"],
            "warnings": len(warnings),
            "duration_ms": duration_ms,
        },
    )
    return IngestResponse(
        docs=n_docs,
        chunks=len(chunks),
        retrieval_mode=index["retrieval_mode"],
        embedding_backend=index["embedding_backend"],
        embedding_model=index["embedding_model"],
        vector_store=index["vector_store"],
        duration_ms=duration_ms,
        warnings=warnings,
    )


def _load_chunks(docs_dir: Path, warnings: list[str]) -> tuple[list[Chunk], int]:
    """Load and chunk every document; returns the chunks and the number of documents kept."""
    manifest = loaders.load_manifest(docs_dir)
    files = _source_files(docs_dir, warnings)
    for name in sorted(set(manifest) - {path.name for path in files}):
        # Usually a typo in the manifest: the metadata it carries (possibly a client
        # restriction) is then applied to no file.
        warnings.append(f"manifest.json : aucun fichier ne correspond à l'entrée « {name} ».")

    chunks: list[Chunk] = []
    n_docs = 0
    for path in files:
        try:
            doc = loaders.load_document(path, manifest)
        except (OSError, ValueError) as exc:
            logger.warning(
                "document_skipped", extra={"doc": path.name, "error": type(exc).__name__}
            )
            warnings.append(f"{path.name} : fichier illisible, document ignoré.")
            continue
        doc_chunks = chunk.chunk_document(doc)
        if not doc_chunks:
            # Typically a scanned PDF: there is no text layer and no OCR here.
            warnings.append(f"{path.name} : aucun texte exploitable, document ignoré.")
            continue
        chunks.extend(doc_chunks)
        n_docs += 1
    return chunks, n_docs


def _source_files(docs_dir: Path, warnings: list[str]) -> list[Path]:
    """Supported files of `docs_dir`, sorted by name (sub-directories are not visited)."""
    if not docs_dir.is_dir():
        return []
    files = []
    for path in sorted(docs_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == loaders.MANIFEST_NAME or path.name.startswith("."):
            continue
        if path.suffix.lower() in loaders.SUPPORTED_SUFFIXES:
            files.append(path)
        else:
            warnings.append(f"{path.name} : format non pris en charge, fichier ignoré.")
    return files


def _build_vector_index(chunks: list[Chunk], warnings: list[str]) -> dict:
    """Embed the chunks and store the vectors.

    Returns the manifest fields describing the vector index, or `{}` (plus a warning)
    when the embedding model or the vector store is unavailable: the index is then
    lexical only.
    """
    try:
        embedder = embeddings.get_embedder()
        vectors = embedder.embed([hybrid.search_text(item) for item in chunks])
        store = vector_store.get_vector_store()
        store.replace(chunks, vectors)
    except embeddings.EmbeddingUnavailable as exc:
        logger.warning("ingest_degraded", extra={"part": "embeddings", "cause": str(exc)})
        warnings.append(
            "Modèle d'embeddings indisponible : index construit en mode lexical (BM25) seul."
        )
        return {}
    except vector_store.VectorStoreUnavailable as exc:
        logger.warning("ingest_degraded", extra={"part": "vector_store", "cause": str(exc)})
        warnings.append(
            "Base vectorielle indisponible : index construit en mode lexical (BM25) seul."
        )
        return {}
    return {
        "retrieval_mode": "hybrid",
        "embedding_backend": get_settings().embedding_backend,
        "embedding_model": embedder.name,
        "dim": int(vectors.shape[1]),
        "vector_store": store.name,
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
