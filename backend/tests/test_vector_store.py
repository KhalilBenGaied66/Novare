"""Both vector stores against the same cases; Qdrant runs embedded (`:memory:`)."""

import json
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import reset_settings
from app.core.types import Chunk
from app.ingestion.ingest import ingest_docs
from app.retrieval import vector_store
from app.retrieval.hybrid import get_retriever
from app.retrieval.vector_store import (
    NumpyVectorStore,
    QdrantVectorStore,
    VectorStoreUnavailable,
    get_vector_store,
    is_visible,
    point_id,
)


def make_chunk(chunk_id, client_id=None):
    return Chunk(
        chunk_id=chunk_id, doc=chunk_id.split("#")[0], title="Titre", doc_type="procedure",
        page=1, section="Section", text=f"Texte de {chunk_id}", client_id=client_id,
    )  # fmt: skip


CHUNKS = [
    make_chunk("public_a.md#p1-1"),
    make_chunk("public_b.md#p1-1"),
    make_chunk("contrat_C-12.md#p1-1", "C-12"),
    make_chunk("contrat_C-34.md#p1-1", "C-34"),
]
# Deliberately not unit length: the stores must return cosine similarities anyway.
VECTORS = np.array(
    [
        [2.0, 0.0, 0.0],  # cosine with QUERY: 1.0
        [0.0, 3.0, 0.0],  # 0.0
        [1.0, 1.0, 0.0],  # 1 / sqrt(2) = 0.7071
        [3.0, 0.0, 4.0],  # 3 / 5 = 0.6
    ],
    dtype=np.float32,
)
QUERY = np.array([5.0, 0.0, 0.0], dtype=np.float32)

PUBLIC_A = ("public_a.md#p1-1", 1.0)
PUBLIC_B = ("public_b.md#p1-1", 0.0)
C12 = ("contrat_C-12.md#p1-1", 0.70711)
C34 = ("contrat_C-34.md#p1-1", 0.6)


@pytest.fixture(params=["numpy", "qdrant"])
def store(request, tmp_path):
    if request.param == "numpy":
        return NumpyVectorStore(tmp_path / "index")
    return QdrantVectorStore(QdrantClient(":memory:"), "chunks_test")


def rounded(hits):
    return [(chunk_id, round(score, 5)) for chunk_id, score in hits]


# --- same behaviour in both stores -------------------------------------------------


def test_store_names():
    assert (NumpyVectorStore.name, QdrantVectorStore.name) == ("numpy", "qdrant")


@pytest.mark.parametrize(
    ("client_id", "expected"),
    [
        (None, [PUBLIC_A, PUBLIC_B]),  # anonymous request: public chunks only
        ("C-12", [PUBLIC_A, C12, PUBLIC_B]),
        ("C-34", [PUBLIC_A, C34, PUBLIC_B]),
        ("C-99", [PUBLIC_A, PUBLIC_B]),  # another client sees no contract at all
    ],
)
def test_search_returns_visible_chunks_by_cosine_best_first(store, client_id, expected):
    store.replace(CHUNKS, VECTORS)
    assert rounded(store.search(QUERY, top_n=10, client_id=client_id)) == expected


def test_top_n_limits_the_result(store):
    store.replace(CHUNKS, VECTORS)
    assert rounded(store.search(QUERY, top_n=2, client_id="C-12")) == [PUBLIC_A, C12]
    # The limit applies after the access rule: the two best visible chunks are returned.
    assert rounded(store.search(QUERY, top_n=2)) == [PUBLIC_A, PUBLIC_B]


def test_replace_is_a_full_rebuild(store):
    store.replace(CHUNKS, VECTORS)
    new_chunks = [make_chunk("nouveau.md#p1-1"), make_chunk("nouveau.md#p1-2")]
    new_vectors = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # another dimension
    store.replace(new_chunks, new_vectors)
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_n=10, client_id="C-12")
    assert rounded(hits) == [("nouveau.md#p1-2", 1.0), ("nouveau.md#p1-1", 0.0)]


def test_empty_store_returns_nothing(store):
    store.replace([], np.zeros((0, 3), dtype=np.float32))
    assert store.search(QUERY, top_n=5) == []


def test_replace_rejects_mismatched_shapes(store):
    with pytest.raises(ValueError):
        store.replace(CHUNKS, VECTORS[:2])
    with pytest.raises(ValueError):
        store.replace(CHUNKS[:1], VECTORS[0])  # one-dimensional array


def test_query_of_another_dimension_is_reported_as_unavailable(store):
    store.replace(CHUNKS, VECTORS)
    with pytest.raises(VectorStoreUnavailable):
        store.search(np.array([1.0, 0.0], dtype=np.float32), top_n=5)


@pytest.mark.parametrize(
    ("chunk_client", "request_client", "visible"),
    [
        (None, None, True),
        (None, "C-12", True),
        ("C-12", "C-12", True),
        ("C-12", "C-34", False),
        ("C-12", None, False),
    ],
)
def test_is_visible(chunk_client, request_client, visible):
    assert is_visible(chunk_client, request_client) is visible


# --- NumPy store -------------------------------------------------------------------


def test_numpy_store_persists_to_files(tmp_path):
    NumpyVectorStore(tmp_path).replace(CHUNKS, VECTORS)
    assert np.load(tmp_path / "vectors.npy").shape == (4, 3)
    ids = json.loads((tmp_path / "vector_ids.json").read_text(encoding="utf-8"))
    assert ids == {
        "ids": [chunk.chunk_id for chunk in CHUNKS],
        "client_ids": [None, None, "C-12", "C-34"],
    }
    # A new instance (another process, a restart) reads the same index back.
    reloaded = NumpyVectorStore(tmp_path)
    assert rounded(reloaded.search(QUERY, top_n=10, client_id="C-34")) == [PUBLIC_A, C34, PUBLIC_B]


def test_numpy_store_without_files_is_unavailable(tmp_path):
    with pytest.raises(VectorStoreUnavailable) as error:
        NumpyVectorStore(tmp_path).search(QUERY, top_n=5)
    assert str(error.value) == "FileNotFoundError"


def test_numpy_store_with_files_out_of_step_is_unavailable(tmp_path):
    NumpyVectorStore(tmp_path).replace(CHUNKS, VECTORS)
    (tmp_path / "vector_ids.json").write_text(
        json.dumps({"ids": ["a"], "client_ids": [None]}), encoding="utf-8"
    )
    with pytest.raises(VectorStoreUnavailable):
        NumpyVectorStore(tmp_path).search(QUERY, top_n=5)

    (tmp_path / "vector_ids.json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(VectorStoreUnavailable):
        NumpyVectorStore(tmp_path).search(QUERY, top_n=5)


# --- Qdrant store ------------------------------------------------------------------


def test_qdrant_points_have_uuid5_ids_and_the_chunk_as_payload():
    client = QdrantClient(":memory:")
    QdrantVectorStore(client, "chunks_test").replace(CHUNKS, VECTORS)

    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "contrat_C-12.md#p1-1"))
    assert point_id("contrat_C-12.md#p1-1") == expected_id
    assert point_id("contrat_C-12.md#p1-1") != point_id("contrat_C-34.md#p1-1")

    assert client.count("chunks_test").count == 4
    (point,) = client.retrieve("chunks_test", ids=[expected_id], with_payload=True)
    assert point.payload == CHUNKS[2].to_dict()
    assert Chunk.from_dict(point.payload) == CHUNKS[2]


def test_qdrant_missing_collection_is_unavailable():
    store = QdrantVectorStore(QdrantClient(":memory:"), "jamais_creee")
    with pytest.raises(VectorStoreUnavailable) as error:
        store.search(QUERY, top_n=5)
    assert str(error.value) == "ValueError"


class UnreachableClient:
    """Fails like qdrant-client does when the server cannot be reached."""

    def collection_exists(self, name):
        raise ResponseHandlingException(ConnectionError("connection refused"))

    def query_points(self, name, **kwargs):
        raise ResponseHandlingException(ConnectionError("connection refused"))


def test_qdrant_connection_errors_are_reported_as_unavailable():
    store = QdrantVectorStore(UnreachableClient(), "chunks_test")
    with pytest.raises(VectorStoreUnavailable) as error:
        store.replace(CHUNKS, VECTORS)
    assert str(error.value) == "ResponseHandlingException"
    with pytest.raises(VectorStoreUnavailable):
        store.search(QUERY, top_n=5)


# --- get_vector_store --------------------------------------------------------------


def test_get_vector_store_defaults_to_numpy_files_in_index_dir(env):
    store = get_vector_store()
    assert isinstance(store, NumpyVectorStore)
    store.replace(CHUNKS, VECTORS)
    assert (env.index_dir / "vectors.npy").exists()


def test_get_vector_store_uses_qdrant_when_url_is_set(monkeypatch):
    created = []

    class RecordingClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    # The real constructor starts a background version check against the server.
    monkeypatch.setattr("qdrant_client.QdrantClient", RecordingClient)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "dossierops_test")
    reset_settings()
    store = get_vector_store()
    assert isinstance(store, QdrantVectorStore)
    assert store._collection == "dossierops_test"
    assert created == [{"url": "http://qdrant:6333", "api_key": None}]

    monkeypatch.setenv("QDRANT_API_KEY", "secret-de-test")
    reset_settings()
    get_vector_store()
    assert created[-1] == {"url": "http://qdrant:6333", "api_key": "secret-de-test"}


def test_get_vector_store_with_malformed_url_is_unavailable(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "ftp://qdrant")
    reset_settings()
    with pytest.raises(VectorStoreUnavailable):
        get_vector_store()


# --- Qdrant through the ingestion and the retriever --------------------------------


def test_ingest_and_hybrid_search_through_qdrant(mini_corpus, monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        vector_store, "get_vector_store", lambda: QdrantVectorStore(client, "dossierops_test")
    )
    result = ingest_docs()
    assert (result.retrieval_mode, result.vector_store) == ("hybrid", "qdrant")
    assert client.count("dossierops_test").count == result.chunks

    retriever = get_retriever()
    assert retriever.mode == "hybrid"
    hits = retriever.search("franchise par sinistre", client_id="C-34")
    assert hits[0].chunk.doc == "contrat_C-34.md"
    assert hits[0].dense_rank == 1 and hits[0].bm25_rank == 1
    for client_id in (None, "C-12"):
        docs = {hit.chunk.doc for hit in retriever.search("franchise par sinistre", 10, client_id)}
        assert "contrat_C-34.md" not in docs
