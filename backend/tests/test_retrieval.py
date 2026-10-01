"""Retriever: RRF fusion, rank and score fields, access rule, degradation, index lifecycle."""

import logging

import numpy as np
import pytest

from app.core.config import reset_settings
from app.core.types import Chunk
from app.ingestion.ingest import ingest_docs
from app.retrieval import embeddings, hybrid
from app.retrieval.embeddings import EmbeddingUnavailable
from app.retrieval.hybrid import (
    RRF_K,
    IndexNotReady,
    Retriever,
    get_retriever,
    index_status,
    reset_retriever,
    search_text,
)
from app.retrieval.vector_store import VectorStoreUnavailable

P_QUESTION = "Quel est le délai d'intervention pour une priorité P1, P2 ou P3 ?"


def make_chunk(chunk_id, text, client_id=None):
    return Chunk(
        chunk_id=chunk_id, doc=chunk_id.split("#")[0], title="", doc_type="procedure",
        page=1, section="", text=text, client_id=client_id,
    )  # fmt: skip


class FakeStore:
    """A vector store that returns a fixed ranking, whatever the query vector."""

    name = "numpy"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def replace(self, chunks, vectors):
        raise NotImplementedError

    def search(self, vector, top_n, client_id=None):
        self.calls.append((top_n, client_id))
        return self.hits[:top_n]


class FailingEmbedder:
    name = "hash-256"
    dim = 256

    def embed(self, texts):
        raise EmbeddingUnavailable("RuntimeError")


# --- index lifecycle ---------------------------------------------------------------


def test_index_not_ready_before_ingestion(env):
    with pytest.raises(IndexNotReady):
        get_retriever()
    assert index_status() == {"ready": False, "chunks": 0, "mode": "none"}


def test_index_status_after_ingestion(indexed):
    status = index_status()
    created_at = status.pop("created_at")
    assert created_at.endswith("+00:00")
    assert status == {
        "ready": True,
        "chunks": 7,
        "mode": "hybrid",
        "docs": 4,
        "embedding_model": "hash-256",
        "vector_store": "numpy",
    }


def test_retriever_is_cached_until_reset(indexed):
    first = get_retriever()
    assert get_retriever() is first
    reset_retriever()
    assert get_retriever() is not first


def test_reset_settings_also_resets_the_retriever(indexed):
    first = get_retriever()
    reset_settings()
    assert get_retriever() is not first


def test_retriever_sees_the_new_corpus_after_a_new_ingestion(indexed, mini_corpus):
    assert get_retriever().search("télésurveillance des chaufferies") == []
    (mini_corpus / "note_telesurveillance.md").write_text(
        "# Télésurveillance\n\nLa télésurveillance des chaufferies est active 24 h sur 24.\n",
        encoding="utf-8",
    )
    ingest_docs()
    hits = get_retriever().search("télésurveillance des chaufferies")
    assert hits[0].chunk.doc == "note_telesurveillance.md"


@pytest.mark.parametrize("content", ["{pas du json", '{"un": "objet"}', '[{"chunk_id": "x"}]'])
def test_unreadable_index_is_not_ready(indexed, env, content):
    (env.index_dir / "chunks.json").write_text(content, encoding="utf-8")
    reset_retriever()
    with pytest.raises(IndexNotReady):
        get_retriever()
    assert index_status()["ready"] is False


def test_missing_manifest_is_not_ready(indexed, env):
    (env.index_dir / "manifest.json").unlink()
    reset_retriever()
    with pytest.raises(IndexNotReady):
        get_retriever()


# --- relevance on the mini corpus --------------------------------------------------


def test_priority_question_retrieves_the_sav_procedure_first_in_hybrid_mode(indexed):
    retriever = get_retriever()
    assert retriever.mode == "hybrid"
    hits = retriever.search(P_QUESTION)
    assert hits[0].chunk.doc == "procedure_sav.md"
    assert hits[0].chunk.section == "Délais d'intervention"
    assert "| Premium | 2 h | 4 h | 24 h |" in hits[0].chunk.text
    assert (hits[0].bm25_rank, hits[0].dense_rank) == (1, 1)


def test_priority_question_retrieves_the_sav_procedure_first_in_bm25_mode(mini_corpus, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    reset_settings()
    ingest_docs()
    retriever = get_retriever()
    assert retriever.mode == "bm25"
    hits = retriever.search(P_QUESTION)
    assert hits[0].chunk.chunk_id == "procedure_sav.md#p1-2"
    assert [hit.chunk.doc for hit in hits[:3]] == ["procedure_sav.md"] * 3


def test_inflected_and_unaccented_wording_still_matches(indexed):
    hits = get_retriever().search("majorations des deplacements le weekend ?")
    assert hits[0].chunk.chunk_id == "grille_tarifs.md#p1-1"


# --- scores and ranks --------------------------------------------------------------


def test_hybrid_results_carry_ranks_scores_and_rrf_score(indexed):
    hits = get_retriever().search(P_QUESTION, top_k=10)
    assert len(hits) >= 3
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    for hit in hits:
        expected = sum(
            1 / (RRF_K + rank) for rank in (hit.bm25_rank, hit.dense_rank) if rank is not None
        )
        assert hit.score == pytest.approx(expected)
        assert (hit.bm25_rank is None) == (hit.bm25_score is None)
        assert (hit.dense_rank is None) == (hit.dense_score is None)
        assert hit.bm25_rank is not None or hit.dense_rank is not None
        if hit.dense_score is not None:
            assert 0 < hit.dense_score <= 1.0001
    assert hits[0].score == pytest.approx(2 / 61)  # first of both lists
    bm25_ranks = [hit.bm25_rank for hit in hits if hit.bm25_rank is not None]
    assert sorted(bm25_ranks) == list(range(1, len(bm25_ranks) + 1))


def test_bm25_mode_scores_are_bm25_scores(mini_corpus, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    reset_settings()
    ingest_docs()
    hits = get_retriever().search(P_QUESTION)
    assert [hit.bm25_rank for hit in hits] == list(range(1, len(hits) + 1))
    for hit in hits:
        assert hit.score == hit.bm25_score > 0
        assert (hit.dense_rank, hit.dense_score) == (None, None)
    assert get_retriever().search("?? !") == []  # no token: nothing to match


def test_query_without_token_returns_nothing_with_the_hash_backend(indexed):
    assert get_retriever().search("?? !") == []


def test_top_k_defaults_to_settings_and_can_be_overridden(indexed, monkeypatch):
    query = "contrat formule priorité déplacement forfait"
    assert len(get_retriever().search(query)) == 4  # settings.top_k
    assert len(get_retriever().search(query, top_k=2)) == 2
    monkeypatch.setenv("TOP_K", "1")
    reset_settings()
    assert len(get_retriever().search(query)) == 1


def test_search_text_joins_title_section_and_text():
    chunk = Chunk("d.md#p1-1", "d.md", "Grille tarifaire", "tarif", 1, "Forfaits", "120 € HT")
    assert search_text(chunk) == "Grille tarifaire\nForfaits\n120 € HT"


# --- Reciprocal Rank Fusion on a controlled case -----------------------------------


@pytest.fixture
def fusion_chunks():
    return [
        make_chunk("a.md#p1-1", "astreinte astreinte hiver"),
        make_chunk("b.md#p1-1", "astreinte tarif tarif tarif tarif"),
        make_chunk("c.md#p1-1", "planning des équipes"),
        make_chunk("d.md#p1-1", "tarif"),
    ]


def test_rrf_orders_by_the_sum_of_reciprocal_ranks(fusion_chunks):
    # Lexical ranking for "astreinte": a (rank 1), b (rank 2). Vector ranking: c, b, a.
    store = FakeStore([("c.md#p1-1", 0.9), ("b.md#p1-1", 0.8), ("a.md#p1-1", 0.7)])
    retriever = Retriever(fusion_chunks, "hybrid", store)
    hits = retriever.search("astreinte", top_k=10)

    by_id = {hit.chunk.chunk_id: hit for hit in hits}
    assert (by_id["a.md#p1-1"].bm25_rank, by_id["a.md#p1-1"].dense_rank) == (1, 3)
    assert (by_id["b.md#p1-1"].bm25_rank, by_id["b.md#p1-1"].dense_rank) == (2, 2)
    assert (by_id["c.md#p1-1"].bm25_rank, by_id["c.md#p1-1"].dense_rank) == (None, 1)
    assert by_id["a.md#p1-1"].score == pytest.approx(1 / 61 + 1 / 63)
    assert by_id["b.md#p1-1"].score == pytest.approx(1 / 62 + 1 / 62)
    assert by_id["c.md#p1-1"].score == pytest.approx(1 / 61)
    assert by_id["c.md#p1-1"].dense_score == 0.9
    assert by_id["c.md#p1-1"].bm25_score is None
    # 1/61 + 1/63 = 0.032266 is just above 2/62 = 0.032258: a, then b, then c.
    assert [hit.chunk.chunk_id for hit in hits] == ["a.md#p1-1", "b.md#p1-1", "c.md#p1-1"]


def test_rrf_tie_keeps_corpus_order_and_top_k_cuts_after_fusion(fusion_chunks):
    # "tarif", average length 3: b (tf 4, length 5) scores idf * 10 / 6.25 = 1.6 idf and
    # d (tf 1, length 1) scores idf * 2.5 / 1.75 = 1.43 idf, so lexical = b then d.
    # Vector: only a. a and b tie at 1/61 and keep corpus order; d (1/62) comes third.
    retriever = Retriever(fusion_chunks, "hybrid", FakeStore([("a.md#p1-1", 0.5)]))
    hits = retriever.search("tarif", top_k=3)
    assert [hit.chunk.chunk_id for hit in hits] == ["a.md#p1-1", "b.md#p1-1", "d.md#p1-1"]
    assert hits[0].score == hits[1].score == pytest.approx(1 / 61)
    assert hits[2].score == pytest.approx(1 / 62)
    assert len(retriever.search("tarif", top_k=2)) == 2


def test_candidate_k_is_requested_from_both_lists(fusion_chunks, monkeypatch):
    monkeypatch.setenv("CANDIDATE_K", "1")
    reset_settings()
    store = FakeStore([("c.md#p1-1", 0.9), ("b.md#p1-1", 0.8)])
    hits = Retriever(fusion_chunks, "hybrid", store).search("astreinte", top_k=10, client_id="C-12")
    assert store.calls == [(1, "C-12")]
    assert [hit.chunk.chunk_id for hit in hits] == ["a.md#p1-1", "c.md#p1-1"]


def test_vector_hits_with_no_similarity_are_not_candidates(fusion_chunks):
    store = FakeStore([("c.md#p1-1", 0.4), ("d.md#p1-1", 0.0), ("b.md#p1-1", -0.2)])
    hits = Retriever(fusion_chunks, "hybrid", store).search("planning", top_k=10)
    assert [hit.chunk.chunk_id for hit in hits] == ["c.md#p1-1"]


def test_hybrid_mode_requires_a_store(fusion_chunks):
    with pytest.raises(ValueError):
        Retriever(fusion_chunks, "hybrid", None)


# --- access rule -------------------------------------------------------------------


def use_mode(monkeypatch, mode):
    monkeypatch.setenv("RETRIEVAL_MODE", mode)
    reset_settings()
    ingest_docs()
    assert get_retriever().mode == mode


@pytest.mark.parametrize("mode", ["hybrid", "bm25"])
def test_client_chunk_is_invisible_to_other_clients_and_anonymous_requests(
    mini_corpus, monkeypatch, mode
):
    use_mode(monkeypatch, mode)
    retriever = get_retriever()
    query = "franchise de 250 € par sinistre, formule Essentiel"

    own = retriever.search(query, top_k=10, client_id="C-34")
    assert own[0].chunk.doc == "contrat_C-34.md"

    for client_id in (None, "C-12", "C-99"):
        hits = retriever.search(query, top_k=10, client_id=client_id)
        assert all(hit.chunk.client_id in (None, client_id) for hit in hits)
        assert "contrat_C-34.md" not in {hit.chunk.doc for hit in hits}


@pytest.mark.parametrize("mode", ["hybrid", "bm25"])
def test_client_sees_public_documents_and_its_own_contract(mini_corpus, monkeypatch, mode):
    use_mode(monkeypatch, mode)
    hits = get_retriever().search("astreinte week-end formule Confort", top_k=10, client_id="C-12")
    docs = [hit.chunk.doc for hit in hits]
    assert docs[0] == "contrat_C-12.md"
    assert "grille_tarifs.md" in docs
    assert "contrat_C-34.md" not in docs


def test_access_rule_is_checked_again_on_vector_hits(fusion_chunks):
    chunks = [*fusion_chunks, make_chunk("contrat_C-34.md#p1-1", "franchise", client_id="C-34")]
    # A store out of step with chunks.json: it returns another client's chunk and an
    # id that the index does not know.
    store = FakeStore([("contrat_C-34.md#p1-1", 0.9), ("inconnu.md#p1-1", 0.8)])
    retriever = Retriever(chunks, "hybrid", store)
    assert retriever.search("franchise", client_id="C-12") == []
    assert retriever.search("franchise", client_id=None) == []
    own = retriever.search("franchise", client_id="C-34")
    assert [hit.chunk.chunk_id for hit in own] == ["contrat_C-34.md#p1-1"]


# --- degradation at query time -----------------------------------------------------


def test_embedding_failure_at_search_time_degrades_to_bm25(indexed, monkeypatch, caplog):
    retriever = get_retriever()
    assert retriever.mode == "hybrid"
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FailingEmbedder())
    with caplog.at_level(logging.WARNING, logger="app.retrieval.hybrid"):
        hits = retriever.search(P_QUESTION)

    assert hits[0].chunk.chunk_id == "procedure_sav.md#p1-2"
    for hit in hits:
        assert hit.score == hit.bm25_score  # BM25 score, as in bm25 mode
        assert hit.dense_rank is None
    record = next(r for r in caplog.records if r.message == "dense_search_degraded")
    assert (record.error, record.cause) == ("EmbeddingUnavailable", "RuntimeError")
    assert P_QUESTION not in caplog.text  # the query itself is never logged


def test_vector_store_failure_at_search_time_degrades_to_bm25(fusion_chunks, caplog):
    class DownStore(FakeStore):
        def search(self, vector, top_n, client_id=None):
            raise VectorStoreUnavailable("ResponseHandlingException")

    retriever = Retriever(fusion_chunks, "hybrid", DownStore([]))
    with caplog.at_level(logging.WARNING, logger="app.retrieval.hybrid"):
        hits = retriever.search("astreinte")
    assert [hit.chunk.chunk_id for hit in hits] == ["a.md#p1-1", "b.md#p1-1"]
    assert all(hit.dense_rank is None and hit.score == hit.bm25_score for hit in hits)
    assert "dense_search_degraded" in caplog.text


def test_missing_vector_files_degrade_to_bm25(indexed, env):
    (env.index_dir / "vectors.npy").unlink()
    reset_retriever()
    retriever = get_retriever()
    assert retriever.mode == "hybrid"  # the manifest still announces vectors
    hits = retriever.search(P_QUESTION)
    assert hits[0].chunk.doc == "procedure_sav.md"
    assert all(hit.dense_rank is None for hit in hits)


# --- index built with other settings -----------------------------------------------


def test_index_built_with_another_embedding_backend_is_used_in_bm25_mode(
    indexed, monkeypatch, caplog
):
    monkeypatch.setenv("EMBEDDING_BACKEND", "fastembed")  # index vectors come from "hash"
    reset_settings()
    with caplog.at_level(logging.WARNING, logger="app.retrieval.hybrid"):
        retriever = get_retriever()
    assert retriever.mode == "bm25"
    assert "index_settings_mismatch" in caplog.text
    assert index_status()["mode"] == "bm25"
    hits = retriever.search(P_QUESTION)  # no model is loaded for this search
    assert hits[0].chunk.doc == "procedure_sav.md"
    assert hits[0].dense_rank is None


def test_bm25_setting_ignores_the_vectors_of_a_hybrid_index(indexed, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    reset_settings()
    assert get_retriever().mode == "bm25"


def test_hybrid_setting_on_a_bm25_index_stays_bm25(mini_corpus, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    reset_settings()
    ingest_docs()
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    reset_settings()
    assert get_retriever().mode == "bm25"


def test_dense_search_uses_the_configured_embedder(indexed, monkeypatch):
    seen = []

    class SpyEmbedder:
        name = "hash-256"
        dim = 256

        def embed(self, texts):
            seen.append(texts)
            return np.zeros((len(texts), 256), dtype=np.float32)

    monkeypatch.setattr(hybrid.embeddings, "get_embedder", lambda: SpyEmbedder())
    hits = get_retriever().search("majoration week-end")
    assert seen == [["majoration week-end"]]
    # A zero query vector matches nothing: only the lexical list contributes.
    assert hits and all(hit.dense_rank is None and hit.bm25_rank is not None for hit in hits)
