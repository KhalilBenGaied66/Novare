"""Ingestion end to end on the mini corpus: index files, idempotence, degradation, warnings."""

import json
from email.message import EmailMessage

import numpy as np
import pytest
from fpdf import FPDF

from app.core.config import reset_settings
from app.core.types import Chunk
from app.ingestion.ingest import ingest_docs
from app.retrieval import embeddings, vector_store
from app.retrieval.embeddings import EmbeddingUnavailable
from app.retrieval.hybrid import get_retriever
from app.retrieval.vector_store import VectorStoreUnavailable


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_pdf(path, pages):
    pdf = FPDF()
    pdf.set_font("helvetica", size=12)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


def test_ingest_writes_chunks_vectors_and_manifest(mini_corpus, env):
    result = ingest_docs()

    assert (result.docs, result.chunks) == (4, 7)
    assert result.retrieval_mode == "hybrid"
    assert result.embedding_backend == "hash"
    assert result.embedding_model == "hash-256"
    assert result.vector_store == "numpy"
    assert result.warnings == []
    assert result.duration_ms >= 0

    chunks = [Chunk.from_dict(item) for item in read_json(env.index_dir / "chunks.json")]
    assert [chunk.chunk_id for chunk in chunks] == [
        "contrat_C-12.md#p1-1",
        "contrat_C-34.md#p1-1",
        "grille_tarifs.md#p1-1",
        "grille_tarifs.md#p1-2",
        "procedure_sav.md#p1-1",
        "procedure_sav.md#p1-2",
        "procedure_sav.md#p1-3",
    ]
    assert [chunk.client_id for chunk in chunks] == ["C-12", "C-34", None, None, None, None, None]
    delays = chunks[5]
    assert (delays.title, delays.doc_type, delays.section) == (
        "Procédure SAV",
        "procedure",
        "Délais d'intervention",
    )
    assert "| Confort | 4 h ouvrées | 8 h ouvrées | 48 h ouvrées |" in delays.text
    assert all("titre:" not in chunk.text for chunk in chunks)  # front matter is not indexed

    manifest = read_json(env.index_dir / "manifest.json")
    created_at = manifest.pop("created_at")
    assert created_at.endswith("+00:00")  # UTC timestamp
    assert manifest == {
        "docs": 4,
        "chunks": 7,
        "retrieval_mode": "hybrid",
        "embedding_backend": "hash",
        "embedding_model": "hash-256",
        "dim": 256,
        "vector_store": "numpy",
    }

    assert np.load(env.index_dir / "vectors.npy").shape == (7, 256)
    assert read_json(env.index_dir / "vector_ids.json")["ids"] == [c.chunk_id for c in chunks]


def test_ingest_is_idempotent(mini_corpus, env):
    first = ingest_docs()
    chunks_before = (env.index_dir / "chunks.json").read_bytes()
    vectors_before = np.load(env.index_dir / "vectors.npy")

    second = ingest_docs()
    assert (second.docs, second.chunks) == (first.docs, first.chunks)
    assert (env.index_dir / "chunks.json").read_bytes() == chunks_before
    assert np.array_equal(np.load(env.index_dir / "vectors.npy"), vectors_before)


def test_ingest_is_a_full_rebuild(mini_corpus, env):
    ingest_docs()
    (mini_corpus / "contrat_C-34.md").unlink()
    result = ingest_docs()
    assert (result.docs, result.chunks) == (3, 6)
    docs = {item["doc"] for item in read_json(env.index_dir / "chunks.json")}
    assert "contrat_C-34.md" not in docs
    assert len(read_json(env.index_dir / "vector_ids.json")["ids"]) == 6


def test_bm25_setting_builds_no_vector_index(mini_corpus, env, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    reset_settings()
    result = ingest_docs()
    assert result.retrieval_mode == "bm25"
    assert (result.embedding_backend, result.embedding_model, result.vector_store) == (
        None,
        None,
        None,
    )
    assert result.warnings == []
    assert not (mini_corpus.parent / "index" / "vectors.npy").exists()
    manifest = read_json(mini_corpus.parent / "index" / "manifest.json")
    assert (manifest["retrieval_mode"], manifest["dim"]) == ("bm25", None)


class FailingEmbedder:
    name = "hash-256"
    dim = 256

    def embed(self, texts):
        raise EmbeddingUnavailable("ConnectionError")


def test_ingest_degrades_to_bm25_when_the_embedder_is_unavailable(mini_corpus, env, monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FailingEmbedder())
    result = ingest_docs()

    assert (result.docs, result.chunks) == (4, 7)
    assert result.retrieval_mode == "bm25"
    assert result.embedding_model is None
    assert result.warnings == [
        "Modèle d'embeddings indisponible : index construit en mode lexical (BM25) seul."
    ]
    assert read_json(env.index_dir / "manifest.json")["retrieval_mode"] == "bm25"
    assert not (env.index_dir / "vectors.npy").exists()

    retriever = get_retriever()
    assert retriever.mode == "bm25"
    assert retriever.search("majoration week-end")[0].chunk.doc == "grille_tarifs.md"


def test_ingest_degrades_to_bm25_when_the_vector_store_is_unavailable(
    mini_corpus, env, monkeypatch
):
    class BrokenStore:
        name = "qdrant"

        def replace(self, chunks, vectors):
            raise VectorStoreUnavailable("ResponseHandlingException")

    monkeypatch.setattr(vector_store, "get_vector_store", lambda: BrokenStore())
    result = ingest_docs()
    assert result.retrieval_mode == "bm25"
    assert result.vector_store is None
    assert result.warnings == [
        "Base vectorielle indisponible : index construit en mode lexical (BM25) seul."
    ]
    assert get_retriever().mode == "bm25"


def test_degraded_ingest_after_a_hybrid_one_leaves_a_bm25_index(mini_corpus, env, monkeypatch):
    assert ingest_docs().retrieval_mode == "hybrid"
    assert get_retriever().mode == "hybrid"
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FailingEmbedder())
    assert ingest_docs().retrieval_mode == "bm25"
    # The vectors of the first run are still on disk but the manifest rules them out.
    assert get_retriever().mode == "bm25"


def test_broken_files_are_skipped_with_a_warning(mini_corpus, env):
    (mini_corpus / "rapport_casse.pdf").write_bytes(b"%PDF-1.4 fichier tronque")
    (mini_corpus / "export_latin1.txt").write_bytes("Procédure dégradée".encode("latin-1"))
    result = ingest_docs()

    assert (result.docs, result.chunks) == (4, 7)  # the four valid documents are indexed
    assert result.warnings == [
        "export_latin1.txt : fichier illisible, document ignoré.",
        "rapport_casse.pdf : fichier illisible, document ignoré.",
    ]
    docs = {item["doc"] for item in read_json(env.index_dir / "chunks.json")}
    assert docs == {"contrat_C-12.md", "contrat_C-34.md", "grille_tarifs.md", "procedure_sav.md"}


def test_unsupported_and_hidden_files(mini_corpus, env):
    (mini_corpus / "tableau.xlsx").write_bytes(b"PK")
    (mini_corpus / ".gitkeep").write_text("", encoding="utf-8")
    (mini_corpus / "archives").mkdir()
    (mini_corpus / "archives" / "ancien.md").write_text("# Ancien\nTexte.\n", encoding="utf-8")
    result = ingest_docs()
    assert result.docs == 4  # sub-directories are not visited
    assert result.warnings == ["tableau.xlsx : format non pris en charge, fichier ignoré."]


def test_pdf_and_eml_are_indexed_with_manifest_metadata(mini_corpus, env):
    write_pdf(
        mini_corpus / "guide_planification.pdf",
        ["Les créneaux sont proposés sous 48 h ouvrées.", "Annulation sans frais 24 h avant."],
    )
    message = EmailMessage()
    message["Subject"] = "Renfort de l'astreinte hivernale"
    message["From"] = "direction@novare.example"
    message.set_content("L'astreinte est renforcée du 1er décembre au 28 février.\n")
    (mini_corpus / "note_astreinte.eml").write_bytes(message.as_bytes())
    (mini_corpus / "manifest.json").write_text(
        json.dumps(
            {"guide_planification.pdf": {"titre": "Guide de planification", "type": "guide"}}
        ),
        encoding="utf-8",
    )

    result = ingest_docs()
    assert (result.docs, result.warnings) == (6, [])  # manifest.json is not a document
    chunks = {item["chunk_id"]: item for item in read_json(env.index_dir / "chunks.json")}
    assert chunks["guide_planification.pdf#p2-1"]["text"] == "Annulation sans frais 24 h avant."
    assert chunks["guide_planification.pdf#p2-1"]["page"] == 2
    assert chunks["guide_planification.pdf#p1-1"]["title"] == "Guide de planification"
    assert chunks["guide_planification.pdf#p1-1"]["doc_type"] == "guide"
    mail = chunks["note_astreinte.eml#p1-1"]
    assert (mail["title"], mail["doc_type"]) == ("Renfort de l'astreinte hivernale", "email")
    assert mail["text"].startswith("Objet : Renfort de l'astreinte hivernale\n\nL'astreinte")
    assert "novare.example" not in mail["text"]


def test_manifest_can_restrict_a_document_to_a_client(mini_corpus, env):
    (mini_corpus / "manifest.json").write_text(
        json.dumps({"grille_tarifs.md": {"client_id": "C-27"}}), encoding="utf-8"
    )
    ingest_docs()
    owners = {item["doc"]: item["client_id"] for item in read_json(env.index_dir / "chunks.json")}
    assert owners["grille_tarifs.md"] == "C-27"
    assert get_retriever().search("majoration week-end", client_id=None) == []
    assert get_retriever().search("majoration week-end", client_id="C-27")[0].chunk.doc == (
        "grille_tarifs.md"
    )


def test_manifest_entry_without_file_gives_a_warning(mini_corpus):
    (mini_corpus / "manifest.json").write_text(
        json.dumps({"contrat_c-99.pdf": {"client_id": "C-99"}}), encoding="utf-8"
    )
    result = ingest_docs()
    assert result.warnings == [
        "manifest.json : aucun fichier ne correspond à l'entrée « contrat_c-99.pdf »."
    ]


def test_unreadable_manifest_stops_the_ingestion(mini_corpus, env):
    (mini_corpus / "manifest.json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_docs()
    assert not (env.index_dir / "chunks.json").exists()  # nothing was published


def test_pdf_without_text_layer_is_reported(mini_corpus):
    write_pdf(mini_corpus / "scan.pdf", [""])
    result = ingest_docs()
    assert result.docs == 4
    assert result.warnings == ["scan.pdf : aucun texte exploitable, document ignoré."]


def test_empty_or_missing_directory_gives_an_empty_index(env, tmp_path):
    for docs_dir in (env.docs_dir, tmp_path / "absent"):
        result = ingest_docs(docs_dir)
        assert (result.docs, result.chunks, result.retrieval_mode) == (0, 0, "hybrid")
        assert result.warnings == ["Aucun document indexable n'a été trouvé : l'index est vide."]
        assert read_json(env.index_dir / "chunks.json") == []
        assert get_retriever().search("majoration week-end") == []


def test_docs_dir_argument_overrides_settings(env, tmp_path):
    other = tmp_path / "autres_docs"
    other.mkdir()
    (other / "note.md").write_text("# Note\n\nAstreinte renforcée cet hiver.\n", encoding="utf-8")
    result = ingest_docs(other)
    assert (result.docs, result.chunks) == (1, 1)
    assert read_json(env.index_dir / "chunks.json")[0]["chunk_id"] == "note.md#p1-1"


def test_ingest_resets_the_cached_retriever(mini_corpus):
    ingest_docs()
    before = get_retriever()
    assert get_retriever() is before
    (mini_corpus / "note.md").write_text("# Note\n\nAstreinte renforcée.\n", encoding="utf-8")
    ingest_docs()
    after = get_retriever()
    assert after is not before
    assert len(after.chunks) == len(before.chunks) + 1
