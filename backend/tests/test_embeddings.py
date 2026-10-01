"""Embedders: the offline hash backend, and the fastembed adapter against a fake model.

The real model is never loaded here (tests stay offline): `fastembed` is replaced by a
small fake module wherever `FastEmbedEmbedder` needs it.
"""

import sys
import types
import zlib

import numpy as np
import pytest

from app.core.config import reset_settings
from app.retrieval import embeddings
from app.retrieval.embeddings import (
    EmbeddingUnavailable,
    FastEmbedEmbedder,
    HashEmbedder,
    l2_normalise,
)


def test_l2_normalise_gives_unit_rows_and_keeps_zero_rows():
    out = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert out.tolist() == [[pytest.approx(0.6), pytest.approx(0.8)], [0.0, 0.0]]


# --- HashEmbedder ------------------------------------------------------------------


def test_hash_embedder_shape_dtype_and_norm():
    embedder = HashEmbedder()
    vectors = embedder.embed(["Majoration week-end : +35 %", "Forfait diagnostic : 120 € HT"])
    assert (embedder.name, embedder.dim) == ("hash-256", 256)
    assert vectors.shape == (2, 256)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_hash_embedder_is_deterministic_and_uses_crc32_buckets():
    vector = HashEmbedder().embed(["tarif"])[0]
    # One token -> a single bucket set to 1.0, at an index that does not depend on the process.
    assert vector[zlib.crc32(b"tarif") % 256] == 1.0
    assert np.count_nonzero(vector) == 1
    assert np.array_equal(HashEmbedder().embed(["tarif"]), HashEmbedder().embed(["tarif"]))


def test_hash_embedder_works_on_stems_not_surface_forms():
    a, b = HashEmbedder().embed(["résiliation du contrat", "Résilier les contrats"])
    assert float(a @ b) == pytest.approx(1.0, abs=1e-6)


def test_hash_embedder_ranks_overlapping_text_first():
    query, close, far = HashEmbedder().embed(
        [
            "délai d'intervention priorité P1",
            "Les délais d'intervention dépendent de la priorité : P1 sous 4 h.",
            "Forfait diagnostic : 120 € HT, déduit si le devis est accepté.",
        ]
    )
    assert float(query @ close) > 0.5 > float(query @ far)


def test_hash_embedder_empty_inputs():
    embedder = HashEmbedder()
    assert embedder.embed([]).shape == (0, 256)
    # No token at all: an all-zero vector, not NaN.
    assert not embedder.embed(["?? !"]).any()


# --- get_embedder ------------------------------------------------------------------


def test_get_embedder_follows_settings_and_is_cached(monkeypatch):
    first = embeddings.get_embedder()
    assert isinstance(first, HashEmbedder)
    assert embeddings.get_embedder() is first

    monkeypatch.setenv("EMBEDDING_BACKEND", "fastembed")
    monkeypatch.setenv("EMBEDDING_MODEL", "org/some-model")
    reset_settings()  # runs the reset hook: the cached embedder is dropped
    second = embeddings.get_embedder()
    assert isinstance(second, FastEmbedEmbedder)
    assert second.name == "org/some-model"


# --- FastEmbedEmbedder -------------------------------------------------------------


class FakeTextEmbedding:
    """Stands for fastembed.TextEmbedding: unnormalised float64 vectors, like the real one."""

    instances = 0
    embedding_size = 3

    def __init__(self, model_name, cache_dir=None):
        if model_name == "missing/model":
            raise ValueError("Model missing/model is not supported")
        type(self).instances += 1
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, texts):
        for text in texts:
            if text == "boom":
                raise RuntimeError("onnx failure")
            yield np.array([len(text), 0.0, 0.0], dtype=np.float64) * 2.0


@pytest.fixture
def fake_fastembed(monkeypatch):
    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)
    monkeypatch.setattr(FakeTextEmbedding, "instances", 0)
    return FakeTextEmbedding


def test_fastembed_embedder_loads_lazily_and_once(fake_fastembed, tmp_path):
    embedder = FastEmbedEmbedder("org/model", tmp_path / "models")
    assert fake_fastembed.instances == 0  # nothing loaded by the constructor

    assert embedder.dim == 3
    embedder.embed(["a"])
    embedder.embed(["b"])
    assert fake_fastembed.instances == 1
    model = embedder._load()
    assert model.model_name == "org/model"
    assert model.cache_dir == str(tmp_path / "models")


def test_fastembed_embedder_normalises_to_float32(fake_fastembed, tmp_path):
    embedder = FastEmbedEmbedder("org/model", tmp_path)
    vectors = embedder.embed(["abc", "abcdefgh"])
    assert vectors.dtype == np.float32
    assert vectors.tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert embedder.embed([]).shape == (0, 3)


def test_fastembed_load_failure_raises_embedding_unavailable(fake_fastembed, tmp_path):
    embedder = FastEmbedEmbedder("missing/model", tmp_path)
    with pytest.raises(EmbeddingUnavailable) as error:
        embedder.embed(["texte"])
    assert str(error.value) == "ValueError"  # class name only, never the provider message
    with pytest.raises(EmbeddingUnavailable):
        _ = embedder.dim


def test_fastembed_runtime_failure_raises_embedding_unavailable(fake_fastembed, tmp_path):
    embedder = FastEmbedEmbedder("org/model", tmp_path)
    with pytest.raises(EmbeddingUnavailable) as error:
        embedder.embed(["boom"])
    assert str(error.value) == "RuntimeError"


def test_fastembed_missing_package_raises_embedding_unavailable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` fail
    with pytest.raises(EmbeddingUnavailable) as error:
        FastEmbedEmbedder("org/model", tmp_path).embed(["texte"])
    assert str(error.value) == "ModuleNotFoundError"
