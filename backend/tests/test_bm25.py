"""BM25 against scores computed by hand on a three-document corpus.

Corpus (already tokenised), N = 3, lengths 3 / 2 / 4, average length 3:
    d0 = chaudier pann pann
    d1 = chaudier tarif
    d2 = tarif astreint astreint astreint
With k1 = 1.5 and b = 0.75:
    idf(df=1) = ln(1 + 2.5 / 1.5) = ln(8/3)  = 0.980829
    idf(df=2) = ln(1 + 1.5 / 2.5) = ln(1.6)  = 0.470004
    length term k1 * (1 - b + b * len / avg): d0 -> 1.5, d1 -> 1.125, d2 -> 1.875
    pann     in d0 (tf 2): 0.980829 * 2 * 2.5 / (2 + 1.5)   = 1.401185
    chaudier in d0 (tf 1): 0.470004 * 2.5 / (1 + 1.5)       = 0.470004
    chaudier in d1 (tf 1): 0.470004 * 2.5 / (1 + 1.125)     = 0.552945
    tarif    in d1 (tf 1): same as above                    = 0.552945
    tarif    in d2 (tf 1): 0.470004 * 2.5 / (1 + 1.875)     = 0.408699
    astreint in d2 (tf 3): 0.980829 * 3 * 2.5 / (3 + 1.875) = 1.508968
"""

import pytest

from app.retrieval.bm25 import BM25Index

DOCS = [
    ["chaudier", "pann", "pann"],
    ["chaudier", "tarif"],
    ["tarif", "astreint", "astreint", "astreint"],
]


@pytest.fixture
def index():
    return BM25Index(DOCS)


def approx(value):
    return pytest.approx(value, abs=1e-6)


def test_idf_matches_hand_computation(index):
    assert index.idf("pann") == approx(0.980829)
    assert index.idf("chaudier") == approx(0.470004)
    # A term absent from the corpus has df = 0: ln(1 + 3.5 / 0.5) = ln(8).
    assert index.idf("inconnu") == approx(2.079442)


def test_single_term_scores(index):
    assert index.search(["pann"], top_n=10) == [(0, approx(1.401185))]
    # The shorter document wins for the same term frequency.
    assert index.search(["chaudier"], top_n=10) == [(1, approx(0.552945)), (0, approx(0.470004))]
    assert index.search(["tarif"], top_n=10) == [(1, approx(0.552945)), (2, approx(0.408699))]
    assert index.search(["astreint"], top_n=10) == [(2, approx(1.508968))]


def test_multi_term_scores_add_up_and_rank_best_first(index):
    hits = index.search(["chaudier", "pann"], top_n=10)
    assert hits == [(0, approx(0.470004 + 1.401185)), (1, approx(0.552945))]

    hits = index.search(["tarif", "astreint", "pann"], top_n=10)
    assert [doc for doc, _ in hits] == [2, 0, 1]
    assert hits[0][1] == approx(0.408699 + 1.508968)


def test_repeated_query_term_counts_once(index):
    assert index.search(["pann", "pann", "pann"], top_n=10) == index.search(["pann"], top_n=10)


def test_only_matching_documents_are_returned(index):
    assert index.search(["inconnu"], top_n=10) == []
    assert index.search([], top_n=10) == []
    assert all(score > 0 for _, score in index.search(["chaudier", "tarif"], top_n=10))


def test_top_n_limits_the_result(index):
    assert len(index.search(["chaudier", "tarif"], top_n=10)) == 3
    assert index.search(["chaudier", "tarif"], top_n=1) == [(1, approx(2 * 0.552945))]
    assert index.search(["chaudier"], top_n=0) == []


def test_allowed_restricts_candidates_without_changing_scores(index):
    assert index.search(["chaudier"], top_n=10, allowed={0, 2}) == [(0, approx(0.470004))]
    assert index.search(["chaudier"], top_n=10, allowed=set()) == []
    assert index.search(["chaudier"], top_n=10, allowed=None) == index.search(["chaudier"], 10)


def test_equal_scores_keep_document_order():
    index = BM25Index([["tarif"], ["tarif"], ["tarif"]])
    assert [doc for doc, _ in index.search(["tarif"], top_n=3)] == [0, 1, 2]


def test_custom_parameters():
    # b = 0 removes the length normalisation: d0 and d1 get the same score for "chaudier".
    index = BM25Index(DOCS, k1=1.2, b=0.0)
    hits = index.search(["chaudier"], top_n=10)
    assert hits == [(0, approx(0.470004 * 2.2 / 2.2)), (1, approx(0.470004 * 2.2 / 2.2))]


def test_empty_corpus_and_empty_documents():
    assert BM25Index([]).search(["tarif"], top_n=5) == []
    index = BM25Index([[], ["tarif"]])
    assert [doc for doc, _ in index.search(["tarif"], top_n=5)] == [1]
