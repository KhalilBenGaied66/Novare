"""Confidence score, prompt source block, reference parsing and citations."""

import pytest

from app.core.guardrails import (
    build_citations,
    extract_refs,
    format_sources,
    retrieval_confidence,
    valid_refs,
)
from app.core.types import Chunk, RetrievedChunk


def hit(
    text: str,
    *,
    doc: str = "grille_tarifs.md",
    title: str = "Grille tarifaire 2026",
    section: str = "Déplacements",
    page: int = 1,
    n: int = 1,
    score: float = 1.0,
    dense_score: float | None = None,
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{doc}#p{page}-{n}",
        doc=doc,
        title=title,
        doc_type="tarif",
        page=page,
        section=section,
        text=text,
    )
    return RetrievedChunk(chunk=chunk, score=score, dense_score=dense_score)


TARIFS = "Déplacement en journée : 89 € HT.\nMajoration week-end : +35 % sur le déplacement."

# --- retrieval_confidence ----------------------------------------------------------


def test_confidence_is_full_when_every_query_token_is_in_the_sources():
    assert retrieval_confidence("Quelle est la majoration le week-end ?", [hit(TARIFS)]) == 1.0


def test_confidence_is_the_share_of_query_tokens_found():
    # Content tokens: major(ation), week, end, chaudi(ère) -> 3 of 4 are in the chunk.
    query = "majoration week-end chaudière"
    assert retrieval_confidence(query, [hit(TARIFS)]) == pytest.approx(0.75)


def test_confidence_ignores_accents_and_inflections():
    results = [hit("Les délais dépendent de la formule.", section="Délais d'intervention")]
    assert retrieval_confidence("delai des interventions", results) == 1.0


def test_confidence_counts_title_and_section_tokens():
    results = [hit("89 € HT.", title="Grille tarifaire 2026", section="Déplacements")]
    assert retrieval_confidence("grille tarifaire déplacement", results) == 1.0


def test_confidence_uses_all_results():
    results = [hit(TARIFS), hit("Forfait diagnostic : 120 € HT.", section="Forfaits", n=2)]
    assert retrieval_confidence("majoration et forfait diagnostic", results) == 1.0


def test_confidence_is_zero_without_results_or_without_content_tokens():
    assert retrieval_confidence("majoration week-end", []) == 0.0
    assert retrieval_confidence("est-ce que vous ?", [hit(TARIFS)]) == 0.0
    assert retrieval_confidence("", [hit(TARIFS)]) == 0.0


def test_confidence_is_zero_for_an_unrelated_question():
    assert retrieval_confidence("recette de la tarte aux pommes", [hit(TARIFS)]) == 0.0


@pytest.mark.parametrize(
    ("dense_score", "expected"),
    [
        (0.10, 0.30),  # below the floor: the dense part is 0
        (0.25, 0.30),
        (0.475, 0.50),  # halfway between 0.25 and 0.70
        (0.70, 0.70),
        (0.95, 0.70),  # above the ceiling: the dense part is capped at 1
    ],
)
def test_confidence_blends_coverage_and_dense_score(dense_score, expected):
    # Coverage is 0.5: "majoration" is in the chunk, "chaudière" is not.
    results = [hit(TARIFS, dense_score=dense_score)]
    assert retrieval_confidence("majoration chaudière", results) == pytest.approx(expected)


def test_confidence_takes_the_best_dense_score():
    results = [hit(TARIFS, dense_score=0.30), hit("Autre passage.", n=2, dense_score=0.70)]
    assert retrieval_confidence("majoration chaudière", results) == pytest.approx(0.70)


def test_paraphrase_without_shared_words_keeps_the_dense_part():
    results = [hit(TARIFS, dense_score=0.70)]
    assert retrieval_confidence("supplément samedi dimanche", results) == pytest.approx(0.40)


# --- format_sources ----------------------------------------------------------------


def test_format_sources_numbers_each_source():
    results = [
        hit(TARIFS),
        hit("  Guide de planification.\n", doc="guide.pdf", title="Guide", section="", page=2),
    ]
    assert format_sources(results) == (
        "[1] Grille tarifaire 2026 — grille_tarifs.md, p. 1, Déplacements\n"
        "Déplacement en journée : 89 € HT.\n"
        "Majoration week-end : +35 % sur le déplacement.\n"
        "\n"
        "[2] Guide — guide.pdf, p. 2\n"
        "Guide de planification."
    )


def test_format_sources_of_nothing_is_empty():
    assert format_sources([]) == ""


# --- extract_refs / valid_refs -----------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "refs"),
    [
        ("Le déplacement coûte 89 € HT [1].", [1]),
        ("Majoration de 35 % [2, 3] le week-end.", [2, 3]),
        ("Délai de 4 h [3][1], confirmé [1] puis [2 ,3].", [3, 1, 2]),
        ("Voir la source [12].", [12]),
        ("Aucune référence ici.", []),
        # Not references: masking tags, placeholders, empty or malformed brackets.
        ("Contact [TEL] ou [EMAIL], forme [n], [], [1.5], [a1], (2).", []),
    ],
)
def test_extract_refs(answer, refs):
    assert extract_refs(answer) == refs


def test_valid_refs_keeps_only_existing_sources():
    answer = "Tarif [1], délai [4], autre [0], encore [2, 7]."
    assert valid_refs(answer, 3) == [1, 2]
    assert valid_refs(answer, 4) == [1, 4, 2]
    assert valid_refs(answer, 0) == []


# --- build_citations ---------------------------------------------------------------


def test_build_citations_without_refs_cites_every_result_in_order():
    results = [hit(TARIFS, score=0.0327868), hit("Forfait diagnostic.", section="Forfaits", n=2)]
    citations = build_citations(results)
    assert [c.ref for c in citations] == [1, 2]
    first = citations[0]
    assert first.doc == "grille_tarifs.md"
    assert first.title == "Grille tarifaire 2026"
    assert first.page == 1
    assert first.section == "Déplacements"
    assert first.chunk_id == "grille_tarifs.md#p1-1"
    assert first.score == 0.0328
    assert (
        first.excerpt
        == "Déplacement en journée : 89 € HT. Majoration week-end : +35 % sur le déplacement."
    )


def test_build_citations_keeps_only_the_given_refs_in_their_order():
    results = [hit("un", n=1), hit("deux", n=2), hit("trois", n=3)]
    citations = build_citations(results, refs=[3, 1])
    assert [(c.ref, c.excerpt) for c in citations] == [(3, "trois"), (1, "un")]


def test_build_citations_ignores_invalid_and_duplicate_refs():
    results = [hit("un", n=1), hit("deux", n=2)]
    citations = build_citations(results, refs=[2, 0, 5, -1, 2])
    assert [c.ref for c in citations] == [2]
    assert build_citations(results, refs=[]) == []


def test_excerpt_is_single_spaced_and_limited_to_300_characters():
    text = "| Formule | P1 |\n|---|---|\n\n" + "mot  " * 100
    excerpt = build_citations([hit(text)])[0].excerpt
    assert len(excerpt) == 300
    assert excerpt.startswith("| Formule | P1 | |---|---| mot mot ")
    assert "  " not in excerpt
    assert "\n" not in excerpt
