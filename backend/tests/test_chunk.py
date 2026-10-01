"""Chunker properties: size limit, whole lines, sections, overlap, stable ids."""

import pytest

from app.core.config import reset_settings
from app.ingestion.chunk import chunk_document
from app.ingestion.loaders import LoadedDoc, Page

SAMPLE = """Document fictif, sans valeur contractuelle.

# Procédure d'astreinte

## Déclenchement

L'astreinte est déclenchée pour toute panne critique signalée hors horaires ouvrés.
Le client appelle le numéro d'astreinte indiqué dans son contrat.

- Étape 1 : qualification de la priorité par le régulateur.
- Étape 2 : envoi d'un technicien sous le délai contractuel.
- Étape 3 : compte rendu au gestionnaire le jour ouvré suivant.

## Tarifs

| Prestation | Tarif HT | Majoration |
|---|---|---|
| Déplacement en journée | 89 € | aucune |
| Déplacement le week-end | 89 € | +35 % |
| Déplacement de nuit | 89 € | +50 % |
| Déplacement un jour férié | 89 € | +100 % |
| Forfait diagnostic | 120 € | aucune |
| Main-d'œuvre | 68 € / h | selon créneau |

## Escalade

En cas de doute, le dossier est transmis à un gestionnaire.
"""

SOURCE_LINES = [line for line in SAMPLE.splitlines() if line.strip()]
# Every size is above the longest line of SAMPLE (83 characters): no line has to be cut.
SIZES = [(100, 0), (160, 60), (200, 50), (400, 100), (900, 150)]


def make_doc(text, name="procedure_astreinte.md", client_id=None):
    return LoadedDoc(
        doc=name, title="Procédure d'astreinte", doc_type="procedure", client_id=client_id,
        pages=[Page(1, text)],
    )  # fmt: skip


def lines_of(chunk):
    return [line for line in chunk.text.splitlines() if line]


def chunk_with(text_fragment, chunks):
    return next(chunk for chunk in chunks if text_fragment in chunk.text)


@pytest.mark.parametrize(("size", "overlap"), SIZES)
def test_no_chunk_exceeds_size(size, overlap):
    chunks = chunk_document(make_doc(SAMPLE), size=size, overlap=overlap)
    assert chunks
    assert max(len(chunk.text) for chunk in chunks) <= size


@pytest.mark.parametrize(("size", "overlap"), SIZES)
def test_lines_and_table_rows_are_never_cut(size, overlap):
    chunks = chunk_document(make_doc(SAMPLE), size=size, overlap=overlap)
    for chunk in chunks:
        for line in lines_of(chunk):
            assert line in SOURCE_LINES
            if line.startswith("|"):
                assert line.endswith("|") and line.count("|") == 4


@pytest.mark.parametrize(("size", "overlap"), SIZES)
def test_all_source_text_is_preserved(size, overlap):
    chunks = chunk_document(make_doc(SAMPLE), size=size, overlap=overlap)
    produced = [line for chunk in chunks for line in lines_of(chunk)]
    if overlap == 0:
        assert produced == SOURCE_LINES  # same lines, same order, nothing repeated
    else:
        assert set(produced) == set(SOURCE_LINES)


def test_large_table_is_cut_between_rows_only():
    chunks = chunk_document(make_doc(SAMPLE), size=160, overlap=0)
    table_chunks = [chunk for chunk in chunks if "| " in chunk.text]
    assert len(table_chunks) > 1  # 330 characters of table cannot fit in 160
    rows = [line for chunk in table_chunks for line in lines_of(chunk) if line.startswith("|")]
    assert rows == [line for line in SOURCE_LINES if line.startswith("|")]


def test_small_table_stays_in_one_chunk():
    chunks = chunk_document(make_doc(SAMPLE), size=900, overlap=150)
    table = chunk_with("| Prestation |", chunks)
    assert all(row in table.text for row in SOURCE_LINES if row.startswith("|"))


def test_section_is_the_nearest_heading_above():
    chunks = chunk_document(make_doc(SAMPLE), size=160, overlap=60)
    assert chunk_with("Document fictif", chunks).section == ""  # before any heading
    assert chunk_with("Étape 3", chunks).section == "Déclenchement"
    assert {c.section for c in chunks if c.text.count("|") >= 4} == {"Tarifs"}
    assert chunk_with("En cas de doute", chunks).section == "Escalade"


def test_heading_starts_a_new_chunk_and_title_joins_first_section():
    chunks = chunk_document(make_doc(SAMPLE), size=900, overlap=150)
    assert [chunk.section for chunk in chunks] == ["", "Déclenchement", "Tarifs", "Escalade"]
    # The document title has no text of its own: it is kept with the first section
    # instead of forming a chunk made of a heading alone.
    assert chunks[1].text.startswith("# Procédure d'astreinte\n\n## Déclenchement\n\nL'astreinte")
    assert chunks[2].text.startswith("## Tarifs\n\n| Prestation |")
    assert chunks[3].text == (
        "## Escalade\n\nEn cas de doute, le dossier est transmis à un gestionnaire."
    )


@pytest.mark.parametrize(("size", "overlap"), [(160, 60), (200, 50), (400, 100), (900, 150)])
def test_no_chunk_is_made_of_headings_alone(size, overlap):
    for chunk in chunk_document(make_doc(SAMPLE), size=size, overlap=overlap):
        assert any(not line.startswith("#") for line in lines_of(chunk))


def test_heading_keeps_the_first_line_of_a_paragraph_that_does_not_fit_beside_it():
    # Title + heading (41 characters) and the first paragraph (149) exceed 160 together.
    chunks = chunk_document(make_doc(SAMPLE), size=160, overlap=0)
    assert chunks[1].text == (
        "# Procédure d'astreinte\n\n## Déclenchement\n\n"
        "L'astreinte est déclenchée pour toute panne critique signalée hors horaires ouvrés."
    )
    assert chunks[2].text.startswith("Le client appelle le numéro d'astreinte")


def test_chunk_ids_are_stable_and_unique():
    first = chunk_document(make_doc(SAMPLE), size=160, overlap=60)
    second = chunk_document(make_doc(SAMPLE), size=160, overlap=60)
    ids = [chunk.chunk_id for chunk in first]
    assert ids == [f"procedure_astreinte.md#p1-{n}" for n in range(1, len(first) + 1)]
    assert first == second


def test_metadata_is_copied_on_every_chunk():
    chunks = chunk_document(make_doc(SAMPLE, name="contrat_C-12.md", client_id="C-12"), 200, 50)
    assert {(c.doc, c.title, c.doc_type, c.page, c.client_id) for c in chunks} == {
        ("contrat_C-12.md", "Procédure d'astreinte", "procedure", 1, "C-12")
    }


def test_overlap_repeats_whole_trailing_lines():
    lines = [f"Ligne {n:02d} : contrôle." for n in range(1, 7)]  # 20 characters each
    assert {len(line) for line in lines} == {20}
    text = "## Liste\n" + "\n".join(lines)
    chunks = chunk_document(make_doc(text), size=65, overlap=25)
    # 25 characters of overlap hold exactly one 20-character line.
    assert [chunk.text for chunk in chunks] == [
        "## Liste\n\n" + "\n".join(lines[0:2]),
        "\n".join(lines[1:4]),
        "\n".join(lines[3:6]),
    ]
    assert {chunk.section for chunk in chunks} == {"Liste"}


def test_overlap_zero_repeats_nothing():
    lines = [f"Ligne {n:02d} : contrôle." for n in range(1, 7)]
    chunks = chunk_document(make_doc("\n".join(lines)), size=65, overlap=0)
    assert [chunk.text for chunk in chunks] == ["\n".join(lines[0:3]), "\n".join(lines[3:6])]


def test_overlap_gives_way_rather_than_exceed_size_or_cut_a_paragraph():
    first, second, third = "A" * 40, "B" * 20, "C" * 60
    chunks = chunk_document(make_doc(f"{first}\n\n{second}\n\n{third}"), size=65, overlap=25)
    # `second` would be repeated before `third`, but 20 + 2 + 60 characters exceed 65.
    assert [chunk.text for chunk in chunks] == [f"{first}\n\n{second}", third]


def test_overlap_does_not_cross_a_section_boundary():
    chunks = chunk_document(make_doc(SAMPLE), size=160, overlap=60)
    escalade = chunk_with("En cas de doute", chunks)
    assert escalade.text.startswith("## Escalade")
    assert "Main-d'œuvre" not in escalade.text


def test_overlap_stays_within_budget_on_the_sample():
    size, overlap = 160, 60
    chunks = chunk_document(make_doc(SAMPLE), size=size, overlap=overlap)
    repeated = 0
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = [line for line in lines_of(current) if line in lines_of(previous)]
        assert len("\n".join(shared)) <= overlap
        repeated += len(shared)
    assert repeated > 0


def test_line_longer_than_size_is_cut_at_sentence_ends():
    sentences = [
        "Le technicien d'astreinte intervient sous quatre heures pour une panne critique.",
        "Il rédige un compte rendu d'intervention transmis au gestionnaire du contrat.",
        "Le client reçoit ensuite un devis si des pièces hors contrat sont nécessaires.",
        "La facturation suit la grille tarifaire en vigueur ; les majorations s'appliquent.",
    ]
    line = " ".join(sentences)
    chunks = chunk_document(make_doc(line), size=120, overlap=0)
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks).split() == line.split()
    assert chunks[0].text == sentences[0]


def test_sentence_and_word_longer_than_size_are_still_bounded():
    sentence = " ".join(["intervention"] * 30)  # 389 characters, no punctuation
    blob = "A1b2" * 100  # a 400-character "word"
    chunks = chunk_document(make_doc(f"{sentence}\n\n{blob}"), size=100, overlap=0)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert " ".join(c.text for c in chunks if "interv" in c.text).split() == sentence.split()
    assert "".join(c.text for c in chunks if "A1b2" in c.text) == blob


def test_pages_restart_numbering_and_carry_the_section():
    doc = LoadedDoc(
        doc="guide.pdf", title="Guide", doc_type="document", client_id=None,
        pages=[
            Page(1, "## Planification\nLes créneaux sont proposés sous 48 h."),
            Page(3, "Annulation sans frais jusqu'à 24 h avant."),
        ],
    )  # fmt: skip
    chunks = chunk_document(doc, size=900, overlap=150)
    assert [(c.chunk_id, c.page, c.section) for c in chunks] == [
        ("guide.pdf#p1-1", 1, "Planification"),
        ("guide.pdf#p3-1", 3, "Planification"),
    ]


def test_document_without_page_gives_no_chunk():
    doc = LoadedDoc(doc="scan.pdf", title="Scan", doc_type="document", client_id=None, pages=[])
    assert chunk_document(doc) == []


def test_defaults_come_from_settings(monkeypatch):
    assert len(chunk_document(make_doc(SAMPLE))) == 4  # default size 900: one chunk per section
    monkeypatch.setenv("CHUNK_SIZE", "160")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    reset_settings()
    chunks = chunk_document(make_doc(SAMPLE))
    assert chunks == chunk_document(make_doc(SAMPLE), size=160, overlap=0)
    assert len(chunks) > 4


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, -1)])
def test_invalid_size_or_overlap_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(make_doc(SAMPLE), size=size, overlap=overlap)
