"""Loaders: Markdown / text front matter, PDF pages, .eml messages, manifest overrides."""

import json
from email.message import EmailMessage

import pytest
from fpdf import FPDF

from app.ingestion.loaders import (
    SUPPORTED_SUFFIXES,
    Page,
    load_document,
    load_manifest,
)


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def make_pdf(path, pages, title=None):
    """Write a PDF with one page per string; an empty string gives a page without text."""
    pdf = FPDF()
    if title:
        pdf.set_title(title)
    pdf.set_font("helvetica", size=12)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 8, text)
    pdf.output(str(path))
    return path


def make_eml(path, subject, body, html=None, crlf=False):
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "Direction technique <direction.technique@novare.example>"
    message["To"] = "equipe.astreinte@novare.example"
    message.set_content(body)
    if html:
        message.add_alternative(html, subtype="html")
    raw = message.as_bytes()
    path.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)
    return path


def test_supported_suffixes():
    assert SUPPORTED_SUFFIXES == (".md", ".txt", ".pdf", ".eml")


# --- Markdown / text ---------------------------------------------------------------


def test_markdown_front_matter_gives_metadata_and_is_removed_from_the_text(tmp_path):
    path = write(
        tmp_path,
        "contrat_C-12.md",
        "---\ntitre: Contrat de maintenance — client C-12\ntype: contrat\nclient_id: c-12\n---\n"
        "# Contrat\n\nFormule Confort, astreinte week-end non incluse.\n",
    )
    doc = load_document(path)
    assert doc.doc == "contrat_C-12.md"
    assert doc.title == "Contrat de maintenance — client C-12"
    assert doc.doc_type == "contrat"
    assert doc.client_id == "C-12"  # upper-cased, like the client id of a request
    assert doc.pages == [Page(1, "# Contrat\n\nFormule Confort, astreinte week-end non incluse.")]


def test_front_matter_value_may_contain_a_colon_and_quotes(tmp_path):
    path = write(tmp_path, "note.md", '---\ntitre: "Note : astreinte hiver"\n---\nTexte.\n')
    assert load_document(path).title == "Note : astreinte hiver"


def test_markdown_without_front_matter_uses_first_heading_then_file_stem(tmp_path):
    with_heading = write(tmp_path, "faq.md", "Intro.\n\n# Foire aux questions\n\n## Tarifs\n")
    doc = load_document(with_heading)
    assert (doc.title, doc.doc_type, doc.client_id) == ("Foire aux questions", "document", None)
    assert doc.pages[0].text.startswith("Intro.")

    without_heading = write(tmp_path, "memo_interne.txt", "Rappel : astreinte renforcée.\n")
    assert load_document(without_heading).title == "memo_interne"


def test_front_matter_title_wins_over_heading(tmp_path):
    path = write(tmp_path, "a.md", "---\ntitre: Titre officiel\n---\n# Autre titre\nTexte.\n")
    assert load_document(path).title == "Titre officiel"


def test_unclosed_front_matter_is_plain_text(tmp_path):
    path = write(tmp_path, "regle.md", "---\nUne ligne de séparation puis du texte.\n")
    doc = load_document(path)
    assert doc.pages[0].text == "---\nUne ligne de séparation puis du texte."
    assert doc.title == "regle"


def test_byte_order_mark_does_not_hide_the_front_matter(tmp_path):
    path = tmp_path / "bom.md"
    path.write_text("---\ntitre: Avec BOM\n---\nTexte.\n", encoding="utf-8-sig")
    doc = load_document(path)
    assert doc.title == "Avec BOM"
    assert doc.pages[0].text == "Texte."


def test_empty_document_has_no_page(tmp_path):
    assert load_document(write(tmp_path, "vide.md", "---\ntitre: Vide\n---\n\n")).pages == []


def test_non_utf8_text_file_raises_value_error(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("Procédure dégradée".encode("latin-1"))
    with pytest.raises(ValueError):
        load_document(path)


# --- PDF ---------------------------------------------------------------------------


def test_pdf_one_page_object_per_pdf_page_and_empty_pages_skipped(tmp_path):
    path = make_pdf(
        tmp_path / "guide.pdf",
        [
            "Guide de planification\nLes créneaux sont proposés sous 48 h ouvrées.",
            "",
            "Annulation sans frais jusqu'à 24 h avant le créneau.",
        ],
        title="Guide de planification des interventions",
    )
    doc = load_document(path)
    assert doc.title == "Guide de planification des interventions"
    assert doc.doc_type == "document"
    assert doc.client_id is None
    assert [page.number for page in doc.pages] == [1, 3]
    assert "Les créneaux sont proposés sous 48 h ouvrées." in doc.pages[0].text
    assert doc.pages[1].text == "Annulation sans frais jusqu'à 24 h avant le créneau."


def test_pdf_without_title_metadata_uses_file_stem(tmp_path):
    doc = load_document(make_pdf(tmp_path / "planning_2026.pdf", ["Planning."]))
    assert doc.title == "planning_2026"


def test_corrupt_pdf_raises_value_error(tmp_path):
    path = tmp_path / "casse.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a real pdf")
    with pytest.raises(ValueError, match="unreadable PDF"):
        load_document(path)


# --- .eml --------------------------------------------------------------------------


def test_eml_subject_and_plain_body_without_addresses(tmp_path):
    path = make_eml(
        tmp_path / "note_astreinte.eml",
        "Renfort de l'astreinte hivernale",
        "Bonjour,\n\nL'astreinte est renforcée du 1er décembre au 28 février.\n",
        html="<p>Version HTML à ignorer</p>",
    )
    doc = load_document(path)
    assert doc.title == "Renfort de l'astreinte hivernale"
    assert doc.doc_type == "email"
    assert len(doc.pages) == 1
    assert doc.pages[0].text == (
        "Objet : Renfort de l'astreinte hivernale\n\n"
        "Bonjour,\n\nL'astreinte est renforcée du 1er décembre au 28 février."
    )
    assert "novare.example" not in doc.pages[0].text
    assert "HTML" not in doc.pages[0].text


def test_eml_crlf_line_endings_are_normalised(tmp_path):
    path = make_eml(tmp_path / "crlf.eml", "Sujet", "Ligne 1\nLigne 2\n", crlf=True)
    assert load_document(path).pages[0].text == "Objet : Sujet\n\nLigne 1\nLigne 2"


def test_eml_without_subject_falls_back_to_file_stem(tmp_path):
    path = tmp_path / "sans_objet.eml"
    path.write_bytes(b"From: a@novare.example\n\nCorps du message.\n")
    doc = load_document(path)
    assert doc.title == "sans_objet"
    assert doc.pages[0].text == "Corps du message."


def test_eml_with_unknown_charset_raises_value_error(tmp_path):
    path = tmp_path / "charset.eml"
    path.write_bytes(b"Subject: x\nContent-Type: text/plain; charset=inconnu-42\n\ncorps\n")
    with pytest.raises(ValueError, match="charset"):
        load_document(path)


# --- manifest ----------------------------------------------------------------------


def test_load_manifest_missing_file_is_empty(tmp_path):
    assert load_manifest(tmp_path) == {}


def test_load_manifest_reads_entries(tmp_path):
    entries = {"guide.pdf": {"titre": "Guide", "type": "procedure"}}
    write(tmp_path, "manifest.json", json.dumps(entries))
    assert load_manifest(tmp_path) == entries


@pytest.mark.parametrize("content", ["{not json", "[]", '{"guide.pdf": "Guide"}'])
def test_load_manifest_rejects_invalid_content(tmp_path, content):
    write(tmp_path, "manifest.json", content)
    with pytest.raises(ValueError):
        load_manifest(tmp_path)


def test_manifest_overrides_metadata_of_any_file_type(tmp_path):
    manifest = {
        "guide.pdf": {"titre": "Guide de planification", "type": "procedure"},
        "contrat.md": {"client_id": "C-27"},
        "note.eml": {"titre": "Note de service", "type": "note"},
    }
    pdf = load_document(make_pdf(tmp_path / "guide.pdf", ["Texte."], title="Titre PDF"), manifest)
    assert (pdf.title, pdf.doc_type, pdf.client_id) == ("Guide de planification", "procedure", None)

    md = write(tmp_path, "contrat.md", "---\ntitre: Contrat\ntype: contrat\n---\nTexte.\n")
    doc = load_document(md, manifest)
    assert (doc.title, doc.doc_type, doc.client_id) == ("Contrat", "contrat", "C-27")

    eml = load_document(make_eml(tmp_path / "note.eml", "Objet du mail", "Corps."), manifest)
    assert (eml.title, eml.doc_type) == ("Note de service", "note")
    assert eml.pages[0].text.startswith("Objet : Objet du mail")


def test_blank_manifest_client_id_does_not_make_a_scoped_document_public(tmp_path):
    md = write(tmp_path, "contrat.md", "---\nclient_id: C-34\n---\nClause particulière.\n")
    doc = load_document(md, {"contrat.md": {"client_id": None, "titre": ""}})
    assert doc.client_id == "C-34"
    assert doc.title == "contrat"


# --- unsupported -------------------------------------------------------------------


def test_unsupported_suffix_raises_value_error(tmp_path):
    path = write(tmp_path, "tableau.csv", "a;b\n")
    with pytest.raises(ValueError, match="unsupported"):
        load_document(path)


def test_suffix_match_is_case_insensitive(tmp_path):
    assert load_document(write(tmp_path, "NOTE.MD", "# Note\nTexte.\n")).title == "Note"
