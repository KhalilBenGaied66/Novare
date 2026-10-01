"""Build data/sample_docs/guide_planification_interventions.pdf with fpdf2.

The text of the guide lives in this file, so the PDF of the corpus can be reviewed and
rebuilt like any other source: `python scripts/make_sample_pdf.py`.

Constraints that shape the code:
- only the core Helvetica font is used (no font file to ship, same result on every OS).
  Core fonts are limited to Latin-1, so the text avoids the euro sign, the "œ" ligature
  and typographic dashes or apostrophes;
- one authored line is one rendered line (no automatic wrapping), so that text
  extraction gives back whole sentences instead of lines cut at the right margin;
- the creation date is fixed, so two runs produce the same bytes.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sample_docs" / "guide_planification_interventions.pdf"

TITLE = "Guide de planification des interventions"
CREATION_DATE = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)

# One string per page. A line starting with "# " is the document title and a line
# starting with "## " is a section heading; the marker itself is not printed.
PAGES = [
    """\
# Guide de planification des interventions
Référence : GU-PLA-05, version 2, applicable au 1er janvier 2026.
Service émetteur : direction de l'exploitation, Novare Services, agence de Lyon.
## 1. Objet
Ce guide fixe les règles de planification des visites préventives et des dépannages.
Il s'adresse aux planificateurs et aux gestionnaires SAV de l'agence de Lyon.
La qualification des demandes et les délais d'intervention relèvent de la procédure SAV.
## 2. Créneaux d'intervention
Les interventions en heures ouvrées ont lieu du lundi au vendredi, de 8 h à 18 h.
Créneau du matin : de 8 h à 12 h.
Créneau de l'après-midi : de 13 h à 18 h.
Le créneau est annoncé au client avec une plage d'arrivée de deux heures.
Un technicien réalise au plus cinq interventions par jour, trajets compris.
Le temps de trajet retenu entre deux sites est de 30 minutes.
## 3. Visites préventives
Les visites préventives sont planifiées au moins 15 jours à l'avance.
Le planificateur confirme le rendez-vous au contact du site 48 heures avant la visite.
Les visites préventives des chaudières ont lieu de mai à septembre, hors période de chauffe.
Nombre de visites par an : une en Essentiel, deux en Confort, quatre en Premium.
Une visite préventive non réalisée dans l'année est reprise en priorité l'année suivante.
## 4. Interventions correctives
Une demande P1 est planifiée immédiatement, avant toute visite préventive.
Une demande P2 est planifiée dans la journée.
Une demande P3 est planifiée dans la semaine.
Si aucun technicien n'est disponible dans le délai, une visite préventive est déplacée.
Le client dont la visite préventive est déplacée est prévenu le jour même.
""",
    """\
## 5. Affectation des techniciens
Le planificateur affecte un technicien qui détient l'habilitation exigée par l'équipement.
- Chaudières et brûleurs gaz : habilitation gaz.
- Groupes froids et climatiseurs : attestation de capacité pour les fluides frigorigènes.
- Armoires électriques : habilitation électrique.
À habilitation égale, le technicien le plus proche du site est retenu.
Un établissement de santé reçoit toujours un binôme de techniciens.
## 6. Durées standard
- Diagnostic : 1 heure.
- Visite préventive d'une chaudière : 2 heures.
- Visite préventive d'une centrale de traitement d'air : 3 heures.
- Remplacement d'un circulateur : 2 heures.
- Remplacement d'un brûleur : 4 heures.
La durée réelle de chaque intervention figure dans le rapport du technicien.
## 7. Report et annulation par le client
Un report ou une annulation est gratuit jusqu'à 24 heures ouvrées avant le créneau.
Passé ce délai, le déplacement est facturé 89 euros HT.
Le client demande le report sur le portail ou auprès du planificateur.
## 8. Report par Novare Services
Novare Services peut reporter une visite préventive pour traiter une demande P1.
Un nouveau créneau est alors proposé au client sous 5 jours ouvrés.
## 9. Absence du client
Si le site est inaccessible à l'heure convenue, le technicien attend 15 minutes.
Le déplacement est alors facturé et un nouveau rendez-vous est proposé au client.
""",
    """\
## 10. Période de chauffe
La période de chauffe s'étend du 1er octobre au 30 avril.
Pendant la période de chauffe, les demandes de chauffage sont planifiées en premier.
Les remplacements de chaudière sont planifiés hors période de chauffe, sauf urgence.
## 11. Suivi du planning
Le planning de la semaine suivante est figé chaque jeudi à 16 h.
Le respect des créneaux annoncés est mesuré chaque mois, avec un objectif de 90 %.
Chaque intervention planifiée figure dans le ticket, que le client consulte sur le portail.
## 12. Rôles
- Planificateur : construit le planning, affecte les techniciens et prévient les clients.
- Gestionnaire SAV : qualifie la demande et transmet la priorité au planificateur.
- Responsable d'exploitation : arbitre lorsque deux demandes P1 visent le même technicien.
Document fictif - contenu de démonstration, sans valeur contractuelle.
""",
]

# (font style, font size in points, line height in mm) for each kind of line.
STYLES = {
    "title": ("B", 16, 10),
    "heading": ("B", 12, 8),
    "text": ("", 10, 5.5),
}
SPACE_BEFORE_HEADING_MM = 3


def parse_line(line: str) -> tuple[str, str]:
    """Return (kind, printed text) for one authored line."""
    if line.startswith("## "):
        return "heading", line[3:]
    if line.startswith("# "):
        return "title", line[2:]
    return "text", line


def page_lines(page: str) -> list[tuple[str, str]]:
    return [parse_line(line) for line in page.splitlines()]


def build_pdf() -> FPDF:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=False)
    pdf.set_title(TITLE)
    pdf.set_author("Novare Services")
    pdf.set_lang("fr-FR")
    pdf.set_creation_date(CREATION_DATE)

    for page in PAGES:
        pdf.add_page()
        for kind, text in page_lines(page):
            style, size, height = STYLES[kind]
            pdf.set_font("Helvetica", style=style, size=size)
            if pdf.get_string_width(text) > pdf.epw:
                raise ValueError(f"Line too long for the page width, shorten it: {text!r}")
            if kind == "heading":
                pdf.ln(SPACE_BEFORE_HEADING_MM)
            pdf.cell(w=0, h=height, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if pdf.get_y() > pdf.h - pdf.b_margin:
            raise ValueError(f"Page {pdf.page_no()} overflows, move lines to the next page")
    return pdf


def check_extraction(path: Path) -> None:
    """Fail when pypdf does not give back exactly the authored lines, page by page."""
    reader = PdfReader(str(path))
    if len(reader.pages) != len(PAGES):
        raise ValueError(f"Expected {len(PAGES)} pages, found {len(reader.pages)}")
    for number, (pdf_page, page) in enumerate(zip(reader.pages, PAGES, strict=True), start=1):
        extracted = [line.strip() for line in pdf_page.extract_text().splitlines()]
        expected = [text for _, text in page_lines(page)]
        if extracted != expected:
            raise ValueError(f"Page {number}: extracted text differs from the source text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PDF file to write")
    args = parser.parse_args()

    build_pdf().output(str(args.output))
    check_extraction(args.output)
    print(f"{args.output} : {len(PAGES)} pages, texte extrait conforme")


if __name__ == "__main__":
    main()
