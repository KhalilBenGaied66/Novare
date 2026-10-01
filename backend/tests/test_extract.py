import pytest

from app.agents.extract import extract_amounts, extract_client_id, format_amount


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Je suis le client C-12.", "C-12"),
        ("contrat c-27, site de Villeurbanne", "C-27"),  # upper-cased
        ("(C-45) demande de devis", "C-45"),
        ("Dossier C-123456 en cours", "C-123456"),
        ("Clients C-12 et C-34", "C-12"),  # the first one
        ("Aucun identifiant ici", None),
        ("Référence ABC-12 du fabricant", None),  # inside a longer reference
        ("Référence C-1234567", None),  # more than six digits
        ("Ticket T-000123, priorité P1", None),
        ("Vitamine C - 12 comprimés", None),
    ],
)
def test_extract_client_id(text, expected):
    assert extract_client_id(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Je conteste la facture de 120 €.", [120.0]),
        ("Facture de 120€ reçue hier", [120.0]),
        ("Montant : 1 250,50 €", [1250.5]),
        ("Montant : 1 250,50 €", [1250.5]),  # no-break spaces
        ("Montant : 1 250 €", [1250.0]),  # narrow no-break space
        ("Total de 1.250,50 euros", [1250.5]),
        ("Total de 1.250 €", [1250.0]),  # dot + three digits = thousands
        ("Total de 120.50 €", [120.5]),  # dot + two digits = decimals
        ("Un trop-perçu de 60 euros", [60.0]),
        ("Un écart de 1 euro", [1.0]),
        ("Montant 300 EUR, client C-12", [300.0]),
        ("montant 300 eur", [300.0]),
        ("Facture de 499,99 €", [499.99]),
        ("Facture de 245,5 €", [245.5]),
        ("Facture de 12 500 € pour la chaudière", [12500.0]),
        ("Facture de 0 €", [0.0]),
        ("Facturé 480 € au lieu des 300 € prévus", [480.0, 300.0]),
        ("120 € puis encore 120,00 €", [120.0, 120.0]),  # duplicates are kept
    ],
)
def test_extract_amounts(text, expected):
    assert extract_amounts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Intervention sous 4 h, majoration de 35 %",
        "Rappel au 06 12 34 56 78 avant le 12/03/2026",
        "Facture F-2026-0412 du client C-12",
        "Ticket T-000123 ouvert en 2026",
        "Le marché européen compte 12 Europe et 3 eurodéputés",
        "",
    ],
)
def test_numbers_without_currency_are_not_amounts(text):
    assert extract_amounts(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The digits of a reference must not be glued to the amount that follows it.
        ("Facture F-2026-0412 de 120 €", [120.0]),
        ("Client C-12 120 €", [120.0]),
        ("Ticket T-000123 45,50 €", [45.5]),
        ("Priorité P1 500 €", [500.0]),
        ("En 2026 120 € ont été facturés", [120.0]),
    ],
)
def test_amount_next_to_a_reference(text, expected):
    assert extract_amounts(text) == expected


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(120.0, "120,00 €"), (1250.5, "1250,50 €"), (499.99, "499,99 €"), (0.0, "0,00 €")],
)
def test_format_amount_uses_the_french_decimal_comma(amount, expected):
    assert format_amount(amount) == expected
