"""PII masking on realistic French requests: what must be masked and what must survive."""

import pytest

from app.core.pii import redact

# Check digits used below (verified by hand or with the public test numbers):
# - FR76 3000 6000 0112 3456 7890 189, DE89 3704 0044 0532 0130 00,
#   GB29 NWBK 6016 1331 9268 19 and BE68 5390 0754 7034 pass mod-97;
# - 4111 1111 1111 1111 (Visa test number) and 3782 822463 10005 (Amex test number)
#   pass Luhn; 732 829 320 00074 is a valid SIRET.

MASKED = [
    # --- TEL: every common way of writing a French number -------------------------
    ("Rappelez-moi au 06 12 34 56 78 svp.", "Rappelez-moi au [TEL] svp.", ["TEL"]),
    ("Mon portable : 0612345678", "Mon portable : [TEL]", ["TEL"]),
    ("Tél. 06.12.34.56.78", "Tél. [TEL]", ["TEL"]),
    ("Joignable au 06-12-34-56-78 après 18 h", "Joignable au [TEL] après 18 h", ["TEL"]),
    ("Depuis l'étranger : +33 6 12 34 56 78", "Depuis l'étranger : [TEL]", ["TEL"]),
    ("Contact +33612345678", "Contact [TEL]", ["TEL"]),
    ("Contact 0033 6 12 34 56 78", "Contact [TEL]", ["TEL"]),
    ("Contact +33 (0)6 12 34 56 78", "Contact [TEL]", ["TEL"]),
    ("Portable 07 81 23 45 67", "Portable [TEL]", ["TEL"]),
    ("Standard Paris 01 42 68 53 00", "Standard Paris [TEL]", ["TEL"]),
    ("Agence de Nantes 02 40 12 34 56", "Agence de Nantes [TEL]", ["TEL"]),
    ("Agence de Strasbourg 03 88 12 34 56", "Agence de Strasbourg [TEL]", ["TEL"]),
    ("Le gardien est au 04 72 00 11 22.", "Le gardien est au [TEL].", ["TEL"]),
    ("Agence de Bordeaux 05 56 12 34 56", "Agence de Bordeaux [TEL]", ["TEL"]),
    ("Ligne box 09 72 12 34 56", "Ligne box [TEL]", ["TEL"]),
    # No-break spaces, as pasted from an e-mail signature.
    ("Tél. : 04\u00a072\u00a000\u00a011\u00a022", "Tél. : [TEL]", ["TEL"]),
    (
        "Bureau 04 72 00 11 22 ou portable 06 12 34 56 78",
        "Bureau [TEL] ou portable [TEL]",
        ["TEL"],
    ),
    ("04 72 00 11 22 06 12 34 56 78", "[TEL] [TEL]", ["TEL"]),
    # --- EMAIL --------------------------------------------------------------------
    (
        "Merci de répondre à jean.dupont@syndic-lumiere.fr.",
        "Merci de répondre à [EMAIL].",
        ["EMAIL"],
    ),
    ("Copie à hélène_martin+sav@société.example.com", "Copie à [EMAIL]", ["EMAIL"]),
    # --- IBAN: valid mod-97 only, any country, any case, with or without spaces ---
    (
        "Remboursez sur FR76 3000 6000 0112 3456 7890 189 merci de faire vite",
        "Remboursez sur [IBAN] merci de faire vite",
        ["IBAN"],
    ),
    ("iban : fr7630006000011234567890189.", "iban : [IBAN].", ["IBAN"]),
    ("Compte allemand DE89 3704 0044 0532 0130 00", "Compte allemand [IBAN]", ["IBAN"]),
    ("Compte anglais GB29 NWBK 6016 1331 9268 19", "Compte anglais [IBAN]", ["IBAN"]),
    ("BE68 5390 0754 7034 immédiatement", "[IBAN] immédiatement", ["IBAN"]),
    (
        "Ancien FR76 3000 6000 0112 3456 7890 189 et DE89 3704 0044 0532 0130 00 pour le nouveau",
        "Ancien [IBAN] et [IBAN] pour le nouveau",
        ["IBAN"],
    ),
    # --- CARTE: valid Luhn only ---------------------------------------------------
    ("Ma carte 4111 1111 1111 1111 a été débitée", "Ma carte [CARTE] a été débitée", ["CARTE"]),
    ("CB 4111111111111111 12/26", "CB [CARTE] 12/26", ["CARTE"]),
    ("Carte 4111-1111-1111-1111", "Carte [CARTE]", ["CARTE"]),
    ("Amex 3782 822463 10005", "Amex [CARTE]", ["CARTE"]),
    # Digits just before the number must not hide it.
    ("lot 12 4111 1111 1111 1111", "lot 12 [CARTE]", ["CARTE"]),
    # --- SIRET: 14 digits, valid Luhn only ----------------------------------------
    ("Notre SIRET : 732 829 320 00074", "Notre SIRET : [SIRET]", ["SIRET"]),
    (
        "SIRET 73282932000074 12 rue de la République",
        "SIRET [SIRET] 12 rue de la République",
        ["SIRET"],
    ),
    # --- NIR ----------------------------------------------------------------------
    ("Mon numéro de sécu : 1 85 05 78 006 084 91", "Mon numéro de sécu : [NIR]", ["NIR"]),
    ("NIR 185057800608491", "NIR [NIR]", ["NIR"]),
    ("Née en Corse : 2 69 05 2A 588 157 80", "Née en Corse : [NIR]", ["NIR"]),
    (
        "NIR 1 85 05 78 006 084 91 12 rue des Lilas",
        "NIR [NIR] 12 rue des Lilas",
        ["NIR"],
    ),
]

UNCHANGED = [
    # Identifiers of the application.
    "Client C-12, ticket T-000123, règle RG-03.",
    # Amounts.
    "Je conteste la facture de 120 € et l'avoir de 1 250,50 €.",
    "Déplacement 89 € HT, forfait diagnostic 120 € HT, main-d'œuvre 68 € HT/h.",
    # Dates and times.
    "Intervention du 01/02/2026 confirmée pour le 2026-06-30 à 14h30.",
    "Le 01.02.2026 10 techniciens sont passés.",
    "Contrat valable du 1er janvier 2025 au 31 décembre 2028.",
    # Durations, percentages, time ranges.
    "Délai de 4 h ouvrées, majoration de +35 % le week-end et de 50 % la nuit (22 h - 6 h).",
    # Postal codes and street numbers.
    "Chaufferie au 12 rue de la République, 69003 Lyon.",
    # Document numbers.
    "Facture FA-2026-00412, contrat n° 2025-0012, devis 2026-118.",
    # Equipment references followed by words have the shape of an IBAN, not its checksum.
    "Chaudière modèle DT25 en panne depuis hier matin",
    "Référence TH45 modele standard ne fonctionne plus",
    # A VAT number is company data, not a bank account.
    "TVA intracommunautaire FR40 303 265 045",
    # Service numbers (08) are not personal data.
    "Numéro vert 0 800 123 456",
    # Check digits: one wrong digit and the number is not recognised.
    "IBAN FR76 3000 6000 0112 3456 7890 180",
    "Carte 4111 1111 1111 1112",
    "SIRET 732 829 320 00075",
    # Too long to be a card number.
    "Compteur 12345678901234567890123",
    "",
]


@pytest.mark.parametrize(("text", "expected", "types"), MASKED)
def test_redact_masks_personal_identifiers(text, expected, types):
    assert redact(text) == (expected, types)


@pytest.mark.parametrize("text", UNCHANGED)
def test_redact_leaves_other_text_unchanged(text):
    assert redact(text) == (text, [])


def test_types_are_sorted_and_unique():
    text = (
        "Bonjour, je suis joignable au 06 12 34 56 78 ou au 04 72 00 11 22, "
        "ou par mail : j.dupont@example.fr. Remboursement de 120 € sur "
        "FR76 3000 6000 0112 3456 7890 189 (société SIRET 732 829 320 00074, client C-12)."
    )
    masked, types = redact(text)
    assert masked == (
        "Bonjour, je suis joignable au [TEL] ou au [TEL], "
        "ou par mail : [EMAIL]. Remboursement de 120 € sur "
        "[IBAN] (société SIRET [SIRET], client C-12)."
    )
    assert types == ["EMAIL", "IBAN", "SIRET", "TEL"]


def test_redact_is_idempotent():
    once, types = redact("Carte 4111 1111 1111 1111, tél. 06 12 34 56 78, a@b.fr")
    assert types == ["CARTE", "EMAIL", "TEL"]
    assert redact(once) == (once, [])


def test_fourteen_digit_luhn_number_is_labelled_siret_not_card():
    # 3056 9309 0259 04 is a 14-digit card test number: masked, under the SIRET label.
    assert redact("Numéro 30569309025904") == ("Numéro [SIRET]", ["SIRET"])


def test_nir_is_masked_without_checking_its_key():
    # Same number as above with a wrong key (36 instead of 91): still personal data.
    assert redact("NIR 1 85 05 78 006 084 36") == ("NIR [NIR]", ["NIR"])
