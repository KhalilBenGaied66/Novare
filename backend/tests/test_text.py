"""French text normalisation: the behaviours the triage and the retrieval rely on."""

import pytest

from app.core.text import (
    STOPWORDS,
    has_stem,
    normalize,
    split_sentences,
    stem,
    strip_accents,
    tokenize,
    words,
)

# --- strip_accents / normalize / words ---------------------------------------------


def test_strip_accents():
    assert strip_accents("Élève, ça, naïve, où, pénalité") == "Eleve, ca, naive, ou, penalite"


def test_normalize_lowercases_removes_accents_and_collapses_whitespace():
    assert normalize("  Mise   en\n DEMEURE\treçue cet été ") == "mise en demeure recue cet ete"


def test_normalize_keeps_apostrophes_for_phrase_matching():
    assert normalize("Droit d'accès et droit à l'oubli") == "droit d'acces et droit a l'oubli"


def test_words_are_lowercase_keep_accents_and_split_on_punctuation():
    assert words("L'ÉTÉ à Lyon : client C-12, 4h30 !") == [
        "l",
        "été",
        "à",
        "lyon",
        "client",
        "c",
        "12",
        "4h30",
    ]


# --- tokenize ------------------------------------------------------------------------


def test_tokenize_removes_stopwords():
    assert tokenize("Bonjour, je ne suis pas sûr que le ou la, et vous ?") == ["sur"]
    assert all(tokenize(word) == [] for word in STOPWORDS)


def test_tokenize_keeps_tokens_with_digits_as_they_are():
    assert tokenize("Priorité P2 : 4h, 500 € en 2026") == ["priorit", "p2", "4h", "500", "2026"]


def test_tokenize_drops_single_letters_but_not_single_digits():
    assert tokenize("4 h le 5 x") == ["4", "5"]


def test_tokenize_is_case_and_accent_insensitive():
    assert tokenize("PÉNALITÉS de retard") == tokenize("pénalités DE RETARD") == ["penal", "retard"]
    assert tokenize("Délais") == tokenize("delais")


def test_tokenize_stems_singular_and_plural_to_the_same_token():
    assert tokenize("facture") == tokenize("factures") == tokenize("facturation") == ["factur"]
    assert tokenize("panne") == tokenize("pannes") == ["pann"]
    assert tokenize("litige") == tokenize("litiges") == ["litig"]


def test_tokenize_works_on_whole_words_not_substrings():
    # "ticket" contains "et", "projet" contains "et" and "pro", "stock" contains "st".
    tokens = tokenize("Où en est mon ticket et le projet de stock ?")
    assert tokens == ["ticket", "projet", "stock"]
    assert "et" not in tokens


# --- stem / has_stem -----------------------------------------------------------------


def test_stem_has_no_accent():
    assert stem("résiliation") == "resili"
    assert stem("Pénalités") == "penal"


@pytest.mark.parametrize(
    "word",
    ["résiliation", "résilier", "resilie", "résilie", "résilié", "résilions", "RESILIATION"],
)
def test_every_form_of_resilier_meets_on_the_resil_prefix(word):
    assert has_stem(tokenize(word), ("resil",))


@pytest.mark.parametrize(
    ("text", "prefixes"),
    [
        ("Des pénalités seront appliquées", ("penal",)),
        ("penalite de retard", ("penal",)),
        ("Je conteste cette facture", ("contest",)),
        ("contestation de la facturation", ("factur",)),
        ("demande de remboursement", ("rembours",)),
        ("La chaudière est en panne", ("pann",)),
        ("Il y a une fuite d'eau", ("fuit",)),
        ("défaillance du brûleur", ("defaill",)),
        ("dysfonctionnement de la ventilation", ("dysfonction",)),
        ("merci de planifier un passage", ("planifi",)),
        ("assignation devant le tribunal", ("assign", "tribunal")),
        ("notre avocat", ("avocat",)),
        ("conformité RGPD et CNIL", ("rgpd",)),
    ],
)
def test_has_stem_finds_the_vocabulary_whatever_the_inflection(text, prefixes):
    assert has_stem(tokenize(text), prefixes)


def test_has_stem_matches_from_the_start_of_a_token_only():
    # A prefix must not match in the middle of a word: "et" is not a prefix of "ticket",
    # "pann" is not a prefix of "dépannage".
    assert not has_stem(tokenize("mon ticket"), ("et",))
    assert not has_stem(tokenize("dépannage"), ("pann",))
    assert has_stem(tokenize("dépannage"), ("depann",))
    assert not has_stem(tokenize("Quel est le tarif du déplacement ?"), ("resil", "penal", "pann"))


def test_has_stem_accepts_a_set_and_an_empty_input():
    assert has_stem({"factur", "client"}, ("factur",))
    assert not has_stem([], ("factur",))


# --- split_sentences -------------------------------------------------------------------


def test_split_sentences_splits_on_sentence_punctuation():
    text = "Le délai est de 4 h. Le déplacement coûte 89 € HT ! Est-ce clair ? Oui."
    assert split_sentences(text) == [
        "Le délai est de 4 h.",
        "Le déplacement coûte 89 € HT !",
        "Est-ce clair ?",
        "Oui.",
    ]


def test_split_sentences_does_not_split_inside_numbers_or_before_lowercase():
    assert split_sentences("Majoration de 1,5 % soit 12.50 € env. par visite.") == [
        "Majoration de 1,5 % soit 12.50 € env. par visite."
    ]


def test_split_sentences_keeps_table_rows_whole():
    table = (
        "| Formule | P1 | P2 | P3 |\n"
        "|---|---|---|---|\n"
        "| Premium | 2 h | 4 h | 24 h |\n"
        "| Confort | 4 h ouvrées | 8 h ouvrées | 48 h ouvrées |"
    )
    assert split_sentences(table) == [
        "| Formule | P1 | P2 | P3 |",
        "|---|---|---|---|",
        "| Premium | 2 h | 4 h | 24 h |",
        "| Confort | 4 h ouvrées | 8 h ouvrées | 48 h ouvrées |",
    ]


def test_split_sentences_keeps_list_items_and_headings_whole():
    text = (
        "## Priorités\n"
        "- P1 (critique) : arrêt total d'un équipement ou risque pour la sécurité.\n"
        "- P2 (majeure) : fonctionnement dégradé.\n"
        "\n"
        "Majoration de nuit (22 h - 6 h) : +50 %."
    )
    assert split_sentences(text) == [
        "## Priorités",
        "- P1 (critique) : arrêt total d'un équipement ou risque pour la sécurité.",
        "- P2 (majeure) : fonctionnement dégradé.",
        "Majoration de nuit (22 h - 6 h) : +50 %.",
    ]


def test_split_sentences_ignores_blank_lines_and_surrounding_spaces():
    assert split_sentences("  Ligne un  \n\n\n  Ligne deux\r\n") == ["Ligne un", "Ligne deux"]
    assert split_sentences("") == []
    assert split_sentences(" \n ") == []
