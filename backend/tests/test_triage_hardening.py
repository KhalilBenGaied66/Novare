"""Triage on the wordings an independent review used to get past the rules.

Clients known to the reference data of the fixture: C-12, C-27, C-34. C-99 is unknown.
"""

import unicodedata

import pytest

from app.agents import extract
from app.agents.triage import triage
from app.core.schemas import AskRequest

pytestmark = pytest.mark.usefixtures("mini_corpus")


def run(q: str, client_id: str | None = None, montant: float | None = None):
    return triage(AskRequest(q=q, client_id=client_id, montant=montant))


# --- RG-03 needs an explicit dispute ---------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "Merci pour la facture de 120 €, bien reçue.",
        "Pouvez-vous me renvoyer la facture de 120 € du mois dernier ?",
        "J'ai réglé la facture de 120 € hier par virement.",
        "Merci pour le remboursement de 120 € reçu ce matin.",
        "Je ne conteste pas la facture de 120 €, je demande seulement un duplicata.",
    ],
)
def test_mentioning_an_invoice_is_not_a_dispute(q):
    decision = run(q, "C-12")

    assert decision.route != "automation"
    assert "Facturation, non automatisé : pas de contestation explicite" in decision.reasons


@pytest.mark.parametrize(
    "q",
    [
        "Je conteste la facture de 120 €.",
        "Il y a une erreur de facturation de 120 € sur le dernier relevé.",
        "Vous m'avez facturé deux fois le déplacement, soit 89 €.",
        "Je vous adresse une demande de remboursement de 120 €.",
        "Merci d'établir un avoir de 120 €.",
        "Ce trop-perçu de 60 € doit m'être rendu.",
    ],
)
def test_explicit_dispute_of_a_small_amount_is_automated(q):
    decision = run(q, "C-12")

    assert (decision.route, decision.rule) == ("automation", "RG-03")


def test_billing_at_or_above_the_threshold_goes_to_a_person_even_without_dispute_words():
    decision = run("Pouvez-vous m'expliquer la facture de 1 200 € ?", "C-12")

    assert (decision.route, decision.rule) == ("human", "RG-03b")


def test_dispute_with_a_breakdown_is_not_filed_as_a_billing_ticket():
    decision = run("Je conteste la facture de 120 € : la chaudière est toujours en panne.", "C-12")

    assert (decision.route, decision.rule) == ("agent", "RG-05")
    assert "Facturation, non automatisé : la demande signale aussi une panne" in decision.reasons


def test_reasons_say_why_billing_was_not_automated():
    no_amount = run("Je conteste ma dernière facture.", "C-12")
    assert "Facturation, non automatisé : pas de montant exploitable" in no_amount.reasons
    unknown = run("Je conteste la facture de 150 €.", "C-99")
    assert "Facturation, non automatisé : pas de client connu" in unknown.reasons


def test_amount_is_compared_after_rounding_to_cents():
    decision = run("Je conteste cette facture.", "C-12", 499.999)

    assert decision.montant == 500.0
    assert (decision.route, decision.rule) == ("human", "RG-03b")


# --- RG-04 on wordings outside the first list ------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "Je vous mets en demeure de corriger la facture de 120 €.",
        "Sans réponse, nous irons en justice pour ces 120 € facturés à tort.",
        "Nous engagerons des poursuites si la facture de 120 € n'est pas annulée.",
        "Nous saisirons le médiateur pour ce litige de 120 €.",
        "Un procès est envisagé.",
        "Nous allons mettre fin au contrat, et contestons la facture de 120 €.",
        "Nous dénonçons le contrat à son échéance.",
        "Nous ne renouvellerons pas le contrat.",
        "Merci de supprimer mes coordonnées de vos fichiers.",
        "Je demande l'effacement de mes données.",
        "Je veux exercer mon droit d'opposition.",
        "Qui est votre DPO ?",
        "Je signale une fuite de données sur votre portail.",
        "Conformément au R.G.P.D., communiquez-moi les informations me concernant.",
    ],
)
def test_sensitive_subject_in_other_words_goes_to_a_person(q):
    decision = run(q, "C-12")

    assert (decision.route, decision.rule) == ("human", "RG-04")


@pytest.mark.parametrize(
    "q",
    [
        "Quelle est la référence de la facture de mars ?",
        "Quel est le processus de traitement d'une demande ?",
        "Le contrat de maintenance est-il renouvelé par tacite reconduction ?",
        "Un justificatif immédiat est-il nécessaire ?",
    ],
)
def test_lookalike_words_are_not_sensitive(q):
    assert run(q).rule == "DEFAULT"


def test_proces_verbal_is_the_documented_false_positive():
    # "procès" is matched as a word: a "procès-verbal" goes to a person. A false
    # positive of RG-04 only costs a handler's look.
    assert run("Où trouver le procès-verbal d'intervention ?").rule == "RG-04"


# --- Unicode forms ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "route"),
    [
        ("Je veux résilier mon contrat, facture de 120 € contestée.", "human"),
        ("Des pénalités seront appliquées.", "human"),
        ("La chaudière présente une défaillance depuis hier.", "agent"),
    ],
)
def test_decomposed_accents_and_invisible_characters_do_not_change_the_route(q, route):
    decomposed = unicodedata.normalize("NFD", q)
    with_zero_width = q.replace("é", "é​")

    assert decomposed != q
    assert run(decomposed, "C-12").route == route
    assert run(with_zero_width, "C-12").route == route


def test_request_text_is_stored_in_composed_form_without_invisible_characters():
    request = AskRequest(q=unicodedata.normalize("NFD", "  Délai​ prévu ?\u0000 "))

    assert request.q == "Délai prévu ?"


# --- RG-05 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "Nous n'avons plus de chauffage depuis ce matin.",
        "Le radiateur du hall ne chauffe plus.",
        "Il n'y a plus d'eau chaude dans l'immeuble.",
        "La pompe ne démarre plus.",
        "La ventilation ne fonctionne pas.",
        "La chaudière est en défaut.",
        "Il y a une odeur de gaz dans la chaufferie.",
        "Le local technique est inondé.",
        "Brûleur H.S. depuis hier.",
    ],
)
def test_breakdown_in_other_words_goes_to_the_agent(q):
    decision = run(q, "C-12")

    assert (decision.route, decision.rule) == ("agent", "RG-05")


@pytest.mark.parametrize(
    "q",
    [
        "Que faire en cas de panne de chaudière ?",
        "Comment planifier une intervention ?",
        "Qui contacter pour une fuite le week-end ?",
        "Comment ouvrir un ticket ?",
    ],
)
def test_question_without_client_is_documentary_even_with_breakdown_words(q):
    decision = run(q)

    assert (decision.route, decision.rule) == ("rag", "DEFAULT")
    assert any(reason.startswith("RG-05 non appliquée") for reason in decision.reasons)


def test_same_question_from_an_identified_client_is_a_dossier():
    assert run("La chaudière est en panne, que faire ?", "C-12").route == "agent"
    assert run("Chaudière en panne sur le site C-12, que faire ?").route == "agent"


# --- Where the client comes from -------------------------------------------------------


def test_client_written_in_the_text_is_declared_not_scoped():
    in_text = run("Quelle est la formule du contrat C-12 ?")
    in_field = run("Quelle est la formule de mon contrat ?", "C-12")

    assert (in_text.client_id, in_text.client_origin, in_text.scoped_client_id) == (
        "C-12",
        "texte",
        None,
    )
    assert (in_field.client_origin, in_field.scoped_client_id) == ("champ", "C-12")
    assert run("Formule du contrat ?", "C-99").scoped_client_id is None  # unknown client


# --- Amounts with unusual separators ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "1  250 €",  # two spaces
        "1\n250 €",
        "1\t250 €",
        "1 250 €",  # thin space
        "1 250 €",  # figure space
        "1'250 €",
        "1 250 €",
        "1 250   euros",
    ],
)
def test_thousands_separator_never_hides_the_thousands(text):
    assert extract.extract_amounts(f"Je conteste la facture de {text}.") == [1250.0]


def test_dispute_of_1250_with_an_odd_separator_is_not_automated():
    decision = run("Je conteste la facture de 1 250 €.", "C-12")

    assert (decision.route, decision.rule) == ("human", "RG-03b")
