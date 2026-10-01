"""Triage rules on realistic requests: every rule, the order of the rules, and the
wordings that must NOT trigger a rule.

Clients known to the reference data of the fixture: C-12, C-27, C-34. C-99 is unknown.
"""

import pytest

from app.agents.triage import triage
from app.core.config import reset_settings
from app.core.schemas import AskRequest

pytestmark = pytest.mark.usefixtures("mini_corpus")


def run(q: str, client_id: str | None = None, montant: float | None = None):
    return triage(AskRequest(q=q, client_id=client_id, montant=montant))


# (request text, client_id field, montant field, expected route, expected rule)
CASES = [
    # --- RG-04: sensitive subjects go to a person --------------------------------
    ("Je souhaite la résiliation de notre contrat fin mars.", "C-12", None, "human", "RG-04"),
    ("je veux resilier le contrat", None, None, "human", "RG-04"),
    ("Je résilie mon contrat dès aujourd'hui.", "C-12", None, "human", "RG-04"),
    ("Nous résilions le contrat au 31 décembre.", "C-27", None, "human", "RG-04"),
    ("Le contrat a été résilié par courrier recommandé.", None, None, "human", "RG-04"),
    ("RESILIATION IMMEDIATE DU CONTRAT C-34", None, None, "human", "RG-04"),
    ("Quelles pénalités s'appliquent si le délai P1 est dépassé ?", None, None, "human", "RG-04"),
    ("vos penalites de retard sont inacceptables", None, None, "human", "RG-04"),
    ("Nous appliquerons la pénalité prévue au contrat.", "C-27", None, "human", "RG-04"),
    ("Notre avocat vous adressera un courrier.", "C-27", None, "human", "RG-04"),
    ("Sans réponse, nous saisirons le tribunal de commerce.", None, None, "human", "RG-04"),
    ("Un huissier constatera la panne demain.", "C-12", None, "human", "RG-04"),
    ("Notre service juridique étudie le dossier.", None, None, "human", "RG-04"),
    ("Une procédure judiciaire est envisagée.", None, None, "human", "RG-04"),
    ("Nous préparons une assignation pour les retards répétés.", "C-27", None, "human", "RG-04"),
    ("Une plainte sera déposée dès lundi.", None, None, "human", "RG-04"),
    ("Votre service contentieux nous relance à tort.", "C-12", None, "human", "RG-04"),
    ("Ceci vaut mise en demeure d'intervenir sous 48 h.", "C-12", None, "human", "RG-04"),
    ("MISE EN DEMEURE avant poursuites", None, None, "human", "RG-04"),
    ("Conformément au RGPD, envoyez-moi mes informations.", None, None, "human", "RG-04"),
    ("Je vais saisir la CNIL.", None, None, "human", "RG-04"),
    ("Quelles données personnelles conservez-vous sur moi ?", None, None, "human", "RG-04"),
    ("Je souhaite exercer mon droit d'accès.", None, None, "human", "RG-04"),
    ("Je souhaite exercer mon droit d’accès.", None, None, "human", "RG-04"),  # typographic '
    ("J'invoque mon droit à l'oubli.", None, None, "human", "RG-04"),
    ("Merci de procéder à la suppression de mes données.", "C-12", None, "human", "RG-04"),
    # --- RG-04 is evaluated first: never automated, never sent to the agent ------
    (
        "Bonjour, je conteste la facture de 120 € et je résilie mon contrat. "
        "Mon avocat vous contactera.",
        "C-12",
        None,
        "human",
        "RG-04",
    ),
    ("Je conteste la facture de 150 € et je saisis le tribunal.", "C-12", None, "human", "RG-04"),
    ("Remboursez cette facture, sinon nous résilierons.", "C-12", 80.0, "human", "RG-04"),
    ("La chaudière est en panne, notre avocat est prévenu.", "C-12", None, "human", "RG-04"),
    ("Litige de 900 € : mise en demeure à suivre.", "C-12", None, "human", "RG-04"),
    # --- RG-03: small dispute, known client -> automation ------------------------
    ("Je conteste la facture du mois dernier.", "C-12", 120.0, "automation", "RG-03"),
    ("Je conteste la facture F-2026-0412 de 120 €.", "C-12", None, "automation", "RG-03"),
    ("Litige de facturation pour le client C-27.", None, 120.0, "automation", "RG-03"),
    ("Ici le client C-27 : je conteste la facture de 245,50 €.", None, None, "automation", "RG-03"),
    ("Contestation de la facture : 300 EUR, client c-12.", None, None, "automation", "RG-03"),
    ("Demande de remboursement : trop-perçu de 60 euros.", "C-27", None, "automation", "RG-03"),
    ("Un trop perçu de 60 € figure sur mon relevé.", "C-27", None, "automation", "RG-03"),
    ("Erreur de facturation de 45 € sur le déplacement.", "C-12", None, "automation", "RG-03"),
    ("Merci d'émettre un avoir de 80 € (visite annulée).", "C-12", None, "automation", "RG-03"),
    ("Les avoirs promis, soit 90 €, ne sont pas arrivés.", "C-12", None, "automation", "RG-03"),
    ("Facture contestée : 499,99 €.", "C-12", None, "automation", "RG-03"),
    ("Nous contestons cette facture.", "C-12", 0.01, "automation", "RG-03"),
    # the contract of C-34 has expired: RG-03 only needs a known client
    ("Je conteste la facture du diagnostic : 90 €.", "C-34", None, "automation", "RG-03"),
    # the same amount written twice is one distinct amount
    ("Facture de 120 € : je conteste ces 120,00 €.", "C-12", None, "automation", "RG-03"),
    # the explicit field wins over the amounts of the text
    ("Facturé 480 € au lieu de 300 € : je conteste.", "C-12", 180.0, "automation", "RG-03"),
    # --- RG-03b: dispute at or above the threshold -> billing manager -------------
    ("Litige sur la facture de mars.", "C-12", 500.0, "human", "RG-03b"),
    ("Je conteste la facture de 500 €.", "C-12", None, "human", "RG-03b"),
    ("Je conteste la facture de 1 250,50 €.", "C-27", None, "human", "RG-03b"),
    ("Contestation de la facture de 780 €.", "C-99", None, "human", "RG-03b"),  # unknown client
    ("Contestation de la facture de 780 €.", None, None, "human", "RG-03b"),  # no client
    ("Remboursement demandé pour la chaudière en panne.", "C-12", 2000.0, "human", "RG-03b"),
    # --- dispute that cannot be automated: falls through to the next rules -------
    ("Je conteste ma dernière facture, comment procéder ?", "C-12", None, "rag", "DEFAULT"),
    ("Je conteste la facture de 150 €.", "C-99", None, "rag", "DEFAULT"),  # unknown client
    ("Je conteste la facture de 150 €.", None, None, "rag", "DEFAULT"),  # no client at all
    ("Je conteste la facture de 0 €.", "C-12", None, "rag", "DEFAULT"),  # zero is not an amount
    ("Facturé 480 € au lieu des 300 € prévus : je conteste.", "C-12", None, "rag", "DEFAULT"),
    ("Dans quel délai une facture peut-elle être contestée ?", None, None, "rag", "DEFAULT"),
    ("La chaudière est en panne et je conteste la facture.", "C-12", None, "agent", "RG-05"),
    # an amount and a known client without any dispute vocabulary
    ("Pourrais-je avoir un devis pour un circulateur à 300 € ?", "C-12", None, "rag", "DEFAULT"),
    ("Le forfait diagnostic est-il bien de 120 € ?", "C-12", None, "rag", "DEFAULT"),
    # --- RG-05: breakdown -> agent -----------------------------------------------
    ("La chaudière est en panne depuis ce matin.", "C-12", None, "agent", "RG-05"),
    ("Plusieurs pannes cette semaine sur la ventilation.", "C-27", None, "agent", "RG-05"),
    ("Chaudière HS, il fait 14 °C dans les bureaux.", "C-12", None, "agent", "RG-05"),
    ("chaudiere hs", None, None, "agent", "RG-05"),
    ("L'aérotherme ne fonctionne plus.", "C-34", None, "agent", "RG-05"),
    ("Notre pompe à chaleur ne marche plus depuis hier.", None, None, "agent", "RG-05"),
    ("La ventilation est en arrêt complet.", "C-27", None, "agent", "RG-05"),
    ("Le compresseur est à l'arrêt.", "C-27", None, "agent", "RG-05"),
    ("Le groupe froid est hors service depuis 6 h.", "C-27", None, "agent", "RG-05"),
    ("Fuite d'eau importante sous la chaudière.", "C-12", None, "agent", "RG-05"),
    ("Le circuit fuit au niveau du circulateur.", "C-12", None, "agent", "RG-05"),
    ("Défaillance du brûleur de la chaudière numéro 2.", "C-12", None, "agent", "RG-05"),
    ("Dysfonctionnement de la régulation de chaufferie.", "C-12", None, "agent", "RG-05"),
    ("La régulation dysfonctionne depuis lundi.", "C-12", None, "agent", "RG-05"),
    # --- RG-05: action requested on the dossier -> agent -------------------------
    ("Merci de planifier le remplacement du circulateur.", "C-27", None, "agent", "RG-05"),
    ("Planifiez une visite d'entretien la semaine prochaine.", "C-12", None, "agent", "RG-05"),
    ("Pouvez-vous programmer la visite annuelle ?", "C-12", None, "agent", "RG-05"),
    ("Merci de dépanner la résidence au plus vite.", "C-12", None, "agent", "RG-05"),
    ("Nous avons besoin d'un dépannage aujourd'hui.", "C-12", None, "agent", "RG-05"),
    ("Merci de créer un ticket pour le site de Bron.", "C-12", None, "agent", "RG-05"),
    ("Merci d'ouvrir un ticket.", "C-12", None, "agent", "RG-05"),
    ("Il faut envoyer un technicien sur place.", "C-12", None, "agent", "RG-05"),
    ("Bruit anormal, intervention urgente demandée.", "C-27", None, "agent", "RG-05"),
    ("Pouvez-vous vérifier si notre contrat couvre le brûleur ?", "C-12", None, "agent", "RG-05"),
    ("Vérifiez notre contrat, s'il vous plaît.", "C-12", None, "agent", "RG-05"),
    # --- DEFAULT: documentary questions, including the wordings that fooled v0 ----
    ("Quel est le délai pour un ticket P3 ?", None, None, "rag", "DEFAULT"),
    ("Mon projet est en retard, quel délai ?", None, None, "rag", "DEFAULT"),
    ("Le ticket et le projet : les pièces sont-elles en stock ?", None, None, "rag", "DEFAULT"),
    ("Quel est le délai d'intervention P1 pour la formule Confort ?", None, None, "rag", "DEFAULT"),
    ("Combien coûte une intervention le week-end ?", None, None, "rag", "DEFAULT"),
    ("Comment le client peut-il suivre l'avancement de son ticket ?", None, None, "rag", "DEFAULT"),
    ("Quand le planning de la semaine suivante est-il figé ?", None, None, "rag", "DEFAULT"),
    ("Où les données sont-elles hébergées ?", None, None, "rag", "DEFAULT"),
    ("Entretenez-vous les panneaux rayonnants ?", None, None, "rag", "DEFAULT"),  # not "panne"
    ("Merci de vérifier la pression du circuit.", "C-12", None, "rag", "DEFAULT"),  # no contract
    ("Notre contrat est-il encore valable ?", "C-12", None, "rag", "DEFAULT"),  # no check asked
    ("Les majorations de nuit et de week-end se cumulent-elles ?", None, None, "rag", "DEFAULT"),
    ("Quelle est la recette de la tarte aux pommes ?", None, None, "rag", "DEFAULT"),
]


@pytest.mark.parametrize(("q", "client_id", "montant", "route", "rule"), CASES)
def test_route_and_rule(q, client_id, montant, route, rule):
    decision = run(q, client_id, montant)
    assert (decision.route, decision.rule) == (route, rule)


def test_every_rule_is_covered_by_the_table():
    assert {case[4] for case in CASES} == {"RG-04", "RG-03", "RG-03b", "RG-05", "DEFAULT"}


# --- entities ------------------------------------------------------------------


def test_client_and_amount_are_read_from_the_text():
    decision = run("Bonjour, ici le client c-27 : je conteste la facture de 245,50 €.")
    assert decision.client_id == "C-27"
    assert decision.client_known is True
    assert decision.montant == 245.5


def test_explicit_fields_win_over_the_text():
    decision = run("Le client C-34 conteste une facture de 300 €.", "C-12", 120.0)
    assert decision.client_id == "C-12"
    assert decision.montant == 120.0


def test_unknown_client_is_reported_as_unknown():
    decision = run("Je conteste la facture de 150 €.", "C-99")
    assert decision.client_id == "C-99"
    assert decision.client_known is False
    assert decision.route != "automation"


def test_without_client_nothing_is_assumed():
    decision = run("Je conteste la facture de 150 €.")
    assert decision.client_id is None
    assert decision.client_known is False


def test_several_different_amounts_resolve_to_none():
    decision = run("Facturé 480 € au lieu des 300 € prévus : je conteste.", "C-12")
    assert decision.montant is None
    assert any("2 montants différents" in reason for reason in decision.reasons)


def test_missing_reference_file_never_automates(env):
    (env.reference_dir / "clients.json").unlink()
    reset_settings()
    decision = run("Je conteste la facture de 120 €.", "C-12")
    assert decision.client_known is False
    assert decision.route == "rag"


# --- thresholds ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("montant", "route", "rule"),
    [
        (499.99, "automation", "RG-03"),
        (500.0, "human", "RG-03b"),
        (500.01, "human", "RG-03b"),
        (25000.0, "human", "RG-03b"),
    ],
)
def test_threshold_is_strict(montant, route, rule):
    decision = run("Je conteste cette facture.", "C-12", montant)
    assert (decision.route, decision.rule) == (route, rule)


def test_threshold_comes_from_the_settings(monkeypatch):
    monkeypatch.setenv("RG03_MAX_AMOUNT", "200")
    reset_settings()
    assert run("Je conteste la facture de 199 €.", "C-12").rule == "RG-03"
    assert run("Je conteste la facture de 250 €.", "C-12").rule == "RG-03b"


# --- decision log ----------------------------------------------------------------


def test_reasons_explain_the_decision_in_french():
    decision = run("Je conteste la facture de 120 €.", "C-12")
    assert decision.reasons == [
        "Client : C-12 (champ), connu du référentiel",
        "Montant : 120,00 € (texte)",
        "RG-03 : litige de facturation de 120,00 €, inférieur au seuil de 500,00 €, "
        "pour un client connu : ticket standard",
    ]


def test_reasons_name_every_sensitive_subject():
    decision = run("Je résilie le contrat, mon avocat réclamera des pénalités.", "C-12")
    assert decision.reasons[-1] == (
        "RG-04 : sujet sensible (résiliation, pénalités, contentieux), transmis à un gestionnaire"
    )


def test_reasons_of_the_agent_route_name_what_matched():
    decision = run("La chaudière est en panne, merci d'envoyer un technicien.", "C-12")
    assert decision.reasons[-1] == (
        "RG-05 : panne ou dysfonctionnement, action demandée : dossier confié à l'agent"
    )


@pytest.mark.parametrize(("q", "client_id", "montant", "route", "rule"), CASES)
def test_reasons_never_quote_the_request(q, client_id, montant, route, rule):
    # Words of the request longer than a client id or a number must not leak into the
    # decision log, which is stored and displayed.
    secret = "Dupont-Martin"
    decision = run(f"{q} Signé {secret}, joignable au 06 12 34 56 78.", client_id, montant)
    log = " ".join(decision.reasons)
    assert secret not in log
    assert "06 12 34 56 78" not in log
