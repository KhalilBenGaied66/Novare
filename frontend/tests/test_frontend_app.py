"""Streamlit page driven through AppTest, with the API client replaced by a recorder.

The page must send exactly what the form shows: an empty client or amount is `None`,
never a default client and never `0.0`. Those arguments are asserted literally here.
"""

from pathlib import Path

import api_client
import pytest
import streamlit_app
from streamlit.components.v2 import manifest_scanner
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"

DISPUTE_Q = "Je conteste ma facture du mois de mars : le déplacement a été facturé deux fois."
WEEKEND_Q = "Quelle est la majoration appliquée pour un déplacement le week-end ?"
BOILER_Q = (
    "La chaudière de l'immeuble est en panne depuis ce matin. "
    "Pouvez-vous planifier une intervention ?"
)

NO_LLM_USAGE = {
    "model": None,
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost_eur": 0.0,
}
WEEKEND_CITATION = {
    "ref": 1,
    "doc": "grille_tarifaire_2026.md",
    "title": "Grille tarifaire 2026",
    "page": 1,
    "section": "Déplacements",
    "chunk_id": "grille_tarifaire_2026.md#p1-1",
    "excerpt": "Majoration week-end : +35 % sur le déplacement et la main-d'œuvre.",
    "score": 0.0328,
}
PENDING_ACTION = {
    "action_id": "9f1c2ab4",
    "type": "create_ticket",
    "status": "pending",
    "payload": {
        "subject": "Panne de chaudière, immeuble Lumière",
        "summary": "Chaudière à l'arrêt depuis ce matin, intervention à planifier.",
        "priority": "P1",
        "client_id": "C-12",
    },
    "ticket_id": None,
}


def ask_response(**overrides) -> dict:
    """An `AskResponse` payload as the API serialises it (documentary answer, no LLM)."""
    response = {
        "request_id": "req-0001",
        "route": "rag",
        "mode": "extractive",
        "answer": (
            "Extrait des documents, sans reformulation par un LLM :\n"
            "Majoration week-end : +35 % sur le déplacement et la main-d'œuvre. [1]"
        ),
        "citations": [WEEKEND_CITATION],
        "confidence": 0.72,
        "needs_validation": True,
        "proposed_action": None,
        "ticket_id": None,
        "decision_log": [
            "Aucune donnée personnelle détectée",
            "Triage : règle DEFAULT, route rag",
            "Recherche hybride : 4 sources, confiance 0,72",
        ],
        "pii_redacted": [],
        "usage": NO_LLM_USAGE,
        "latency_ms": 42,
        "retrieval_mode": "hybrid",
    }
    response.update(overrides)
    return response


def agent_response() -> dict:
    return ask_response(
        request_id="req-0002",
        route="agent",
        mode="llm",
        answer="Bonjour,\nUne intervention est proposée sous 4 h ouvrées [1].",
        proposed_action=PENDING_ACTION,
        pii_redacted=["EMAIL", "TEL"],
        usage={
            "model": "anthropic/claude-sonnet-5-5",
            "llm_calls": 3,
            "prompt_tokens": 1840,
            "completion_tokens": 312,
            "cost_eur": 0.0123,
        },
        latency_ms=2310,
    )


class FakeApi:
    """Replaces the `api_client` functions; records every call in order."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.ask_result = ask_response()
        self.ready_result = {"ready": True, "checks": {"database": "ok", "index": "ok"}}
        self.failures: dict[str, str] = {}  # function name -> ApiError message to raise

    def _record(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if name in self.failures:
            raise api_client.ApiError(self.failures[name])

    def args_of(self, name: str) -> list[tuple]:
        return [call[1:] for call in self.calls if call[0] == name]

    def ask(self, q, client_id, montant) -> dict:
        self._record("ask", q, client_id, montant)
        return self.ask_result

    def approve(self, action_id, validator) -> dict:
        self._record("approve", action_id, validator)
        decided = {**PENDING_ACTION, "status": "approved", "ticket_id": "T-000042"}
        return {"action": decided, "ticket": {"ticket_id": "T-000042"}}

    def reject(self, action_id, validator, reason) -> dict:
        self._record("reject", action_id, validator, reason)
        return {"action": {**PENDING_ACTION, "status": "rejected"}, "ticket": None}

    def feedback(self, request_id, ok, comment) -> dict:
        self._record("feedback", request_id, ok, comment)
        return {"saved": True, "request_id": request_id, "route": "rag"}

    def ready(self) -> dict:
        self._record("ready")
        return self.ready_result

    def metrics(self) -> dict:
        self._record("metrics")
        return {
            "requests": 12,
            "by_route": {"automation": 3, "rag": 5, "agent": 2, "human": 2},
            "by_mode": {"rule": 3, "extractive": 5, "deterministic": 2, "none": 2},
            "latency_ms_p50": 38,
            "latency_ms_p95": 410,
            "cost_eur_total": 0.0,
            "cost_eur_avg": 0.0,
            "llm_call_share": 0.0,
            "escalation_rate": 0.1667,
            "errors": 1,
            "feedback": {"n": 4, "ok_rate": 0.75, "by_route": {"rag": {"n": 4, "ok": 3}}},
            "pending_actions": 2,
        }


@pytest.fixture(autouse=True)
def no_component_scan(monkeypatch):
    """Skip Streamlit's search for custom components when a page starts.

    Every new AppTest reads the metadata of all installed packages to find custom
    components (about 0.25 s each time). The page uses none, and without the scan the
    whole file runs in a few seconds instead of fifteen.
    """
    monkeypatch.setattr(manifest_scanner, "scan_component_manifests", lambda: [])


@pytest.fixture
def fake_api(monkeypatch) -> FakeApi:
    fake = FakeApi()
    for name in ("ask", "approve", "reject", "feedback", "ready", "metrics"):
        monkeypatch.setattr(api_client, name, getattr(fake, name))
    return fake


def start_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    return app


def request_fields(app: AppTest) -> tuple:
    """The request, client and amount widgets: the first ones of the page.

    They are found by position because their keys change with every example loaded.
    """
    return app.text_area[0], app.text_input[0], app.text_input[1]


def submit(app: AppTest, q: str | None = None, client: str | None = None, amount=None) -> None:
    """Type into the given fields (None = leave the field as it is) and submit the form."""
    q_field, client_field, amount_field = request_fields(app)
    if q is not None:
        q_field.input(q)
    if client is not None:
        client_field.input(client)
    if amount is not None:
        amount_field.input(amount)
    app.button(key="submit_ask").click().run()
    assert not app.exception


def form_values(app: AppTest) -> tuple[str, str, str]:
    q_field, client_field, amount_field = request_fields(app)
    assert (q_field.label, client_field.label, amount_field.label) == (
        "Demande du client",
        "Identifiant client (facultatif)",
        "Montant du litige en € (facultatif)",
    )
    return q_field.value, client_field.value, amount_field.value


def main_texts(app: AppTest) -> list[str]:
    """Every caption and Markdown block of the main area (badges are Markdown)."""
    return [element.value for element in [*app.main.markdown, *app.main.caption]]


# --- What the form sends -------------------------------------------------------------


def test_form_is_empty_and_nothing_is_sent_on_first_display(fake_api):
    app = start_app()

    assert form_values(app) == ("", "", "")
    assert fake_api.args_of("ask") == []
    assert not app.main.subheader  # no result section before a request


def test_question_alone_sends_null_client_and_null_amount(fake_api):
    app = start_app()

    submit(app, q=WEEKEND_Q)

    assert fake_api.args_of("ask") == [(WEEKEND_Q, None, None)]
    _, client_id, montant = fake_api.args_of("ask")[0]
    assert client_id is None
    assert montant is None


EXAMPLE_CASES = [
    # button key, values shown in the form, exact arguments sent to the API
    ("example_0", (DISPUTE_Q, "C-12", "120"), (DISPUTE_Q, "C-12", 120.0)),
    ("example_1", (WEEKEND_Q, "", ""), (WEEKEND_Q, None, None)),
    ("example_2", (BOILER_Q, "C-12", ""), (BOILER_Q, "C-12", None)),
]


@pytest.mark.parametrize(("button_key", "shown", "sent"), EXAMPLE_CASES)
def test_example_button_fills_the_form_and_sends_exactly_those_values(
    fake_api, button_key, shown, sent
):
    app = start_app()

    app.button(key=button_key).click().run()

    assert form_values(app) == shown
    assert fake_api.args_of("ask") == []  # an example fills the form, it does not send it

    submit(app)

    assert fake_api.args_of("ask") == [sent]
    assert type(fake_api.args_of("ask")[0][2]) is type(sent[2])  # float or None, never 0.0


def test_the_three_examples_send_three_different_requests(fake_api):
    app = start_app()

    for button_key, _, _ in EXAMPLE_CASES:
        app.button(key=button_key).click().run()
        submit(app)

    assert fake_api.args_of("ask") == [sent for _, _, sent in EXAMPLE_CASES]


def test_example_without_client_erases_the_client_and_amount_of_the_previous_one(fake_api):
    app = start_app()
    app.button(key="example_0").click().run()
    assert form_values(app) == (DISPUTE_Q, "C-12", "120")

    app.button(key="example_1").click().run()
    submit(app)

    assert form_values(app) == (WEEKEND_Q, "", "")
    assert fake_api.args_of("ask") == [(WEEKEND_Q, None, None)]


def test_example_replaces_values_typed_by_hand(fake_api):
    app = start_app()
    submit(app, q="Bonjour, où en est mon dossier ?", client="C-27", amount="80")

    app.button(key="example_2").click().run()
    submit(app)

    assert fake_api.args_of("ask")[-1] == (BOILER_Q, "C-12", None)


def test_example_replaces_values_typed_but_not_yet_sent(fake_api):
    app = start_app()
    _, client_field, amount_field = request_fields(app)
    client_field.input("C-27")
    amount_field.input("80")

    app.button(key="example_1").click().run()
    submit(app)

    assert form_values(app) == (WEEKEND_Q, "", "")
    assert fake_api.args_of("ask") == [(WEEKEND_Q, None, None)]


def test_each_example_is_shown_in_a_new_form(fake_api):
    """An example must not be assigned to the existing fields.

    In a browser, a field of a form that was edited but not submitted keeps the typed
    value: assigning an example to it changes what is displayed, not what is sent.
    AppTest has no browser and cannot show that difference, so this test pins the
    mechanism that avoids it: every example gets new widgets (new keys).
    """
    app = start_app()
    assert [field.key for field in request_fields(app)] == ["q_0", "client_id_0", "montant_0"]

    app.button(key="example_0").click().run()
    assert [field.key for field in request_fields(app)] == ["q_1", "client_id_1", "montant_1"]

    app.button(key="example_0").click().run()
    assert [field.key for field in request_fields(app)] == ["q_2", "client_id_2", "montant_2"]
    assert form_values(app) == (DISPUTE_Q, "C-12", "120")


def test_example_values_can_be_edited_before_sending(fake_api):
    app = start_app()
    app.button(key="example_0").click().run()

    submit(app, client="C-27", amount="")

    assert fake_api.args_of("ask") == [(DISPUTE_Q, "C-27", None)]


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("120,50", 120.5),
        ("120", 120.0),
        ("120.50", 120.5),
        ("1 250,50", 1250.5),
        ("89 €", 89.0),
    ],
)
def test_typed_amount_is_sent_as_a_number(fake_api, typed, expected):
    app = start_app()

    submit(app, q=DISPUTE_Q, client="C-12", amount=typed)

    assert fake_api.args_of("ask") == [(DISPUTE_Q, "C-12", expected)]
    assert isinstance(fake_api.args_of("ask")[0][2], float)


@pytest.mark.parametrize("typed", ["abc", "0", "0,00", "-5", "12,345", "1.250,50", "1e3", "nan"])
def test_unreadable_amount_is_refused_and_nothing_is_sent(fake_api, typed):
    app = start_app()

    submit(app, q=DISPUTE_Q, client="C-12", amount=typed)

    assert fake_api.args_of("ask") == []
    assert len(app.main.error) == 1
    assert app.main.error[0].value.startswith("Montant invalide.")


def test_client_is_trimmed_and_blank_client_is_null(fake_api):
    app = start_app()

    submit(app, q=WEEKEND_Q, client="  C-12 ")
    submit(app, client="   ", amount="  ")

    assert fake_api.args_of("ask") == [(WEEKEND_Q, "C-12", None), (WEEKEND_Q, None, None)]


def test_empty_question_is_not_sent(fake_api):
    app = start_app()

    submit(app, q="   ", client="C-12", amount="120")

    assert fake_api.args_of("ask") == []
    assert app.main.warning[0].value == (
        "Saisissez la demande du client avant de lancer le traitement."
    )


# --- Result display ------------------------------------------------------------------


def test_documentary_answer_is_displayed_with_its_sources(fake_api):
    app = start_app()

    submit(app, q=WEEKEND_Q)

    texts = main_texts(app)
    assert ":blue-badge[Route : Réponse documentaire]" in texts
    assert ":gray-badge[Mode : extraits des documents, sans LLM]" in texts
    assert "Demande envoyée avec : client non renseigné ; montant du litige non renseigné." in texts
    # The answer keeps its two lines (Markdown hard break).
    assert (
        "Extrait des documents, sans reformulation par un LLM :  \n"
        "Majoration week-end : +35 % sur le déplacement et la main-d'œuvre. [1]"
    ) in texts
    assert "Réponse à relire par un gestionnaire avant tout envoi au client." in texts
    assert "Aucun LLM utilisé pour cette demande : coût nul." in texts
    assert "Recherche documentaire : hybride (lexicale et vectorielle)." in texts
    assert "Identifiant de la demande : req-0001" in texts

    metrics = {metric.label: metric.value for metric in app.main.metric}
    assert metrics == {"Confiance": "72 %", "Latence": "42 ms", "Coût LLM": "0,0000 €"}

    labels = [expander.label for expander in app.main.expander]
    assert labels == [
        "[1] Grille tarifaire 2026 — grille_tarifaire_2026.md, p. 1",
        "Journal de décision (3 étapes)",
    ]
    source, decision_log = app.main.expander
    assert source.caption[0].value == "Section : Déplacements"
    assert source.text[0].value == WEEKEND_CITATION["excerpt"]
    assert decision_log.text[0].value == (
        "1. Aucune donnée personnelle détectée\n"
        "2. Triage : règle DEFAULT, route rag\n"
        "3. Recherche hybride : 4 sources, confiance 0,72"
    )


def test_llm_usage_and_masked_personal_data_are_displayed(fake_api):
    fake_api.ask_result = agent_response()
    app = start_app()

    submit(app, q=BOILER_Q, client="C-12")

    texts = main_texts(app)
    assert ":violet-badge[Route : Agent de traitement de dossier]" in texts
    assert ":gray-badge[Mode : rédaction par un LLM]" in texts
    assert "Demande envoyée avec : client C-12 ; montant du litige non renseigné." in texts
    assert (
        "Modèle anthropic/claude-sonnet-5-5 : 3 appel(s), 1840 jetons en entrée, "
        "312 en sortie, coût 0,0123 €."
    ) in texts
    assert "Données personnelles masquées avant traitement : EMAIL, TEL." in texts
    metrics = {metric.label: metric.value for metric in app.main.metric}
    assert metrics == {"Confiance": "72 %", "Latence": "2310 ms", "Coût LLM": "0,0123 €"}


def test_automation_result_shows_the_ticket_and_no_action_form(fake_api):
    fake_api.ask_result = ask_response(
        route="automation",
        mode="rule",
        answer="Litige enregistré selon la règle RG-03 : ticket T-000007.",
        citations=[],
        confidence=None,
        needs_validation=False,
        ticket_id="T-000007",
        retrieval_mode="none",
    )
    app = start_app()

    submit(app, q=DISPUTE_Q, client="C-12", amount="120")

    texts = main_texts(app)
    assert ":green-badge[Route : Automatisation par règle métier]" in texts
    assert ":gray-badge[Mode : règle métier, sans LLM]" in texts
    assert "Demande envoyée avec : client C-12 ; montant du litige 120,00 €." in texts
    assert "Aucune source citée." in texts
    assert "Réponse à relire par un gestionnaire avant tout envoi au client." not in texts
    assert [message.value for message in app.main.success] == [
        "Ticket T-000007 créé automatiquement."
    ]
    assert {metric.label: metric.value for metric in app.main.metric}["Confiance"] == "—"
    assert [button.key for button in app.button if button.key == "approve_action"] == []


def test_api_unreachable_message_is_shown_and_no_result_is_displayed(fake_api):
    message = "API injoignable (http://localhost:8000). Vérifiez que le service est démarré."
    fake_api.failures["ask"] = message
    app = start_app()

    submit(app, q=WEEKEND_Q)

    assert [error.value for error in app.main.error] == [message]
    assert not app.main.subheader
    assert form_values(app) == (WEEKEND_Q, "", "")  # the request is kept for a retry


def test_failed_request_removes_the_previous_result(fake_api):
    app = start_app()
    submit(app, q=WEEKEND_Q)
    assert app.main.subheader[0].value == "Résultat"

    fake_api.failures["ask"] = "L'API n'a pas répondu dans le délai imparti."
    submit(app, q=BOILER_Q)

    assert not app.main.subheader
    assert app.main.error[0].value == "L'API n'a pas répondu dans le délai imparti."


def test_loading_an_example_removes_the_previous_result(fake_api):
    app = start_app()
    submit(app, q=WEEKEND_Q)
    assert app.main.subheader[0].value == "Résultat"

    app.button(key="example_2").click().run()

    assert not app.main.subheader


# --- Proposed action: approve / reject -----------------------------------------------


def start_with_pending_action(fake_api: FakeApi) -> AppTest:
    fake_api.ask_result = agent_response()
    app = start_app()
    submit(app, q=BOILER_Q, client="C-12")
    return app


def test_pending_action_is_displayed_and_nothing_is_decided_by_default(fake_api):
    app = start_with_pending_action(fake_api)

    assert [subheader.value for subheader in app.main.subheader] == [
        "Résultat",
        "Action proposée : création d'un ticket",
        "Votre avis sur cette réponse",
    ]
    assert (
        "Client : C-12\n"
        "Priorité : P1\n"
        "Objet : Panne de chaudière, immeuble Lumière\n"
        "Résumé : Chaudière à l'arrêt depuis ce matin, intervention à planifier."
    ) in [text.value for text in app.main.text]
    assert app.button(key="approve_action").label == "Valider la création du ticket"
    assert app.button(key="reject_action").label == "Refuser"
    assert fake_api.args_of("approve") == []
    assert fake_api.args_of("reject") == []


def test_approve_sends_the_validator_and_shows_the_created_ticket(fake_api):
    app = start_with_pending_action(fake_api)

    app.text_input(key="validator_9f1c2ab4").input(" Marie Durand ")
    app.button(key="approve_action").click().run()

    assert not app.exception
    assert fake_api.args_of("approve") == [("9f1c2ab4", "Marie Durand")]
    assert fake_api.args_of("reject") == []
    assert [message.value for message in app.main.success] == [
        "Ticket T-000042 créé après validation."
    ]
    # Once decided, the form is gone: the proposal cannot be decided twice from the page.
    assert [button.key for button in app.button if button.key.endswith("_action")] == []


def test_reject_sends_validator_and_reason_and_creates_no_ticket(fake_api):
    app = start_with_pending_action(fake_api)

    app.text_input(key="validator_9f1c2ab4").input("Marie Durand")
    app.text_input(key="reason_9f1c2ab4").input("Doublon du ticket T-000031")
    app.button(key="reject_action").click().run()

    assert not app.exception
    assert fake_api.args_of("reject") == [
        ("9f1c2ab4", "Marie Durand", "Doublon du ticket T-000031")
    ]
    assert fake_api.args_of("approve") == []
    assert [message.value for message in app.main.info] == [
        "Proposition refusée : aucun ticket n'a été créé."
    ]
    assert not app.main.success


@pytest.mark.parametrize("button_key", ["approve_action", "reject_action"])
def test_decision_without_validator_name_is_not_sent(fake_api, button_key):
    app = start_with_pending_action(fake_api)

    app.button(key=button_key).click().run()

    assert fake_api.args_of("approve") == []
    assert fake_api.args_of("reject") == []
    assert app.main.warning[0].value == (
        "Indiquez le nom du valideur avant de valider ou de refuser."
    )


def test_refused_decision_shows_the_api_message_and_keeps_the_form(fake_api):
    fake_api.failures["approve"] = "Cette action a déjà été refusée."
    app = start_with_pending_action(fake_api)

    app.text_input(key="validator_9f1c2ab4").input("Marie Durand")
    app.button(key="approve_action").click().run()

    assert [error.value for error in app.main.error] == ["Cette action a déjà été refusée."]
    assert not app.main.success
    assert app.button(key="approve_action").label == "Valider la création du ticket"


def test_validator_name_is_proposed_again_for_the_next_action(fake_api):
    app = start_with_pending_action(fake_api)
    app.text_input(key="validator_9f1c2ab4").input("Marie Durand")
    app.button(key="approve_action").click().run()

    next_action = {**PENDING_ACTION, "action_id": "77aa01"}
    fake_api.ask_result = ask_response(route="agent", proposed_action=next_action)
    submit(app)

    assert app.text_input(key="validator_77aa01").value == "Marie Durand"
    assert not app.main.success  # the outcome of the previous action is not shown again


# --- Feedback ------------------------------------------------------------------------


def test_positive_feedback_sends_request_id_and_comment(fake_api):
    app = start_app()
    submit(app, q=WEEKEND_Q)

    app.text_area(key="comment_req-0001").input("Réponse exacte, source correcte.")
    app.button(key="feedback_ok").click().run()

    assert not app.exception
    assert fake_api.args_of("feedback") == [("req-0001", True, "Réponse exacte, source correcte.")]
    assert [message.value for message in app.main.success] == ["Avis enregistré. Merci."]
    assert [button.key for button in app.button if button.key.startswith("feedback_")] == []


def test_negative_feedback_without_comment(fake_api):
    app = start_app()
    submit(app, q=WEEKEND_Q)

    app.button(key="feedback_ko").click().run()

    assert fake_api.args_of("feedback") == [("req-0001", False, "")]
    assert [message.value for message in app.main.success] == ["Avis enregistré. Merci."]


def test_feedback_failure_is_reported_and_the_form_stays(fake_api):
    fake_api.failures["feedback"] = "Demande inconnue."
    app = start_app()
    submit(app, q=WEEKEND_Q)

    app.button(key="feedback_ok").click().run()

    assert [error.value for error in app.main.error] == ["Demande inconnue."]
    assert not app.main.success
    assert app.button(key="feedback_ok").label == "Réponse correcte (OK)"


def test_feedback_can_be_given_again_for_a_new_request(fake_api):
    app = start_app()
    submit(app, q=WEEKEND_Q)
    app.button(key="feedback_ok").click().run()

    fake_api.ask_result = ask_response(request_id="req-0009")
    submit(app)
    app.button(key="feedback_ko").click().run()

    assert fake_api.args_of("feedback") == [("req-0001", True, ""), ("req-0009", False, "")]


# --- Sidebar -------------------------------------------------------------------------


def test_sidebar_shows_readiness_and_key_metrics(fake_api):
    app = start_app()

    assert [message.value for message in app.sidebar.success] == ["API prête"]
    assert {metric.label: metric.value for metric in app.sidebar.metric} == {
        "Demandes": "12",
        "Escalade humaine": "17 %",
        "Latence médiane": "38 ms",
        "Latence p95": "410 ms",
        "Demandes avec LLM": "0 %",
        "Actions en attente": "2",
        "Avis positifs": "75 %",
        "Erreurs": "1",
    }
    captions = [caption.value for caption in app.sidebar.caption]
    assert "database : ok" in captions
    assert "Coût LLM total : 0,0000 €. Avis reçus : 4." in captions
    assert "Réponse documentaire : 5" in captions


def test_sidebar_reports_an_api_that_is_not_ready(fake_api):
    fake_api.ready_result = {"ready": False, "checks": {"database": "ok", "index": "absent"}}

    app = start_app()

    assert [message.value for message in app.sidebar.warning] == ["API non prête"]
    assert not app.sidebar.success
    assert "index : absent" in [caption.value for caption in app.sidebar.caption]


def test_sidebar_reports_an_unreachable_api_without_asking_for_metrics(fake_api):
    message = "API injoignable (http://localhost:8000). Vérifiez que le service est démarré."
    fake_api.failures["ready"] = message

    app = start_app()

    assert [error.value for error in app.sidebar.error] == [message]
    assert fake_api.args_of("metrics") == []
    assert not app.sidebar.metric
    assert form_values(app) == ("", "", "")  # the form is still usable


def test_sidebar_reports_a_metrics_failure(fake_api):
    fake_api.failures["metrics"] = "Clé API invalide ou absente."

    app = start_app()

    assert [message.value for message in app.sidebar.success] == ["API prête"]
    assert [message.value for message in app.sidebar.warning] == ["Clé API invalide ou absente."]
    assert not app.sidebar.metric


def test_metrics_are_read_after_the_request_so_that_they_include_it(fake_api):
    app = start_app()
    fake_api.calls.clear()

    submit(app, q=WEEKEND_Q)

    assert [call[0] for call in fake_api.calls] == ["ask", "ready", "metrics"]


# --- Pure helpers --------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\t", " "])
def test_parse_amount_empty_field_is_none_not_zero(text):
    assert streamlit_app.parse_amount(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("120", 120.0),
        ("120,50", 120.5),
        ("120.5", 120.5),
        (" 120,50 € ", 120.5),
        ("1 250,50", 1250.5),
        ("1 250,50 €", 1250.5),
        ("0,01", 0.01),
    ],
)
def test_parse_amount_accepts_french_notation(text, expected):
    assert streamlit_app.parse_amount(text) == expected


@pytest.mark.parametrize(
    "text", ["abc", "0", "0,0", "-12", "+12", "12,", ",5", "12,345", "1.250,50", "1e3", "inf", "€"]
)
def test_parse_amount_refuses_everything_else(text):
    with pytest.raises(ValueError):
        streamlit_app.parse_amount(text)


def test_examples_are_the_three_documented_ones():
    assert [(e.q, e.client_id, e.montant) for e in streamlit_app.EXAMPLES] == [
        (DISPUTE_Q, "C-12", "120"),
        (WEEKEND_Q, "", ""),
        (BOILER_Q, "C-12", ""),
    ]


def test_formatting_helpers():
    assert streamlit_app.format_number(1250.5, 2) == "1250,50"
    assert streamlit_app.format_percent(None) == "—"
    assert streamlit_app.format_percent(0.1667) == "17 %"
    assert streamlit_app.format_ms(None) == "—"
    assert streamlit_app.format_ms(42) == "42 ms"
