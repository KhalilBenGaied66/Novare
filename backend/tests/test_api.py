"""HTTP API, exercised through the real pipeline (offline: no LLM, hash embeddings)."""

import pytest
from fastapi.testclient import TestClient

from app.agents import dossier_agent
from app.core.config import reset_settings
from app.db import repositories, session
from app.main import create_app

DISPUTE = {"q": "Je conteste la facture de 120 € reçue ce mois-ci.", "client_id": "C-12"}
WEEKEND = {"q": "Quelle est la majoration pour un déplacement le week-end ?"}
BREAKDOWN = {
    "q": "La chaudière est en panne depuis ce matin, merci d'intervenir.",
    "client_id": "C-12",
}
SENSITIVE = {
    "q": "Je veux résilier le contrat, mon avocat vous écrira. Facture de 120 €.",
    "client_id": "C-12",
}
OUT_OF_SCOPE = {"q": "Quelle est la recette de la tarte aux pommes ?"}


def ask(api: TestClient, payload: dict) -> dict:
    response = api.post("/api/v1/ask", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# --- The four routes, end to end -------------------------------------------------------


def test_small_dispute_of_a_known_client_creates_a_ticket_without_llm(api):
    body = ask(api, DISPUTE)

    assert (body["route"], body["mode"]) == ("automation", "rule")
    assert body["needs_validation"] is False
    assert body["usage"]["llm_calls"] == 0
    assert body["ticket_id"] in body["answer"] and "RG-03" in body["answer"]

    ticket = api.get(f"/api/v1/tickets/{body['ticket_id']}").json()
    assert ticket["kind"] == "litige_standard"
    assert ticket["client_id"] == "C-12"
    assert ticket["created_by"] == "system:RG-03"
    assert [t["ticket_id"] for t in api.get("/api/v1/tickets").json()] == [body["ticket_id"]]


def test_documentary_question_is_answered_with_a_quoted_and_cited_passage(api):
    body = ask(api, WEEKEND)

    assert (body["route"], body["mode"]) == ("rag", "extractive")
    assert "+35 %" in body["answer"]
    assert "grille_tarifs.md" in {citation["doc"] for citation in body["citations"]}
    assert body["needs_validation"] is True
    assert body["confidence"] >= 0.35
    assert body["usage"] == {
        "model": None,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_eur": 0.0,
    }


def test_breakdown_goes_to_the_agent_and_the_ticket_exists_only_after_approval(api):
    body = ask(api, BREAKDOWN)

    assert (body["route"], body["mode"]) == ("agent", "deterministic")
    assert "Confort" in body["answer"]
    action = body["proposed_action"]
    assert action["status"] == "pending" and action["ticket_id"] is None
    assert api.get("/api/v1/tickets").json() == []
    assert [a["action_id"] for a in api.get("/api/v1/actions?status=pending").json()] == [
        action["action_id"]
    ]

    approve_url = f"/api/v1/actions/{action['action_id']}/approve"
    approved = api.post(approve_url, json={"validator": "n.durand"}).json()
    assert approved["action"]["status"] == "approved"
    assert approved["ticket"]["kind"] == "intervention"
    assert approved["ticket"]["created_by"] == "n.durand"

    again = api.post(approve_url, json={"validator": "someone.else"})
    assert again.status_code == 200
    assert again.json()["ticket"]["ticket_id"] == approved["ticket"]["ticket_id"]
    assert len(api.get("/api/v1/tickets").json()) == 1

    rejected = api.post(
        f"/api/v1/actions/{action['action_id']}/reject", json={"validator": "n.durand"}
    )
    assert rejected.status_code == 409
    assert "décision inverse" in rejected.json()["detail"]


def test_sensitive_request_is_escalated_even_with_a_small_amount(api):
    body = ask(api, SENSITIVE)

    assert (body["route"], body["mode"]) == ("human", "none")
    assert body["ticket_id"] is None and body["citations"] == []
    assert any(line.startswith("RG-04") for line in body["decision_log"])
    assert api.get("/api/v1/tickets").json() == []


def test_question_outside_the_corpus_is_escalated_without_citation(api):
    body = ask(api, OUT_OF_SCOPE)

    assert (body["route"], body["mode"]) == ("human", "none")
    assert body["citations"] == []
    assert body["confidence"] < 0.35


def test_contract_of_another_client_is_never_quoted(api):
    body = ask(
        api, {"q": "Quelle franchise prévoit le contrat du client C-34 ?", "client_id": "C-12"}
    )

    assert "contrat_C-34.md" not in {citation["doc"] for citation in body["citations"]}
    assert "250" not in body["answer"]


# --- Request validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"q": "ab"},
        {"q": "x" * 2001},
        {"q": "Question valide", "client_id": "12"},
        {"q": "Question valide", "client_id": "C-1234567"},
        {"q": "Question valide", "montant": 0},
        {"q": "Question valide", "montant": -5},
        {},
    ],
)
def test_invalid_requests_are_rejected(api, payload):
    assert api.post("/api/v1/ask", json=payload).status_code == 422


def test_no_get_route_takes_free_text(api):
    assert api.get("/api/v1/ask", params={"q": "Quel délai ?"}).status_code == 405


# --- Identifier, personal data, storage --------------------------------------------------


def test_request_id_is_generated_by_the_server_and_echoed(api):
    response = api.post("/api/v1/ask", json=WEEKEND, headers={"X-Request-ID": "chosen-by-caller"})

    request_id = response.json()["request_id"]
    assert response.headers["X-Request-ID"] == request_id
    assert request_id != "chosen-by-caller" and len(request_id) == 32


def test_personal_data_never_reaches_the_request_log_or_the_ticket(api):
    body = ask(
        api,
        {
            "q": "Je conteste la facture de 120 €. Joignez-moi au 06 39 98 12 34, "
            "IBAN FR76 3000 6000 0112 3456 7890 189.",
            "client_id": "C-12",
        },
    )

    assert body["pii_redacted"] == ["IBAN", "TEL"]
    with session.session_scope() as db:
        row = repositories.get_request(db, body["request_id"])
        assert "[TEL]" in row.q_masked and "[IBAN]" in row.q_masked
        assert "39 98" not in row.q_masked and "FR76" not in row.q_masked
        assert row.pii_types == ["IBAN", "TEL"]
    ticket = api.get(f"/api/v1/tickets/{body['ticket_id']}").json()
    assert "39 98" not in ticket["body"] and "FR76" not in ticket["body"]


def test_masking_tags_do_not_lower_the_retrieval_confidence(api):
    plain = ask(api, WEEKEND)
    with_phone = ask(api, {"q": WEEKEND["q"] + " Mon numéro : 06 39 98 12 34."})

    assert with_phone["route"] == "rag"
    assert with_phone["pii_redacted"] == ["TEL"]
    assert with_phone["confidence"] >= plain["confidence"] - 0.2


# --- Failure handling ------------------------------------------------------------------


def test_failure_inside_a_route_becomes_an_escalation_without_details(api, monkeypatch):
    def broken(*_args):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(dossier_agent, "run_agent", broken)
    body = ask(api, BREAKDOWN)

    assert (body["route"], body["mode"]) == ("human", "none")
    assert "secret internal detail" not in str(body)
    with session.session_scope() as db:
        assert repositories.get_request(db, body["request_id"]).error == "RuntimeError"


def test_missing_index_gives_503_with_a_french_message(mini_corpus, db):
    with TestClient(create_app()) as api:
        ready = api.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["ready"] is False
        assert ready.json()["checks"]["database"] == "ok"

        response = api.post("/api/v1/ask", json=WEEKEND)
        assert response.status_code == 503
        assert "POST /api/v1/ingest" in response.json()["detail"]

        ingest = api.post("/api/v1/ingest").json()
        assert ingest["docs"] == 4 and ingest["chunks"] > 0
        assert api.get("/ready").json()["ready"] is True
        assert ask(api, WEEKEND)["route"] == "rag"


def test_index_is_built_at_startup_when_auto_ingest_is_on(mini_corpus, db, monkeypatch):
    monkeypatch.setenv("AUTO_INGEST", "true")
    reset_settings()
    with TestClient(create_app()) as api:
        assert api.get("/ready").json()["ready"] is True


# --- Authentication and rate limit -----------------------------------------------------


def test_api_key_protects_the_api_but_not_the_health_checks(indexed, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    reset_settings()
    with TestClient(create_app()) as api:
        assert api.get("/health").json() == {"status": "ok"}
        assert api.get("/ready").status_code == 200

        missing = api.post("/api/v1/ask", json=WEEKEND)
        assert missing.status_code == 401
        assert missing.json() == {"detail": "Clé API invalide ou absente"}
        assert api.get("/api/v1/metrics", headers={"X-API-Key": "wrong"}).status_code == 401

        ok = api.post("/api/v1/ask", json=WEEKEND, headers={"X-API-Key": "test-key"})
        assert ok.status_code == 200


def test_rate_limit_returns_429_with_retry_after(indexed, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    reset_settings()
    with TestClient(create_app()) as api:
        assert api.get("/api/v1/metrics").status_code == 200
        assert api.get("/api/v1/metrics").status_code == 200
        limited = api.get("/api/v1/metrics")
        assert limited.status_code == 429
        assert int(limited.headers["Retry-After"]) >= 1
        assert api.get("/health").status_code == 200


# --- Tickets, actions, feedback --------------------------------------------------------


def test_unknown_ticket_and_action_give_404(api):
    assert api.get("/api/v1/tickets/T-000999").json() == {"detail": "Ticket introuvable"}
    unknown = api.post("/api/v1/actions/nope/approve", json={"validator": "n.durand"})
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Action introuvable"}


def test_rejecting_an_action_creates_no_ticket(api):
    action = ask(api, BREAKDOWN)["proposed_action"]

    rejected = api.post(
        f"/api/v1/actions/{action['action_id']}/reject",
        json={"validator": "n.durand", "reason": "Doublon du ticket précédent"},
    ).json()

    assert rejected["action"]["status"] == "rejected" and rejected["ticket"] is None
    assert api.get("/api/v1/tickets").json() == []


def test_feedback_is_linked_to_the_request_and_counted_by_route(api):
    rag = ask(api, WEEKEND)
    agent = ask(api, BREAKDOWN)

    saved = api.post(
        "/api/v1/feedback",
        json={"request_id": rag["request_id"], "ok": True, "comment": "Exact, tél 06 39 98 12 34"},
    ).json()
    api.post("/api/v1/feedback", json={"request_id": agent["request_id"], "ok": False})
    unknown = api.post("/api/v1/feedback", json={"request_id": "unknown", "ok": True})

    assert saved == {"saved": True, "request_id": rag["request_id"], "route": "rag"}
    assert unknown.status_code == 404
    assert api.get("/api/v1/feedback/stats").json() == {
        "n": 2,
        "ok_rate": 0.5,
        "by_route": {"agent": {"n": 1, "ok": 0}, "rag": {"n": 1, "ok": 1}},
    }
    with session.session_scope() as db:
        from app.db.models import FeedbackRow

        comments = [row.comment for row in db.query(FeedbackRow).all()]
    assert "Exact, tél [TEL]" in comments
