"""`handle_ask`: masking, dispatch, request log, failure boundary."""

import pytest

from app.core import llm
from app.core.schemas import AskRequest
from app.core.types import LLMResult
from app.db import repositories, session
from app.retrieval import hybrid, rag
from app.services import ask


def logged(request_id: str):
    with session.session_scope() as db:
        return repositories.get_request(db, request_id)


def test_every_request_is_logged_with_its_route_rule_and_mode(indexed):
    response = ask.handle_ask(AskRequest(q="Je conteste la facture de 120 €.", client_id="C-12"))

    row = logged(response.request_id)
    assert (row.route, row.rule, row.mode) == ("automation", "RG-03", "rule")
    assert row.client_id == "C-12"
    assert row.needs_validation is False
    assert row.error is None
    assert row.latency_ms == response.latency_ms


def test_decision_log_starts_with_masking_then_triage_then_the_route(indexed):
    response = ask.handle_ask(
        AskRequest(q="Quelle majoration le week-end ? Écrivez-moi : client@example.com")
    )

    log = response.decision_log
    assert log[0] == "Données personnelles masquées avant traitement : EMAIL"
    assert log[1] == "Client : non renseigné"
    assert any(line.startswith("DEFAULT") for line in log)
    assert log[-1].startswith("Génération")
    assert "client@example.com" not in " ".join(log)


def test_the_caller_supplied_identifier_is_used_as_request_id(indexed):
    response = ask.handle_ask(AskRequest(q="Quelle majoration le week-end ?"), request_id="abc123")

    assert response.request_id == "abc123"
    assert logged("abc123").route == "rag"


def test_llm_usage_is_reported_and_logged(indexed, monkeypatch):
    def fake_complete(task, messages, **_kwargs):
        return LLMResult(
            text="La majoration week-end est de +35 % [1].",
            model="mistral/mistral-small-latest",
            prompt_tokens=420,
            completion_tokens=18,
            cost_eur=0.00021,
            latency_ms=12,
        )

    monkeypatch.setattr(llm, "is_enabled", lambda task: True)
    monkeypatch.setattr(llm, "complete", fake_complete)

    response = ask.handle_ask(AskRequest(q="Quelle majoration le week-end ?"))

    assert (response.route, response.mode) == ("rag", "llm")
    assert response.usage.llm_calls == 1
    assert response.usage.model == "mistral/mistral-small-latest"
    assert response.usage.cost_eur == 0.00021
    row = logged(response.request_id)
    assert (row.llm_calls, row.prompt_tokens, row.completion_tokens) == (1, 420, 18)
    assert row.cost_eur == 0.00021 and row.model == "mistral/mistral-small-latest"


def test_only_the_masked_text_is_sent_to_the_model(indexed, monkeypatch):
    sent = []

    def fake_complete(task, messages, **_kwargs):
        sent.extend(messages)
        return LLMResult(text="Majoration de +35 % le week-end [1].", model="m")

    monkeypatch.setattr(llm, "is_enabled", lambda task: True)
    monkeypatch.setattr(llm, "complete", fake_complete)

    ask.handle_ask(AskRequest(q="Quelle majoration le week-end ? Mon numéro : 06 39 98 12 34"))

    prompt = " ".join(message["content"] for message in sent)
    assert "[TEL]" in prompt
    assert "39 98" not in prompt


def test_failure_of_a_route_is_logged_and_escalated(indexed, monkeypatch):
    def broken(*_args):
        raise KeyError("internal")

    monkeypatch.setattr(rag, "answer_with_rag", broken)

    response = ask.handle_ask(AskRequest(q="Quelle majoration le week-end ?"))

    assert (response.route, response.mode) == ("human", "none")
    assert "internal" not in response.answer
    assert response.decision_log[-1].startswith("Erreur interne")
    assert logged(response.request_id).error == "KeyError"


def test_missing_index_is_not_swallowed(mini_corpus, db):
    with pytest.raises(hybrid.IndexNotReady):
        ask.handle_ask(AskRequest(q="Quelle majoration le week-end ?"))
