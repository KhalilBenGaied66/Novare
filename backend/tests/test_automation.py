"""RG-03 automation: ticket creation, duplicate protection, wording, and no model call."""

import re
from datetime import timedelta

import pytest

from app.agents.triage import triage
from app.core import pii
from app.core.schemas import AskRequest
from app.core.types import TriageDecision
from app.db import repositories
from app.db.models import utcnow
from app.db.session import session_scope
from app.retrieval import hybrid
from app.retrieval.hybrid import IndexNotReady
from app.services.automation import handle_automation
from app.services.context import RequestContext

pytestmark = pytest.mark.usefixtures("mini_corpus", "db")

DISPUTE = "Je conteste la facture F-2026-0412 de 120 € : le déplacement a été compté deux fois."


def submit(q: str = DISPUTE, client_id: str | None = "C-12", montant: float | None = None):
    """Run a request through PII masking, triage and the automation, as the API does."""
    req = AskRequest(q=q, client_id=client_id, montant=montant)
    ctx = RequestContext()
    ctx.q_masked, ctx.pii_types = pii.redact(req.q)
    decision = triage(req)
    assert decision.route == "automation"
    return handle_automation(req, decision, ctx), ctx


def all_tickets():
    with session_scope() as session:
        return repositories.list_tickets(session)


def test_creates_a_standard_dispute_ticket():
    response, ctx = submit()

    assert response.route == "automation"
    assert response.mode == "rule"
    assert response.needs_validation is False
    assert response.retrieval_mode == "none"
    assert response.citations == []
    assert response.confidence is None
    assert response.proposed_action is None
    assert response.request_id == ctx.request_id
    assert response.ticket_id == "T-000001"

    (ticket,) = all_tickets()
    assert ticket.ticket_id == "T-000001"
    assert ticket.client_id == "C-12"
    assert ticket.subject == "Litige facturation 120.00 € — C-12"
    assert ticket.body == DISPUTE
    assert ticket.kind == "litige_standard"
    assert ticket.source == "automation"
    assert ticket.created_by == "system:RG-03"
    assert ticket.status == "open"
    assert ticket.priority is None
    assert ticket.request_id == ctx.request_id
    assert re.fullmatch(r"[0-9a-f]{40}", ticket.dedupe_key)


def test_answer_names_the_ticket_the_rule_and_the_amount():
    response, ctx = submit()
    assert response.answer == (
        "Votre contestation de 120,00 € est enregistrée : le ticket T-000001 a été créé pour "
        "le client C-12. Un litige de facturation inférieur à 500,00 € suit la procédure "
        "standard, sans validation préalable (règle RG-03)."
    )
    assert ctx.decision_log == ["Automatisation RG-03 : ticket T-000001 créé (litige standard)"]


def test_same_request_twice_reuses_the_ticket():
    first, _ = submit()
    second, ctx = submit()

    assert second.ticket_id == first.ticket_id == "T-000001"
    assert len(all_tickets()) == 1
    assert second.route == "automation"
    assert second.needs_validation is False
    assert "déjà enregistrée" in second.answer
    assert "T-000001" in second.answer
    assert "RG-03" in second.answer
    assert "aucun nouveau ticket" in second.answer
    assert ctx.decision_log == [
        "Automatisation RG-03 : doublon détecté, ticket T-000001 réutilisé "
        "(même demande dans les 7 derniers jours)"
    ]


def test_same_words_in_another_order_are_a_duplicate():
    first, _ = submit(DISPUTE)
    second, _ = submit(
        "Le déplacement a été compté deux fois : je conteste la facture F-2026-0412 de 120 €."
    )
    assert second.ticket_id == first.ticket_id
    assert len(all_tickets()) == 1


@pytest.mark.parametrize(
    ("q", "client_id", "montant"),
    [
        (DISPUTE, "C-27", None),  # another client
        (DISPUTE.replace("120 €", "130 €"), "C-12", None),  # another amount
        ("Je conteste la facture de 120 € : la majoration de nuit est injustifiée.", "C-12", None),
    ],
)
def test_a_different_dispute_gets_its_own_ticket(q, client_id, montant):
    first, _ = submit()
    second, ctx = submit(q, client_id, montant)
    assert second.ticket_id == "T-000002" != first.ticket_id
    assert len(all_tickets()) == 2
    assert "créé" in ctx.decision_log[-1]


def backdate_tickets(days: int) -> None:
    with session_scope() as session:
        for ticket in repositories.list_tickets(session):
            ticket.created_at = utcnow() - timedelta(days=days)


def test_ticket_inside_the_window_is_reused():
    submit()
    backdate_tickets(days=6)
    response, _ = submit()
    assert response.ticket_id == "T-000001"
    assert len(all_tickets()) == 1


def test_ticket_older_than_the_window_is_not_reused():
    submit()
    backdate_tickets(days=8)  # ticket_dedupe_days is 7
    response, _ = submit()
    assert response.ticket_id == "T-000002"
    assert len(all_tickets()) == 2
    assert "a été créé" in response.answer


def test_ticket_body_is_the_masked_text():
    iban = "FR76 3000 6000 0112 3456 7890 189"
    response, ctx = submit(f"Je conteste la facture de 180 €. Remboursez sur l'IBAN {iban}.")

    assert ctx.pii_types == ["IBAN"]
    (ticket,) = all_tickets()
    assert ticket.body == "Je conteste la facture de 180 €. Remboursez sur l'IBAN [IBAN]."
    assert "3000 6000" not in ticket.body + ticket.subject + response.answer
    assert "3000 6000" not in " ".join(ctx.decision_log)


def test_no_model_and_no_retrieval_are_used(monkeypatch, env):
    """The route must work without any index, and must not even ask whether an LLM is set."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the automation must not call a model")

    monkeypatch.setattr("app.core.llm.complete", forbidden)
    monkeypatch.setattr("app.core.llm.is_enabled", forbidden)
    # Nothing was ingested: any document search would raise IndexNotReady.
    assert not (env.index_dir / "chunks.json").exists()
    with pytest.raises(IndexNotReady):
        hybrid.get_retriever()

    response, ctx = submit()

    assert response.ticket_id == "T-000001"
    assert ctx.usage.llm_calls == 0
    assert ctx.usage.cost_eur == 0.0


@pytest.mark.parametrize(("client_id", "montant"), [(None, 120.0), ("C-12", None)])
def test_decision_without_client_or_amount_is_refused(client_id, montant):
    decision = TriageDecision(
        route="automation",
        rule="RG-03",
        reasons=[],
        client_id=client_id,
        montant=montant,
        client_known=client_id is not None,
    )
    ctx = RequestContext(q_masked=DISPUTE)
    with pytest.raises(ValueError, match="needs a client and an amount"):
        handle_automation(AskRequest(q=DISPUTE), decision, ctx)
    assert all_tickets() == []
