"""Human approval: a ticket exists only after approval, exactly once, and decisions are final."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from app.agents.dossier_agent import run_agent
from app.core.schemas import ActionDecision, AskRequest
from app.core.types import TriageDecision
from app.db import repositories
from app.db.session import session_scope
from app.services.actions import ActionConflict, ActionNotFound, approve_action, reject_action
from app.services.context import RequestContext

PAYLOAD = {
    "subject": "Chaudière à l'arrêt — immeuble Lumière",
    "summary": "Arrêt total de la chaudière signalé ce matin. Rappeler au [TEL].",
    "priority": "P1",
    "client_id": "C-12",
}
MARIE = ActionDecision(validator="Marie Dupont")
PAUL = ActionDecision(validator="Paul Martin", reason="Doublon du ticket ouvert hier")


def pending_action(payload: dict | None = None) -> str:
    with session_scope() as session:
        row = repositories.create_action(
            session, request_id="req-1", type="create_ticket", payload=payload or PAYLOAD
        )
        return row.id


def stored_tickets() -> list:
    with session_scope() as session:
        return [repositories.to_ticket(row) for row in repositories.list_tickets(session)]


def stored_action(action_id: str):
    with session_scope() as session:
        return repositories.get_action(session, action_id)


# --- Approve -------------------------------------------------------------------


def test_approve_creates_the_ticket_from_the_payload(db):
    action_id = pending_action()
    before = datetime.now(UTC)

    result = approve_action(action_id, MARIE)

    ticket = result.ticket
    assert ticket.ticket_id == "T-000001"
    assert (ticket.kind, ticket.source, ticket.created_by) == (
        "intervention",
        "agent",
        "Marie Dupont",
    )
    assert (ticket.client_id, ticket.priority, ticket.status) == ("C-12", "P1", "open")
    assert ticket.subject == PAYLOAD["subject"]
    assert ticket.body == PAYLOAD["summary"]
    assert ticket.request_id == "req-1"
    assert stored_tickets() == [ticket]

    assert result.action.action_id == action_id
    assert (result.action.status, result.action.ticket_id) == ("approved", "T-000001")
    assert result.action.payload == PAYLOAD
    row = stored_action(action_id)
    assert (row.status, row.decided_by, row.ticket_id) == ("approved", "Marie Dupont", "T-000001")
    assert row.decided_at >= before


def test_approve_is_idempotent(db):
    action_id = pending_action()
    first = approve_action(action_id, MARIE)

    again = approve_action(action_id, MARIE)
    by_someone_else = approve_action(action_id, ActionDecision(validator="Paul Martin"))

    assert again == first
    assert by_someone_else == first
    assert len(stored_tickets()) == 1
    assert stored_action(action_id).decided_by == "Marie Dupont", "the first decision stands"


def test_simultaneous_approvals_create_a_single_ticket(db):
    action_id = pending_action()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: approve_action(action_id, MARIE), range(8)))

    assert {result.ticket.ticket_id for result in results} == {"T-000001"}
    assert {result.action.status for result in results} == {"approved"}
    assert len(stored_tickets()) == 1


def test_each_action_gets_its_own_ticket(db):
    first = approve_action(pending_action(), MARIE)
    second = approve_action(pending_action({**PAYLOAD, "client_id": None}), MARIE)

    assert (first.ticket.ticket_id, second.ticket.ticket_id) == ("T-000001", "T-000002")
    assert second.ticket.client_id is None


def test_failed_approval_leaves_the_action_pending(db, monkeypatch):
    action_id = pending_action()

    def failing_create_ticket(session, **fields):
        raise RuntimeError("database unavailable")

    with monkeypatch.context() as patch, pytest.raises(RuntimeError):
        patch.setattr(repositories, "create_ticket", failing_create_ticket)
        approve_action(action_id, MARIE)

    assert stored_action(action_id).status == "pending"
    assert stored_tickets() == []
    assert approve_action(action_id, MARIE).ticket.ticket_id == "T-000001"


# --- Reject --------------------------------------------------------------------


def test_reject_creates_no_ticket(db):
    action_id = pending_action()

    result = reject_action(action_id, PAUL)

    assert result.ticket is None
    assert (result.action.status, result.action.ticket_id) == ("rejected", None)
    row = stored_action(action_id)
    assert (row.status, row.decided_by, row.reason) == (
        "rejected",
        "Paul Martin",
        "Doublon du ticket ouvert hier",
    )
    assert row.decided_at is not None
    assert stored_tickets() == []


def test_reject_is_idempotent(db):
    action_id = pending_action()
    first = reject_action(action_id, PAUL)

    again = reject_action(action_id, ActionDecision(validator="Marie Dupont", reason="Autre motif"))

    assert again == first
    row = stored_action(action_id)
    assert (row.decided_by, row.reason) == ("Paul Martin", "Doublon du ticket ouvert hier")


def test_personal_data_in_the_reason_is_masked_before_storage(db):
    action_id = pending_action()
    decision = ActionDecision(
        validator="Paul Martin", reason="Le client a rappelé du 06 12 34 56 78, déjà traité."
    )
    reject_action(action_id, decision)
    assert stored_action(action_id).reason == "Le client a rappelé du [TEL], déjà traité."


# --- Conflicts and unknown ids ---------------------------------------------------


def test_approving_a_rejected_action_is_a_conflict(db):
    action_id = pending_action()
    reject_action(action_id, PAUL)

    with pytest.raises(ActionConflict) as error:
        approve_action(action_id, MARIE)

    assert str(error.value) == "Action déjà refusée : la décision ne peut plus être modifiée"
    assert stored_tickets() == []
    row = stored_action(action_id)
    assert (row.status, row.decided_by) == ("rejected", "Paul Martin")


def test_rejecting_an_approved_action_is_a_conflict(db):
    action_id = pending_action()
    approved = approve_action(action_id, MARIE)

    with pytest.raises(ActionConflict) as error:
        reject_action(action_id, PAUL)

    assert str(error.value) == "Action déjà approuvée : la décision ne peut plus être modifiée"
    assert stored_tickets() == [approved.ticket]
    row = stored_action(action_id)
    assert (row.status, row.ticket_id, row.reason) == ("approved", "T-000001", "")


@pytest.mark.parametrize("decide", [approve_action, reject_action])
@pytest.mark.parametrize("action_id", ["inconnu", "", "T-000001", "' OR 1=1 --"])
def test_unknown_action_id(db, decide, action_id):
    pending_action()
    with pytest.raises(ActionNotFound) as error:
        decide(action_id, MARIE)
    assert str(error.value) == "Action introuvable"
    assert stored_tickets() == []


# --- With the agent --------------------------------------------------------------


def test_agent_proposal_becomes_a_ticket_only_after_approval(indexed):
    q_masked = "La chaudière est à l'arrêt total, c'est urgent. Rappeler au [TEL]."
    ctx = RequestContext(q_masked=q_masked)
    decision = TriageDecision("agent", "RG-05", [], "C-12", None, True)
    response = run_agent(AskRequest(q=q_masked, client_id="C-12"), decision, ctx)

    action = response.proposed_action
    assert action.status == "pending"
    assert stored_tickets() == [], "nothing is created before a human approves"

    result = approve_action(action.action_id, MARIE)

    (ticket,) = stored_tickets()
    assert result.ticket == ticket
    assert ticket.subject == "Demande d'intervention P1 — C-12"
    assert ticket.body == q_masked
    assert (ticket.client_id, ticket.priority, ticket.kind) == ("C-12", "P1", "intervention")
    assert (ticket.source, ticket.created_by) == ("agent", "Marie Dupont")
    assert ticket.request_id == ctx.request_id
    with session_scope() as session:
        assert repositories.list_actions(session, status="pending") == []
