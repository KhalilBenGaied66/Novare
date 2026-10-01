"""Human decision on the actions proposed by the agent.

The agent only proposes a ticket; the ticket exists once a person has approved the
action. A decision is final: an approved action cannot be rejected afterwards, nor the
reverse. Repeating the same decision is harmless and returns the first result, so a
double click or a retried HTTP call never creates a second ticket.
"""

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core import pii
from app.core.logging import get_logger
from app.core.schemas import ActionDecision, ActionResult
from app.db import repositories
from app.db.models import ActionRow
from app.db.session import session_scope

logger = get_logger(__name__)


class ActionNotFound(Exception):
    """No action has this identifier."""


class ActionConflict(Exception):
    """The action already carries the opposite decision."""


def approve_action(action_id: str, decision: ActionDecision) -> ActionResult:
    """Approve a proposed action: create its ticket and link the two.

    Raises `ActionNotFound` for an unknown id and `ActionConflict` when the action was
    rejected. An action that is already approved returns its existing ticket.
    """
    with session_scope() as session:
        if _claim(session, action_id, "approved"):
            action = repositories.get_action(session, action_id)
            payload = action.payload
            ticket = repositories.create_ticket(
                session,
                client_id=payload.get("client_id"),
                subject=payload["subject"],
                body=payload["summary"],
                kind="intervention",
                # The validator's priority, when given, replaces the proposed one.
                priority=decision.priority or payload.get("priority"),
                source="agent",
                created_by=decision.validator,
                request_id=action.request_id,
            )
            repositories.decide_action(
                session,
                action,
                status="approved",
                decided_by=decision.validator,
                reason=_masked(decision.reason),
                ticket_id=ticket.ticket_id,
            )
            logger.info(
                "action_approved", extra={"action_id": action.id, "ticket_id": ticket.ticket_id}
            )
        else:
            action = _already_decided(session, action_id, "approved")
            ticket = repositories.get_ticket(session, action.ticket_id or "")
        return ActionResult(
            action=repositories.to_action(action),
            ticket=repositories.to_ticket(ticket) if ticket is not None else None,
        )


def reject_action(action_id: str, decision: ActionDecision) -> ActionResult:
    """Reject a proposed action: no ticket is created.

    Raises `ActionNotFound` for an unknown id and `ActionConflict` when the action was
    approved. An action that is already rejected is returned unchanged.
    """
    with session_scope() as session:
        if _claim(session, action_id, "rejected"):
            action = repositories.get_action(session, action_id)
            repositories.decide_action(
                session,
                action,
                status="rejected",
                decided_by=decision.validator,
                reason=_masked(decision.reason),
            )
            logger.info("action_rejected", extra={"action_id": action.id})
        else:
            action = _already_decided(session, action_id, "rejected")
        return ActionResult(action=repositories.to_action(action))


def _claim(session: Session, action_id: str, status: str) -> bool:
    """Move the action from "pending" to `status`; False when it is not pending.

    A single conditional UPDATE, so that two simultaneous decisions cannot both find
    the action pending: the database lets one of them through, and the other one sees
    zero rows updated once the first transaction has committed. Reading the status
    first and writing afterwards would leave a window for two tickets.
    """
    result = session.execute(
        update(ActionRow)
        .where(ActionRow.id == action_id, ActionRow.status == "pending")
        .values(status=status)
    )
    return result.rowcount == 1


def _already_decided(session: Session, action_id: str, status: str) -> ActionRow:
    """The action, when its recorded decision is `status`; raises otherwise."""
    action = repositories.get_action(session, action_id)
    if action is None:
        raise ActionNotFound("Action introuvable")
    if action.status != status:
        decided = "approuvée" if action.status == "approved" else "refusée"
        raise ActionConflict(f"Action déjà {decided} : la décision ne peut plus être modifiée")
    return action


def _masked(reason: str) -> str:
    """The reason is free text typed by the validator: stored with personal data masked."""
    masked, _ = pii.redact(reason)
    return masked
