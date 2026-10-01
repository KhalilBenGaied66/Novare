"""Persistence functions. Each takes the session of the caller's transaction.

Rows are flushed before they are returned, so generated values (ticket number,
defaults) are available immediately; the commit belongs to `session_scope`.
"""

import re
from datetime import datetime
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core import schemas
from app.db.models import ActionRow, FeedbackRow, RequestLogRow, TicketRow, utcnow

# "T-000012". At most 9 digits so the number always fits a 32-bit integer column.
_TICKET_ID = re.compile(r"T-(\d{1,9})", re.IGNORECASE)

_DECISIONS = ("approved", "rejected")


# --- Tickets ---------------------------------------------------------------


def create_ticket(
    session: Session,
    *,
    client_id: str | None,
    subject: str,
    body: str,
    kind: str,
    priority: str | None = None,
    source: str,
    created_by: str,
    request_id: str | None = None,
    dedupe_key: str | None = None,
) -> TicketRow:
    row = TicketRow(
        client_id=client_id,
        subject=subject,
        body=body,
        kind=kind,
        priority=priority,
        source=source,
        created_by=created_by,
        request_id=request_id,
        dedupe_key=dedupe_key,
    )
    session.add(row)
    session.flush()
    return row


def find_recent_ticket(session: Session, dedupe_key: str, since: datetime) -> TicketRow | None:
    """Latest ticket with this dedupe key created at or after `since` (UTC)."""
    statement = (
        select(TicketRow)
        .where(TicketRow.dedupe_key == dedupe_key, TicketRow.created_at >= since)
        .order_by(TicketRow.id.desc())
        .limit(1)
    )
    return session.scalars(statement).first()


def get_ticket(session: Session, ticket_id: str) -> TicketRow | None:
    """Look a ticket up by its public identifier (`T-000012`); None when unknown or malformed."""
    match = _TICKET_ID.fullmatch(ticket_id.strip())
    if match is None:
        return None
    return session.get(TicketRow, int(match.group(1)))


def list_tickets(session: Session, limit: int = 50) -> list[TicketRow]:
    statement = select(TicketRow).order_by(TicketRow.id.desc()).limit(limit)
    return list(session.scalars(statement))


# --- Proposed actions ------------------------------------------------------


def create_action(session: Session, *, request_id: str, type: str, payload: dict) -> ActionRow:
    row = ActionRow(request_id=request_id, type=type, payload=payload, status="pending")
    session.add(row)
    session.flush()
    return row


def get_action(session: Session, action_id: str) -> ActionRow | None:
    return session.get(ActionRow, action_id)


def list_actions(session: Session, status: str | None = None, limit: int = 50) -> list[ActionRow]:
    """Newest first; `status` filters on "pending", "approved" or "rejected"."""
    statement = select(ActionRow)
    if status is not None:
        statement = statement.where(ActionRow.status == status)
    statement = statement.order_by(ActionRow.created_at.desc()).limit(limit)
    return list(session.scalars(statement))


def decide_action(
    session: Session,
    action: ActionRow,
    *,
    status: str,
    decided_by: str,
    reason: str = "",
    ticket_id: str | None = None,
) -> ActionRow:
    """Record a human decision. Whether the action may still be decided is the caller's rule."""
    if status not in _DECISIONS:
        raise ValueError(f"status must be one of {_DECISIONS}, got {status!r}")
    action.status = status
    action.decided_by = decided_by
    action.decided_at = utcnow()
    action.reason = reason
    action.ticket_id = ticket_id
    session.flush()
    return action


# --- Request log and feedback ----------------------------------------------


def log_request(session: Session, **fields: Any) -> RequestLogRow:
    """Insert one request log row; `fields` are `RequestLogRow` column names."""
    row = RequestLogRow(**fields)
    session.add(row)
    session.flush()
    return row


def get_request(session: Session, request_id: str) -> RequestLogRow | None:
    return session.get(RequestLogRow, request_id)


def list_requests(session: Session, limit: int = 5000) -> list[RequestLogRow]:
    statement = select(RequestLogRow).order_by(RequestLogRow.ts.desc()).limit(limit)
    return list(session.scalars(statement))


def save_feedback(session: Session, *, request_id: str, ok: bool, comment: str) -> FeedbackRow:
    row = FeedbackRow(request_id=request_id, ok=ok, comment=comment)
    session.add(row)
    session.flush()
    return row


def feedback_stats(session: Session) -> dict:
    """Feedback counts overall and per route of the request that was rated.

    Returns `{"n": int, "ok_rate": float | None, "by_route": {route: {"n": int, "ok": int}}}`.
    """
    statement = (
        select(
            RequestLogRow.route,
            func.count(FeedbackRow.id),
            func.sum(case((FeedbackRow.ok, 1), else_=0)),
        )
        .join(RequestLogRow, RequestLogRow.request_id == FeedbackRow.request_id)
        .group_by(RequestLogRow.route)
        .order_by(RequestLogRow.route)
    )
    by_route = {route: {"n": int(n), "ok": int(ok)} for route, n, ok in session.execute(statement)}
    total = sum(counts["n"] for counts in by_route.values())
    total_ok = sum(counts["ok"] for counts in by_route.values())
    return {
        "n": total,
        "ok_rate": round(total_ok / total, 4) if total else None,
        "by_route": by_route,
    }


# --- Rows to API models ----------------------------------------------------


def to_ticket(row: TicketRow) -> schemas.Ticket:
    return schemas.Ticket(
        ticket_id=row.ticket_id,
        client_id=row.client_id,
        subject=row.subject,
        body=row.body,
        kind=row.kind,
        priority=row.priority,
        source=row.source,
        status=row.status,
        created_at=row.created_at,
        created_by=row.created_by,
        request_id=row.request_id,
    )


def to_action(row: ActionRow) -> schemas.ProposedAction:
    return schemas.ProposedAction(
        action_id=row.id,
        type=row.type,
        status=row.status,
        payload=row.payload,
        ticket_id=row.ticket_id,
    )
