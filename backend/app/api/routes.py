"""HTTP routes under /api/v1. No GET route carries user text: requests go in POST bodies."""

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core import pii
from app.core.logging import request_id_var
from app.core.schemas import (
    ActionDecision,
    ActionResult,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackSaved,
    FeedbackStats,
    IngestResponse,
    MetricsResponse,
    ProposedAction,
    Ticket,
)
from app.core.security import rate_limit, require_api_key
from app.db import repositories, session
from app.ingestion import ingest
from app.services import actions, ask, metrics

# The key is checked first so that a refused request does not use up the caller's quota.
router = APIRouter(dependencies=[Depends(require_api_key), Depends(rate_limit)])


@router.post("/ask", response_model=AskResponse)
def post_ask(body: AskRequest) -> AskResponse:
    return ask.handle_ask(body, request_id=request_id_var.get() or None)


@router.post("/ingest", response_model=IngestResponse)
def post_ingest() -> IngestResponse:
    return ingest.ingest_docs()


@router.get("/actions", response_model=list[ProposedAction])
def get_actions(
    status_filter: Literal["pending", "approved", "rejected"] | None = Query(
        default=None, alias="status"
    ),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[ProposedAction]:
    with session.session_scope() as db:
        rows = repositories.list_actions(db, status=status_filter, limit=limit)
        return [repositories.to_action(row) for row in rows]


@router.post("/actions/{action_id}/approve", response_model=ActionResult)
def post_approve(action_id: str, body: ActionDecision) -> ActionResult:
    return _decide(actions.approve_action, action_id, body)


@router.post("/actions/{action_id}/reject", response_model=ActionResult)
def post_reject(action_id: str, body: ActionDecision) -> ActionResult:
    return _decide(actions.reject_action, action_id, body)


def _decide(
    decide: Callable[[str, ActionDecision], ActionResult], action_id: str, body: ActionDecision
) -> ActionResult:
    try:
        return decide(action_id, body)
    except actions.ActionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Action introuvable") from exc
    except actions.ActionConflict as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Cette action a déjà reçu la décision inverse : elle ne peut plus être modifiée",
        ) from exc


@router.get("/tickets", response_model=list[Ticket])
def get_tickets(limit: int = Query(default=50, ge=1, le=500)) -> list[Ticket]:
    with session.session_scope() as db:
        return [repositories.to_ticket(row) for row in repositories.list_tickets(db, limit=limit)]


@router.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: str) -> Ticket:
    with session.session_scope() as db:
        row = repositories.get_ticket(db, ticket_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Ticket introuvable")
        return repositories.to_ticket(row)


@router.post("/feedback", response_model=FeedbackSaved)
def post_feedback(body: FeedbackRequest) -> FeedbackSaved:
    comment, _ = pii.redact(body.comment)
    with session.session_scope() as db:
        request = repositories.get_request(db, body.request_id)
        if request is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail="Demande introuvable pour cet identifiant"
            )
        repositories.save_feedback(db, request_id=body.request_id, ok=body.ok, comment=comment)
        return FeedbackSaved(request_id=body.request_id, route=request.route)


@router.get("/feedback/stats", response_model=FeedbackStats)
def get_feedback_stats() -> FeedbackStats:
    with session.session_scope() as db:
        return FeedbackStats(**repositories.feedback_stats(db))


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    return metrics.compute_metrics()
