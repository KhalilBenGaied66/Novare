"""Entry point of a request: mask, triage, dispatch to the route, record what happened.

This is the one place where an unexpected failure of a route is turned into an
escalation to a person: a request is never lost and never answered with a stack trace.
"""

from app.agents import dossier_agent, triage
from app.core import pii
from app.core.logging import get_logger, request_id_var
from app.core.schemas import AskRequest, AskResponse
from app.core.types import TriageDecision
from app.db import repositories, session
from app.retrieval import hybrid, rag
from app.services import automation
from app.services.context import RequestContext

logger = get_logger(__name__)

_SENSITIVE_ANSWER = (
    "Cette demande relève d'un sujet qui doit être traité par un gestionnaire. "
    "Elle lui est transmise sans réponse automatique."
)
_ERROR_ANSWER = (
    "La demande n'a pas pu être traitée automatiquement. Elle est transmise à un gestionnaire."
)


def handle_ask(req: AskRequest, request_id: str | None = None) -> AskResponse:
    """Handle one request end to end and write its row in the request log.

    `request_id` is the identifier chosen by the HTTP layer, when there is one.
    `IndexNotReady` is the only exception that leaves this function: the API answers 503.
    """
    ctx = RequestContext() if request_id is None else RequestContext(request_id=request_id)
    request_id_var.set(ctx.request_id)
    ctx.q_masked, ctx.pii_types = pii.redact(req.q)
    if ctx.pii_types:
        ctx.log(f"Données personnelles masquées avant traitement : {', '.join(ctx.pii_types)}")

    decision = triage.triage(req)
    for reason in decision.reasons:
        ctx.log(reason)

    error = None
    try:
        response = _dispatch(req, decision, ctx)
    except hybrid.IndexNotReady:
        raise
    except Exception as exc:  # boundary: any failure of a route becomes an escalation
        error = type(exc).__name__
        logger.exception("ask_failed", extra={"route": decision.route, "error": error})
        ctx.log("Erreur interne pendant le traitement : demande transmise à un gestionnaire")
        response = AskResponse(
            request_id=ctx.request_id, route="human", mode="none", answer=_ERROR_ANSWER
        )

    response.request_id = ctx.request_id
    response.decision_log = list(ctx.decision_log)
    response.pii_redacted = list(ctx.pii_types)
    response.usage = ctx.usage
    response.latency_ms = ctx.elapsed_ms()
    _record(response, decision, ctx, error)
    logger.info(
        "ask_done",
        extra={
            "route": response.route,
            "rule": decision.rule,
            "mode": response.mode,
            "latency_ms": response.latency_ms,
            "llm_calls": response.usage.llm_calls,
            "q_len": len(req.q),
        },
    )
    return response


def _dispatch(req: AskRequest, decision: TriageDecision, ctx: RequestContext) -> AskResponse:
    if decision.route == "automation":
        return automation.handle_automation(req, decision, ctx)
    if decision.route == "rag":
        return rag.answer_with_rag(req, decision, ctx)
    if decision.route == "agent":
        return dossier_agent.run_agent(req, decision, ctx)
    return AskResponse(
        request_id=ctx.request_id, route="human", mode="none", answer=_SENSITIVE_ANSWER
    )


def _record(
    response: AskResponse, decision: TriageDecision, ctx: RequestContext, error: str | None
) -> None:
    """One row per request. Only the masked text is stored."""
    with session.session_scope() as db:
        repositories.log_request(
            db,
            request_id=ctx.request_id,
            route=response.route,
            rule=decision.rule,
            mode=response.mode,
            q_masked=ctx.q_masked,
            client_id=decision.client_id,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            llm_calls=response.usage.llm_calls,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_eur=response.usage.cost_eur,
            model=response.usage.model,
            retrieval_mode=response.retrieval_mode,
            pii_types=list(ctx.pii_types),
            needs_validation=response.needs_validation,
            error=error,
        )
