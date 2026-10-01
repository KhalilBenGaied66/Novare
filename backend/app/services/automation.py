"""RG-03 automation: a small billing dispute of a known client becomes a standard ticket.

No model and no document search are involved: triage has already checked the rule
(dispute vocabulary, amount below the threshold, client in the reference data), so the
ticket is created without human validation.
"""

import hashlib
from datetime import timedelta

from app.agents import extract
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import AskRequest, AskResponse
from app.core.text import tokenize
from app.core.types import TriageDecision
from app.db import repositories, session
from app.db.models import utcnow
from app.services.context import RequestContext

logger = get_logger(__name__)


def handle_automation(
    req: AskRequest, decision: TriageDecision, ctx: RequestContext
) -> AskResponse:
    """Create the dispute ticket, or reuse the one already opened for the same request.

    The ticket body is `ctx.q_masked`: the request text is stored only after masking.
    """
    client_id, montant = decision.client_id, decision.montant
    if client_id is None or montant is None:
        # Triage only chooses this route with both values resolved.
        raise ValueError("RG-03 automation needs a client and an amount")

    settings = get_settings()
    dedupe_key = _dedupe_key(client_id, montant, ctx.q_masked)
    window_start = utcnow() - timedelta(days=settings.ticket_dedupe_days)

    # Search and creation share one transaction. Two identical requests arriving at the
    # very same moment could still both create a ticket: the window is a rule on dates,
    # which a unique constraint cannot express.
    with session.session_scope() as db:
        ticket = repositories.find_recent_ticket(db, dedupe_key, window_start)
        reused = ticket is not None
        if ticket is None:
            ticket = repositories.create_ticket(
                db,
                client_id=client_id,
                subject=f"Litige facturation {montant:.2f} € — {client_id}",
                body=ctx.q_masked,
                kind="litige_standard",
                source="automation",
                created_by="system:RG-03",
                request_id=ctx.request_id,
                dedupe_key=dedupe_key,
            )
        ticket_id = ticket.ticket_id

    amount = extract.format_amount(montant)
    threshold = extract.format_amount(settings.rg03_max_amount)
    if reused:
        ctx.log(
            f"Automatisation RG-03 : doublon détecté, ticket {ticket_id} réutilisé "
            f"(même demande dans les {settings.ticket_dedupe_days} derniers jours)"
        )
        answer = (
            f"Cette contestation de {amount} est déjà enregistrée pour le client {client_id} : "
            f"le ticket {ticket_id} existant est réutilisé, aucun nouveau ticket n'est créé "
            f"(règle RG-03, litige de facturation inférieur à {threshold})."
        )
    else:
        ctx.log(f"Automatisation RG-03 : ticket {ticket_id} créé (litige standard)")
        answer = (
            f"Votre contestation de {amount} est enregistrée : le ticket {ticket_id} a été créé "
            f"pour le client {client_id}. Un litige de facturation inférieur à {threshold} "
            "suit la procédure standard, sans validation préalable (règle RG-03)."
        )
    logger.info(
        "automation_ticket",
        extra={"ticket_id": ticket_id, "client_id": client_id, "reused": reused},
    )

    return AskResponse(
        request_id=ctx.request_id,
        route="automation",
        mode="rule",
        answer=answer,
        needs_validation=False,
        ticket_id=ticket_id,
        retrieval_mode="none",
    )


def _dedupe_key(client_id: str, montant: float, q_masked: str) -> str:
    """Fingerprint of a dispute: same client, same amount, same words in any order.

    Tokens are stemmed and sorted, so a request resent with its sentences swapped or
    with another punctuation gives the same key.
    """
    tokens = " ".join(sorted(tokenize(q_masked)))
    fingerprint = f"{client_id}|{montant:.2f}|{tokens}"
    # SHA-1 only shortens the fingerprint to a fixed-size index key; nothing relies on
    # it being hard to forge.
    return hashlib.sha1(fingerprint.encode("utf-8"), usedforsecurity=False).hexdigest()
