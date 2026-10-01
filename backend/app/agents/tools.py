"""Tools of the dossier agent: document search, contract, SLA deadline, ticket proposal.

Least privilege: no tool takes a client identifier. Every tool acts on the client the
triage resolved for the dossier (`ToolContext.client`), so a request cannot talk the
model into reading another client's contract. No tool writes to the database either:
`propose_ticket` only records a proposal, and the ticket is created after a human
approves it (see `app.services.actions`).

Tool results are French text written for the planner (an LLM or the scripted plan).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core import pii
from app.core.logging import get_logger
from app.core.types import RetrievedChunk
from app.domain import clients, sla
from app.domain.clients import Client
from app.retrieval import hybrid

logger = get_logger(__name__)

PRIORITIES = ("P1", "P2", "P3")

# SLA hours are counted on the wall clock of the business (Lyon), whatever the time
# zone of the server: a container usually runs in UTC.
BUSINESS_TIMEZONE = "Europe/Paris"

_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


@dataclass
class ToolContext:
    """State of one dossier, shared by the tool calls of a request."""

    request_id: str
    q_masked: str  # request text after PII redaction
    client: Client | None  # resolved by the triage; None when the client is unknown
    # Chunks found by `search_docs`, in order of discovery: source [n] is sources[n - 1].
    sources: list[RetrievedChunk] = field(default_factory=list)
    proposal: dict | None = None  # {"subject", "summary", "priority"}, set by propose_ticket
    steps: list[str] = field(default_factory=list)  # French trace lines, no raw user text


def _schema(name: str, description: str, properties: dict[str, dict]) -> dict:
    """One tool in the OpenAI function-calling format. Every parameter is required."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


_PRIORITY_PARAMETER = {
    "type": "string",
    "enum": list(PRIORITIES),
    "description": "Priorité de l'intervention : P1 (critique), P2 (majeure) ou P3 (mineure).",
}

TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_docs",
        "Recherche dans la documentation interne (procédures, grille tarifaire, conditions "
        "générales, contrat du client du dossier). Renvoie des extraits numérotés [n] à "
        "citer dans la réponse.",
        {"query": {"type": "string", "description": "Question ou mots-clés, en français."}},
    ),
    _schema(
        "get_contract",
        "Résumé du contrat du client du dossier : formule, dates de validité, contrat "
        "actif ou échu, astreinte week-end incluse ou non.",
        {},
    ),
    _schema(
        "compute_deadline",
        "Délai contractuel d'intervention du client du dossier pour une priorité, et "
        "échéance correspondante à partir de maintenant.",
        {"priority": _PRIORITY_PARAMETER},
    ),
    _schema(
        "propose_ticket",
        "Enregistre une proposition de ticket d'intervention pour le client du dossier. "
        "Ne crée aucun ticket : la proposition attend la validation d'un gestionnaire.",
        {
            "subject": {"type": "string", "description": "Objet du ticket, en une ligne."},
            "summary": {"type": "string", "description": "Résumé du problème et du contexte."},
            "priority": _PRIORITY_PARAMETER,
        },
    ),
]


def run_tool(name: str, arguments: dict, tc: ToolContext) -> str:
    """Run one tool call and return its result as text for the planner. Never raises.

    An unknown tool or invalid arguments give an error text instead of an exception:
    the planner reads it as the tool result and can correct its call.
    """
    tool = _TOOLS.get(name)
    if tool is None:
        tc.steps.append("Outil inconnu demandé par le planificateur : appel ignoré")
        return f"Erreur : cet outil n'existe pas. Outils disponibles : {', '.join(_TOOLS)}."
    problem = _argument_problem(name, arguments)
    if problem is not None:
        tc.steps.append(f"{name} : arguments invalides, appel non exécuté")
        return f"Erreur dans les arguments de {name} : {problem}."
    try:
        return tool(tc, **arguments)
    except Exception as exc:
        # Boundary of the agent loop: a failing tool must not abort the request. The
        # planner is told that the result is missing; details go to the log only.
        logger.exception("tool_failed", extra={"tool": name, "error": type(exc).__name__})
        tc.steps.append(f"{name} : erreur interne ({type(exc).__name__}), aucun résultat")
        return (
            f"Erreur interne de l'outil {name} : aucun résultat disponible. "
            "Cette information est à signaler comme manquante."
        )


def _argument_problem(name: str, arguments: dict) -> str | None:
    """French description of the first problem in `arguments`, or None when they are valid.

    The check reads the schema sent to the model, so the two cannot drift apart.
    Every parameter of every tool is a required, non-empty string.
    """
    if not isinstance(arguments, dict):
        return "les arguments doivent être un objet JSON"
    properties = _PARAMETERS[name]
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        # Refused rather than ignored: this is how a call such as
        # get_contract(client_id="C-34") is answered.
        expected = ", ".join(properties) or "aucun"
        return f"argument(s) non reconnu(s) : {', '.join(unknown)} (attendu : {expected})"
    for key, spec in properties.items():
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"l'argument « {key} » est obligatoire (texte non vide)"
        if "enum" in spec and value not in spec["enum"]:
            return f"l'argument « {key} » doit valoir {' ou '.join(spec['enum'])}"
    return None


# --- Tools -------------------------------------------------------------------


def _search_docs(tc: ToolContext, query: str) -> str:
    client_id = tc.client.client_id if tc.client else None
    results = hybrid.get_retriever().search(pii.strip_tags(query), client_id=client_id)
    if not results:
        tc.steps.append("search_docs : aucun extrait trouvé")
        return "Aucun extrait trouvé dans la documentation pour cette recherche."

    known = len(tc.sources)
    blocks = [_format_source(_source_number(tc, result), result) for result in results]
    docs = ", ".join(dict.fromkeys(result.chunk.doc for result in results))
    tc.steps.append(
        f"search_docs : {len(results)} extrait(s), dont {len(tc.sources) - known} "
        f"nouveau(x) ({docs})"
    )
    return "\n\n".join(blocks)


def _source_number(tc: ToolContext, result: RetrievedChunk) -> int:
    """Number of the chunk among the dossier's sources.

    A chunk found by an earlier search keeps its number; a new one is appended.
    """
    for number, source in enumerate(tc.sources, start=1):
        if source.chunk.chunk_id == result.chunk.chunk_id:
            return number
    tc.sources.append(result)
    return len(tc.sources)


def _format_source(number: int, result: RetrievedChunk) -> str:
    """One numbered source, in the layout of `guardrails.format_sources`.

    That function numbers from 1 on every call, while the numbers here run over all the
    searches of a dossier.
    """
    chunk = result.chunk
    header = f"[{number}] {chunk.title} — {chunk.doc}, p. {chunk.page}"
    if chunk.section:
        header += f", {chunk.section}"
    return f"{header}\n{chunk.text.strip()}"


def _get_contract(tc: ToolContext) -> str:
    client = tc.client
    if client is None:
        tc.steps.append("get_contract : client non identifié")
        return "Client non identifié : aucun contrat ne peut être rattaché à ce dossier."

    today = _now().date()
    status = _contract_status(client, today)
    on_call = "incluse" if client.astreinte_weekend else "non incluse"
    tc.steps.append(f"get_contract : {client.client_id}, formule {client.formule}, {status}")
    lines = [
        f"Client {client.client_id} — {client.nom}",
        f"Formule : {client.formule}",
        f"Validité : du {_fr_date(client.date_debut)} au {_fr_date(client.date_fin)}",
        f"Situation au {_fr_date(today)} : {status}",
        f"Astreinte week-end : {on_call}",
    ]
    if client.sites:
        lines.append(f"Sites couverts : {' ; '.join(client.sites)}")
    return "\n".join(lines)


def _compute_deadline(tc: ToolContext, priority: str) -> str:
    quotation = "Un devis préalable est nécessaire avant toute intervention."
    client = tc.client
    if client is None:
        tc.steps.append("compute_deadline : aucun délai applicable, client non identifié")
        return f"Client non identifié : aucun délai contractuel ne s'applique. {quotation}"

    now = _now()
    if not client.is_active(now.date()):
        status = _contract_status(client, now.date())
        tc.steps.append(f"compute_deadline : aucun délai applicable, {status}")
        return f"Aucun délai garanti ne s'applique : {status}. {quotation}"

    hours = clients.get_client_repository().sla_hours(client.formule, priority)
    if hours is None:
        tc.steps.append(f"compute_deadline : aucun délai défini pour la formule {client.formule}")
        return f"Aucun délai n'est défini pour la formule {client.formule} en priorité {priority}."

    deadline = sla.sla_deadline(now, hours, around_the_clock=client.astreinte_weekend)
    if client.astreinte_weekend:
        delay = f"{hours} h (astreinte incluse : décompte 24 h/24 et 7 j/7)"
    else:
        delay = f"{hours} h ouvrées (du lundi au vendredi, de 8 h à 18 h)"
    tc.steps.append(
        f"compute_deadline : {priority}, formule {client.formule}, {hours} h, "
        f"échéance {_fr_datetime(deadline)}"
    )
    return (
        f"Formule {client.formule}, priorité {priority} : délai contractuel d'intervention "
        f"de {delay}. Pour une demande prise en compte le {_fr_datetime(now)}, "
        f"intervention au plus tard le {_fr_datetime(deadline)}."
    )


def _propose_ticket(tc: ToolContext, subject: str, summary: str, priority: str) -> str:
    replaced = tc.proposal is not None
    # The subject is a one-line label: line breaks and repeated spaces are collapsed.
    tc.proposal = {
        "subject": " ".join(subject.split()),
        "summary": summary.strip(),
        "priority": priority,
    }
    note = " (remplace la proposition précédente)" if replaced else ""
    tc.steps.append(
        f"propose_ticket : ticket proposé en priorité {priority}{note}, "
        "en attente de validation humaine"
    )
    return (
        f"Proposition de ticket enregistrée en priorité {priority}{note}. Aucun ticket "
        "n'est créé à ce stade : la proposition attend la validation d'un gestionnaire."
    )


_TOOLS: dict[str, Callable[..., str]] = {
    "search_docs": _search_docs,
    "get_contract": _get_contract,
    "compute_deadline": _compute_deadline,
    "propose_ticket": _propose_ticket,
}

# Parameter specifications by tool name, read from the schemas sent to the model.
_PARAMETERS: dict[str, dict[str, dict]] = {
    schema["function"]["name"]: schema["function"]["parameters"]["properties"]
    for schema in TOOL_SCHEMAS
}


# --- Helpers -----------------------------------------------------------------


def _now() -> datetime:
    """Current wall-clock time of the business, as a naive datetime (what `sla` expects)."""
    try:
        return datetime.now(ZoneInfo(BUSINESS_TIMEZONE)).replace(tzinfo=None)
    except ZoneInfoNotFoundError:
        # No time zone database on this machine (Windows or a minimal image without the
        # `tzdata` package): degrade to the server's local time rather than fail.
        logger.warning("timezone_data_missing", extra={"timezone": BUSINESS_TIMEZONE})
        return datetime.now()


def _contract_status(client: Client, today: date) -> str:
    if client.is_active(today):
        return "contrat actif"
    if today > client.date_fin:
        return f"contrat échu (fin de validité le {_fr_date(client.date_fin)})"
    return f"contrat non encore en vigueur (début le {_fr_date(client.date_debut)})"


def _fr_date(day: date) -> str:
    return f"{day:%d/%m/%Y}"


def _fr_datetime(moment: datetime) -> str:
    """E.g. "lundi 05/10/2026 à 11 h 00". Written by hand: independent of the locale."""
    return f"{_WEEKDAYS[moment.weekday()]} {_fr_date(moment)} à {moment.hour} h {moment:%M}"
