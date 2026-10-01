"""Dossier agent: a planner calls tools until it can write a draft reply for a handler.

The loop is a LangGraph state graph with two nodes:

    START -> plan -> tools -> plan -> ... -> END

`plan` asks the planner for the next assistant message. When that message contains
tool calls, `tools` runs them and the loop goes back to `plan`; otherwise the message
is the final draft and the graph ends.

Two planners share one interface: `LLMPlanner` (a model chooses the tool calls) and
`ScriptedPlanner` (a fixed plan, no LLM), which is used when no LLM is configured and
as a fallback when the model fails or does not conclude within the step limit.

The agent never creates a ticket: a proposal made through `propose_ticket` is stored as
a pending action that a human approves or rejects (see `app.services.actions`).
"""

import json
import operator
from typing import Annotated, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import tools
from app.agents.tools import ToolContext
from app.core import guardrails, llm
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import loader
from app.core.schemas import AskRequest, AskResponse, ProposedAction
from app.core.text import stem, word_sequence
from app.core.types import LLMResult, ToolCall, TriageDecision
from app.db import repositories
from app.db.session import session_scope
from app.domain import clients
from app.domain.clients import Client
from app.retrieval import hybrid
from app.services.context import RequestContext

logger = get_logger(__name__)

PRIORITY_LABELS = {"P1": "critique", "P2": "majeure", "P3": "mineure"}

# Priority heuristic of the scripted plan: P1 when the request mentions an emergency, a
# safety risk or a total stop of the equipment. Matched on whole words and word
# sequences, never on raw substrings ("bien arrêté" is not "en arrêt").
_P1_STEMS = ("urge", "secur", "dang", "incend", "inond")
_P1_PHRASES = tuple(
    f" {word_sequence(phrase)} "
    for phrase in (
        "arrêt total",
        "arrêt complet",
        "à l'arrêt",
        "en arrêt",
        "hors service",
        "panne totale",
        "ne fonctionne plus",
        "ne marche plus",
        "odeur de gaz",
        "fuite de gaz",
    )
)
# "rien d'urgent", "pas urgent", "sans danger": a cue negated by one of the two words
# before it does not count.
_NEGATIONS = frozenset({"pas", "rien", "sans", "aucun", "aucune", "non", "ni"})


class Planner(Protocol):
    def next_message(self, messages: list[dict]) -> LLMResult:
        """Next assistant turn for the conversation `messages` (OpenAI chat format)."""
        ...


class LLMPlanner:
    """A model chooses the tool calls and writes the final draft."""

    def __init__(self, ctx: RequestContext) -> None:
        self._ctx = ctx

    def next_message(self, messages: list[dict]) -> LLMResult:
        result = llm.complete(
            "agent",
            messages,
            tools=tools.TOOL_SCHEMAS,
            max_tokens=get_settings().agent_max_tokens,
        )
        self._ctx.add_llm(result)
        if result.finish_reason == "length":
            # A draft cut at the token limit must not be shown as a finished one.
            raise llm.LLMError("OutputTruncated")
        return result


class ScriptedPlanner:
    """No LLM: a fixed sequence of tool calls, then a draft assembled from their results.

    One tool call per turn, in the order a handler would follow: documentation,
    contract, deadline, ticket proposal. The step is derived from the number of tool
    results already in the conversation, so the planner keeps no state of its own.
    """

    def __init__(self, tc: ToolContext) -> None:
        self._tc = tc
        self._priority = guess_priority(tc.q_masked)
        client_label = tc.client.client_id if tc.client else "client non identifié"
        self._plan: list[tuple[str, dict]] = [
            ("search_docs", {"query": tc.q_masked}),
            ("get_contract", {}),
            ("compute_deadline", {"priority": self._priority}),
            (
                "propose_ticket",
                {
                    "subject": f"Demande d'intervention {self._priority} — {client_label}",
                    "summary": tc.q_masked,
                    "priority": self._priority,
                },
            ),
        ]
        self.turns = len(self._plan) + 1  # one turn per tool call, then the draft

    def next_message(self, messages: list[dict]) -> LLMResult:
        results = [message["content"] for message in messages if message["role"] == "tool"]
        if len(results) < len(self._plan):
            name, arguments = self._plan[len(results)]
            return _tool_call_turn(f"scripted_{len(results) + 1}", name, arguments)
        draft = self._draft(contract=results[1], deadline=results[2])
        return LLMResult(text=draft, raw_message={"role": "assistant", "content": draft})

    def _draft(self, contract: str, deadline: str) -> str:
        priority = self._priority
        lines = [
            "Brouillon préparé sans LLM à partir des données du dossier, à relire avant envoi.",
            "",
            "Bonjour,",
            "",
            f"Nous avons bien reçu votre demande. Elle est qualifiée en priorité {priority} "
            f"({PRIORITY_LABELS[priority]}), sous réserve de confirmation par un gestionnaire.",
            "",
            "Contrat :",
            contract,
            "",
            "Délai d'intervention :",
            deadline,
            "",
        ]
        if self._tc.proposal is not None:
            lines += [
                f"Suite proposée : un ticket d'intervention en priorité {priority} est soumis "
                "à la validation d'un gestionnaire. Aucun ticket n'est créé à ce stade.",
                "",
            ]
        lines.append(_sources_line(self._tc))
        return "\n".join(lines)


def guess_priority(text: str) -> str:
    """P1 when the text mentions an emergency, a safety risk or a total stop, else P2.

    A deliberately simple rule for the mode without LLM: it does not understand the
    sentence ("en arrêt maladie" reads as a stop). The handler who validates the
    proposal confirms the priority or corrects it.
    """
    sequence = word_sequence(text)
    if any(phrase in f" {sequence} " for phrase in _P1_PHRASES):
        return "P1"
    sequence_words = sequence.split()
    for position, word in enumerate(sequence_words):
        negated = _NEGATIONS & set(sequence_words[max(0, position - 2) : position])
        if stem(word).startswith(_P1_STEMS) and not negated:
            return "P1"
    return "P2"


def _sources_line(tc: ToolContext) -> str:
    if not tc.sources:
        return "Aucun extrait de la documentation n'a été trouvé pour cette demande."
    labels = []
    for number, source in enumerate(tc.sources, start=1):
        chunk = source.chunk
        section = f", {chunk.section}" if chunk.section else ""
        labels.append(f"{chunk.title}{section} [{number}]")
    return f"Documentation consultée : {' ; '.join(labels)}."


def _tool_call_turn(call_id: str, name: str, arguments: dict) -> LLMResult:
    """An assistant turn holding one tool call, in the shape `llm.complete` returns."""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }
    return LLMResult(text="", tool_calls=[ToolCall(call_id, name, arguments)], raw_message=message)


# --- Graph ---------------------------------------------------------------------


class AgentState(TypedDict):
    # Conversation in OpenAI chat format. A node returns only the messages it adds;
    # the reducer appends them to the history.
    messages: Annotated[list[dict], operator.add]
    pending: list[ToolCall]  # tool calls of the latest planner turn, replaced every turn
    turns: int  # planner turns so far


def build_graph(planner: Planner, tc: ToolContext, max_steps: int) -> CompiledStateGraph:
    """The plan/tools loop for one request; stops after `max_steps` planner turns."""

    def plan(state: AgentState) -> dict:
        result = planner.next_message(state["messages"])
        return {
            "messages": [result.raw_message],
            "pending": result.tool_calls,
            "turns": state["turns"] + 1,
        }

    def run_tools(state: AgentState) -> dict:
        # One tool message per call, in order: a provider rejects a history in which an
        # assistant tool call has no matching result.
        return {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tools.run_tool(call.name, call.arguments, tc),
                }
                for call in state["pending"]
            ]
        }

    def after_plan(state: AgentState) -> str:
        if state["pending"] and state["turns"] < max_steps:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan, ["tools", END])
    graph.add_edge("tools", "plan")
    return graph.compile()


def _run(planner: Planner, tc: ToolContext, messages: list[dict], max_steps: int) -> str | None:
    """Run the loop; returns the final draft, or None when the planner did not conclude."""
    graph = build_graph(planner, tc, max_steps)
    state = graph.invoke(
        {"messages": messages, "pending": [], "turns": 0},
        # `max_steps` turns take 2 * max_steps - 1 graph steps. LangGraph's own limit is
        # set just above: a second guard that the turn counter always reaches first.
        config={"recursion_limit": 2 * max_steps + 1},
    )
    if state["pending"]:
        return None  # still asking for tools at the step limit
    draft = (state["messages"][-1].get("content") or "").strip()
    return draft or None


# --- Entry point ---------------------------------------------------------------


def run_agent(req: AskRequest, decision: TriageDecision, ctx: RequestContext) -> AskResponse:
    """Handle a dossier: gather facts with tools, draft a reply, propose a ticket.

    `req` is part of the signature shared by the route handlers and is not read: the
    agent works on the PII-masked text (`ctx.q_masked`) only.
    """
    # Checked first so that a missing index fails like the RAG route (`IndexNotReady`,
    # mapped to 503 by the API) instead of producing a draft without documentation.
    retriever = hybrid.get_retriever()
    # Contract data is read only for the client given in the request's client field.
    client = clients.get_client_repository().get(decision.scoped_client_id)
    if decision.client_known and client is None:
        ctx.log(
            f"Client {decision.client_id} cité dans le texte seulement : son contrat n'est pas "
            "consulté, le client est à confirmer"
        )

    answer = None
    if llm.is_enabled("agent"):
        tc, answer = _run_llm_planner(ctx, client)
    mode = "llm" if answer is not None else "deterministic"
    if answer is None:
        tc, answer = _run_scripted_planner(ctx, client)

    refs = guardrails.valid_refs(answer, len(tc.sources))
    unknown_refs = [ref for ref in guardrails.extract_refs(answer) if ref not in refs]
    if unknown_refs:
        ctx.log(f"Brouillon : référence(s) à des sources inexistantes {unknown_refs}, à vérifier")
    # A draft that cites nothing is shown with every source gathered, so the handler
    # can check it against the documentation.
    citations = guardrails.build_citations(tc.sources, refs or None)

    proposed = _save_proposal(ctx, tc)
    settings = get_settings()
    if ctx.usage.cost_eur > settings.max_cost_eur_per_request:
        ctx.log(
            f"RG-08 : coût de la requête ({ctx.usage.cost_eur:.4f} €) supérieur au budget "
            f"({settings.max_cost_eur_per_request:.4f} €)"
        )
    logger.info(
        "agent_done",
        extra={
            "mode": mode,
            "sources": len(tc.sources),
            "citations": len(citations),
            "proposal": proposed is not None,
        },
    )
    return AskResponse(
        request_id=ctx.request_id,
        route="agent",
        mode=mode,
        answer=answer,
        citations=citations,
        needs_validation=True,
        proposed_action=proposed,
        decision_log=list(ctx.decision_log),
        usage=ctx.usage,
        retrieval_mode=retriever.mode if tc.sources else "none",
    )


def _run_llm_planner(ctx: RequestContext, client: Client | None) -> tuple[ToolContext, str | None]:
    """Run the loop with the LLM planner; the draft is None when it failed to conclude."""
    tc = ToolContext(request_id=ctx.request_id, q_masked=ctx.q_masked, client=client)
    max_steps = max(1, get_settings().agent_max_steps)
    ctx.log(f"Agent : planification par LLM ({llm.model_for('agent')}), {max_steps} tour(s) max")
    answer = None
    failure = f"pas de réponse finale en {max_steps} tour(s)"
    try:
        answer = _run(LLMPlanner(ctx), tc, _initial_messages(tc), max_steps)
    except llm.LLMError as exc:
        failure = f"appel LLM en échec ({exc})"  # the message is the provider error class
    except llm.LLMUnavailable:
        failure = "LLM indisponible"
    for step in tc.steps:  # also after a failure: the tools that ran are part of the trail
        ctx.log(step)
    if answer is None:
        ctx.log(f"Agent : {failure}, reprise avec le plan déterministe")
    return tc, answer


def _run_scripted_planner(ctx: RequestContext, client: Client | None) -> tuple[ToolContext, str]:
    # A fresh context: source numbers and the proposal of a failed LLM attempt must
    # not leak into the scripted draft.
    tc = ToolContext(request_id=ctx.request_id, q_masked=ctx.q_masked, client=client)
    planner = ScriptedPlanner(tc)
    ctx.log("Agent : plan déterministe (sans LLM)")
    # The scripted plan has a fixed length and costs nothing: it runs to its end even
    # when `agent_max_steps` (the bound on LLM turns) is smaller.
    answer = _run(planner, tc, _initial_messages(tc), planner.turns)
    for step in tc.steps:
        ctx.log(step)
    if answer is None:
        # Cannot happen with the fixed plan; an empty answer must never be returned.
        raise RuntimeError("scripted plan ended without a draft")
    return tc, answer


def _initial_messages(tc: ToolContext) -> list[dict]:
    client_id = tc.client.client_id if tc.client else "non identifié"
    request = guardrails.neutralize_tags(tc.q_masked)
    return [
        {"role": "system", "content": loader.load("system_agent")},
        {
            "role": "user",
            "content": f"<demande>\n{request}\n</demande>\nClient du dossier : {client_id}",
        },
    ]


def _save_proposal(ctx: RequestContext, tc: ToolContext) -> ProposedAction | None:
    """Store the ticket proposal as a pending action; nothing else is written."""
    if tc.proposal is None:
        return None
    payload = {**tc.proposal, "client_id": tc.client.client_id if tc.client else None}
    with session_scope() as session:
        row = repositories.create_action(
            session, request_id=ctx.request_id, type="create_ticket", payload=payload
        )
        proposed = repositories.to_action(row)
    ctx.log(
        f"Action {proposed.action_id} : création de ticket proposée, validation humaine requise"
    )
    return proposed
