"""Dossier agent: priority heuristic, the graph loop, both planners and the fallbacks."""

import copy
import json
from datetime import datetime

import pytest

from app.agents import dossier_agent, tools
from app.agents.dossier_agent import ScriptedPlanner, build_graph, guess_priority, run_agent
from app.agents.tools import TOOL_SCHEMAS, ToolContext
from app.core import llm
from app.core.config import reset_settings
from app.core.prompts import loader
from app.core.schemas import AskRequest
from app.core.types import LLMResult, ToolCall, TriageDecision
from app.db import repositories
from app.db.session import session_scope
from app.domain.clients import get_client_repository
from app.retrieval.hybrid import IndexNotReady
from app.services.context import RequestContext

FRIDAY_17H = datetime(2026, 10, 2, 17, 0)

BREAKDOWN = "La chaudière de l'immeuble fonctionne mal depuis ce matin, merci d'intervenir."
URGENT = "La chaudière de l'immeuble est à l'arrêt total, c'est urgent. Rappeler au [TEL]."


@pytest.fixture(autouse=True)
def friday(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda: FRIDAY_17H)


def ask(q_masked: str, client_id: str | None):
    """Run the agent the way `handle_ask` does, with a hand-built triage decision."""
    ctx = RequestContext(q_masked=q_masked)
    known = get_client_repository().get(client_id) is not None
    decision = TriageDecision(
        route="agent",
        rule="RG-05",
        reasons=["RG-05 : vocabulaire de panne"],
        client_id=client_id,
        montant=None,
        client_known=known,
    )
    response = run_agent(AskRequest(q=q_masked, client_id=client_id), decision, ctx)
    return response, ctx


def stored_actions() -> list:
    with session_scope() as session:
        return [repositories.to_action(row) for row in repositories.list_actions(session)]


def stored_tickets() -> list:
    with session_scope() as session:
        return [repositories.to_ticket(row) for row in repositories.list_tickets(session)]


# --- Fake LLM ------------------------------------------------------------------


def tool_turn(*calls: tuple[str, str, dict]) -> LLMResult:
    """An assistant turn asking for tools, shaped like the result of `llm.complete`."""
    return LLMResult(
        text="",
        tool_calls=[ToolCall(call_id, name, arguments) for call_id, name, arguments in calls],
        model="fake/agent-model",
        prompt_tokens=100,
        completion_tokens=20,
        cost_eur=0.001,
        raw_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                for call_id, name, arguments in calls
            ],
        },
    )


def final_turn(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        model="fake/agent-model",
        prompt_tokens=300,
        completion_tokens=80,
        cost_eur=0.002,
        raw_message={"role": "assistant", "content": text},
    )


class FakeLLM:
    """Replaces `llm.complete`: plays scripted turns and records what it was sent."""

    def __init__(self, turns: list):
        self.turns = turns
        self.calls: list[dict] = []

    def complete(self, task, messages, *, tools=None, max_tokens=None):
        # A deep copy: the history must be checked as it was at the time of the call.
        self.calls.append(
            {
                "task": task,
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "max_tokens": max_tokens,
            }
        )
        turn = self.turns[min(len(self.calls), len(self.turns)) - 1]  # the last turn repeats
        if isinstance(turn, Exception):
            raise turn
        return turn


@pytest.fixture
def fake_llm(monkeypatch):
    def install(turns: list) -> FakeLLM:
        fake = FakeLLM(turns)
        monkeypatch.setattr(llm, "is_enabled", lambda task: True)
        monkeypatch.setattr(llm, "complete", fake.complete)
        return fake

    return install


def assert_valid_chat_history(messages: list[dict]) -> None:
    """OpenAI chat format: system, user, then assistant tool-call turns, each followed
    by exactly one tool message per call, in the same order."""
    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    i = 2
    while i < len(messages):
        assistant = messages[i]
        assert assistant["role"] == "assistant"
        calls = assistant["tool_calls"]  # a final answer is never sent back to the model
        assert calls
        for call in calls:
            assert call["type"] == "function"
            assert isinstance(json.loads(call["function"]["arguments"]), dict)
        results = messages[i + 1 : i + 1 + len(calls)]
        assert [m["role"] for m in results] == ["tool"] * len(calls)
        assert [m["tool_call_id"] for m in results] == [call["id"] for call in calls]
        assert all(isinstance(m["content"], str) and m["content"] for m in results)
        i += 1 + len(calls)


# --- Priority heuristic --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "C'est urgent, merci d'intervenir.",
        "Intervention en URGENCE demandée",
        "Il y a un risque pour la sécurité des occupants.",
        "Situation dangereuse dans la chaufferie",
        "La chaufferie est à l'arrêt depuis ce matin.",
        "La chaufferie est à l’arrêt depuis ce matin.",  # typographic apostrophe
        "Arrêt total de la ventilation",
        "Le groupe froid est hors service.",
        "La pompe ne fonctionne plus.",
        "Forte odeur de gaz au sous-sol",
        "Début d'inondation dans le local technique",
    ],
)
def test_guess_priority_p1(text):
    assert guess_priority(text) == "P1"


@pytest.mark.parametrize(
    "text",
    [
        "La chaudière fonctionne mal depuis ce matin.",
        "Une chaudière sur deux est en défaut.",
        "Pouvez-vous programmer la visite annuelle ?",
        "La chaudière à gaz fait un bruit anormal.",
        "Le voyant d'entretien est allumé, rien de pressant.",
    ],
)
def test_guess_priority_p2_by_default(text):
    assert guess_priority(text) == "P2"


# --- Graph loop ----------------------------------------------------------------


class StubPlanner:
    def __init__(self, turns: list[LLMResult]):
        self.turns = turns
        self.seen: list[int] = []

    def next_message(self, messages):
        self.seen.append(len(messages))
        return self.turns[len(self.seen) - 1]


def run_graph(planner, tc, max_steps):
    start = {"messages": [{"role": "user", "content": "x"}], "pending": [], "turns": 0}
    return build_graph(planner, tc, max_steps).invoke(start)


def test_graph_has_two_nodes_and_one_conditional_edge(db):
    tc = ToolContext("req-1", "x", None)
    drawn = build_graph(StubPlanner([]), tc, 3).get_graph()
    assert set(drawn.nodes) == {"__start__", "plan", "tools", "__end__"}
    edges = {(edge.source, edge.target, edge.conditional) for edge in drawn.edges}
    assert edges == {
        ("__start__", "plan", False),
        ("plan", "tools", True),
        ("plan", "__end__", True),
        ("tools", "plan", False),
    }


def test_graph_runs_tools_then_ends_on_a_message_without_tool_calls(db):
    tc = ToolContext("req-1", "x", None)
    planner = StubPlanner(
        [
            tool_turn(("a", "get_contract", {}), ("b", "compute_deadline", {"priority": "P2"})),
            final_turn("Brouillon."),
        ]
    )
    state = run_graph(planner, tc, max_steps=6)

    assert state["turns"] == 2
    assert state["pending"] == []
    assert [m["role"] for m in state["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [m.get("tool_call_id") for m in state["messages"][2:4]] == ["a", "b"]
    assert state["messages"][2]["content"].startswith("Client non identifié")
    assert state["messages"][-1]["content"] == "Brouillon."
    assert planner.seen == [1, 4], "the planner sees the tool results of the previous turn"


def test_graph_gives_tool_errors_back_to_the_planner(db):
    tc = ToolContext("req-1", "x", None)
    planner = StubPlanner(
        [
            tool_turn(("a", "drop_database", {})),
            tool_turn(("b", "search_docs", {})),
            final_turn("Fin."),
        ]
    )
    state = run_graph(planner, tc, max_steps=6)

    tool_results = [m["content"] for m in state["messages"] if m["role"] == "tool"]
    assert tool_results[0].startswith("Erreur : cet outil n'existe pas.")
    assert tool_results[1].startswith("Erreur dans les arguments de search_docs")
    assert state["messages"][-1]["content"] == "Fin."


def test_graph_stops_after_max_steps_planner_turns(db):
    tc = ToolContext("req-1", "x", None)
    planner = StubPlanner([tool_turn((f"call-{n}", "get_contract", {})) for n in range(10)])
    state = run_graph(planner, tc, max_steps=3)

    assert state["turns"] == 3
    assert len(planner.seen) == 3
    assert state["pending"], "the third turn still asked for a tool: no final answer"
    assert len(tc.steps) == 2, "the tool calls of the last turn are not executed"


# --- Scripted planner ----------------------------------------------------------


def test_scripted_planner_emits_the_fixed_sequence(indexed):
    tc = ToolContext("req-1", URGENT, get_client_repository().get("C-12"))
    planner = ScriptedPlanner(tc)
    messages = [{"role": "user", "content": "x"}]
    emitted = []
    for _ in range(4):
        result = planner.next_message(messages)
        (call,) = result.tool_calls
        emitted.append((call.name, call.arguments))
        messages.append(result.raw_message)
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": f"résultat {call.name}"}
        )

    assert emitted == [
        ("search_docs", {"query": URGENT}),
        ("get_contract", {}),
        ("compute_deadline", {"priority": "P1"}),
        (
            "propose_ticket",
            {"subject": "Demande d'intervention P1 — C-12", "summary": URGENT, "priority": "P1"},
        ),
    ]
    assert_valid_chat_history([{"role": "system", "content": "s"}, *messages])
    final = planner.next_message(messages)
    assert final.tool_calls == []
    assert "résultat get_contract" in final.text and "résultat compute_deadline" in final.text
    assert final.raw_message == {"role": "assistant", "content": final.text}


def test_scripted_agent_end_to_end(indexed):
    response, ctx = ask(URGENT, "C-12")

    assert (response.route, response.mode) == ("agent", "deterministic")
    assert response.needs_validation is True
    assert response.request_id == ctx.request_id
    assert response.retrieval_mode == "hybrid"
    assert response.usage.llm_calls == 0 and response.usage.cost_eur == 0.0

    answer = response.answer
    assert answer.startswith("Brouillon préparé sans LLM")
    assert "priorité P1 (critique)" in answer
    assert "Formule : Confort" in answer
    assert "contrat actif" in answer
    assert "délai contractuel d'intervention de 4 h ouvrées" in answer
    assert "intervention au plus tard le lundi 05/10/2026 à 11 h 00" in answer
    assert "Aucun ticket n'est créé à ce stade" in answer

    # Every gathered source is cited in the draft and returned as a citation.
    assert response.citations
    assert [c.ref for c in response.citations] == list(range(1, len(response.citations) + 1))
    for citation in response.citations:
        assert f"[{citation.ref}]" in answer
    docs = {c.doc for c in response.citations}
    assert "contrat_C-12.md" in docs
    assert "contrat_C-34.md" not in docs

    # The proposal is stored as a pending action; no ticket exists.
    expected_payload = {
        "subject": "Demande d'intervention P1 — C-12",
        "summary": URGENT,
        "priority": "P1",
        "client_id": "C-12",
    }
    action = response.proposed_action
    assert (action.type, action.status, action.ticket_id) == ("create_ticket", "pending", None)
    assert action.payload == expected_payload
    assert stored_actions() == [action]
    with session_scope() as session:
        row = repositories.get_action(session, action.action_id)
        assert (row.request_id, row.status, row.payload) == (
            ctx.request_id,
            "pending",
            expected_payload,
        )
    assert stored_tickets() == []


def test_scripted_agent_decision_log(indexed):
    response, ctx = ask(BREAKDOWN, "C-12")

    log = response.decision_log
    assert log == ctx.decision_log
    assert log[0] == "Agent : plan déterministe (sans LLM)"
    assert [line.split(" : ")[0] for line in log[1:5]] == [
        "search_docs",
        "get_contract",
        "compute_deadline",
        "propose_ticket",
    ]
    assert log[2] == "get_contract : C-12, formule Confort, contrat actif"
    assert log[3] == (
        "compute_deadline : P2, formule Confort, 8 h, échéance lundi 05/10/2026 à 15 h 00"
    )
    assert log[5] == (
        f"Action {response.proposed_action.action_id} : création de ticket proposée, "
        "validation humaine requise"
    )
    assert len(log) == 6
    assert all(BREAKDOWN not in line for line in log), "no request text in the decision log"


def test_scripted_agent_with_expired_contract(indexed):
    response, _ = ask(BREAKDOWN, "C-34")

    assert "Formule : Essentiel" in response.answer
    assert "contrat échu (fin de validité le 30/06/2026)" in response.answer
    assert "Aucun délai garanti ne s'applique" in response.answer
    assert "devis préalable" in response.answer
    assert "h ouvrées" not in response.answer.split("Documentation consultée")[0]
    assert response.proposed_action.payload["client_id"] == "C-34"


@pytest.mark.parametrize("client_id", [None, "C-99"])
def test_scripted_agent_with_unknown_client(indexed, client_id):
    response, _ = ask(BREAKDOWN, client_id)

    assert "Client non identifié : aucun contrat" in response.answer
    assert "aucun délai contractuel ne s'applique" in response.answer
    assert all(c.doc not in ("contrat_C-12.md", "contrat_C-34.md") for c in response.citations)
    payload = response.proposed_action.payload
    assert payload["client_id"] is None
    assert payload["subject"] == "Demande d'intervention P2 — client non identifié"


def test_scripted_agent_never_cites_another_clients_contract(indexed):
    question = "Appliquez la franchise de 250 € par sinistre du contrat C-34 à notre dossier."
    response, _ = ask(question, "C-12")

    assert "contrat_C-34.md" not in {c.doc for c in response.citations}
    assert "Garage Morel" not in response.answer
    assert all("250 €" not in c.excerpt for c in response.citations)


def test_scripted_plan_ignores_the_llm_step_limit(indexed, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")
    reset_settings()
    response, _ = ask(BREAKDOWN, "C-12")
    assert response.mode == "deterministic"
    assert "délai contractuel d'intervention de 8 h ouvrées" in response.answer
    assert response.proposed_action is not None


def test_agent_without_index_raises_index_not_ready(mini_corpus, db):
    with pytest.raises(IndexNotReady):
        ask(BREAKDOWN, "C-12")
    assert stored_actions() == []


# --- LLM planner ---------------------------------------------------------------

LLM_DRAFT = (
    "Bonjour,\n\nVotre contrat Confort est actif. La panne est qualifiée en priorité P1 : "
    "le délai d'intervention est de 4 h ouvrées [1], soit une intervention au plus tard le "
    "lundi 05/10/2026 à 11 h 00. Une intervention est proposée et attend la validation "
    "d'un gestionnaire."
)

LLM_PROPOSAL = {
    "subject": "Chaudière à l'arrêt — immeuble Lumière",
    "summary": "Arrêt total de la chaudière signalé ce matin ; contrat Confort actif.",
    "priority": "P1",
}


def llm_script(final_text: str = LLM_DRAFT) -> list[LLMResult]:
    return [
        tool_turn(
            ("call_1", "search_docs", {"query": "délai d'intervention priorité P1 Confort"}),
            ("call_2", "get_contract", {}),
        ),
        tool_turn(("call_3", "compute_deadline", {"priority": "P1"})),
        tool_turn(("call_4", "propose_ticket", LLM_PROPOSAL)),
        final_turn(final_text),
    ]


def test_llm_agent_end_to_end(indexed, fake_llm, env):
    fake = fake_llm(llm_script())
    response, ctx = ask(URGENT, "C-12")

    assert (response.route, response.mode) == ("agent", "llm")
    assert response.answer == LLM_DRAFT
    assert response.needs_validation is True

    # Only the source cited by the draft is returned.
    assert [c.ref for c in response.citations] == [1]
    assert response.citations[0].doc == "procedure_sav.md"

    assert response.proposed_action.status == "pending"
    assert response.proposed_action.payload == {**LLM_PROPOSAL, "client_id": "C-12"}
    assert stored_actions() == [response.proposed_action]
    assert stored_tickets() == []

    usage = response.usage
    assert (usage.llm_calls, usage.model) == (4, "fake/agent-model")
    assert (usage.prompt_tokens, usage.completion_tokens) == (600, 140)
    assert usage.cost_eur == pytest.approx(0.005)

    assert len(fake.calls) == 4
    for call in fake.calls:
        assert call["task"] == "agent"
        assert call["tools"] == TOOL_SCHEMAS
        assert call["max_tokens"] == env.agent_max_tokens

    log = response.decision_log
    assert log[0].startswith("Agent : planification par LLM (")
    assert [line.split(" : ")[0] for line in log[1:5]] == [
        "search_docs",
        "get_contract",
        "compute_deadline",
        "propose_ticket",
    ]
    assert not any("déterministe" in line for line in log)


def test_llm_agent_sends_a_valid_chat_history(indexed, fake_llm):
    fake = fake_llm(llm_script())
    ask(URGENT, "C-12")

    first = fake.calls[0]["messages"]
    assert first == [
        {"role": "system", "content": loader.load("system_agent")},
        {"role": "user", "content": f"<demande>\n{URGENT}\n</demande>\nClient du dossier : C-12"},
    ]
    for call in fake.calls:
        assert_valid_chat_history(call["messages"])
    # History length per call: 2, +1 assistant +2 tools, +1 assistant +1 tool, +1 +1.
    assert [len(call["messages"]) for call in fake.calls] == [2, 5, 7, 9]

    last = fake.calls[-1]["messages"]
    by_id = {m["tool_call_id"]: m["content"] for m in last if m["role"] == "tool"}
    assert set(by_id) == {"call_1", "call_2", "call_3", "call_4"}
    assert by_id["call_1"].startswith("[1] Procédure SAV — procedure_sav.md")
    assert "Formule : Confort" in by_id["call_2"]
    assert "lundi 05/10/2026 à 11 h 00" in by_id["call_3"]
    assert "Aucun ticket n'est créé à ce stade" in by_id["call_4"]


def test_llm_agent_user_message_for_an_unknown_client(indexed, fake_llm):
    fake = fake_llm([final_turn("Client à identifier avant toute intervention.")])
    ask(BREAKDOWN, "C-99")
    assert fake.calls[0]["messages"][1]["content"] == (
        f"<demande>\n{BREAKDOWN}\n</demande>\nClient du dossier : non identifié"
    )


def test_llm_draft_without_citation_cites_every_gathered_source(indexed, fake_llm):
    fake_llm(llm_script("Bonjour, une intervention est proposée."))
    response, _ = ask(URGENT, "C-12")

    assert response.mode == "llm"
    assert len(response.citations) >= 2
    assert [c.ref for c in response.citations] == list(range(1, len(response.citations) + 1))


def test_llm_draft_citing_an_unknown_source_is_flagged(indexed, fake_llm):
    fake_llm(llm_script("Le délai est de 4 h ouvrées [1] et le déplacement est offert [9]."))
    response, _ = ask(URGENT, "C-12")

    assert [c.ref for c in response.citations] == [1]
    assert "Brouillon : référence(s) à des sources inexistantes [9], à vérifier" in (
        response.decision_log
    )


def test_llm_agent_without_proposal_stores_no_action(indexed, fake_llm):
    fake_llm([tool_turn(("call_1", "get_contract", {})), final_turn("Contrat actif.")])
    response, _ = ask("Pouvez-vous vérifier notre contrat ?", "C-12")

    assert response.mode == "llm"
    assert response.proposed_action is None
    assert stored_actions() == []
    assert response.retrieval_mode == "none", "no document search was made"
    assert response.citations == []


def test_llm_cost_above_the_budget_is_flagged(indexed, fake_llm, monkeypatch):
    monkeypatch.setenv("MAX_COST_EUR_PER_REQUEST", "0.004")
    reset_settings()
    fake_llm(llm_script())
    response, _ = ask(URGENT, "C-12")
    assert response.decision_log[-1] == (
        "RG-08 : coût de la requête (0.0050 €) supérieur au budget (0.0040 €)"
    )


# --- Fallbacks -------------------------------------------------------------------


def test_step_limit_falls_back_to_the_scripted_plan(indexed, fake_llm, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    reset_settings()
    # A model that never concludes: it proposes a ticket, then searches again and again.
    fake = fake_llm(
        [
            tool_turn(("call_1", "propose_ticket", LLM_PROPOSAL)),
            tool_turn(("call_n", "search_docs", {"query": "délai d'intervention"})),
        ]
    )
    response, _ = ask(URGENT, "C-12")

    assert len(fake.calls) == 3, "hard stop after agent_max_steps planner turns"
    assert response.mode == "deterministic"
    assert response.answer.startswith("Brouillon préparé sans LLM")
    assert "intervention au plus tard le lundi 05/10/2026 à 11 h 00" in response.answer
    assert response.usage.llm_calls == 3, "the failed attempt is still accounted for"

    log = response.decision_log
    fallback = "Agent : pas de réponse finale en 3 tour(s), reprise avec le plan déterministe"
    assert fallback in log
    assert log.index(fallback) < log.index("Agent : plan déterministe (sans LLM)")

    # Only the proposal of the scripted plan is stored, not the one of the failed attempt.
    (action,) = stored_actions()
    assert action.payload["subject"] == "Demande d'intervention P1 — C-12"
    assert response.proposed_action == action
    # Source numbers of the draft come from the scripted run alone.
    assert [c.ref for c in response.citations] == list(range(1, len(response.citations) + 1))


def test_llm_error_falls_back_to_the_scripted_plan(indexed, fake_llm):
    fake = fake_llm(
        [tool_turn(("call_1", "propose_ticket", LLM_PROPOSAL)), llm.LLMError("RateLimitError")]
    )
    response, _ = ask(URGENT, "C-12")

    assert len(fake.calls) == 2
    assert response.mode == "deterministic"
    assert response.answer.startswith("Brouillon préparé sans LLM")
    assert "délai contractuel d'intervention de 4 h ouvrées" in response.answer

    log = response.decision_log
    assert log[1].startswith("propose_ticket : "), "tools run before the failure stay in the trail"
    assert log[2] == (
        "Agent : appel LLM en échec (RateLimitError), reprise avec le plan déterministe"
    )
    assert log[3] == "Agent : plan déterministe (sans LLM)"

    (action,) = stored_actions()
    assert action.payload["subject"] == "Demande d'intervention P1 — C-12"
    assert response.usage.llm_calls == 1
    assert stored_tickets() == []


def test_llm_unavailable_falls_back_to_the_scripted_plan(indexed, fake_llm):
    fake_llm([llm.LLMUnavailable("no key")])
    response, _ = ask(BREAKDOWN, "C-12")

    assert response.mode == "deterministic"
    assert "Agent : LLM indisponible, reprise avec le plan déterministe" in response.decision_log
    assert "no key" not in " ".join(response.decision_log)


def test_empty_final_message_falls_back_to_the_scripted_plan(indexed, fake_llm):
    fake_llm([final_turn("   ")])
    response, _ = ask(BREAKDOWN, "C-12")
    assert response.mode == "deterministic"
    assert response.answer.startswith("Brouillon préparé sans LLM")


def test_llm_is_not_called_when_disabled(indexed, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("llm.complete must not be called when the LLM is disabled")

    monkeypatch.setattr(llm, "complete", forbidden)
    response, _ = ask(BREAKDOWN, "C-12")
    assert response.mode == "deterministic"
    assert dossier_agent.llm.is_enabled("agent") is False
