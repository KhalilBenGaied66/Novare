"""Operating figures computed from the request log."""

from app.core.schemas import AskRequest
from app.db import repositories, session
from app.services import ask, metrics
from app.services.metrics import _percentile


def test_percentile_uses_the_nearest_rank():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    assert _percentile([], 50) is None
    assert _percentile([7], 95) == 7
    assert _percentile(values, 50) == 50
    assert _percentile(values, 95) == 100
    assert _percentile(values, 10) == 10


def test_metrics_of_an_empty_log(db):
    result = metrics.compute_metrics()

    assert result.requests == 0
    assert result.by_route == {} and result.by_mode == {}
    assert result.latency_ms_p50 is None and result.cost_eur_avg is None
    assert result.llm_call_share is None and result.escalation_rate is None
    assert result.feedback.n == 0 and result.pending_actions == 0


def test_metrics_count_routes_modes_escalations_and_pending_actions(indexed):
    ask.handle_ask(AskRequest(q="Je conteste la facture de 120 €.", client_id="C-12"))
    ask.handle_ask(AskRequest(q="Quelle est la majoration pour un déplacement le week-end ?"))
    ask.handle_ask(AskRequest(q="La chaudière est en panne, merci d'intervenir.", client_id="C-12"))
    ask.handle_ask(AskRequest(q="Je souhaite résilier mon contrat."))

    result = metrics.compute_metrics()

    assert result.requests == 4
    assert result.by_route == {"automation": 1, "rag": 1, "agent": 1, "human": 1}
    assert result.by_mode == {"rule": 1, "extractive": 1, "deterministic": 1, "none": 1}
    assert result.escalation_rate == 0.25
    assert result.llm_call_share == 0.0
    assert result.cost_eur_total == 0.0 and result.cost_eur_avg == 0.0
    assert result.errors == 0
    assert result.pending_actions == 1
    assert result.latency_ms_p95 >= result.latency_ms_p50 >= 0


def test_metrics_report_llm_cost_and_errors_from_the_log(db):
    common = {"route": "rag", "mode": "llm", "retrieval_mode": "hybrid"}
    with session.session_scope() as database:
        repositories.log_request(
            database, request_id="a", latency_ms=100, llm_calls=1, cost_eur=0.004, **common
        )
        repositories.log_request(
            database, request_id="b", latency_ms=300, llm_calls=2, cost_eur=0.006, **common
        )
        repositories.log_request(
            database, request_id="c", route="human", mode="none", latency_ms=5, error="ValueError"
        )

    result = metrics.compute_metrics()

    assert result.requests == 3
    assert result.cost_eur_total == 0.01
    assert result.cost_eur_avg == round(0.01 / 3, 6)
    assert result.llm_call_share == round(2 / 3, 4)
    assert result.errors == 1
    assert (result.latency_ms_p50, result.latency_ms_p95) == (100, 300)
