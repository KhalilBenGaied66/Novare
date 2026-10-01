"""Operating figures computed from the request log, the feedback and the pending actions."""

import math
from collections import Counter

from app.core.schemas import FeedbackStats, MetricsResponse
from app.db import repositories, session


def compute_metrics() -> MetricsResponse:
    """Figures over the most recent requests (at most 5000, the repository's limit)."""
    with session.session_scope() as db:
        rows = repositories.list_requests(db)
        feedback = repositories.feedback_stats(db)
        pending = len(repositories.list_actions(db, status="pending", limit=10_000))
        n = len(rows)
        latencies = sorted(row.latency_ms for row in rows)
        cost_total = sum(row.cost_eur for row in rows)
        return MetricsResponse(
            requests=n,
            by_route=dict(Counter(row.route for row in rows)),
            by_mode=dict(Counter(row.mode for row in rows)),
            latency_ms_p50=_percentile(latencies, 50),
            latency_ms_p95=_percentile(latencies, 95),
            cost_eur_total=round(cost_total, 6),
            cost_eur_avg=round(cost_total / n, 6) if n else None,
            llm_call_share=_share(sum(1 for row in rows if row.llm_calls > 0), n),
            escalation_rate=_share(sum(1 for row in rows if row.route == "human"), n),
            errors=sum(1 for row in rows if row.error),
            feedback=FeedbackStats(**feedback),
            pending_actions=pending,
        )


def _percentile(sorted_values: list[int], percent: int) -> int | None:
    """Nearest-rank percentile: the smallest value with at least `percent` % at or below it."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(percent / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def _share(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None
