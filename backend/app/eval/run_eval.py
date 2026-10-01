"""Evaluation of the real pipeline on the golden set.

Every case goes through `triage` and `handle_ask` exactly as an API request would, on an
index rebuilt from the configured corpus and with a temporary database. The report gives
the metrics per split and lists every failed check. Rules and thresholds are tuned on
`dev`; `test` holds cases written afterwards, except where docs/05-evaluation.md says a
comparison used both. The gate compares the metrics of the whole set with
`evals/thresholds.json`.

    python -m app.eval.run_eval [--split dev|test|all] [--judge] [--no-gate]

One report per configuration is written under evals/reports: `bm25`, `hybrid`, with the
suffix `-llm` when a model answered.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from app.agents import triage
from app.core import guardrails, llm, pii
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.schemas import AskRequest
from app.db import session
from app.eval import judge as judge_module
from app.eval import metrics
from app.ingestion import ingest
from app.retrieval import hybrid
from app.services import ask

# Metrics that must stay at or above their threshold / at or below it.
MIN_METRICS = (
    "routing_accuracy",
    "final_route_accuracy",
    "retrieval_hit_rate",
    "retrieval_mrr",
    "citation_hit_rate",
    "fact_accuracy",
    "pii_accuracy",
)
MAX_METRICS = ("forbidden_doc_violations", "forbidden_fact_violations")


@dataclass
class CaseResult:
    id: str
    split: str
    category: str
    expected_route: str
    route: str  # route chosen by triage
    final_route: str  # route of the response
    mode: str
    latency_ms: int
    cost_eur: float
    # One entry per check that applies to the case: True = passed.
    checks: dict[str, bool] = field(default_factory=dict)
    reciprocal_rank: float | None = None
    faithful: bool | None = None
    failures: list[str] = field(default_factory=list)


def load_golden_set(path: Path | None = None) -> list[dict]:
    path = path or get_settings().evals_dir / "golden_set.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_case(case: dict, judge: bool = False) -> CaseResult:
    """Run one golden case through the pipeline and score every check that applies."""
    req = AskRequest(**case["input"])
    decision = triage.triage(req)
    response = ask.handle_ask(req)
    result = CaseResult(
        id=case["id"],
        split=case["split"],
        category=case["category"],
        expected_route=case["expected_route"],
        route=decision.route,
        final_route=response.route,
        mode=response.mode,
        latency_ms=response.latency_ms,
        cost_eur=response.usage.cost_eur,
    )

    def check(name: str, passed: bool, detail: str) -> None:
        result.checks[name] = passed
        if not passed:
            result.failures.append(detail)

    check(
        "routing",
        decision.route == case["expected_route"],
        f"route {decision.route}, attendue {case['expected_route']}",
    )
    check(
        "final_route",
        response.route == case["expected_final_route"],
        f"route finale {response.route}, attendue {case['expected_final_route']}",
    )
    check(
        "pii",
        response.pii_redacted == sorted(case["expected_pii"]),
        f"données masquées {response.pii_redacted}, attendues {sorted(case['expected_pii'])}",
    )

    # Same search as the pipeline: masked question without tags, client-scoped.
    query = pii.strip_tags(pii.redact(req.q)[0])
    client_id = decision.scoped_client_id
    retrieved = hybrid.get_retriever().search(query, client_id=client_id)
    retrieved_docs = [item.chunk.doc for item in retrieved]
    cited_docs = [citation.doc for citation in response.citations]

    if case["expected_docs"]:
        check(
            "retrieval",
            metrics.hit(case["expected_docs"], retrieved_docs),
            f"aucun document attendu dans les sources ({', '.join(dict.fromkeys(retrieved_docs))})",
        )
        result.reciprocal_rank = metrics.reciprocal_rank(case["expected_docs"], retrieved_docs)
        if case["expected_final_route"] in ("rag", "agent"):
            # A request that should have been answered and was escalated cites nothing:
            # it counts as a miss here, not as a case left out.
            check(
                "citation",
                metrics.hit(case["expected_docs"], cited_docs),
                f"aucun document attendu cité ({', '.join(dict.fromkeys(cited_docs)) or 'rien'})",
            )
    if case["forbidden_docs"]:
        leaked = sorted(set(case["forbidden_docs"]) & set(retrieved_docs + cited_docs))
        check("forbidden_docs", not leaked, f"document interdit atteint : {', '.join(leaked)}")
    if case["expected_facts"]:
        missing = metrics.missing_facts(response.answer, case["expected_facts"])
        check("facts", not missing, f"faits absents de la réponse : {missing}")
    if case["forbidden_facts"]:
        present = metrics.present_facts(response.answer, case["forbidden_facts"])
        check("forbidden_facts", not present, f"faits interdits dans la réponse : {present}")

    if judge and response.route == "rag" and response.mode == "llm":
        sources = guardrails.format_sources(retrieved)
        result.faithful = judge_module.judge_faithfulness(query, sources, response.answer)
        if result.faithful is False:
            result.failures.append("réponse jugée non fidèle aux sources")
    return result


def summarise(results: list[CaseResult]) -> dict:
    """Metrics of a list of case results."""

    def outcomes(name: str) -> list[bool]:
        return [r.checks[name] for r in results if name in r.checks]

    ranks = [r.reciprocal_rank for r in results if r.reciprocal_rank is not None]
    verdicts = [r.faithful for r in results if r.faithful is not None]
    latencies = [r.latency_ms for r in results]
    cost = sum(r.cost_eur for r in results)
    return {
        "cases": len(results),
        "routing_accuracy": metrics.rate(outcomes("routing")),
        "final_route_accuracy": metrics.rate(outcomes("final_route")),
        "retrieval_hit_rate": metrics.rate(outcomes("retrieval")),
        "retrieval_mrr": round(sum(ranks) / len(ranks), 4) if ranks else None,
        "citation_hit_rate": metrics.rate(outcomes("citation")),
        "fact_accuracy": metrics.rate(outcomes("facts")),
        "pii_accuracy": metrics.rate(outcomes("pii")),
        "forbidden_doc_violations": outcomes("forbidden_docs").count(False),
        "forbidden_fact_violations": outcomes("forbidden_facts").count(False),
        "faithfulness": metrics.rate(verdicts),
        "judged_answers": len(verdicts),
        "latency_ms_mean": round(sum(latencies) / len(latencies)) if latencies else None,
        "latency_ms_p95": metrics.percentile(latencies, 95),
        "cost_eur_total": round(cost, 6),
        "cost_eur_mean": round(cost / len(results), 6) if results else None,
        "routing_confusion": metrics.confusion([(r.expected_route, r.route) for r in results]),
    }


def check_gate(summary: dict, thresholds: dict) -> list[str]:
    """Threshold misses, in French; empty when the gate passes."""
    misses = []
    for name, minimum in thresholds.get("min", {}).items():
        value = summary.get(name)
        if value is None or value < minimum:
            misses.append(f"{name} = {value} (minimum {minimum})")
    for name, maximum in thresholds.get("max", {}).items():
        value = summary.get(name)
        if value is None or value > maximum:
            misses.append(f"{name} = {value} (maximum {maximum})")
    return misses


def run_eval(split: str = "all", judge: bool = False) -> dict:
    """Evaluate the golden set and return the report (also written under evals/reports)."""
    # The application database must not receive the tickets created by the cases.
    with session.temporary_database():
        setup_logging()  # resetting the settings removed the log handler
        _ensure_index()
        cases = [c for c in load_golden_set() if split == "all" or c["split"] == split]
        results = [evaluate_case(case, judge=judge) for case in cases]
        report = _build_report(results, split)
    _write_report(report)
    return report


def _ensure_index() -> None:
    """Rebuild the index, so that the evaluation measures the corpus as it is now.

    An index left on disk could predate a change of documents, of chunking or of
    retrieval mode.
    """
    ingest.ingest_docs()


def _build_report(results: list[CaseResult], split: str) -> dict:
    settings = get_settings()
    status = hybrid.index_status()
    llm_models = {
        task: llm.model_for(task) for task in ("rag", "agent", "judge") if llm.is_enabled(task)
    }
    thresholds_path = settings.evals_dir / "thresholds.json"
    thresholds = (
        json.loads(thresholds_path.read_text(encoding="utf-8")) if thresholds_path.exists() else {}
    )
    summary = summarise(results)
    # The gate is defined on the whole set: a partial run is measured, not gated.
    misses = check_gate(summary, thresholds) if split == "all" else []
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "split": split,
            "retrieval_mode": status["mode"],
            "embedding_model": status.get("embedding_model"),
            "vector_store": status.get("vector_store"),
            "chunks": status["chunks"],
            "top_k": settings.top_k,
            "min_confidence": settings.min_confidence,
            "llm": llm_models or None,
        },
        "metrics": {
            "all": summary,
            "dev": summarise([r for r in results if r.split == "dev"]),
            "test": summarise([r for r in results if r.split == "test"]),
        },
        "gate": {"applied": split == "all", "passed": not misses, "misses": misses},
        "failures": [
            {"id": r.id, "split": r.split, "category": r.category, "reasons": r.failures}
            for r in results
            if r.failures
        ],
        "results": [asdict(r) for r in results],
    }


def report_name(report: dict) -> str:
    """File name of a report: the configuration it measured ("bm25", "hybrid-llm")."""
    config = report["config"]
    return config["retrieval_mode"] + ("-llm" if config["llm"] else "")


def _write_report(report: dict) -> None:
    reports_dir = get_settings().evals_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = report_name(report)
    (reports_dir / f"{name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / f"{name}.md").write_text(render_markdown(report), encoding="utf-8")


_METRIC_LABELS = {
    "cases": "Cas",
    "routing_accuracy": "Routage (triage)",
    "final_route_accuracy": "Route finale",
    "retrieval_hit_rate": "Document attendu retrouvé (top-k)",
    "retrieval_mrr": "MRR",
    "citation_hit_rate": "Document attendu cité (demandes à répondre)",
    "fact_accuracy": "Faits attendus dans la réponse",
    "pii_accuracy": "Données personnelles détectées",
    "forbidden_doc_violations": "Documents interdits atteints",
    "forbidden_fact_violations": "Faits interdits dans la réponse",
    "faithfulness": "Fidélité (juge LLM)",
    "latency_ms_mean": "Latence moyenne (ms, indicatif)",
    "latency_ms_p95": "Latence p95 (ms, indicatif)",
    "cost_eur_total": "Coût total (€)",
}


def render_markdown(report: dict) -> str:
    config = report["config"]
    llm_line = (
        ", ".join(f"{task} = {model}" for task, model in config["llm"].items())
        if config["llm"]
        else "aucun LLM (réponses extractives et plan d'agent déterministe)"
    )
    lines = [
        "# Rapport d'évaluation",
        "",
        f"- Date : {report['generated_at']}",
        f"- Recherche : {config['retrieval_mode']}"
        + (f" ({config['embedding_model']})" if config["retrieval_mode"] == "hybrid" else ""),
        f"- Index : {config['chunks']} passages, top-k {config['top_k']}, "
        f"seuil de confiance {config['min_confidence']}",
        f"- Modèles : {llm_line}",
        "",
        "| Mesure | Tous | dev | test |",
        "|---|---|---|---|",
    ]
    for name, label in _METRIC_LABELS.items():
        cells = [_cell(report["metrics"][split].get(name)) for split in ("all", "dev", "test")]
        lines.append(f"| {label} | {' | '.join(cells)} |")

    gate = report["gate"]
    lines += ["", "## Seuils"]
    if not gate["applied"]:
        lines.append("Non appliqués : exécution partielle.")
    elif gate["passed"]:
        lines.append("Tous les seuils de `evals/thresholds.json` sont respectés.")
    else:
        lines += [f"- Non respecté : {miss}" for miss in gate["misses"]]

    lines += ["", f"## Cas en échec ({len(report['failures'])})"]
    for failure in report["failures"]:
        reasons = " ; ".join(failure["reasons"])
        lines.append(f"- {failure['id']} ({failure['split']}, {failure['category']}) : {reasons}")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return "—" if value is None else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Évalue le pipeline sur le jeu de référence.")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--judge", action="store_true", help="juge de fidélité (LLM requis)")
    parser.add_argument("--no-gate", action="store_true", help="ne pas échouer sous les seuils")
    args = parser.parse_args(argv)

    report = run_eval(split=args.split, judge=args.judge)
    sys.stdout.reconfigure(encoding="utf-8")
    print(render_markdown(report))
    if report["gate"]["applied"] and not report["gate"]["passed"] and not args.no_gate:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
