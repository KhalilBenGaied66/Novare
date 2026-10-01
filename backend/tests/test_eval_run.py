"""The evaluation harness runs the real pipeline, and its gate can fail."""

import json

import pytest

from app.core import llm
from app.core.types import LLMResult
from app.db import repositories, session
from app.eval import judge, metrics, run_eval


def case(case_id: str, q: str, route: str, **overrides) -> dict:
    base = {
        "id": case_id,
        "split": "dev",
        "category": route,
        "tags": [],
        "input": {"q": q, "client_id": None, "montant": None},
        "expected_route": route,
        "expected_final_route": route,
        "expected_docs": [],
        "forbidden_docs": [],
        "expected_facts": [],
        "forbidden_facts": [],
        "expected_pii": [],
        "note": "",
    }
    return base | overrides


def good_cases() -> list[dict]:
    return [
        case(
            "T-1",
            "Je conteste la facture de 120 €.",
            "automation",
            input={"q": "Je conteste la facture de 120 €.", "client_id": "C-12", "montant": None},
            expected_facts=["rg-03", r"t-\d{6}"],
        ),
        case(
            "T-2",
            "Quelle est la majoration pour un déplacement le week-end ?",
            "rag",
            expected_docs=["grille_tarifs.md"],
            expected_facts=["35 ?%"],
        ),
        case("T-3", "Je souhaite résilier mon contrat.", "human", split="test"),
        case(
            "T-4",
            "Quelle est la recette de la tarte aux pommes ?",
            "rag",
            expected_final_route="human",
            split="test",
        ),
        case(
            "T-5",
            "Quelle majoration le week-end ? Mon numéro : 06 39 98 12 34",
            "rag",
            expected_pii=["TEL"],
        ),
    ]


THRESHOLDS = {
    "min": {"routing_accuracy": 1.0, "fact_accuracy": 1.0},
    "max": {"forbidden_doc_violations": 0, "forbidden_fact_violations": 0},
}


@pytest.fixture
def golden(indexed, env):
    """Write a golden set and thresholds under the temporary evals directory."""

    def write(cases: list[dict], thresholds: dict = THRESHOLDS) -> None:
        (env.evals_dir / "golden_set.json").write_text(
            json.dumps(cases, ensure_ascii=False), encoding="utf-8"
        )
        (env.evals_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

    return write


def failures(report: dict) -> dict[str, list[str]]:
    return {failure["id"]: failure["reasons"] for failure in report["failures"]}


def test_correct_pipeline_passes_every_check_and_the_gate(golden):
    golden(good_cases())

    report = run_eval.run_eval()

    assert report["failures"] == []
    assert report["gate"] == {"applied": True, "passed": True, "misses": []}
    summary = report["metrics"]["all"]
    assert summary["cases"] == 5
    assert summary["routing_accuracy"] == 1.0
    assert summary["final_route_accuracy"] == 1.0
    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["fact_accuracy"] == 1.0
    assert summary["pii_accuracy"] == 1.0
    assert summary["cost_eur_total"] == 0.0
    assert report["metrics"]["dev"]["cases"] == 3 and report["metrics"]["test"]["cases"] == 2
    assert report["config"]["llm"] is None


def test_wrong_expected_route_fails_the_case_and_the_gate(golden):
    cases = good_cases()
    cases[2]["expected_route"] = cases[2]["expected_final_route"] = "rag"
    golden(cases)

    report = run_eval.run_eval()

    assert failures(report)["T-3"] == [
        "route human, attendue rag",
        "route finale human, attendue rag",
    ]
    assert report["metrics"]["all"]["routing_accuracy"] == 0.8
    assert report["gate"]["passed"] is False
    assert report["gate"]["misses"] == ["routing_accuracy = 0.8 (minimum 1.0)"]


def test_missing_fact_fails_the_case_and_the_gate(golden):
    cases = good_cases()
    cases[1]["expected_facts"] = ["35 ?%", "50 euros par mois"]
    golden(cases)

    report = run_eval.run_eval()

    assert failures(report)["T-2"] == ["faits absents de la réponse : ['50 euros par mois']"]
    assert report["gate"]["passed"] is False


def test_forbidden_document_and_fact_are_violations(golden):
    cases = good_cases()
    cases[1]["forbidden_docs"] = ["grille_tarifs.md"]
    cases[1]["forbidden_facts"] = ["35 ?%"]
    golden(cases)

    report = run_eval.run_eval()

    summary = report["metrics"]["all"]
    assert summary["forbidden_doc_violations"] == 1
    assert summary["forbidden_fact_violations"] == 1
    assert report["gate"]["passed"] is False
    assert "forbidden_doc_violations = 1 (maximum 0)" in report["gate"]["misses"]


def test_command_line_returns_1_when_the_gate_fails_unless_disabled(golden, capsys):
    cases = good_cases()
    cases[1]["expected_facts"] = ["jamais dans la réponse"]
    golden(cases)

    assert run_eval.main([]) == 1
    assert run_eval.main(["--no-gate"]) == 0
    assert "T-2" in capsys.readouterr().out


def test_partial_run_is_measured_but_not_gated(golden):
    cases = good_cases()
    cases[2]["expected_route"] = "rag"  # a test-split case, outside the dev run
    golden(cases)

    report = run_eval.run_eval(split="dev")

    assert report["metrics"]["all"]["cases"] == 3
    assert report["gate"]["applied"] is False


def test_evaluation_never_writes_to_the_application_database(golden):
    golden(good_cases())

    run_eval.run_eval()

    with session.session_scope() as db:
        assert repositories.list_tickets(db) == []
        assert repositories.list_requests(db) == []


def test_reports_are_written_in_json_and_markdown(golden, env):
    golden(good_cases())

    run_eval.run_eval()

    written = json.loads((env.evals_dir / "reports" / "latest.json").read_text(encoding="utf-8"))
    markdown = (env.evals_dir / "reports" / "latest.md").read_text(encoding="utf-8")
    assert written["metrics"]["all"]["cases"] == 5
    assert "| Routage (triage) | 1.0 | 1.0 | 1.0 |" in markdown
    assert "aucun LLM" in markdown
    assert "## Cas en échec (0)" in markdown


def test_judge_scores_answers_written_by_a_model(golden, monkeypatch):
    def fake_complete(task, messages, **_kwargs):
        if task == "judge":
            return LLMResult(text='{"faithful": false, "unsupported_claims": ["x"]}', model="j")
        return LLMResult(text="La majoration week-end est de +35 % [1].", model="m")

    monkeypatch.setattr(llm, "is_enabled", lambda task: True)
    monkeypatch.setattr(llm, "complete", fake_complete)
    golden([good_cases()[1]], thresholds={})

    report = run_eval.run_eval(judge=True)

    assert report["metrics"]["all"]["faithfulness"] == 0.0
    assert report["metrics"]["all"]["judged_answers"] == 1
    assert failures(report)["T-2"] == ["réponse jugée non fidèle aux sources"]


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ('{"faithful": true, "unsupported_claims": []}', True),
        ('Voici mon avis :\n{"faithful": false, "unsupported_claims": ["délai"]}\nFin.', False),
        ("Je ne peux pas répondre.", None),
        ('{"faithful": "oui"}', None),
        ("{pas du json}", None),
    ],
)
def test_judge_verdict_is_parsed_defensively(reply, verdict):
    assert judge.parse_verdict(reply) is verdict


def test_judge_gives_no_verdict_without_a_model():
    assert judge.judge_faithfulness("q", "sources", "réponse") is None


def test_metric_functions():
    assert metrics.rate([]) is None
    assert metrics.rate([True, True, False, True]) == 0.75
    assert metrics.confusion([("rag", "rag"), ("rag", "human"), ("agent", "agent")]) == {
        "rag": {"rag": 1, "human": 1},
        "agent": {"agent": 1},
    }
    assert metrics.hit(["a.md"], ["b.md", "a.md"]) is True
    assert metrics.hit(["a.md"], ["b.md"]) is False
    assert metrics.reciprocal_rank(["a.md"], ["b.md", "a.md", "a.md"]) == 0.5
    assert metrics.reciprocal_rank(["a.md"], ["b.md"]) == 0.0
    assert metrics.missing_facts("Délai : 4 h ouvrées, +35 %", ["4 ?h", "35 ?%", "8 ?h"]) == [
        "8 ?h"
    ]
    assert metrics.present_facts("Franchise de 250 €", ["250 ?€", "30 juin"]) == ["250 ?€"]
    assert metrics.percentile([], 95) is None
    assert metrics.percentile([5, 1, 3], 50) == 3
    assert metrics.percentile(list(range(1, 101)), 95) == 95
