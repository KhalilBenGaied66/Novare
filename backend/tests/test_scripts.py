"""Command-line scripts under scripts/ (loaded by path: they are not a package)."""

import importlib.util
import json

import pytest

from app.core import llm
from app.core.config import REPO_ROOT
from app.core.schemas import AskRequest
from app.core.types import LLMResult
from app.db import repositories, session
from app.services import ask


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ko_feedback_becomes_a_candidate_case_with_the_masked_question(indexed, env):
    script = load_script("feedback_to_golden")
    liked = ask.handle_ask(AskRequest(q="Quelle est la majoration le week-end ?"))
    disliked = ask.handle_ask(
        AskRequest(q="La chaudière est en panne, rappelez-moi au 06 39 98 12 34.", client_id="C-12")
    )
    with session.session_scope() as db:
        repositories.save_feedback(db, request_id=liked.request_id, ok=True, comment="")
        repositories.save_feedback(
            db, request_id=disliked.request_id, ok=False, comment="Priorité P1 attendue"
        )

    assert script.main() == 0

    (candidate,) = json.loads((env.evals_dir / "candidates.json").read_text(encoding="utf-8"))
    assert candidate["request_id"] == disliked.request_id
    assert candidate["observed_route"] == "agent"
    assert candidate["handler_comment"] == "Priorité P1 attendue"
    assert "[TEL]" in candidate["input"]["q"] and "39 98" not in candidate["input"]["q"]
    assert candidate["input"]["client_id"] == "C-12"
    assert candidate["expected_route"] is None  # left for a person to fill in


def test_llm_smoke_script_stops_without_a_provider_key(indexed, capsys):
    script = load_script("smoke_llm")

    assert script.main() == 1
    assert "Aucune clé de fournisseur" in capsys.readouterr().out


@pytest.mark.parametrize(("cited_answer", "exit_code"), [(True, 0), (False, 1)])
def test_llm_smoke_script_fails_when_a_request_is_not_answered_by_a_model(
    indexed, monkeypatch, capsys, cited_answer, exit_code
):
    script = load_script("smoke_llm")

    def fake_complete(task, messages, **_kwargs):
        if not cited_answer:
            raise llm.LLMError("APIConnectionError")
        text = "Réponse appuyée sur la documentation [1]."
        message = {"role": "assistant", "content": text}
        return LLMResult(text=text, model="stub", raw_message=message)

    monkeypatch.setattr(llm, "is_enabled", lambda task: True)
    monkeypatch.setattr(llm, "complete", fake_complete)

    assert script.main() == exit_code
    output = capsys.readouterr().out
    assert ("ÉCHEC" in output) is (exit_code == 1)
    with session.session_scope() as db:
        assert repositories.list_requests(db) == []  # the application database is untouched
