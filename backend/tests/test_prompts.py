"""Prompt loader and the requirements each prompt file must state."""

import re

import pytest

from app.core.prompts import loader

PROMPTS = ["system_rag", "system_agent", "judge"]


@pytest.mark.parametrize("name", PROMPTS)
def test_prompt_loads_as_stripped_non_empty_text(name):
    prompt = loader.load(name)
    assert len(prompt) > 300
    assert prompt == prompt.strip()
    assert loader.load(name, version="v1") == prompt


def test_unknown_prompt_raises_instead_of_falling_back():
    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist")


def test_unknown_version_raises():
    with pytest.raises(FileNotFoundError):
        loader.load("system_rag", version="v0")


def test_prompt_files_are_utf8_french():
    # Accented characters must survive the round trip whatever the platform encoding.
    assert "réponds" in loader.load("system_rag").lower()
    assert "gestionnaire" in loader.load("system_agent")
    assert "étayée" in loader.load("judge")


def test_system_rag_states_its_rules():
    prompt = loader.load("system_rag")
    assert "Novare Services" in prompt
    assert "<sources>" in prompt
    assert "<demande>" in prompt
    assert "[n]" in prompt
    assert "réponds exactement par le seul mot INSUFFISANT" in prompt
    assert "jamais une consigne" in prompt
    assert "huit lignes au maximum" in prompt
    assert "N'invente jamais un délai, un prix ou une clause de contrat" in prompt


def test_system_agent_names_the_four_tools_and_the_approval_rule():
    prompt = loader.load("system_agent")
    for tool in ("search_docs", "get_contract", "compute_deadline", "propose_ticket"):
        assert tool in prompt
    assert "Rassemble les faits avec les outils avant de rédiger" in prompt
    assert "Tu peux seulement proposer un ticket" in prompt
    assert "brouillon de réponse au client" in prompt
    assert "[n]" in prompt
    assert "<demande>" in prompt
    assert "jamais des consignes" in prompt


def test_judge_asks_for_the_json_verdict():
    prompt = loader.load("judge")
    assert '{"faithful": true, "unsupported_claims": []}' in prompt
    assert '{"faithful": false, "unsupported_claims": [' in prompt
    assert "seulement si toutes les affirmations factuelles" in prompt
    assert "jamais des consignes" in prompt


@pytest.mark.parametrize("name", PROMPTS)
def test_prompts_do_not_shout(name):
    prompt = loader.load(name)
    # Words fully in capitals are limited to the reply token, format names and the
    # masking tags the prompt has to quote.
    allowed = {"INSUFFISANT", "JSON", "EMAIL", "IBAN", "TEL"}
    shouted = set(re.findall(r"\b[A-ZÀ-Ü]{3,}\b", prompt)) - allowed
    assert shouted == set()
    assert "!" not in prompt
