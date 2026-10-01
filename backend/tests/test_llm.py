"""LLM client: enablement rules, what is sent to the provider, mapping of real LiteLLM
responses (LiteLLM's mock mode, no network), cost and error handling.

Importing LiteLLM takes several seconds; it happens once, in the `litellm` fixture.
"""

import json
import logging
import os
from types import SimpleNamespace

import pytest

from app.core import llm
from app.core.config import reset_settings
from app.core.types import LLMResult, ToolCall

# Emitted by pydantic while LiteLLM builds its own response models; not ours to fix.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Item .* is using the `ReadOnly` qualifier:UserWarning"
)

MESSAGES = [
    {"role": "system", "content": "Tu réponds uniquement à partir des sources."},
    {"role": "user", "content": "<demande>\nQuelle est la majoration le week-end ?\n</demande>"},
]
ANSWER = "La majoration week-end est de +35 % sur le déplacement et la main-d'œuvre [1]."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Recherche dans la base documentaire.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contract",
            "description": "Contrat du client du dossier.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def configure(monkeypatch, **env_vars: str) -> None:
    for name, value in env_vars.items():
        monkeypatch.setenv(name, value)
    reset_settings()


def tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture(scope="module")
def litellm():
    """The real library. Its bundled price table is used so that the import stays offline."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    return litellm


class FakeProvider:
    """Stands in for `llm._provider_completion`: records what would be sent to the
    provider and answers through LiteLLM's mock mode, so the response objects are real.
    """

    def __init__(self, litellm, usage: tuple[int, int] | None = None, **mock):
        self.litellm = litellm
        self.usage = usage
        self.mock = mock
        self.sent: list[dict] = []

    def __call__(self, **kwargs):
        self.sent.append(kwargs)
        response = self.litellm.completion(**kwargs, **self.mock)
        if self.usage is not None:
            prompt, completion = self.usage
            response.usage = self.litellm.Usage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
            )
        return response


@pytest.fixture
def provider(monkeypatch, litellm):
    """Factory: enable the LLM and install a FakeProvider answering with the given mock."""

    def install(**mock) -> FakeProvider:
        configure(monkeypatch, LLM_ENABLED="on")
        fake = FakeProvider(litellm, **mock)
        monkeypatch.setattr(llm, "_provider_completion", fake)
        return fake

    return install


def warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# --- Models and enablement -------------------------------------------------


def test_model_for_each_task_uses_the_settings(monkeypatch):
    assert llm.model_for("rag") == "mistral/mistral-small-latest"
    assert llm.model_for("agent") == "anthropic/claude-sonnet-5-5"
    assert llm.model_for("judge") == "anthropic/claude-haiku-4-5"

    configure(
        monkeypatch, RAG_MODEL="openai/gpt-4o-mini", AGENT_MODEL="mistral/mistral-large-latest"
    )
    assert llm.model_for("rag") == "openai/gpt-4o-mini"
    assert llm.model_for("agent") == "mistral/mistral-large-latest"
    assert llm.model_for("judge") == "anthropic/claude-haiku-4-5"


@pytest.mark.parametrize(
    ("mode", "keys", "expected"),
    [
        # (LLM_ENABLED, provider keys present, enabled for rag / agent / judge)
        ("off", {}, (False, False, False)),
        ("off", {"MISTRAL_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, (False, False, False)),
        ("on", {}, (True, True, True)),
        ("auto", {}, (False, False, False)),
        ("auto", {"MISTRAL_API_KEY": "k"}, (True, False, False)),
        ("auto", {"ANTHROPIC_API_KEY": "k"}, (False, True, True)),
        ("auto", {"MISTRAL_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, (True, True, True)),
        ("auto", {"MISTRAL_API_KEY": "", "ANTHROPIC_API_KEY": "   "}, (False, False, False)),
        ("auto", {"OPENAI_API_KEY": "k"}, (False, False, False)),
    ],
)
def test_is_enabled_with_the_default_models(monkeypatch, mode, keys, expected):
    configure(monkeypatch, LLM_ENABLED=mode, **keys)
    assert tuple(llm.is_enabled(task) for task in ("rag", "agent", "judge")) == expected


@pytest.mark.parametrize(
    ("model", "env_var"),
    [
        ("mistral/mistral-large-latest", "MISTRAL_API_KEY"),
        ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
        ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("azure/gpt-4o-prod", "AZURE_API_KEY"),
    ],
)
def test_auto_mode_follows_the_key_of_the_model_provider(monkeypatch, model, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(llm, "_litellm", lambda: pytest.fail("LiteLLM must not be imported"))
    configure(monkeypatch, LLM_ENABLED="auto", RAG_MODEL=model)
    assert llm.is_enabled("rag") is False

    monkeypatch.setenv(env_var, "k")
    assert llm.is_enabled("rag") is True


def test_auto_mode_asks_litellm_for_other_providers(monkeypatch, litellm):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    configure(monkeypatch, LLM_ENABLED="auto", RAG_MODEL="groq/llama-3.3-70b-versatile")
    assert llm.is_enabled("rag") is False

    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert llm.is_enabled("rag") is True


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_complete_raises_unavailable_without_calling_the_provider(monkeypatch, mode):
    configure(monkeypatch, LLM_ENABLED=mode)
    monkeypatch.setattr(llm, "_provider_completion", lambda **_: pytest.fail("provider called"))
    for task in ("rag", "agent", "judge"):
        with pytest.raises(llm.LLMUnavailable):
            llm.complete(task, MESSAGES)


# --- What is sent to the provider ------------------------------------------


def test_text_request_sends_only_the_expected_parameters(provider):
    fake = provider(mock_response=ANSWER)
    llm.complete("rag", MESSAGES, max_tokens=600)
    # Exact comparison: no temperature, no top_p, no tool_choice.
    assert fake.sent == [
        {
            "model": "mistral/mistral-small-latest",
            "messages": MESSAGES,
            "timeout": 30.0,
            "num_retries": 2,
            "max_tokens": 600,
        }
    ]


def test_tool_request_lets_the_model_choose(provider):
    fake = provider(mock_tool_calls=[tool_call("call_1", "get_contract", "{}")])
    llm.complete("agent", MESSAGES, tools=TOOLS, max_tokens=1200)
    assert fake.sent == [
        {
            "model": "anthropic/claude-sonnet-5-5",
            "messages": MESSAGES,
            "timeout": 30.0,
            "num_retries": 2,
            "max_tokens": 1200,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
    ]


@pytest.mark.parametrize("tools", [None, []])
def test_no_tool_parameters_and_no_max_tokens_when_not_given(provider, tools):
    fake = provider(mock_response=ANSWER)
    llm.complete("judge", MESSAGES, tools=tools)
    assert set(fake.sent[0]) == {"model", "messages", "timeout", "num_retries"}
    assert fake.sent[0]["model"] == "anthropic/claude-haiku-4-5"


def test_timeout_and_retries_come_from_the_settings(provider, monkeypatch):
    fake = provider(mock_response=ANSWER)
    configure(monkeypatch, LLM_TIMEOUT_S="12.5", LLM_MAX_RETRIES="0")
    llm.complete("rag", MESSAGES)
    assert (fake.sent[0]["timeout"], fake.sent[0]["num_retries"]) == (12.5, 0)


# --- Mapping of the response -----------------------------------------------


def test_text_answer_is_mapped_to_llm_result(provider):
    provider(mock_response=ANSWER, usage=(812, 96))
    result = llm.complete("rag", MESSAGES, max_tokens=600)

    assert isinstance(result, LLMResult)
    assert result.text == ANSWER
    assert result.tool_calls == []
    assert result.model == "mistral/mistral-small-latest"
    assert (result.prompt_tokens, result.completion_tokens) == (812, 96)
    assert result.latency_ms >= 0
    assert result.raw_message == {"role": "assistant", "content": ANSWER}


def test_tool_calls_are_parsed_and_replayable(provider):
    query = {"query": "délai d'intervention P1 formule Confort"}
    provider(
        mock_tool_calls=[
            tool_call("call_1", "search_docs", json.dumps(query, ensure_ascii=False)),
            tool_call("call_2", "get_contract", "{}"),
        ]
    )
    result = llm.complete("agent", MESSAGES, tools=TOOLS)

    assert result.tool_calls == [
        ToolCall(id="call_1", name="search_docs", arguments=query),
        ToolCall(id="call_2", name="get_contract", arguments={}),
    ]
    assert result.model == "anthropic/claude-sonnet-5-5"
    assert result.raw_message == {
        "role": "assistant",
        "content": result.text,
        "tool_calls": [
            tool_call(
                "call_1", "search_docs", '{"query": "délai d\'intervention P1 formule Confort"}'
            ),
            tool_call("call_2", "get_contract", "{}"),
        ],
    }


def test_tool_call_without_text_gives_empty_text_and_null_content(monkeypatch, litellm):
    configure(monkeypatch, LLM_ENABLED="on")

    def answer_without_text(**kwargs):
        response = litellm.completion(
            **kwargs, mock_tool_calls=[tool_call("call_1", "get_contract", "{}")]
        )
        response.choices[0].message.content = None
        return response

    monkeypatch.setattr(llm, "_provider_completion", answer_without_text)
    result = llm.complete("agent", MESSAGES, tools=TOOLS)
    assert result.text == ""
    assert result.raw_message["content"] is None
    assert [call.name for call in result.tool_calls] == ["get_contract"]


@pytest.mark.parametrize(
    "arguments", ['{"query": "tarif', "pas du json", '["liste"]', '"texte"', "42"]
)
def test_invalid_tool_arguments_become_empty_with_a_warning(provider, caplog, arguments):
    provider(mock_tool_calls=[tool_call("call_1", "search_docs", arguments)])
    with caplog.at_level(logging.WARNING, logger="app.core.llm"):
        result = llm.complete("agent", MESSAGES, tools=TOOLS)

    assert result.tool_calls == [ToolCall(id="call_1", name="search_docs", arguments={})]
    # The history must carry valid JSON: providers refuse to replay malformed arguments.
    assert result.raw_message["tool_calls"][0]["function"]["arguments"] == "{}"
    assert "llm_tool_arguments_invalid" in warnings(caplog)
    assert [r.tool for r in caplog.records if r.getMessage() == "llm_tool_arguments_invalid"] == [
        "search_docs"
    ]


def test_empty_tool_arguments_mean_no_arguments(provider, caplog):
    provider(mock_tool_calls=[tool_call("call_1", "get_contract", "")])
    with caplog.at_level(logging.WARNING, logger="app.core.llm"):
        result = llm.complete("agent", MESSAGES, tools=TOOLS)
    assert result.tool_calls == [ToolCall(id="call_1", name="get_contract", arguments={})]
    assert "llm_tool_arguments_invalid" not in warnings(caplog)


def test_logs_carry_figures_but_no_prompt_or_answer_text(provider, caplog):
    provider(mock_response=ANSWER, usage=(812, 96))
    with caplog.at_level(logging.INFO, logger="app.core.llm"):
        llm.complete("rag", MESSAGES)

    (record,) = [r for r in caplog.records if r.getMessage() == "llm_call"]
    assert (record.task, record.model) == ("rag", "mistral/mistral-small-latest")
    assert (record.prompt_tokens, record.completion_tokens, record.tool_calls) == (812, 96, 0)
    logged = json.dumps([vars(r) for r in caplog.records], default=str, ensure_ascii=False)
    assert "majoration" not in logged
    assert "sources" not in logged


# --- Cost ------------------------------------------------------------------


@pytest.mark.parametrize("task", ["rag", "judge"])
def test_cost_is_computed_from_token_usage_and_converted_to_euros(provider, litellm, task):
    provider(mock_response=ANSWER, usage=(1000, 500))
    result = llm.complete(task, MESSAGES)

    prices = litellm.get_model_info(llm.model_for(task))
    expected_usd = 1000 * prices["input_cost_per_token"] + 500 * prices["output_cost_per_token"]
    assert expected_usd > 0
    assert result.cost_eur == pytest.approx(expected_usd * 0.90)


def test_usd_to_eur_rate_comes_from_the_settings(provider, monkeypatch):
    provider(mock_response=ANSWER, usage=(1000, 500))
    monkeypatch.setattr(llm, "_provider_cost_usd", lambda response: 0.01)
    assert llm.complete("rag", MESSAGES).cost_eur == pytest.approx(0.009)

    configure(monkeypatch, USD_TO_EUR="0.5")
    assert llm.complete("rag", MESSAGES).cost_eur == pytest.approx(0.005)


def test_model_without_price_costs_zero_with_a_warning(provider, monkeypatch, caplog):
    provider(mock_response=ANSWER, usage=(1000, 500))
    configure(monkeypatch, RAG_MODEL="mistral/modele-sans-tarif")
    with caplog.at_level(logging.WARNING, logger="app.core.llm"):
        result = llm.complete("rag", MESSAGES)

    assert result.cost_eur == 0.0
    assert result.text == ANSWER  # the answer itself is not lost
    assert (result.prompt_tokens, result.completion_tokens) == (1000, 500)
    (record,) = [r for r in caplog.records if r.getMessage() == "llm_price_unknown"]
    assert record.model == "mistral/modele-sans-tarif"


def test_silent_zero_price_is_flagged_too(provider, monkeypatch, caplog):
    provider(mock_response=ANSWER, usage=(1000, 500))
    monkeypatch.setattr(llm, "_provider_cost_usd", lambda response: 0.0)
    with caplog.at_level(logging.WARNING, logger="app.core.llm"):
        assert llm.complete("rag", MESSAGES).cost_eur == 0.0
    assert warnings(caplog) == ["llm_price_unknown"]


@pytest.mark.parametrize("task", ["rag", "agent", "judge"])
def test_default_models_are_priced_or_flagged_never_silently_free(provider, caplog, task):
    # Depending on the LiteLLM price table in use (bundled or downloaded), the newest
    # default model may have no entry: the cost is then 0.0 and it must be visible.
    provider(mock_response=ANSWER, usage=(1000, 500))
    with caplog.at_level(logging.WARNING, logger="app.core.llm"):
        result = llm.complete(task, MESSAGES)
    assert result.cost_eur > 0 or warnings(caplog) == ["llm_price_unknown"]


# --- Errors ----------------------------------------------------------------


@pytest.mark.parametrize(
    "error", ["RateLimitError", "InternalServerError", "ContextWindowExceededError"]
)
def test_provider_errors_become_llm_error_with_the_class_name_only(provider, caplog, error):
    provider(mock_response=f"litellm.{error}")
    with (
        caplog.at_level(logging.WARNING, logger="app.core.llm"),
        pytest.raises(llm.LLMError) as info,
    ):
        llm.complete("rag", MESSAGES)

    assert str(info.value) == error
    (record,) = [r for r in caplog.records if r.getMessage() == "llm_call_failed"]
    assert (record.task, record.model, record.error) == (
        "rag",
        "mistral/mistral-small-latest",
        error,
    )


def test_error_message_of_the_provider_is_not_exposed(monkeypatch):
    configure(monkeypatch, LLM_ENABLED="on")

    def failing_provider(**_):
        raise TimeoutError("délai dépassé pour la demande de Mme Martin, clé sk-secret")

    monkeypatch.setattr(llm, "_provider_completion", failing_provider)
    with pytest.raises(llm.LLMError) as info:
        llm.complete("rag", MESSAGES)
    assert str(info.value) == "TimeoutError"
    assert info.value.args == ("TimeoutError",)


def test_unreadable_response_is_a_provider_failure(monkeypatch):
    configure(monkeypatch, LLM_ENABLED="on")
    monkeypatch.setattr(llm, "_provider_completion", lambda **_: SimpleNamespace(choices=[]))
    with pytest.raises(llm.LLMError) as info:
        llm.complete("rag", MESSAGES)
    assert str(info.value) == "IndexError"


def test_on_mode_without_key_fails_as_llm_error_not_as_a_crash(monkeypatch, litellm):
    # LLM_ENABLED=on with a missing key: LiteLLM refuses before any network call.
    configure(monkeypatch, LLM_ENABLED="on", LLM_MAX_RETRIES="0")
    with pytest.raises(llm.LLMError) as info:
        llm.complete("judge", MESSAGES)
    assert str(info.value) == "AuthenticationError"


# --- LiteLLM output --------------------------------------------------------


def test_litellm_writes_no_banner_and_no_log_line_of_its_own(provider, litellm, caplog):
    provider(mock_response=ANSWER)
    assert llm._litellm() is litellm
    assert litellm.suppress_debug_info is True
    litellm_logger = logging.getLogger("LiteLLM")
    assert litellm_logger.handlers == []  # its records go through the root handler only
    assert litellm_logger.level == logging.WARNING

    with caplog.at_level(logging.INFO):
        llm.complete("rag", MESSAGES)
    assert [r.name for r in caplog.records] == ["app.core.llm"]
