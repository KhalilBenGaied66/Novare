"""LLM access through LiteLLM: one provider-neutral call that reports usage and cost.

Each task ("rag", "agent", "judge") has its own model in the settings, so a cheap model
can answer documentary questions while a stronger one drives the agent. Callers handle
two failures: `LLMUnavailable` (no LLM configured: use the offline path) and `LLMError`
(the provider failed: fall back and record it).

Prices come from LiteLLM's price table. LiteLLM downloads the current table when it is
imported and falls back to the copy bundled with the package; setting
`LITELLM_LOCAL_MODEL_COST_MAP=True` forces the bundled copy. A model without a price
entry costs 0.0 here and a warning is logged, so the cost figures are a lower bound.
"""

import json
import os
import time
from typing import Any, Literal

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.types import LLMResult, ToolCall

logger = get_logger(__name__)

Task = Literal["rag", "agent", "judge"]

# Provider key expected for a model id, checked without importing LiteLLM: the import
# takes several seconds and a deployment without any key should never pay for it.
# Models of other providers fall back to LiteLLM's own environment check.
_KEY_BY_PREFIX = (
    ("mistral/", "MISTRAL_API_KEY"),
    ("anthropic/", "ANTHROPIC_API_KEY"),
    ("openai/", "OPENAI_API_KEY"),
    ("gpt-", "OPENAI_API_KEY"),
    ("azure/", "AZURE_API_KEY"),
)


class LLMUnavailable(Exception):
    """The LLM is disabled by configuration, or the provider key of the model is missing."""


class LLMError(Exception):
    """The provider call failed after retries.

    The message is the exception class name only, so it can be written to a decision
    log without leaking provider or prompt content.
    """


def model_for(task: Task) -> str:
    settings = get_settings()
    models = {
        "rag": settings.rag_model,
        "agent": settings.agent_model,
        "judge": settings.judge_model,
    }
    return models[task]


def is_enabled(task: Task) -> bool:
    """Whether `complete(task, ...)` may be called with the current configuration."""
    mode = get_settings().llm_enabled
    if mode == "off":
        return False
    if mode == "on":
        return True
    return _has_provider_key(model_for(task))


def complete(
    task: Task,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """Send `messages` (OpenAI chat format) to the model of `task` and map the answer.

    `max_tokens=None` leaves the output limit to the provider default.
    """
    if not is_enabled(task):
        raise LLMUnavailable(f"LLM disabled or provider key missing for task '{task}'")

    settings = get_settings()
    model = model_for(task)
    # No temperature/top_p of our own: current Claude models reject sampling
    # parameters. What a given model needs beyond the request comes from the settings.
    kwargs: dict[str, Any] = {
        **settings.llm_extra_params,
        "model": model,
        "messages": messages,
        "timeout": settings.llm_timeout_s,
        "num_retries": settings.llm_max_retries,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
        # The model decides whether to call a tool: a forced tool choice is rejected by
        # current Claude models.
        kwargs["tool_choice"] = "auto"

    started = time.perf_counter()
    try:
        response = _provider_completion(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = _to_result(response, model, latency_ms)
    except Exception as exc:
        # Boundary with the provider: network, authentication, rate limit, timeout or
        # an unreadable response all mean the same thing to the caller.
        logger.warning(
            "llm_call_failed",
            extra={"task": task, "model": model, "error": type(exc).__name__},
        )
        raise LLMError(type(exc).__name__) from exc

    logger.info(
        "llm_call",
        extra={
            "task": task,
            "model": model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_eur": result.cost_eur,
            "latency_ms": result.latency_ms,
            "tool_calls": len(result.tool_calls),
        },
    )
    return result


def _has_provider_key(model: str) -> bool:
    for prefix, env_var in _KEY_BY_PREFIX:
        if model.startswith(prefix):
            return bool(os.environ.get(env_var, "").strip())
    # LiteLLM reports nothing missing for a provider it does not know; the call is then
    # attempted and its failure surfaces as LLMError.
    return bool(_litellm().validate_environment(model)["keys_in_environment"])


def _litellm() -> Any:
    """Import LiteLLM on first use (slow import, useless when no LLM is configured)."""
    import litellm

    # The process output must stay the application's log lines. By default LiteLLM
    # prints a help banner when a call fails and logs every call at INFO through a
    # plain-text handler of its own, on top of the root handler. Keep its warnings
    # only, formatted like every other line. These settings are idempotent.
    litellm.suppress_debug_info = True
    litellm_logger = get_logger("LiteLLM")
    litellm_logger.handlers.clear()
    litellm_logger.setLevel("WARNING")
    return litellm


def _provider_completion(**kwargs: Any) -> Any:
    """The only place that calls a provider; tests replace this function."""
    return _litellm().completion(**kwargs)


def _provider_cost_usd(response: Any) -> float:
    return _litellm().completion_cost(completion_response=response)


def _to_result(response: Any, model: str, latency_ms: int) -> LLMResult:
    message = response.choices[0].message
    text = message.content or ""
    tool_calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=_parse_arguments(call.function.arguments, call.function.name),
        )
        for call in message.tool_calls or []
    ]
    usage = getattr(response, "usage", None)
    return LLMResult(
        text=text,
        tool_calls=tool_calls,
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_eur=_cost_eur(response, model),
        latency_ms=latency_ms,
        finish_reason=getattr(response.choices[0], "finish_reason", None) or "",
        raw_message=_assistant_message(text, tool_calls),
    )


def _parse_arguments(raw: str | None, tool_name: str) -> dict:
    """Tool arguments are a JSON string written by the model, so they can be malformed."""
    if not raw:
        return {}  # a tool without parameters
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    logger.warning("llm_tool_arguments_invalid", extra={"tool": tool_name})
    return {}


def _assistant_message(text: str, tool_calls: list[ToolCall]) -> dict:
    """Assistant message in OpenAI chat format, ready to append to the history."""
    if not tool_calls:
        return {"role": "assistant", "content": text}
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                # Serialised again from the parsed arguments: a provider rejects a
                # history that replays malformed JSON.
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in tool_calls
        ],
    }


def _cost_eur(response: Any, model: str) -> float:
    try:
        cost_usd = _provider_cost_usd(response)
    except Exception as exc:
        # LiteLLM raises a plain Exception for a model missing from its price table.
        logger.warning("llm_price_unknown", extra={"model": model, "error": type(exc).__name__})
        return 0.0
    if not cost_usd:
        # Some unpriced models yield 0 instead of an error.
        logger.warning("llm_price_unknown", extra={"model": model, "error": "zero_cost"})
        return 0.0
    # A millionth of a euro is precise enough for a single call.
    return round(cost_usd * get_settings().usd_to_eur, 6)
