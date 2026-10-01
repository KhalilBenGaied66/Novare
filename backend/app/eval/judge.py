"""Optional faithfulness check of a generated answer by a second model.

Only meaningful for answers written by a model: an extractive answer quotes its sources
word for word.
"""

import json

from app.core import llm
from app.core.logging import get_logger
from app.core.prompts import loader

logger = get_logger(__name__)


def judge_faithfulness(question: str, sources: str, answer: str) -> bool | None:
    """True / False as judged by the model; None when no verdict could be obtained."""
    if not llm.is_enabled("judge"):
        return None
    messages = [
        {"role": "system", "content": loader.load("judge")},
        {
            "role": "user",
            "content": (
                f"<question>\n{question}\n</question>\n<sources>\n{sources}\n</sources>\n"
                f"<reponse>\n{answer}\n</reponse>"
            ),
        },
    ]
    try:
        result = llm.complete("judge", messages, max_tokens=400)
    except (llm.LLMError, llm.LLMUnavailable) as exc:
        logger.warning("judge_unavailable", extra={"error": type(exc).__name__})
        return None
    return parse_verdict(result.text)


def parse_verdict(text: str) -> bool | None:
    """Read `{"faithful": true|false, ...}` from a model reply, even wrapped in prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        verdict = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    faithful = verdict.get("faithful") if isinstance(verdict, dict) else None
    return faithful if isinstance(faithful, bool) else None
