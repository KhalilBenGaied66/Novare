"""Per-request context: identifier, decision log and LLM usage accumulated along the way."""

import time
import uuid
from dataclasses import dataclass, field

from app.core.schemas import Usage
from app.core.types import LLMResult


@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    decision_log: list[str] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)
    q_masked: str = ""  # request text after PII redaction; the only form that is stored
    usage: Usage = field(default_factory=Usage)

    def log(self, step: str) -> None:
        """Append one step to the decision log. Never pass raw user text."""
        self.decision_log.append(step)

    def add_llm(self, result: LLMResult) -> None:
        self.usage.model = result.model or self.usage.model
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += result.prompt_tokens
        self.usage.completion_tokens += result.completion_tokens
        self.usage.cost_eur = round(self.usage.cost_eur + result.cost_eur, 6)

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)
