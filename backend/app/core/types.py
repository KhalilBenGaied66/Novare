"""Internal data types shared between layers (ingestion, retrieval, agents, LLM).

API payloads live in `schemas.py`; these dataclasses never cross the HTTP boundary.
"""

from dataclasses import asdict, dataclass, field
from typing import Literal

RouteKind = Literal["automation", "rag", "agent", "human"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc>#p<page>-<n>", stable for a given corpus
    doc: str  # source file name, e.g. "procedure_sav.md"
    title: str  # document title (front matter `titre`, first heading or file name)
    doc_type: str  # "procedure" | "tarif" | "contrat" | "faq" | "note" | "email" | "document"
    page: int  # 1-based page number (PDF); 1 for other formats
    section: str  # nearest heading above the chunk, "" when none
    text: str
    client_id: str | None = None  # set => visible only to requests of that client

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused score (RRF in hybrid mode, BM25 score in bm25 mode)
    bm25_rank: int | None = None  # 1-based rank in the lexical list
    dense_rank: int | None = None  # 1-based rank in the vector list
    bm25_score: float | None = None
    dense_score: float | None = None  # cosine similarity


@dataclass(frozen=True)
class TriageDecision:
    route: RouteKind
    rule: str  # "RG-04" | "RG-03" | "RG-03b" | "RG-05" | "DEFAULT"
    reasons: list[str]  # ordered, human-readable (French), no raw user text
    client_id: str | None  # explicit field or extracted from the text
    montant: float | None  # explicit field or extracted from the text
    client_known: bool  # client_id exists in the client reference data


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_eur: float = 0.0
    latency_ms: int = 0
    # Assistant message in OpenAI chat format, ready to append to the history.
    raw_message: dict = field(default_factory=dict)
