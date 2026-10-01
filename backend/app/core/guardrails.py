"""Grounding helpers for the RAG and agent routes: confidence, source block, citations."""

import re

from app.core.schemas import Citation
from app.core.text import tokenize
from app.core.types import RetrievedChunk

# Share of the confidence given to lexical coverage; the rest goes to the dense score.
_COVERAGE_WEIGHT = 0.6
# Cosine similarity mapped linearly to 0..1: at or below 0.25 counts as no semantic
# match, at or above 0.70 as a full match.
_DENSE_FLOOR = 0.25
_DENSE_SPAN = 0.45

_EXCERPT_CHARS = 300

# "[1]" or "[2, 3]". Tags such as "[TEL]" hold no digits and are not references.
_REF_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def retrieval_confidence(query: str, results: list[RetrievedChunk]) -> float:
    """Confidence in 0..1 that `results` can answer `query`.

    Coverage is the share of the query's content tokens found in the retrieved chunks
    (title, section and text). When dense scores are available, the best cosine
    similarity is blended in, so a paraphrase with little word overlap is not rejected.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens or not results:
        return 0.0
    source_tokens: set[str] = set()
    for result in results:
        chunk = result.chunk
        source_tokens.update(tokenize(f"{chunk.title} {chunk.section} {chunk.text}"))
    coverage = len(query_tokens & source_tokens) / len(query_tokens)

    dense_scores = [r.dense_score for r in results if r.dense_score is not None]
    if not dense_scores:
        return coverage
    dense = min(1.0, max(0.0, (max(dense_scores) - _DENSE_FLOOR) / _DENSE_SPAN))
    return _COVERAGE_WEIGHT * coverage + (1 - _COVERAGE_WEIGHT) * dense


def format_sources(results: list[RetrievedChunk]) -> str:
    """Numbered source block for prompts; the number is the 1-based position in `results`."""
    return "\n\n".join(_format_source(n, r) for n, r in enumerate(results, start=1))


def _format_source(number: int, result: RetrievedChunk) -> str:
    chunk = result.chunk
    header = f"[{number}] {chunk.title} — {chunk.doc}, p. {chunk.page}"
    if chunk.section:
        header += f", {chunk.section}"
    return f"{header}\n{chunk.text.strip()}"


def extract_refs(answer: str) -> list[int]:
    """Source numbers cited in `answer`, in order of first appearance, without duplicates."""
    refs: list[int] = []
    for group in _REF_GROUP.findall(answer):
        for number in group.split(","):
            ref = int(number)
            if ref not in refs:
                refs.append(ref)
    return refs


def valid_refs(answer: str, n_sources: int) -> list[int]:
    """Cited source numbers that exist, i.e. within 1..n_sources."""
    return [ref for ref in extract_refs(answer) if 1 <= ref <= n_sources]


def build_citations(results: list[RetrievedChunk], refs: list[int] | None = None) -> list[Citation]:
    """Citations for the given source numbers, or for every result when `refs` is None.

    `refs` are 1-based positions in `results`; numbers out of range and duplicates are
    ignored.
    """
    if refs is None:
        refs = list(range(1, len(results) + 1))
    citations = []
    for ref in dict.fromkeys(refs):  # removes duplicates, keeps the order
        if 1 <= ref <= len(results):
            citations.append(_citation(ref, results[ref - 1]))
    return citations


def _citation(ref: int, result: RetrievedChunk) -> Citation:
    chunk = result.chunk
    single_spaced = " ".join(chunk.text.split())
    return Citation(
        ref=ref,
        doc=chunk.doc,
        title=chunk.title,
        page=chunk.page,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        excerpt=single_spaced[:_EXCERPT_CHARS],
        score=round(result.score, 4),
    )
