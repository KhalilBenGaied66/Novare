"""Pure scoring functions of the evaluation. No I/O, no pipeline call."""

import re

from app.core.text import normalize


def rate(outcomes: list[bool]) -> float | None:
    """Share of True in `outcomes`; None when there is nothing to score."""
    return round(sum(outcomes) / len(outcomes), 4) if outcomes else None


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """`{expected: {actual: count}}` for (expected, actual) pairs."""
    table: dict[str, dict[str, int]] = {}
    for expected, actual in pairs:
        row = table.setdefault(expected, {})
        row[actual] = row.get(actual, 0) + 1
    return table


def hit(expected_docs: list[str], retrieved_docs: list[str]) -> bool:
    """True when at least one expected document is among the retrieved ones."""
    return bool(set(expected_docs) & set(retrieved_docs))


def reciprocal_rank(expected_docs: list[str], retrieved_docs: list[str]) -> float:
    """1 / rank of the first expected document in `retrieved_docs`, 0.0 when absent."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected_docs:
            return 1 / rank
    return 0.0


def missing_facts(answer: str, patterns: list[str]) -> list[str]:
    """Patterns (regular expressions) that do not match the normalised answer."""
    text = normalize(answer)
    return [pattern for pattern in patterns if not re.search(pattern, text)]


def present_facts(answer: str, patterns: list[str]) -> list[str]:
    """Patterns that match the normalised answer (used for facts that must not appear)."""
    text = normalize(answer)
    return [pattern for pattern in patterns if re.search(pattern, text)]


def percentile(values: list[int], percent: int) -> int | None:
    """Nearest-rank percentile of `values`."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-percent * len(ordered) // 100))  # ceiling division
    return ordered[rank - 1]
