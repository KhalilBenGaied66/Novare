"""Okapi BM25 lexical index over pre-tokenised documents (pure Python, in memory)."""

import math
from collections import Counter


class BM25Index:
    """score(q, d) = sum over the distinct terms t of q of

        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))

    with idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)), which is always positive.
    A term repeated in the query counts once.
    """

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self.doc_lengths = [len(tokens) for tokens in docs_tokens]
        self.avg_length = sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        # term -> {document index: term frequency}
        self.postings: dict[str, dict[int, int]] = {}
        for index, tokens in enumerate(docs_tokens):
            for term, frequency in Counter(tokens).items():
                self.postings.setdefault(term, {})[index] = frequency

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(
        self, query_tokens: list[str], top_n: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Return the `top_n` best `(document index, score)` pairs, best first.

        Only documents sharing at least one term with the query are returned. `allowed`
        restricts the candidates; the statistics (idf, average length) stay those of the
        whole corpus.
        """
        scores: dict[int, float] = {}
        # dict.fromkeys removes duplicates and keeps the query order (a set would sum the
        # floats in a different order from one process to the next).
        for term in dict.fromkeys(query_tokens):
            idf = self.idf(term)
            for index, frequency in self.postings.get(term, {}).items():
                if allowed is not None and index not in allowed:
                    continue
                length_ratio = self.doc_lengths[index] / self.avg_length
                saturation = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                gain = idf * frequency * (self.k1 + 1) / saturation
                scores[index] = scores.get(index, 0.0) + gain
        # Ties are broken by document index so that results are reproducible.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_n]
