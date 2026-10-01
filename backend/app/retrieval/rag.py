"""Documentary answers grounded in the corpus, always with citations.

Two generation modes share the same retrieval and the same confidence check:
- "llm": a model writes the answer from the numbered sources and must cite them;
- "extractive": no model (none configured, or the call failed): the passages of the
  sources that best match the question are quoted as they are.

Whenever the sources cannot support an answer (low retrieval confidence, model replying
INSUFFISANT, answer without citation, no matching passage), the request is escalated to
a person instead of being answered.
"""

import re
from dataclasses import dataclass

from app.core import guardrails, llm, pii
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import loader
from app.core.schemas import AnswerMode, AskRequest, AskResponse
from app.core.text import split_sentences, tokenize
from app.core.types import RetrievedChunk, RouteKind, TriageDecision
from app.retrieval import hybrid
from app.services.context import RequestContext

logger = get_logger(__name__)

_NO_SOURCE_ANSWER = (
    "Aucune source fiable n'a été trouvée dans la documentation pour répondre à cette "
    "demande. Elle est transmise à un gestionnaire."
)
_NOT_ANSWERABLE = (
    "La documentation disponible ne permet pas de répondre à cette demande de façon "
    "vérifiable. Elle est transmise à un gestionnaire."
)
_EXTRACT_NOTICE = (
    "Extraits de la documentation, cités tels quels (réponse sans reformulation par un "
    "modèle de langage) :"
)

_MAX_QUOTES = 3
# A passage is quoted after the best one only if it reaches this share of the best
# score: one precise passage is worth more than three vague ones.
_MIN_SCORE_SHARE = 0.75

# "- item", "* item", "• item", "3. item"
_LIST_MARKER = re.compile(r"^(?:[-*•]|\d+\.)\s+")
# "|---|:---:|" between the header row and the data rows of a Markdown table
_TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|$")


@dataclass(frozen=True)
class _Passage:
    """One quotable unit of a source: a sentence, a list item or a table row."""

    ref: int  # 1-based source number
    text: str  # as shown in the answer
    tokens: frozenset[str]  # tokens of the passage itself
    # Tokens of the header row of a table row. They count only when the row itself
    # matches the question: "P1" in the header must not bring every row of the table.
    context: frozenset[str] = frozenset()
    # Tokens of the document title and of the heading above the passage. They only help
    # a passage qualify (see `_best_passages`); they do not rank it, or every sentence
    # of a section whose title matches would come before the sentence that answers.
    heading: frozenset[str] = frozenset()


def answer_with_rag(req: AskRequest, decision: TriageDecision, ctx: RequestContext) -> AskResponse:
    """Answer `ctx.q_masked` from the corpus; escalate when the sources do not support it.

    A client-scoped document is searched only for a known client given in the request's
    client field (`TriageDecision.scoped_client_id`). `IndexNotReady` is not handled
    here: the API turns it into a 503.
    """
    settings = get_settings()
    retriever = hybrid.get_retriever()
    client_id = decision.scoped_client_id
    # Masking tags are not search terms: the question is searched without them.
    query = pii.strip_tags(ctx.q_masked)
    results = retriever.search(query, client_id=client_id)
    confidence = guardrails.retrieval_confidence(query, results)
    ctx.log(
        f"Recherche documentaire ({retriever.mode}) : {len(results)} source(s), "
        f"confiance {confidence:.2f}"
    )

    def respond(route: RouteKind, mode: AnswerMode, answer: str, refs: list[int]) -> AskResponse:
        return AskResponse(
            request_id=ctx.request_id,
            route=route,
            mode=mode,
            answer=answer,
            citations=guardrails.build_citations(results, refs),
            confidence=round(confidence, 2),
            needs_validation=True,
            retrieval_mode=retriever.mode,
        )

    if not results or confidence < settings.min_confidence:
        ctx.log(
            f"Confiance inférieure au seuil ({settings.min_confidence:.2f}) : "
            "demande transmise à un gestionnaire"
        )
        return respond("human", "none", _NO_SOURCE_ANSWER, [])

    generated = _generate(ctx, results) if llm.is_enabled("rag") else None
    if generated is not None:
        # "startswith": a model sometimes adds a full stop or a comment to the keyword.
        if generated.casefold().startswith("insuffisant"):
            ctx.log("Génération : sources jugées insuffisantes par le modèle, escalade")
            return respond("human", "llm", _NOT_ANSWERABLE, [])
        refs = guardrails.valid_refs(generated, len(results))
        if not refs:
            # The handler could not check an answer that cites none of the sources.
            ctx.log("Génération : réponse du modèle sans citation valide, escalade")
            return respond("human", "llm", _NOT_ANSWERABLE, [])
        unknown = [ref for ref in guardrails.extract_refs(generated) if ref not in refs]
        if unknown:
            ctx.log(f"Génération : référence(s) à des sources inexistantes {unknown}, à vérifier")
        ctx.log(f"Génération : réponse du modèle, {len(refs)} source(s) citée(s)")
        return respond("rag", "llm", generated, refs)

    quotes = _best_passages(query, results)
    if not quotes:
        ctx.log("Génération : aucun passage des sources ne reprend les termes de la demande")
        return respond("human", "none", _NOT_ANSWERABLE, [])
    ctx.log(f"Génération : mode extractif, {len(quotes)} passage(s) cité(s)")
    lines = [_EXTRACT_NOTICE, *(f"- {quote.text} [{quote.ref}]" for quote in quotes)]
    return respond("rag", "extractive", "\n".join(lines), [quote.ref for quote in quotes])


def _generate(ctx: RequestContext, results: list[RetrievedChunk]) -> str | None:
    """Text written by the model, or None when the call failed (extractive fallback)."""
    settings = get_settings()
    sources = guardrails.format_sources(results)
    question = guardrails.neutralize_tags(ctx.q_masked)
    messages = [
        {"role": "system", "content": loader.load("system_rag")},
        {
            "role": "user",
            "content": f"<sources>\n{sources}\n</sources>\n<demande>\n{question}\n</demande>",
        },
    ]
    try:
        result = llm.complete("rag", messages, max_tokens=settings.rag_max_tokens)
    except (llm.LLMError, llm.LLMUnavailable) as exc:
        # Class name only: nothing coming from the provider reaches the decision log.
        logger.warning("rag_llm_fallback", extra={"error": type(exc).__name__})
        ctx.log(f"Génération : modèle indisponible ({type(exc).__name__}), repli extractif")
        return None
    ctx.add_llm(result)  # a truncated answer was paid for too
    if result.finish_reason == "length":
        ctx.log("Génération : réponse du modèle tronquée (limite de tokens), repli extractif")
        return None
    if ctx.usage.cost_eur > settings.max_cost_eur_per_request:
        ctx.log(
            f"RG-08 : coût de la requête ({ctx.usage.cost_eur:.4f} €) supérieur au budget "
            f"({settings.max_cost_eur_per_request:.4f} €)"
        )
    return result.text.strip()


def _best_passages(query: str, results: list[RetrievedChunk]) -> list[_Passage]:
    """The one to three passages sharing the most tokens with the question, best first.

    Empty when no passage shares at least two content tokens with the question (one
    when the question has a single content token), counting the header row of a table
    row, the heading above the passage and the document title. Passages are ranked on
    their own tokens and, for a table row, its header.
    """
    query_tokens = set(tokenize(query))
    # One shared word is not evidence that a passage answers the question.
    required = min(2, len(query_tokens))
    scored: list[tuple[int, _Passage]] = []
    seen: set[str] = set()
    for ref, result in enumerate(results, start=1):
        chunk = result.chunk
        for passage in _passages(ref, chunk.text, chunk.section, chunk.title):
            # Consecutive chunks overlap, so the same sentence can come up twice.
            if passage.text in seen:
                continue
            seen.add(passage.text)
            own = len(query_tokens & passage.tokens)
            if not own:
                continue
            score = own + len(query_tokens & passage.context)
            if score + _in_headings(query_tokens - passage.tokens, passage.heading) >= required:
                scored.append((score, passage))
    if not scored:
        return []
    # Stable sort: equal scores keep the retrieval order, then the order of the text.
    scored.sort(key=lambda item: -item[0])
    floor = scored[0][0] * _MIN_SCORE_SHARE
    return [passage for score, passage in scored[:_MAX_QUOTES] if score >= floor]


def _in_headings(query_tokens: set[str], heading_tokens: frozenset[str]) -> int:
    """Number of query tokens found in a title or heading.

    A title names its subject with another word of the same family as the question's
    ("Grille tarifaire" for "tarifs"), which the stemmer does not always reduce to one
    stem: here a token also matches when it starts the other one or the reverse, from
    five letters up.
    """

    def same_family(a: str, b: str) -> bool:
        short, long = sorted((a, b), key=len)
        return a == b or (len(short) >= 5 and long.startswith(short))

    return sum(1 for token in query_tokens if any(same_family(token, h) for h in heading_tokens))


def _passages(ref: int, text: str, section: str = "", title: str = "") -> list[_Passage]:
    """Split a chunk into quotable passages: sentences, list items and table rows.

    Headings are not quoted: they announce a content, they do not state it. They are
    kept as the context of the passages below them ("Le rappel a lieu sous 20 minutes"
    answers a question on the "délai de rappel" because of its heading); `section` is
    the heading above the chunk and `title` the title of its document. A table row stays
    in one piece and is shown with its
    header row, without which its cells mean nothing. Sentences are expected to hold on
    one line, as in the Markdown, e-mail and PDF sources of the corpus; a PDF with
    hard-wrapped lines would be quoted line by line.
    """
    passages = []
    title_tokens = frozenset(tokenize(title))
    heading = title_tokens | frozenset(tokenize(section))
    header: list[str] | None = None
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            header = None
            if _is_heading(line):
                heading = title_tokens | frozenset(tokenize(line))
            else:
                passages.extend(_sentences(ref, line, heading))
            continue
        if _TABLE_SEPARATOR.match(line):
            continue
        cells = _cells(line)
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if _TABLE_SEPARATOR.match(next_line):
            header = cells  # the row right above the separator names the columns
            continue
        own = frozenset(tokenize(line))
        context = frozenset(tokenize(" ".join(header or []))) - own
        passages.append(_Passage(ref, _row_text(header, cells), own, context, heading - own))
    return passages


def _is_heading(line: str) -> bool:
    """Markdown heading, or numbered title of a plain-text document ("7. Report")."""
    if line.startswith("#"):
        return True
    # A numbered list item is a sentence and ends with punctuation; a title does not.
    return line[:1].isdigit() and bool(_LIST_MARKER.match(line)) and line[-1] not in ".!?"


def _sentences(ref: int, line: str, heading: frozenset[str]) -> list[_Passage]:
    passages = []
    for sentence in split_sentences(_LIST_MARKER.sub("", line)):
        tokens = frozenset(tokenize(sentence))
        # A fragment with a single content word states nothing.
        if len(tokens) >= 2:
            passages.append(_Passage(ref, sentence, tokens, heading=heading - tokens))
    return passages


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def _row_text(header: list[str] | None, cells: list[str]) -> str:
    """A table row as text, each cell after its column name: "Formule : Confort ; P1 : 4 h"."""
    if header is None or len(header) != len(cells):
        # The header is in another chunk, or the row is irregular: quote the cells alone.
        return " ; ".join(cells)
    return " ; ".join(f"{name} : {cell}" for name, cell in zip(header, cells, strict=True))
