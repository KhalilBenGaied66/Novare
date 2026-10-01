"""RAG route: extractive answers on the indexed fixture, then the LLM path with a fake model."""

import pytest

from app.core import guardrails, llm, pii
from app.core.config import reset_settings
from app.core.prompts import loader
from app.core.schemas import AskRequest
from app.core.types import Chunk, LLMResult, RetrievedChunk, TriageDecision
from app.ingestion.ingest import ingest_docs
from app.retrieval import hybrid, rag
from app.retrieval.rag import answer_with_rag
from app.services.context import RequestContext

WEEKEND_QUESTION = "Quelle est la majoration appliquée le week-end sur le déplacement ?"
WEEKEND_SENTENCE = "Majoration week-end : +35 % sur le déplacement et la main-d'œuvre."
NOTICE = (
    "Extraits de la documentation, cités tels quels (réponse sans reformulation par un "
    "modèle de langage) :"
)


def ask(q: str, client_id: str | None = None, client_known: bool | None = None):
    """Call the RAG route as `handle_ask` does: masked text in the context, triage done."""
    req = AskRequest(q=q, client_id=client_id)
    ctx = RequestContext()
    ctx.q_masked, ctx.pii_types = pii.redact(req.q)
    decision = TriageDecision(
        route="rag",
        rule="DEFAULT",
        reasons=[],
        client_id=client_id,
        montant=None,
        client_known=client_id is not None if client_known is None else client_known,
    )
    return answer_with_rag(req, decision, ctx), ctx


def cited_docs(response) -> list[str]:
    return [citation.doc for citation in response.citations]


# --- extractive mode (no LLM) ------------------------------------------------------


def test_weekend_surcharge_is_quoted_with_its_source(indexed):
    response, ctx = ask(WEEKEND_QUESTION)

    assert response.route == "rag"
    assert response.mode == "extractive"
    assert response.needs_validation is True
    assert response.retrieval_mode == "hybrid"
    assert response.request_id == ctx.request_id

    notice, first_quote = response.answer.split("\n")[:2]
    assert notice == NOTICE
    (citation,) = response.citations
    assert citation.doc == "grille_tarifs.md"
    assert citation.title == "Grille tarifaire 2026"
    assert first_quote == f"- {WEEKEND_SENTENCE} [{citation.ref}]"
    assert WEEKEND_SENTENCE in citation.excerpt
    assert ctx.usage.llm_calls == 0


def test_table_row_is_quoted_with_its_header(indexed):
    response, _ = ask("Quel est le délai d'intervention P1 pour la formule Confort ?")

    assert response.mode == "extractive"
    first_quote = response.answer.split("\n")[1]
    assert first_quote.startswith(
        "- Formule : Confort ; P1 : 4 h ouvrées ; P2 : 8 h ouvrées ; P3 : 48 h ouvrées ["
    )
    assert cited_docs(response) == ["procedure_sav.md"]
    # The rows of the other formulas share the header but not the question.
    assert "Premium" not in response.answer
    assert "Essentiel" not in response.answer


def test_confidence_and_decision_log(indexed):
    response, ctx = ask(WEEKEND_QUESTION)

    results = hybrid.get_retriever().search(ctx.q_masked)
    confidence = guardrails.retrieval_confidence(ctx.q_masked, results)
    assert confidence >= 0.35
    assert response.confidence == round(confidence, 2)
    assert ctx.decision_log == [
        f"Recherche documentaire (hybrid) : {len(results)} source(s), confiance {confidence:.2f}",
        "Génération : mode extractif, 1 passage(s) cité(s)",
    ]


def test_out_of_scope_question_is_escalated_without_citation(indexed):
    response, ctx = ask("Quelle est la recette de la tarte aux pommes ?")

    assert response.route == "human"
    assert response.mode == "none"
    assert response.citations == []
    assert response.needs_validation is True
    assert response.confidence < 0.35
    assert response.answer == (
        "Aucune source fiable n'a été trouvée dans la documentation pour répondre à cette "
        "demande. Elle est transmise à un gestionnaire."
    )
    assert "Confiance inférieure au seuil (0.35)" in ctx.decision_log[-1]


def test_bm25_mode_is_reported(mini_corpus, db, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    reset_settings()
    ingest_docs()

    response, ctx = ask(WEEKEND_QUESTION)

    assert response.retrieval_mode == "bm25"
    assert response.route == "rag"
    assert WEEKEND_SENTENCE in response.answer
    assert ctx.decision_log[0].startswith("Recherche documentaire (bm25) : ")


def test_missing_index_is_not_swallowed(db):
    with pytest.raises(hybrid.IndexNotReady):
        ask(WEEKEND_QUESTION)


# --- access control ----------------------------------------------------------------

FRANCHISE_QUESTION = "Quelle franchise par sinistre est prévue par le contrat de maintenance ?"


def test_client_reads_its_own_contract(indexed):
    response, _ = ask(FRANCHISE_QUESTION, client_id="C-34")

    assert response.route == "rag"
    assert "contrat_C-34.md" in cited_docs(response)
    assert "franchise de 250 € par sinistre" in response.answer


@pytest.mark.parametrize(
    ("client_id", "client_known"),
    [
        (None, False),  # anonymous request
        ("C-12", True),  # another client
        ("C-99", False),  # identifier absent from the reference data
        ("C-34", False),  # right identifier, but not confirmed by the reference data
    ],
)
def test_contract_of_another_client_is_never_cited(indexed, client_id, client_known):
    response, _ = ask(FRANCHISE_QUESTION, client_id=client_id, client_known=client_known)

    assert "contrat_C-34.md" not in cited_docs(response)
    assert "250" not in response.answer
    assert all("250" not in citation.excerpt for citation in response.citations)


# --- passage selection ---------------------------------------------------------------


def retrieved(text: str, doc: str = "doc.md", section: str = "") -> RetrievedChunk:
    chunk = Chunk(f"{doc}#p1-1", doc, "Titre", "procedure", 1, section, text)
    return RetrievedChunk(chunk, score=1.0)


def quotes(query: str, *texts: str, section: str = "") -> list[tuple[int, str]]:
    results = [retrieved(text, f"doc{n}.md", section) for n, text in enumerate(texts, start=1)]
    return [(passage.ref, passage.text) for passage in rag._best_passages(query, results)]


def test_headings_are_never_quoted():
    text = "# Délai de rappel\n## Délai de rappel du client\nLe rappel a lieu sous 20 minutes."
    assert quotes("Quel est le délai de rappel ?", text) == [
        (1, "Le rappel a lieu sous 20 minutes.")
    ]


def test_numbered_title_of_a_plain_text_document_is_not_quoted():
    text = "7. Report et annulation\nUn report est gratuit jusqu'à 24 heures avant le créneau."
    assert quotes("Le report est-il gratuit ?", text) == [
        (1, "Un report est gratuit jusqu'à 24 heures avant le créneau.")
    ]


def test_list_markers_are_removed_from_quotes():
    text = (
        "- Portail client : disponible 24 h/24.\n2. Clôture : le ticket est clôturé sous 5 jours."
    )
    assert quotes("Quand le ticket est-il clôturé ?", text) == [
        (1, "Clôture : le ticket est clôturé sous 5 jours.")
    ]


def test_table_row_without_header_in_the_chunk_is_quoted_alone():
    # The chunk starts in the middle of a table: its header row is in the previous chunk.
    # The heading above the chunk still says what the rows are about.
    text = "| Confort | 4 h ouvrées | 8 h ouvrées |\n| Essentiel | 8 h ouvrées | 24 h ouvrées |"
    assert quotes("Quel délai pour la formule Confort ?", text, section="Délais") == [
        (1, "Confort ; 4 h ouvrées ; 8 h ouvrées")
    ]
    assert quotes("Quel délai pour la formule Confort ?", text) == []


def test_one_shared_word_is_not_enough_to_quote_a_passage():
    text = "## Agences\nL'agence de Lyon est ouverte du lundi au vendredi."
    assert quotes("Quel temps fera-t-il demain à Lyon ?", text) == []
    assert quotes("Quand l'agence de Lyon est-elle ouverte ?", text) == [
        (1, "L'agence de Lyon est ouverte du lundi au vendredi.")
    ]


def test_header_row_alone_does_not_make_a_row_match():
    text = "| Formule | Délai P1 |\n|---|---|\n| Premium | 2 h |\n| Confort | 4 h |"
    # "délai" and "P1" are in the header only: no row answers the question by itself.
    assert quotes("Quel est le délai P1 ?", text) == []
    assert quotes("Quel est le délai P1 en Premium ?", text) == [
        (1, "Formule : Premium ; Délai P1 : 2 h")
    ]


def test_at_most_three_passages_best_first():
    text = "\n".join(
        [
            "Le forfait diagnostic coûte 120 €.",
            "Le forfait diagnostic est déduit du devis accepté.",
            "Le forfait diagnostic comprend le déplacement.",
            "Le forfait diagnostic comprend une heure de recherche.",
            "Le forfait diagnostic comprend le rapport.",
            "Le forfait diagnostic comprend la mise en sécurité.",
        ]
    )
    assert quotes("Le forfait diagnostic est-il déduit du devis ?", text) == [
        (1, "Le forfait diagnostic est déduit du devis accepté.")
    ]
    # Four sentences match equally: the first three of the text are kept.
    assert quotes("Que comprend le forfait diagnostic ?", text) == [
        (1, "Le forfait diagnostic comprend le déplacement."),
        (1, "Le forfait diagnostic comprend une heure de recherche."),
        (1, "Le forfait diagnostic comprend le rapport."),
    ]


def test_weak_matches_are_dropped_next_to_a_strong_one():
    strong = "La majoration de nuit est de 50 % sur le déplacement."
    weak = "Le déplacement en journée coûte 89 €."
    assert quotes("Quelle majoration de nuit sur le déplacement ?", f"{weak}\n{strong}") == [
        (1, strong)
    ]


def test_sentence_repeated_by_chunk_overlap_is_quoted_once():
    sentence = "Le rappel du client a lieu sous 20 minutes."
    assert quotes("Sous quel délai a lieu le rappel du client ?", sentence, sentence) == [
        (1, sentence)
    ]


def test_ties_keep_the_retrieval_order():
    first = "Le rappel a lieu sous 20 minutes en astreinte."
    second = "Le rappel a lieu sous 2 heures en journée."
    assert quotes("Quand a lieu le rappel ?", first, second) == [(1, first), (2, second)]


def test_no_shared_word_means_no_quote_and_an_escalation(indexed, monkeypatch):
    unrelated = retrieved("Le rappel a lieu sous 20 minutes.")
    assert quotes("Quel est le prix ?", unrelated.chunk.text) == []

    # Retrieval is confident (for instance through vectors alone) but nothing can be quoted.
    monkeypatch.setattr(hybrid.Retriever, "search", lambda self, query, **kwargs: [unrelated])
    monkeypatch.setattr(guardrails, "retrieval_confidence", lambda query, results: 0.9)
    response, ctx = ask("Quel est le prix ?")

    assert response.route == "human"
    assert response.mode == "none"
    assert response.citations == []
    assert response.confidence == 0.9
    assert "transmise à un gestionnaire" in response.answer
    assert ctx.decision_log[-1] == (
        "Génération : aucun passage des sources ne reprend les termes de la demande"
    )


# --- LLM mode ------------------------------------------------------------------------


class FakeModel:
    """Stands for `llm.complete`: records the calls and replies with a fixed text."""

    def __init__(self, text: str = "", error: Exception | None = None, cost_eur: float = 0.0004):
        self.text = text
        self.error = error
        self.cost_eur = cost_eur
        self.calls: list[dict] = []

    def __call__(self, task, messages, *, tools=None, max_tokens=None) -> LLMResult:
        self.calls.append(
            {"task": task, "messages": messages, "tools": tools, "max_tokens": max_tokens}
        )
        if self.error is not None:
            raise self.error
        return LLMResult(
            text=self.text,
            model="mistral/mistral-small-latest",
            prompt_tokens=850,
            completion_tokens=40,
            cost_eur=self.cost_eur,
        )


def use_model(monkeypatch, **kwargs) -> FakeModel:
    model = FakeModel(**kwargs)
    monkeypatch.setattr(llm, "is_enabled", lambda task: True)
    monkeypatch.setattr(llm, "complete", model)
    return model


def test_llm_answer_keeps_only_the_cited_sources(indexed, monkeypatch):
    text = "La majoration du week-end est de +35 % [2]. Elle porte sur le déplacement [2]."
    model = use_model(monkeypatch, text=text)

    response, ctx = ask(WEEKEND_QUESTION)

    assert response.route == "rag"
    assert response.mode == "llm"
    assert response.answer == text
    assert response.needs_validation is True
    assert response.retrieval_mode == "hybrid"
    assert len(model.calls) == 1
    assert ctx.usage.llm_calls == 1
    assert ctx.usage.model == "mistral/mistral-small-latest"
    assert (ctx.usage.prompt_tokens, ctx.usage.completion_tokens) == (850, 40)
    assert ctx.usage.cost_eur == 0.0004
    assert ctx.decision_log[-1] == "Génération : réponse du modèle, 1 source(s) citée(s)"

    # Several sources were sent to the model; only the one it cites is returned.
    results = hybrid.get_retriever().search(ctx.q_masked)
    assert len(results) > 1
    (citation,) = response.citations
    assert citation.ref == 2
    assert citation.chunk_id == results[1].chunk.chunk_id


def test_llm_citations_follow_the_order_of_the_answer(indexed, monkeypatch):
    use_model(monkeypatch, text="Majoration de +35 % [2, 1]. Hors périmètre [7].")
    response, _ = ask(WEEKEND_QUESTION)
    assert response.mode == "llm"
    assert [citation.ref for citation in response.citations] == [2, 1]


@pytest.mark.parametrize(
    "reply", ["INSUFFISANT", "  insuffisant \n", "Insuffisant.", "INSUFFISANT [1]"]
)
def test_insufficient_sources_are_escalated(indexed, monkeypatch, reply):
    use_model(monkeypatch, text=reply)

    response, ctx = ask(WEEKEND_QUESTION)

    assert response.route == "human"
    assert response.mode == "llm"
    assert response.citations == []
    assert "INSUFFISANT" not in response.answer.upper()
    assert "transmise à un gestionnaire" in response.answer
    assert ctx.usage.llm_calls == 1  # the call was made and is accounted for
    assert "insuffisantes" in ctx.decision_log[-1]


@pytest.mark.parametrize(
    "reply",
    [
        "La majoration du week-end est de 40 %.",  # no reference at all
        "La majoration du week-end est de 40 % [9].",  # reference to a source that does not exist
        "La majoration est de 40 % [0], contactez-nous au [TEL].",
        "",
    ],
)
def test_uncited_answer_is_never_returned(indexed, monkeypatch, reply):
    use_model(monkeypatch, text=reply)

    response, ctx = ask(WEEKEND_QUESTION)

    assert response.route == "human"
    assert response.mode == "llm"
    assert response.citations == []
    assert "40 %" not in response.answer
    assert "sans citation valide" in ctx.decision_log[-1]


@pytest.mark.parametrize(
    "error",
    [llm.LLMError("AuthenticationError: clé sk-secret-123 refusée"), llm.LLMUnavailable("no key")],
)
def test_llm_failure_falls_back_to_the_extractive_mode(indexed, monkeypatch, error):
    model = use_model(monkeypatch, error=error)

    response, ctx = ask(WEEKEND_QUESTION)

    assert len(model.calls) == 1
    assert response.route == "rag"
    assert response.mode == "extractive"
    assert WEEKEND_SENTENCE in response.answer
    assert cited_docs(response) == ["grille_tarifs.md"]
    assert ctx.usage.llm_calls == 0
    # Only the class name is recorded; the message of the error goes nowhere.
    everything = response.model_dump_json() + " ".join(ctx.decision_log)
    assert "sk-secret-123" not in everything
    assert "no key" not in everything
    assert f"modèle indisponible ({type(error).__name__}), repli extractif" in ctx.decision_log[1]
    assert ctx.decision_log[2].startswith("Génération : mode extractif")


def test_cost_over_budget_is_flagged(indexed, monkeypatch):
    use_model(monkeypatch, text="La majoration est de +35 % [1].", cost_eur=0.05)

    response, ctx = ask(WEEKEND_QUESTION)

    assert response.mode == "llm"  # RG-08 flags the overrun, it does not block the answer
    flagged = [line for line in ctx.decision_log if line.startswith("RG-08")]
    assert flagged == ["RG-08 : coût de la requête (0.0500 €) supérieur au budget (0.0200 €)"]


def test_cost_within_budget_is_not_flagged(indexed, monkeypatch):
    use_model(monkeypatch, text="La majoration est de +35 % [1].", cost_eur=0.02)
    _, ctx = ask(WEEKEND_QUESTION)
    assert not any("RG-08" in line for line in ctx.decision_log)


def test_prompt_holds_the_masked_question_and_the_numbered_sources(indexed, monkeypatch, env):
    model = use_model(monkeypatch, text="La majoration est de +35 % [1].")
    question = (
        "Répondez à marie.durand@example.com ou au 06 12 34 56 78. "
        "Quelle est la majoration appliquée le week-end sur le déplacement ?"
    )

    _, ctx = ask(question)

    assert ctx.pii_types == ["EMAIL", "TEL"]
    (call,) = model.calls
    assert call["task"] == "rag"
    assert call["tools"] is None
    assert call["max_tokens"] == env.rag_max_tokens
    system, user = call["messages"]
    assert system == {"role": "system", "content": loader.load("system_rag")}
    assert user["role"] == "user"

    results = hybrid.get_retriever().search(ctx.q_masked)
    assert user["content"] == (
        f"<sources>\n{guardrails.format_sources(results)}\n</sources>\n"
        "<demande>\n"
        "Répondez à [EMAIL] ou au [TEL]. "
        "Quelle est la majoration appliquée le week-end sur le déplacement ?\n"
        "</demande>"
    )
    assert "[1] Grille tarifaire 2026 — grille_tarifs.md, p. 1" in user["content"]
    assert WEEKEND_SENTENCE in user["content"]
    prompt = system["content"] + user["content"]
    assert "marie.durand@example.com" not in prompt
    assert "06 12 34 56 78" not in prompt


def test_prompt_sources_respect_the_client_scope(indexed, monkeypatch):
    model = use_model(monkeypatch, text="INSUFFISANT")

    ask(FRANCHISE_QUESTION, client_id="C-12")
    assert "250 €" not in model.calls[-1]["messages"][1]["content"]

    ask(FRANCHISE_QUESTION, client_id="C-34")
    assert "franchise de 250 € par sinistre" in model.calls[-1]["messages"][1]["content"]


def test_model_is_not_called_when_retrieval_is_not_confident(indexed, monkeypatch):
    model = use_model(monkeypatch, text="La tarte se prépare avec des pommes [1].")

    response, ctx = ask("Quelle est la recette de la tarte aux pommes ?")

    assert model.calls == []
    assert response.route == "human"
    assert response.mode == "none"
    assert ctx.usage.llm_calls == 0
