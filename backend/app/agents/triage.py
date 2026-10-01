"""Deterministic triage: one request, one route, by business rules evaluated in order.

No model is involved, so the same request always takes the same route and every
decision can be explained by the rule that fired.

Order of the rules:
1. RG-04   sensitive subject                                 -> human
2. RG-03   billing dispute below the threshold, known client -> automation
3. RG-03b  billing dispute at or above the threshold         -> human
4. RG-05   breakdown or action requested on a dossier        -> agent
5. DEFAULT documentary question                              -> rag

RG-04 comes first so that a request mentioning a termination or a lawyer is never
handled automatically, even when it also disputes a small invoice.

Vocabulary is matched on word stems, exact words and word sequences, never on raw
substrings: "ticket" must not match "et", nor "en retard" match "en arrêt".
"""

from dataclasses import dataclass

from app.agents import extract
from app.core.config import get_settings
from app.core.schemas import AskRequest
from app.core.text import has_stem, normalize, tokenize, words
from app.core.types import RouteKind, TriageDecision
from app.domain import clients


def _word_sequence(text: str) -> str:
    """Accent-free lowercase words separated by single spaces ("droit d acces").

    Punctuation, hyphens and both kinds of apostrophe disappear, so "trop-perçu" and
    "trop perçu" give the same sequence.
    """
    return " ".join(words(normalize(text)))


def _phrases(*texts: str) -> tuple[str, ...]:
    """Phrases written in plain French, in the padded form `_Request.sequence` is searched for.

    The surrounding spaces make a phrase match whole words only.
    """
    return tuple(f" {_word_sequence(text)} " for text in texts)


@dataclass(frozen=True)
class _Vocabulary:
    """What reveals one subject in a request."""

    label: str  # French name of the subject, written to the decision log
    # Prefixes of `tokenize()` stems. The stemmer does not give one stem per verb
    # ("résilier" -> "resili", "résilie" -> "resil"), hence the shortest common form.
    stems: tuple[str, ...] = ()
    # Exact words, for terms whose stem prefix would match unrelated words.
    words: frozenset[str] = frozenset()
    # Sequences of consecutive words, built with `_phrases`.
    phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Request:
    """The three views of the request text that a vocabulary is matched against."""

    stems: frozenset[str]
    words: frozenset[str]
    sequence: str  # padded with spaces, like the phrases

    @classmethod
    def of(cls, text: str) -> "_Request":
        sequence = _word_sequence(text)
        return cls(
            stems=frozenset(tokenize(text)),
            words=frozenset(sequence.split()),
            sequence=f" {sequence} ",
        )

    def mentions(self, vocabulary: _Vocabulary) -> bool:
        if has_stem(self.stems, vocabulary.stems) or self.words & vocabulary.words:
            return True
        return any(phrase in self.sequence for phrase in vocabulary.phrases)


# RG-04. A false positive only sends the request to a person, so prefixes stay broad.
_SENSITIVE = (
    _Vocabulary("résiliation", stems=("resil",)),
    _Vocabulary("pénalités", stems=("penal",)),
    _Vocabulary(
        "contentieux",
        stems=(
            "contenti",
            "avocat",
            "tribunal",
            "huissi",
            "jurid",
            "judiciair",
            "assign",
            "plaint",
        ),
        phrases=_phrases("mise en demeure"),
    ),
    _Vocabulary(
        "données personnelles",
        stems=("rgpd", "cnil"),
        phrases=_phrases(
            "données personnelles",
            "droit d'accès",
            "droit à l'oubli",
            "suppression de mes données",
        ),
    ),
)

# RG-03 / RG-03b. "avoir" (credit note) is also the verb "to have" and a stopword for
# `tokenize()`: only the plural reaches the stems, the singular is matched as "un avoir".
# "erreur de facturation" needs no phrase: "facturation" already has the stem "factur".
_DISPUTE = _Vocabulary(
    "litige de facturation",
    stems=("factur", "contest", "litig", "avoir", "rembours"),
    phrases=_phrases("trop perçu", "un avoir"),
)

# RG-05. "panne" is matched as a word because the prefix "pann" would also match
# "panneau"; "hs" because its stem is the single letter "h".
_BREAKDOWN = _Vocabulary(
    "panne ou dysfonctionnement",
    stems=("fuit", "defaill", "dysfonction"),
    words=frozenset({"panne", "pannes", "hs"}),
    phrases=_phrases(
        "ne fonctionne plus", "ne marche plus", "en arrêt", "à l'arrêt", "hors service"
    ),
)
_ACTION = _Vocabulary(
    "action demandée",
    stems=("planif", "programm", "depann"),
    phrases=_phrases(
        "créer un ticket",
        "ouvrir un ticket",
        "envoyer un technicien",
        "intervention urgente demandée",
    ),
)


def triage(req: AskRequest) -> TriageDecision:
    """Choose the route of `req`. `reasons` never contains the request text."""
    client_id, client_origin = _resolve_client(req)
    client_known = clients.get_client_repository().get(client_id) is not None
    montant, amount_reason = _resolve_amount(req)

    route, rule, rule_reasons = _apply_rules(_Request.of(req.q), montant, client_known)
    return TriageDecision(
        route=route,
        rule=rule,
        reasons=[_client_reason(client_id, client_origin, client_known), amount_reason]
        + rule_reasons,
        client_id=client_id,
        montant=montant,
        client_known=client_known,
    )


def _apply_rules(
    request: _Request, montant: float | None, client_known: bool
) -> tuple[RouteKind, str, list[str]]:
    """First rule that fires: (route, rule id, decision-log lines)."""
    sensitive = [vocabulary.label for vocabulary in _SENSITIVE if request.mentions(vocabulary)]
    if sensitive:
        subjects = ", ".join(sensitive)
        return (
            "human",
            "RG-04",
            [f"RG-04 : sujet sensible ({subjects}), transmis à un gestionnaire"],
        )

    reasons = []
    if request.mentions(_DISPUTE):
        threshold = get_settings().rg03_max_amount
        if montant is not None and 0 < montant < threshold and client_known:
            reason = (
                f"RG-03 : litige de facturation de {extract.format_amount(montant)}, inférieur "
                f"au seuil de {extract.format_amount(threshold)}, pour un client connu : "
                "ticket standard"
            )
            return "automation", "RG-03", [reason]
        if montant is not None and montant >= threshold:
            reason = (
                f"RG-03b : litige de facturation de {extract.format_amount(montant)}, supérieur "
                f"ou égal au seuil de {extract.format_amount(threshold)} : transmis à un "
                "responsable facturation"
            )
            return "human", "RG-03b", [reason]
        # Not automated: the request goes on through the following rules.
        missing = "montant exploitable" if montant is None or montant <= 0 else "client connu"
        reasons.append(f"Litige de facturation non automatisé : pas de {missing}")

    subjects = [v.label for v in (_BREAKDOWN, _ACTION) if request.mentions(v)]
    # Asking to check something is an action on the dossier only when it is about the contract.
    if has_stem(request.stems, ("verif",)) and has_stem(request.stems, ("contrat",)):
        subjects.append("vérification du contrat")
    if subjects:
        reasons.append(f"RG-05 : {', '.join(subjects)} : dossier confié à l'agent")
        return "agent", "RG-05", reasons

    reasons.append("DEFAULT : aucune règle métier ne s'applique, recherche documentaire")
    return "rag", "DEFAULT", reasons


def _resolve_client(req: AskRequest) -> tuple[str | None, str]:
    """Client identifier and where it comes from; the explicit field wins over the text."""
    if req.client_id:
        return req.client_id, "champ"
    return extract.extract_client_id(req.q), "texte"


def _client_reason(client_id: str | None, origin: str, known: bool) -> str:
    if client_id is None:
        return "Client : non renseigné"
    status = "connu du référentiel" if known else "inconnu du référentiel"
    return f"Client : {client_id} ({origin}), {status}"


def _resolve_amount(req: AskRequest) -> tuple[float | None, str]:
    """Disputed amount and the decision-log line that explains it.

    The explicit field wins. Otherwise the text must hold exactly one distinct amount:
    with several ("480 € au lieu de 300 €") nothing says which one is disputed, and
    guessing would create a ticket for the wrong amount.
    """
    if req.montant is not None:
        return req.montant, f"Montant : {extract.format_amount(req.montant)} (champ)"
    found = sorted(set(extract.extract_amounts(req.q)))
    if not found:
        return None, "Montant : absent"
    if len(found) > 1:
        return None, f"Montant : non résolu ({len(found)} montants différents dans le texte)"
    return found[0], f"Montant : {extract.format_amount(found[0])} (texte)"
