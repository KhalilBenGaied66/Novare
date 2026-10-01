"""Deterministic triage: one request, one route, by business rules evaluated in order.

No model is involved, so the same request always takes the same route and every
decision can be explained by the rule that fired.

Order of the rules:
1. RG-04   sensitive subject                                          -> human
2. RG-03   explicit billing dispute below the threshold, known client -> automation
3. RG-03b  billing subject at or above the threshold                  -> human
4. RG-05   breakdown or action requested on a dossier                 -> agent
5. DEFAULT documentary question                                       -> rag

RG-04 comes first so that a request mentioning a termination or a lawyer is never
handled automatically, even when it also disputes a small invoice.

The rules read vocabulary, not intent: a subject worded outside the lists below is not
recognised. The lists err on the side of the person: a false match of RG-04 or RG-03b
only sends the request to a handler, whereas RG-03, the only rule that acts without
validation, needs an explicit dispute and nothing else going on in the request.

Vocabulary is matched on word stems, exact words and word sequences, never on raw
substrings: "ticket" must not match "et", nor "en retard" match "en arrêt".
"""

from dataclasses import dataclass

from app.agents import extract
from app.core.config import get_settings
from app.core.schemas import AskRequest
from app.core.text import has_stem, tokenize, word_sequence
from app.core.types import RouteKind, TriageDecision
from app.domain import clients


def _phrases(*texts: str) -> tuple[str, ...]:
    """Phrases written in plain French, in the padded form `_Request.sequence` is searched for.

    The surrounding spaces make a phrase match whole words only.
    """
    return tuple(f" {word_sequence(text)} " for text in texts)


@dataclass(frozen=True)
class _Vocabulary:
    """What reveals one subject in a request."""

    label: str  # French name of the subject, written to the decision log
    # Prefixes of `tokenize()` stems. The stemmer does not give one stem per verb
    # ("résilier" -> "resili", "résilie" -> "resil"), hence the shortest common form.
    stems: tuple[str, ...] = ()
    # Exact words (accent-free), for terms whose stem prefix would match unrelated
    # words: "proces" must not match "processus".
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
        sequence = word_sequence(text)
        return cls(
            stems=frozenset(tokenize(text)),
            words=frozenset(sequence.split()),
            sequence=f" {sequence} ",
        )

    def mentions(self, vocabulary: _Vocabulary) -> bool:
        if has_stem(self.stems, vocabulary.stems) or self.words & vocabulary.words:
            return True
        return any(phrase in self.sequence for phrase in vocabulary.phrases)


# RG-04. A false positive only sends the request to a person, so the lists stay broad.
_SENSITIVE = (
    _Vocabulary(
        "résiliation",
        stems=("resil", "denonc"),
        phrases=_phrases(
            "mettre fin au contrat",
            "mettre fin à mon contrat",
            "mettre fin à notre contrat",
            "mettons fin",
            "rompre le contrat",
            "rompre l'engagement",
            "ne renouvellerons pas",
            "ne pas renouveler",
        ),
    ),
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
            "justic",
            "poursuit",
            "mediat",
        ),
        words=frozenset({"proces", "refere", "dgccrf"}),
        phrases=_phrases("en demeure"),
    ),
    _Vocabulary(
        "données personnelles",
        stems=("rgpd", "cnil", "effac", "rectif", "portab"),
        words=frozenset({"dpo"}),
        phrases=_phrases(
            "données personnelles",
            "donnée personnelle",
            "caractère personnel",
            "mes données",
            "informations me concernant",
            "fuite de données",
            "violation de données",
            "droit d'accès",
            "droit à l'oubli",
            "droit d'opposition",
            "supprimer mes coordonnées",
            "r g p d",
        ),
    ),
)

# RG-03b: anything about billing at or above the threshold goes to a billing manager.
# "avoir" (credit note) is also the verb "to have" and a stopword for `tokenize()`: only
# the plural reaches the stems, the singular is matched as "un avoir".
_BILLING = _Vocabulary(
    "facturation",
    stems=("factur", "contest", "litig", "avoir", "rembours"),
    phrases=_phrases("trop perçu", "un avoir"),
)

# RG-03 creates a ticket without validation: the request must dispute something in so
# many words. Mentioning an invoice ("merci pour la facture de 120 €") is not a dispute.
_EXPLICIT_DISPUTE = _Vocabulary(
    "contestation explicite",
    stems=("contest", "litig", "avoir"),
    phrases=_phrases(
        "trop perçu",
        "un avoir",
        "erreur de facturation",
        "erreur sur la facture",
        "erreur sur ma facture",
        "demande de remboursement",
        "facturé à tort",
        "facturée à tort",
        "facturé deux fois",
        "facturée deux fois",
    ),
)
_DISPUTE_DENIED = _phrases(
    "ne conteste pas", "ne contestons pas", "ne conteste aucun", "sans contester"
)

# RG-05. "panne" is matched as a word because the prefix "pann" would also match
# "panneau"; "hs" because its stem is the single letter "h".
_BREAKDOWN = _Vocabulary(
    "panne ou dysfonctionnement",
    stems=("fuit", "defaill", "dysfonction", "inond", "incend"),
    words=frozenset({"panne", "pannes", "hs"}),
    phrases=_phrases(
        "ne fonctionne plus",
        "ne fonctionne pas",
        "ne marche plus",
        "ne marche pas",
        "ne chauffe plus",
        "ne démarre plus",
        "plus de chauffage",
        "plus d'eau chaude",
        "en arrêt",
        "à l'arrêt",
        "en défaut",
        "hors service",
        "h s",
        "odeur de gaz",
        "sent le gaz",
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

    # A question that names no client is about the documentation, even when it uses
    # the vocabulary of a breakdown ("Que faire en cas de panne ?").
    question_without_client = client_id is None and req.q.rstrip().endswith("?")
    route, rule, rule_reasons = _apply_rules(
        _Request.of(req.q), montant, client_known, question_without_client
    )
    return TriageDecision(
        route=route,
        rule=rule,
        reasons=[_client_reason(client_id, client_origin, client_known), amount_reason]
        + rule_reasons,
        client_id=client_id,
        montant=montant,
        client_known=client_known,
        client_origin=client_origin,
    )


def _apply_rules(
    request: _Request, montant: float | None, client_known: bool, question_without_client: bool
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
    breakdown = request.mentions(_BREAKDOWN)
    if request.mentions(_BILLING):
        threshold = get_settings().rg03_max_amount
        if montant is not None and montant >= threshold:
            reason = (
                f"RG-03b : facturation, montant de {extract.format_amount(montant)} supérieur "
                f"ou égal au seuil de {extract.format_amount(threshold)} : transmis à un "
                "responsable facturation"
            )
            return "human", "RG-03b", [reason]
        blocker = _automation_blocker(request, montant, client_known, breakdown)
        if blocker is None:
            reason = (
                f"RG-03 : litige de facturation de {extract.format_amount(montant)}, inférieur "
                f"au seuil de {extract.format_amount(threshold)}, pour un client connu : "
                "ticket standard"
            )
            return "automation", "RG-03", [reason]
        # Not automated: the request goes on through the following rules.
        reasons.append(f"Facturation, non automatisé : {blocker}")

    subjects = [v.label for v in (_BREAKDOWN, _ACTION) if request.mentions(v)]
    # Asking to check something is an action on the dossier only when it is about the contract.
    if has_stem(request.stems, ("verif",)) and has_stem(request.stems, ("contrat",)):
        subjects.append("vérification du contrat")
    if subjects and question_without_client:
        reasons.append(
            f"RG-05 non appliquée ({', '.join(subjects)}) : question sans client identifié"
        )
    elif subjects:
        reasons.append(f"RG-05 : {', '.join(subjects)} : dossier confié à l'agent")
        return "agent", "RG-05", reasons

    reasons.append("DEFAULT : aucune règle métier ne s'applique, recherche documentaire")
    return "rag", "DEFAULT", reasons


def _automation_blocker(
    request: _Request, montant: float | None, client_known: bool, breakdown: bool
) -> str | None:
    """Why RG-03 must not create a ticket for this request; None when it may."""
    if montant is None or montant <= 0:
        return "pas de montant exploitable"
    if not client_known:
        return "pas de client connu"
    denied = any(phrase in request.sequence for phrase in _DISPUTE_DENIED)
    if denied or not request.mentions(_EXPLICIT_DISPUTE):
        return "pas de contestation explicite"
    if breakdown:
        return "la demande signale aussi une panne"
    return None


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
    """Disputed amount, in whole cents, and the decision-log line that explains it.

    The explicit field wins. Otherwise the text must hold exactly one distinct amount:
    with several ("480 € au lieu de 300 €") nothing says which one is disputed, and
    guessing would create a ticket for the wrong amount.
    """
    if req.montant is not None:
        # Rounded before any comparison: 499,999 € is 500,00 € on an invoice.
        montant = round(req.montant, 2)
        return montant, f"Montant : {extract.format_amount(montant)} (champ)"
    found = sorted(set(extract.extract_amounts(req.q)))
    if not found:
        return None, "Montant : absent"
    if len(found) > 1:
        return None, f"Montant : non résolu ({len(found)} montants différents dans le texte)"
    return found[0], f"Montant : {extract.format_amount(found[0])} (texte)"
