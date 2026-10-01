"""French text normalisation shared by triage, retrieval and guardrails.

Matching is done on stems (Snowball French) with accents removed, never on raw
substrings ("ticket" must not match "et"). The stemmer does not give one stem per word
family ("résilier" -> "resili", "résilie" -> "resil"), so callers match on the shortest
common prefix with `has_stem`.
"""

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[0-9a-zà-öø-ÿœæ]+", re.IGNORECASE)
_STEMMER = snowballstemmer.stemmer("french")

STOPWORDS = frozenset(
    """
    a à ai as au aux avec avez avons c ce ces cet cette ceci cela d dans de des du donc elle
    elles en est et eu eux il ils j je l la le les leur leurs lui m ma mais me mes moi mon n
    ne ni nos notre nous on ont ou où par pas pour qu que quel quelle quelles quels qui quoi
    s sa sans se ses si son sont sur t ta te tes toi ton tu un une vos votre vous y été être
    avoir fait faire peut plus moins très bien aussi comme alors afin ainsi tout tous toute
    toutes cher chère bonjour bonsoir merci cordialement svp
    combien comment quand pourquoi lequel laquelle lesquels lesquelles dont
    suis es sommes êtes étais était serai sera serait avais avait aurai aura aurait
    dois doit devons devez doivent peux pouvez pouvons peuvent puis veux veut voulons voulez
    faut vais va allons allez vont chez encore rien déjà jamais toujours ici là ça
    """.split()  # noqa: SIM905
)


# Ligatures have no Unicode decomposition: "main-d'œuvre" must meet "main d'oeuvre".
_LIGATURES = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(_LIGATURES))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lowercase, accent-free, single-spaced text for phrase matching."""
    composed = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", strip_accents(composed.lower())).strip()


def words(text: str) -> list[str]:
    """Lowercase word tokens, accents kept (composed form), no filtering."""
    composed = unicodedata.normalize("NFC", text)
    return [m.group(0).lower() for m in _WORD.finditer(composed)]


def word_sequence(text: str) -> str:
    """Accent-free lowercase words separated by single spaces ("droit d acces").

    Punctuation, hyphens and both kinds of apostrophe disappear, so "trop-perçu" and
    "trop perçu" give the same sequence. Used to match multi-word expressions on whole
    words: search for " expression " in " sequence ".
    """
    return " ".join(words(normalize(text)))


@lru_cache(maxsize=20_000)
def stem(word: str) -> str:
    return strip_accents(_STEMMER.stemWord(word.lower().translate(_LIGATURES)))


def tokenize(text: str) -> list[str]:
    """Content tokens for indexing and matching: stopwords removed, stemmed, accent-free.

    Tokens containing a digit ("p2", "4h", "500") are kept as they are; other tokens
    shorter than two characters are dropped.
    """
    out = []
    for word in words(text):
        if word in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in word):
            out.append(strip_accents(word))
        elif len(word) >= 2:
            out.append(stem(word))
    return out


def has_stem(tokens: list[str] | set[str], prefixes: tuple[str, ...]) -> bool:
    """True when any token starts with one of the stem prefixes."""
    return any(tok.startswith(prefixes) for tok in tokens)


def split_sentences(text: str) -> list[str]:
    """Split on line breaks and sentence punctuation; keeps list items and table rows whole."""
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+(?=[A-ZÀ-Ü0-9]))", text)
    return [p.strip() for p in parts if p and p.strip()]
