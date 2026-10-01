"""Client identifiers and euro amounts read from free text."""

import re

_CLIENT_ID = re.compile(r"\bC-\d{1,6}\b", re.IGNORECASE)

# A number followed by a currency mark: "120 €", "120€", "1 250,50 €", "1.250,50 euros",
# "300 EUR".
# - The number must not start inside a word, a number or a hyphenated reference, so
#   that "C-12" or "F-2026-0412" never gives the first digits of an amount.
# - Integer part: groups of three digits separated by a dot, an apostrophe or any
#   whitespace (plain, no-break or thin space, tab, line break), or plain digits. Being
#   generous with separators errs towards the larger amount: "1  250 €" must be read as
#   1250 €, which a person handles, never as 250 €, which a rule would automate.
# - Decimals: comma or dot followed by one or two digits ("1.250" is one thousand two
#   hundred and fifty, "1.25" is one and a quarter).
_AMOUNT = re.compile(
    r"(?<![\w.,-])"
    r"(?P<integer>\d{1,3}(?:(?:[\s'’]+|\.)\d{3})+|\d+)"
    r"(?:[.,](?P<decimals>\d{1,2}))?"
    r"\s*(?:€|euros?\b|eur\b)",
    re.IGNORECASE,
)
_NOT_A_DIGIT = re.compile(r"\D")


def extract_client_id(text: str) -> str | None:
    """First client identifier of `text` ("C-12"), upper-cased; None when there is none."""
    match = _CLIENT_ID.search(text)
    return match.group(0).upper() if match else None


def extract_amounts(text: str) -> list[float]:
    """Every amount in euros written in `text`, in order of appearance.

    A number without a currency mark (a duration, a percentage, an invoice number) is
    not an amount. Duplicates are kept: the caller decides what several amounts mean.
    A quantity is not multiplied: "2 factures de 300 €" is 300 €.
    """
    amounts = []
    for match in _AMOUNT.finditer(text):
        integer = _NOT_A_DIGIT.sub("", match.group("integer"))
        decimals = match.group("decimals") or "0"
        amounts.append(float(f"{integer}.{decimals}"))
    return amounts


def format_amount(amount: float) -> str:
    """Amount for user-facing text, with a decimal comma: 1250.5 -> "1250,50 €"."""
    return f"{amount:.2f}".replace(".", ",") + " €"
