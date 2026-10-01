"""Masking of personal identifiers before a request is stored, logged or sent to an LLM.

Regex-based, with check digits where the format has them (IBAN, card, SIRET) so that
ordinary numbers of a request (amounts, dates, ticket ids, postal codes) are left alone.

Out of scope: names and postal addresses cannot be recognised reliably with regular
expressions (that needs a named-entity model), and only French phone numbers are
handled.

Known limit: Luhn accepts one random number in ten, so a sequence of 13 to 19 digits
that is not a card (a long reference, a list of numbers separated by single spaces)
is sometimes masked by mistake. Masking too much was preferred to leaking a card number.
"""

import re
from collections.abc import Callable

# Space, no-break space and narrow no-break space: French typography and text pasted
# from e-mails use all three inside numbers.
_SPACES = " \u00a0\u202f"

_EMAIL = re.compile(r"[\w.+%-]+@[\w-]+(?:\.[\w-]+)+")

# National (0X XX XX XX XX) or international (+33 / 0033, optional "(0)") prefix, then
# four pairs of digits. The separator between pairs must be the same throughout, which
# keeps "01.02.2026 10" (a date followed by a number) from looking like a phone number.
# First digit 1-7 or 9: 08 numbers are service numbers, not personal data.
_TEL = re.compile(
    rf"(?<![\w+])"
    rf"(?:(?:\+|00[{_SPACES}]?)33[{_SPACES}.-]?(?:\(0\)|0)?[{_SPACES}.-]?|0)"
    rf"[1-79]"
    rf"(?P<sep>[{_SPACES}.-]?)[0-9]{{2}}(?:(?P=sep)[0-9]{{2}}){{3}}"
    rf"(?![0-9])"
)

# Sex (1/2), year, month, department (2A/2B for Corsica), commune, order number, key.
# The key is not verified: a mistyped social security number is still personal data.
_NIR = re.compile(
    rf"(?<!\w)[12][{_SPACES}]?[0-9]{{2}}[{_SPACES}]?(?:0[1-9]|1[0-2])[{_SPACES}]?"
    rf"(?:[0-9]{{2}}|2[ABab])[{_SPACES}]?[0-9]{{3}}[{_SPACES}]?[0-9]{{3}}[{_SPACES}]?[0-9]{{2}}"
    rf"(?![0-9])"
)

# Candidates only: both patterns are greedy and may run into what follows the number
# ("... 189 merci", "... 00074 12 rue"); `_mask_checked` trims them with the check digits.
_IBAN = re.compile(rf"(?<!\w)[A-Za-z]{{2}}[0-9]{{2}}(?:[{_SPACES}]?[A-Za-z0-9]){{11,30}}(?!\w)")
_DIGIT_RUN = re.compile(rf"(?<!\w)[0-9](?:[{_SPACES}-]?[0-9]){{12,18}}(?![0-9])")
_SEPARATOR = re.compile(rf"[{_SPACES}-]")


def redact(text: str) -> tuple[str, list[str]]:
    """Return `text` with personal identifiers replaced by tags, and the types found.

    Tags: [EMAIL], [TEL], [IBAN], [CARTE], [NIR], [SIRET]. The list of types is sorted
    and without duplicates. Names and postal addresses are not masked (see module doc).
    """
    found: set[str] = set()
    # Order matters: the most specific formats first, so that the digits of an IBAN or
    # of a social security number are never mistaken for a phone or a card number.
    text = _mask(text, _EMAIL, "EMAIL", found)
    text = _mask_checked(text, _IBAN, _iban_kind, found)
    text = _mask(text, _NIR, "NIR", found)
    text = _mask(text, _TEL, "TEL", found)
    text = _mask_checked(text, _DIGIT_RUN, _number_kind, found)
    return text, sorted(found)


def _mask(text: str, pattern: re.Pattern[str], kind: str, found: set[str]) -> str:
    masked, count = pattern.subn(f"[{kind}]", text)
    if count:
        found.add(kind)
    return masked


def _mask_checked(
    text: str, pattern: re.Pattern[str], kind_of: Callable[[str], str | None], found: set[str]
) -> str:
    """Mask the candidates of `pattern` that `kind_of` accepts (check digits).

    A candidate may include unrelated groups after the number, so it is shortened one
    group at a time until it is accepted. When nothing is accepted, the scan resumes
    after the first group: the number may start later in the same run.
    """
    out: list[str] = []
    pos = 0
    while (match := pattern.search(text, pos)) is not None:
        candidate = match.group(0)
        accepted = _longest_valid_prefix(candidate, kind_of)
        if accepted is None:
            first_group = _SEPARATOR.split(candidate, maxsplit=1)[0]
            end = match.start() + len(first_group)
            out.append(text[pos:end])
        else:
            length, kind = accepted
            end = match.start() + length
            out.append(text[pos : match.start()])
            out.append(f"[{kind}]")
            found.add(kind)
        pos = end
    out.append(text[pos:])
    return "".join(out)


_TAG = re.compile(r"\[(?:EMAIL|TEL|IBAN|CARTE|NIR|SIRET)\]")


def strip_tags(text: str) -> str:
    """Remove the masking tags from a masked text.

    Used for document search only: "[TEL]" in a question is not something to look up,
    and counting it as a search term would lower the retrieval confidence.
    """
    return re.sub(r"\s+", " ", _TAG.sub(" ", text)).strip()


def _longest_valid_prefix(
    candidate: str, kind_of: Callable[[str], str | None]
) -> tuple[int, str] | None:
    """Longest prefix of `candidate` ending at a group boundary that `kind_of` accepts."""
    boundaries = [m.start() for m in _SEPARATOR.finditer(candidate)]
    for end in [len(candidate), *reversed(boundaries)]:
        kind = kind_of(_SEPARATOR.sub("", candidate[:end]))
        if kind is not None:
            return end, kind
    return None


def _iban_kind(compact: str) -> str | None:
    """Return "IBAN" when `compact` (no spaces) has a valid ISO 13616 mod-97 checksum."""
    if not 15 <= len(compact) <= 34:
        return None
    # An account number is mostly digits. Without this guard an equipment reference
    # followed by words ("DT25 en panne depuis hier") is a candidate and would pass the
    # checksum by chance about once in 97.
    if sum(ch.isdigit() for ch in compact) * 2 <= len(compact):
        return None
    rearranged = compact[4:] + compact[:4]
    # Letters count as A=10 ... Z=35, which is exactly their value in base 36.
    number = int("".join(str(int(ch, 36)) for ch in rearranged))
    return "IBAN" if number % 97 == 1 else None


def _number_kind(digits: str) -> str | None:
    """Return "SIRET" (14 digits) or "CARTE" (13 to 19 digits) when Luhn passes, else None.

    A 14-digit number that passes Luhn is labelled SIRET: in this context it is far more
    likely than a 14-digit payment card. Either way the number is masked.
    """
    if not 13 <= len(digits) <= 19 or not _luhn_ok(digits):
        return None
    return "SIRET" if len(digits) == 14 else "CARTE"


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:  # every second digit from the right is doubled
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
