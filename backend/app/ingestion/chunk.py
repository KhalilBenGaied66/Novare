"""Split a loaded document into chunks that respect its structure.

Rules, in order of priority:
1. a chunk never exceeds `size` characters;
2. a Markdown heading starts a new chunk and names its `section`; it stays with the
   text that follows it rather than alone in a chunk;
3. a paragraph (consecutive lines: a table, a list, a block of text) stays whole when it
   fits in a chunk; otherwise it is divided between lines, never inside a line, so a
   table is only ever cut between two rows;
4. a single line longer than `size` is cut at sentence ends, then between words;
5. when a section spans several chunks, the last lines of a chunk (up to `overlap`
   characters) are repeated at the start of the next one.

Known limit: when a large table is divided, its header row is not repeated in the
following chunks.
"""

import re
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.types import Chunk
from app.ingestion.loaders import LoadedDoc

_HEADING = re.compile(r"^#{1,6}\s+(.*\S)")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class _Unit:
    """Lines that must stay in the same chunk."""

    lines: list[str]
    heading: str | None = None  # heading title when the unit is a Markdown heading line
    gap: bool = True  # separated from the previous unit by a blank line


def chunk_document(
    doc: LoadedDoc, size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    settings = get_settings()
    size = settings.chunk_size if size is None else size
    overlap = settings.chunk_overlap if overlap is None else overlap
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk size must be positive and overlap must be in [0, size)")

    chunks = []
    section = ""  # kept from one page to the next: a PDF section can span several pages
    for page in doc.pages:
        packer = _PagePacker(size, overlap, section)
        for unit in _units(page.text, size):
            packer.add(unit)
        for number, (chunk_section, text) in enumerate(packer.finish(), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc}#p{page.number}-{number}",
                    doc=doc.doc,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    page=page.number,
                    section=chunk_section,
                    text=text,
                    client_id=doc.client_id,
                )
            )
        section = packer.section
    return chunks


def _units(text: str, size: int) -> Iterator[_Unit]:
    """Yield the headings and paragraphs of a page, in order."""
    paragraph: list[str] = []
    for raw_line in [*text.splitlines(), ""]:  # the final blank line closes the last paragraph
        line = raw_line.rstrip()
        heading = _HEADING.match(line)
        if line and not heading:
            paragraph.append(line)
            continue
        yield from _paragraph_units(paragraph, size)
        paragraph = []
        if heading:
            yield _Unit([line], heading=heading.group(1))


def _paragraph_units(paragraph: list[str], size: int) -> Iterator[_Unit]:
    if not paragraph:
        return
    if len("\n".join(paragraph)) <= size:
        yield _Unit(paragraph)
        return
    # Too large for one chunk: each line becomes its own unit, so cuts fall between lines.
    gap = True
    for line in paragraph:
        for piece in _split_long_line(line, size):
            yield _Unit([piece], gap=gap)
            gap = False


def _split_long_line(line: str, size: int) -> list[str]:
    """Return the line unchanged when it fits, else pieces of at most `size` characters."""
    if len(line) <= size:
        return [line]
    pieces = []
    for sentence in _SENTENCE_END.split(line):
        if len(sentence) <= size:
            pieces.append(sentence)
        else:
            # A sentence longer than a chunk: cut between words (and inside a word only
            # when the word itself is longer than a chunk, e.g. an encoded blob).
            pieces.extend(textwrap.wrap(sentence, width=size, break_on_hyphens=False))
    return pieces


class _PagePacker:
    """Packs the units of one page into chunks of at most `size` characters."""

    def __init__(self, size: int, overlap: int, section: str) -> None:
        self.size = size
        self.overlap = overlap
        self.section = section
        self._lines: list[str] = []  # lines of the chunk being built ("" = blank line)
        self._has_body = False  # the chunk being built holds more than headings
        self._done: list[tuple[str, str]] = []  # (section, text) of the finished chunks

    def add(self, unit: _Unit) -> None:
        if unit.heading is not None:
            if self._has_body:
                # No overlap across sections: it would file text under the wrong heading.
                self._close()
                self._lines = []
            # A heading without text of its own (a title directly followed by a
            # sub-heading) stays with the chunk of the next heading.
            self.section = unit.heading
        elif self._lines and self._length_with(self._lines, unit) > self.size:
            if not self._has_body and len(unit.lines) > 1:
                # Only headings so far: closing now would leave a chunk made of titles
                # alone. The paragraph is added line by line instead, so that its first
                # lines stay with their heading.
                for n, line in enumerate(unit.lines):
                    self.add(_Unit([line], gap=n == 0))
                return
            self._close()
            self._lines = self._overlap_before(unit)
        self._append(unit)

    def finish(self) -> list[tuple[str, str]]:
        if self._lines:
            self._close()
            self._lines = []
        return self._done

    def _append(self, unit: _Unit) -> None:
        if unit.gap and self._lines:
            self._lines.append("")
        self._lines.extend(unit.lines)
        self._has_body = self._has_body or unit.heading is None

    def _close(self) -> None:
        self._done.append((self.section, "\n".join(self._lines)))
        self._has_body = False

    def _length_with(self, lines: list[str], unit: _Unit) -> int:
        """Length of the chunk made of `lines` followed by `unit`."""
        separator = [""] if unit.gap and lines else []
        return len("\n".join([*lines, *separator, *unit.lines]))

    def _overlap_before(self, unit: _Unit) -> list[str]:
        """Last lines of the chunk just closed, to repeat before `unit` in the next chunk."""
        tail: list[str] = []
        for line in reversed(self._lines):
            if len("\n".join([line, *tail])) > self.overlap:
                break
            tail.insert(0, line)
        # Overlap is a bonus: it gives way, oldest line first, rather than push the next
        # chunk above `size`.
        while tail and (not tail[0] or self._length_with(tail, unit) > self.size):
            tail.pop(0)
        return tail
