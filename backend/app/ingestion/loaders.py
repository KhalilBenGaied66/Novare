"""Document loaders: turn a source file into pages of plain text plus metadata.

Supported formats: Markdown and plain text (optional front matter), PDF (text layer
only, no OCR: a scanned PDF yields no page) and .eml messages (subject and text/plain
body). Every loader failure surfaces as `OSError` or `ValueError` so that the ingestion
can skip the file and continue.
"""

import email
import email.policy
import json
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".eml")
MANIFEST_NAME = "manifest.json"

_METADATA_KEYS = ("titre", "type", "client_id")
_FIRST_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class LoadedDoc:
    doc: str  # file name
    title: str
    doc_type: str
    client_id: str | None  # set => the document is visible to that client only
    pages: list[Page]


def load_manifest(docs_dir: Path) -> dict[str, dict]:
    """Read the optional `manifest.json`: `{"<file name>": {"titre", "type", "client_id"}}`.

    An unreadable manifest raises `ValueError` instead of being ignored: it can restrict
    a document to one client, so dropping it silently could publish that document.
    """
    path = docs_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not all(isinstance(entry, dict) for entry in data.values()):
        raise ValueError(f"{MANIFEST_NAME} must map file names to metadata objects")
    return data


def load_document(path: Path, manifest: dict[str, dict] | None = None) -> LoadedDoc:
    """Load one file. Metadata found in the file is overridden by its manifest entry."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        meta, pages = _load_text(path)
    elif suffix == ".pdf":
        meta, pages = _load_pdf(path)
    elif suffix == ".eml":
        meta, pages = _load_eml(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix!r}")

    override = (manifest or {}).get(path.name, {})
    for key in _METADATA_KEYS:
        # Empty or null manifest values are ignored: a blank `client_id` must not turn
        # a client-scoped document into a public one.
        if override.get(key):
            meta[key] = override[key]

    return LoadedDoc(
        doc=path.name,
        title=str(meta.get("titre") or "").strip() or path.stem,
        doc_type=str(meta.get("type") or "").strip() or "document",
        # Upper-cased like the client id of a request, so that the access rule compares equals.
        client_id=str(meta.get("client_id") or "").strip().upper() or None,
        pages=pages,
    )


def _load_text(path: Path) -> tuple[dict, list[Page]]:
    # utf-8-sig is UTF-8 that also drops the byte order mark some Windows editors add;
    # with plain utf-8 the mark would hide the opening `---` of the front matter.
    raw = path.read_text(encoding="utf-8-sig")
    meta, body = _split_front_matter(raw)
    if not meta.get("titre"):
        heading = _FIRST_HEADING.search(body)
        if heading:
            meta["titre"] = heading.group(1)
    return meta, _single_page(body)


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Return (`key: value` pairs found between two `---` lines at the top, remaining body)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            break
    else:
        return {}, text  # no closing line: the first `---` is a horizontal rule, not metadata
    meta = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator:
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, "\n".join(lines[end + 1 :])


def _load_pdf(path: Path) -> tuple[dict, list[Page]]:
    # Imported here: pypdf takes about a second to import and most runs have no PDF to read.
    import pypdf

    try:
        reader = pypdf.PdfReader(str(path))
        texts = [page.extract_text() or "" for page in reader.pages]
        title = reader.metadata.title if reader.metadata else None
    except Exception as exc:  # parsing boundary
        # pypdf also raises outside its own exception hierarchy (an AES-encrypted file
        # needs a package that is not installed): any failure means "unreadable", and
        # the ingestion skips the file instead of stopping.
        raise ValueError(f"unreadable PDF: {type(exc).__name__}") from exc
    pages = [
        Page(number, text.strip()) for number, text in enumerate(texts, start=1) if text.strip()
    ]
    return {"titre": title}, pages


def _load_eml(path: Path) -> tuple[dict, list[Page]]:
    with path.open("rb") as handle:
        message = email.message_from_binary_file(handle, policy=email.policy.default)
    subject = str(message["Subject"] or "").strip()
    part = message.get_body(preferencelist=("plain",))
    try:
        body = part.get_content() if part is not None else ""
    except LookupError as exc:  # the message declares a charset Python does not know
        raise ValueError("unreadable e-mail body: unknown charset") from exc
    # Only the subject and the body are kept: sender and recipient addresses are personal
    # data and bring nothing to the search.
    parts = [f"Objet : {subject}"] if subject else []
    parts.append("\n".join(body.splitlines()))  # splitlines also normalises CRLF endings
    return {"titre": subject, "type": "email"}, _single_page("\n\n".join(parts))


def _single_page(text: str) -> list[Page]:
    text = text.strip()
    return [Page(1, text)] if text else []
