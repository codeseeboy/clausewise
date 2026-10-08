"""Read a document (Markdown, text or PDF) into plain text plus metadata."""

import io
from dataclasses import dataclass, field
from pathlib import Path

"""using it to handle the pdfs files """
from pypdf import PdfReader

SUPPORTED = {".md", ".txt", ".pdf"}


class LoaderError(ValueError):
    """The file cannot be read as a document."""


@dataclass
class LoadedDoc:
    source: str
    text: str
    meta: dict[str, str] = field(default_factory=dict)


def load_file(path: Path) -> LoadedDoc:
    return load_bytes(path.name, path.read_bytes())


def load_bytes(filename: str, data: bytes) -> LoadedDoc:
    """Parse file bytes. Raises LoaderError for unsupported, empty or broken files."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise LoaderError(f"{filename}: unsupported file type '{suffix or 'none'}'")
    if not data.strip():
        raise LoaderError(f"{filename}: file is empty")

    if suffix == ".pdf":
        meta, text = {}, _read_pdf(filename, data)
    else:
        try:
            raw = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LoaderError(f"{filename}: not UTF-8 text") from exc
        meta, text = _split_front_matter(raw)

    if not text.strip():
        raise LoaderError(f"{filename}: no text found")

    stem = Path(filename).stem
    meta.setdefault("doc_id", stem.split("_")[0].upper())
    meta.setdefault("title", stem)
    return LoadedDoc(source=filename, text=text, meta=meta)


def _read_pdf(filename: str, data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises many different types for broken files
        raise LoaderError(f"{filename}: cannot read PDF ({type(exc).__name__})") from exc
    return "\n\n".join(pages)


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split a leading '---' block of 'key: value' lines from the body."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            meta = {}
            for line in lines[1:end]:
                key, sep, value = line.partition(":")
                if sep:
                    meta[key.strip()] = value.strip()
            return meta, "\n".join(lines[end + 1 :])
    return {}, raw
