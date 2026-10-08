"""Split a document into chunks: one per clause, or fixed-size pieces when it has no clauses."""

import re

from app.ingest.loaders import LoadedDoc
from app.schemas import Chunk

# A clause paragraph starts with its id in brackets, e.g. "[TEP-4.2.2] Expenses above ..."
CLAUSE_RE = re.compile(r"^\[([A-Z][A-Z0-9]*-\d+(?:\.\d+)*)\]\s*(.*)$")
HEADING_RE = re.compile(r"^#{2,6}\s+(.*)$")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_document(doc: LoadedDoc) -> list[Chunk]:
    return _clause_chunks(doc) or _fixed_chunks(doc)


def _clause_chunks(doc: LoadedDoc) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    for block in re.split(r"\n\s*\n", doc.text):
        block = " ".join(line.strip() for line in block.strip().splitlines())
        if not block:
            continue
        heading = HEADING_RE.match(block)
        if heading:
            section = heading.group(1).strip()
            continue
        clause = CLAUSE_RE.match(block)
        if clause:
            chunks.append(_make_chunk(doc, clause.group(1), section, clause.group(2)))
        elif chunks:
            # A paragraph without an id continues the clause above it.
            chunks[-1].text += " " + block
    return chunks


def _fixed_chunks(doc: LoadedDoc) -> list[Chunk]:
    text = " ".join(doc.text.split())
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE]
        chunk_id = f"{doc.meta['doc_id']}-c{len(chunks) + 1:03d}"
        chunks.append(_make_chunk(doc, chunk_id, "", piece))
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _make_chunk(doc: LoadedDoc, chunk_id: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.meta["doc_id"],
        doc_title=doc.meta["title"],
        source=doc.source,
        section=section,
        text=text.strip(),
        version=doc.meta.get("version", ""),
        effective_date=doc.meta.get("effective_date", ""),
    )
