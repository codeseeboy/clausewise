import pytest

from app.ingest.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document
from app.ingest.loaders import LoadedDoc, LoaderError, load_bytes, load_file
from app.ingest.pipeline import index_folder
from tests.conftest import CORPUS, FIXTURES, make_pdf


def test_front_matter_is_read():
    doc = load_file(CORPUS / "TEP_travel_expense.md")
    assert doc.meta["doc_id"] == "TEP"
    assert doc.meta["version"] == "2.1"
    assert not doc.text.lstrip().startswith("---")


def test_one_chunk_per_clause_with_section():
    chunks = chunk_document(load_file(CORPUS / "TEP_travel_expense.md"))
    by_id = {c.chunk_id: c for c in chunks}
    assert len(chunks) == 13
    assert by_id["TEP-4.2.2"].section == "4. Approval limits"
    assert by_id["TEP-4.2.2"].text.startswith("Expenses above INR 5,000")


def test_text_without_clauses_is_cut_into_fixed_pieces():
    text = " ".join(f"w{i}" for i in range(500))
    doc = LoadedDoc(source="notes.txt", text=text, meta={"doc_id": "NOTES", "title": "notes"})
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    assert chunks[1].text[: CHUNK_OVERLAP - 20] in chunks[0].text  # neighbouring pieces overlap
    assert chunks[0].chunk_id == "NOTES-c001"


def test_pdf_text_is_read():
    doc = load_bytes("memo.pdf", make_pdf("Gas readings are repeated every two hours"))
    assert "Gas readings" in doc.text
    assert doc.meta["doc_id"] == "MEMO"


@pytest.mark.parametrize(
    ("name", "data", "reason"),
    [
        ("corrupt.pdf", (FIXTURES / "corrupt.pdf").read_bytes(), "cannot read PDF"),
        ("empty.pdf", (FIXTURES / "empty.pdf").read_bytes(), "empty"),
        ("sheet.xlsx", b"PK\x03\x04", "unsupported"),
        ("notes.md", b"   \n  ", "empty"),
        ("latin.txt", "caf\xe9".encode("latin-1"), "not UTF-8"),
    ],
)
def test_bad_files_raise_loader_error(name, data, reason):
    with pytest.raises(LoaderError, match=reason):
        load_bytes(name, data)


def test_index_folder_skips_bad_files(tmp_path, fake_embedder, memory_store):
    (tmp_path / "good.md").write_text("## 1. Rules\n\n[GD-1.1] Helmets are required.\n", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes((FIXTURES / "corrupt.pdf").read_bytes())
    report = index_folder(tmp_path, fake_embedder, memory_store)
    assert report.files == 1
    assert report.chunks == 1
    assert "broken.pdf" in report.failed
