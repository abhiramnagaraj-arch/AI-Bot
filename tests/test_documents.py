import zipfile

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import Settings
from app.documents import clean_text, parse_document


def make_pdf(path, text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as target:
        writer.write(target)


def test_cleanup_preserves_tables_and_headings():
    text = "Chapter 2: Retention\nThe database uses a table of retention periods.\nSee figure 1."
    assert clean_text(text) == text


def test_pdf_page_metadata(tmp_path):
    path = tmp_path / "policy.pdf"
    make_pdf(path, "Employees receive 20 days of annual leave.")
    docs = parse_document(path, Settings(_env_file=None))
    assert "20 days" in docs[0].page_content
    assert docs[0].metadata["page"] == 0


def test_scanned_or_empty_pdf_is_not_success(tmp_path):
    path = tmp_path / "scan.pdf"
    make_pdf(path)
    with pytest.raises(ValueError, match="OCR"):
        parse_document(path, Settings(_env_file=None))


def test_docx_and_markdown(tmp_path):
    path = tmp_path / "policy.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Keep invoices for seven years.</w:t></w:r></w:p></w:body></w:document>',
        )
    docs = parse_document(path, Settings(_env_file=None))
    assert "seven years" in docs[0].page_content
    path = tmp_path / "guide.md"
    path.write_text("# Retention\n\n| Item | Years |\n| Invoice | 7 |\n", encoding="utf-8")
    docs = parse_document(path, Settings(_env_file=None))
    assert "# Retention" in docs[0].page_content
    assert "| Invoice | 7 |" in docs[0].page_content


def test_extraction_limit(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("x" * 2000)
    with pytest.raises(ValueError, match="limit"):
        parse_document(path, Settings(_env_file=None, max_extracted_chars=1000))


def test_production_configuration_requires_keys():
    with pytest.raises(ValueError, match="Production requires"):
        Settings(_env_file=None, environment="production")
    with pytest.raises(ValueError, match="overlap"):
        Settings(_env_file=None, chunk_tokens=100, chunk_overlap=100)


def test_backup_restore_round_trip(tmp_path):
    import hashlib

    from app.storage import Store
    from scripts.backup import backup, restore

    root = tmp_path / "original"
    store = Store(Settings(_env_file=None, data_dir=root))
    source = root / "uploads" / "policy.txt"
    source.write_text("Backup test")
    doc, _ = store.add_document(
        "alice", "policy.txt", source, hashlib.sha256(source.read_bytes()).hexdigest(), "test"
    )
    store.publish(doc, 1, 1)
    token = store.create_login("alice")
    archive = tmp_path / "snapshot.tar.gz"
    backup(root, archive)
    destination = tmp_path / "restored"
    restore(archive, destination)
    restored = Store(Settings(_env_file=None, data_dir=destination))
    path, name = restored.source("alice", doc)
    assert path.read_text() == "Backup test"
    assert path.is_relative_to(destination)
    assert restored.login_owner(token) is None
    with pytest.raises(ValueError, match="empty"):
        restore(archive, destination)


def test_old_catalog_migration_preserves_ready_sources_and_cleans_pending(tmp_path):
    import sqlite3

    from app.storage import Store

    source = tmp_path / "uploads" / "policy.txt"
    source.parent.mkdir()
    source.write_text("Existing private document")
    with sqlite3.connect(tmp_path / "metadata.sqlite3") as db:
        db.executescript("""
            CREATE TABLE documents (id TEXT PRIMARY KEY,active_version TEXT,deleted INTEGER);
            CREATE TABLE versions (id TEXT PRIMARY KEY,document_id TEXT,owner TEXT,filename TEXT,
                path TEXT,sha TEXT,size INTEGER,fingerprint TEXT,status TEXT,chunks INTEGER,
                pages INTEGER,created REAL,summary TEXT);
            INSERT INTO documents VALUES ('doc','version',0);
        """)
        db.execute(
            "INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "version",
                "doc",
                "alice",
                "policy.txt",
                str(source),
                "hash",
                25,
                "old-fingerprint",
                "ready",
                1,
                1,
                1.0,
                "discarded overview",
            ),
        )
    store = Store(Settings(_env_file=None, data_dir=tmp_path))
    assert store.source("alice", "doc", "version")[0].read_text() == "Existing private document"
    assert store.active("alice")[0]["fingerprint"] == "old-fingerprint"
    assert len(Store(store.settings).documents("alice")) == 1
    with store.connect() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='versions'").fetchone()


def test_parser_timeout_and_encrypted_pdf(tmp_path, monkeypatch):
    import subprocess

    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    writer.write(path)
    with pytest.raises(ValueError, match="Password-protected"):
        parse_document(path, Settings(_env_file=None))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("parser", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="time limit"):
        parse_document(path, Settings(_env_file=None))
