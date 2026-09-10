"""Bounded parser subprocess; LangChain loaders preserve source text and page boundaries."""

import json
import sys
import zipfile
from pathlib import Path


def clean_text(text):
    return text.replace("\x00", "").replace("\r\n", "\n").strip()


def load_documents(path, max_pages=1000, max_chars=2000000):
    from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
    from langchain_core.documents import Document

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported. Upload an unlocked copy.")
        if len(reader.pages) > max_pages:
            raise ValueError("This PDF exceeds the configured page limit.")
        loader = PyPDFLoader(str(path), mode="page")
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=False)
    elif suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(i.file_size for i in entries) > 50 * 1024**2:
                raise ValueError("The DOCX archive is too large after expansion.")
            if "word/document.xml" not in archive.namelist():
                raise ValueError("The uploaded file is not a valid DOCX document.")
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError("Supported formats are PDF, TXT, Markdown and DOCX.")
    documents, size, empty_pages = [], 0, 0
    for document in loader.lazy_load():
        text = clean_text(document.page_content)
        size += len(text)
        if size > max_chars:
            raise ValueError("Extracted document text exceeds the configured limit.")
        if not text:
            empty_pages += 1
            continue
        page = document.metadata.get("page", -1)
        documents.append(Document(page_content=text, metadata={"page": page}))
    if not documents:
        raise ValueError("No readable text was found. Scanned PDFs require OCR before upload.")
    # Avoid silently indexing only the text portion of a mixed scanned/text PDF.
    if suffix == ".pdf" and empty_pages:
        raise ValueError(
            "Some PDF pages contain no readable text. Run OCR or remove blank pages before upload."
        )
    return documents


def load_pdf(path):
    return load_documents(path)


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    # Limit parser virtual memory; a separate process is terminated on timeout.
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    try:
        docs = load_documents(sys.argv[1], int(sys.argv[3]), int(sys.argv[4]))
        result = {
            "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        }
    except ValueError as exc:
        result = {"error": str(exc)}
    except Exception:
        result = {
            "error": "The document could not be parsed. Check that it is a valid, readable file."
        }
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
