"""LangChain document loading, token splitting and persistent Chroma indexing."""

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

from langchain_core.documents import Document

from .guardrails import AppError


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



def parse_document(path, settings):
    with tempfile.TemporaryDirectory(prefix="rag-parser-") as directory:
        output = Path(directory) / "parsed.json"
        # Parser stdout/stderr can contain document text: never pass through to logs.
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.documents",
                    str(path),
                    str(output),
                    str(settings.max_pages),
                    str(settings.max_extracted_chars),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=settings.parser_timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise ValueError(
                "Document parsing exceeded the time limit. Split the document into smaller files."
            ) from None
        if completed.returncode or not output.exists():
            raise ValueError(
                "Document parsing exceeded a resource limit or the parser could not start."
            )
        result = json.loads(output.read_text(encoding="utf-8"))
        if "error" in result:
            raise ValueError(result["error"])
        return [Document(**d) for d in result["documents"]]



def create_embeddings(settings):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": settings.embedding_device,
            "revision": settings.embedding_revision,
            "local_files_only": not settings.allow_model_downloads,
        },
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        query_encode_kwargs={"normalize_embeddings": True},
    )


def cosine_relevance(distance):
    # Bound cosine relevance so LangChain warnings never dump source text.
    return max(0.0, min(1.0, 1.0 - float(distance)))


class Documents:
    def __init__(self, settings, catalog, embeddings=None, splitter=None):
        from langchain_chroma import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.settings, self.catalog = settings, catalog
        self.fingerprint = settings.fingerprint
        embeddings = embeddings or create_embeddings(settings)
        if splitter is None:
            model = embeddings._client
            limit = min(settings.chunk_tokens,
                        model.max_seq_length - model.tokenizer.num_special_tokens_to_add())
            if limit <= settings.chunk_overlap:
                raise ValueError("Embedding token limit is smaller than configured overlap")
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                model.tokenizer, chunk_size=limit, chunk_overlap=settings.chunk_overlap,
                add_start_index=True)
        self.splitter = splitter
        self.index = Chroma(
            collection_name=f"rag_v2_{self.fingerprint}", embedding_function=embeddings,
            persist_directory=str(settings.data_dir / "chroma"),
            collection_metadata={"hnsw:space": "cosine"}, relevance_score_fn=cosine_relevance)

    def ingest(self, owner, filename, stream):
        filename = Path((filename or "").replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if not filename or len(filename) > 180 or any(ord(c) < 32 for c in filename):
            raise AppError(422, "invalid_filename", "Choose a file with a valid filename.")
        if suffix not in {".pdf", ".txt", ".md", ".docx"}:
            raise AppError(415, "unsupported_type", "Supported formats: PDF, TXT, Markdown and DOCX.")
        path = self.catalog.root / "uploads" / (uuid.uuid4().hex + suffix)
        published = False
        try:
            size, digest, head = 0, hashlib.sha256(), b""
            with path.open("xb") as target:
                while block := stream.read(1024 * 1024):
                    size += len(block)
                    if size > self.settings.max_upload_mb * 1024**2:
                        raise AppError(413, "upload_too_large", "The file exceeds the upload size limit.")
                    head = head or block[:1024]
                    if suffix in {".txt", ".md"} and b"\x00" in block:
                        raise AppError(415, "invalid_file", "Text files must be UTF-8 without binary content.")
                    digest.update(block)
                    target.write(block)
            if not size:
                raise AppError(422, "empty_file", "The uploaded file is empty.")
            if (suffix == ".pdf" and b"%PDF-" not in head) or (suffix == ".docx" and not head.startswith(b"PK")):
                raise AppError(415, "invalid_file", "The file signature does not match its format.")
            doc_id, created = self.catalog.add_document(owner, filename, path, digest.hexdigest(), self.fingerprint)
            if created:
                pages = parse_document(path, self.settings)
                chunks = self.splitter.split_documents(pages)
                if not chunks or len(chunks) > self.settings.max_chunks:
                    raise ValueError("The document produced too many chunks, or no indexable text.")
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update(owner=owner, document_id=doc_id, version_id=doc_id,
                                          filename=filename, chunk_index=i)
                for start in range(0, len(chunks), 64):
                    batch = chunks[start:start + 64]
                    self.index.add_documents(batch, ids=[f"{doc_id}:{start+i}" for i in range(len(batch))])
                self.catalog.publish(doc_id, len(chunks), len(pages))
                published = True
            return {"document_id": doc_id, "duplicate": not created, "status": "ready"}
        except ValueError as exc:
            raise AppError(422, "invalid_document", str(exc)) from None
        finally:
            if not published:
                path.unlink(missing_ok=True)
                self.cleanup()

    def search(self, query, owner, versions):
        if not versions:
            return []
        return self.index.similarity_search_with_relevance_scores(
            query, k=self.settings.fetch_k, score_threshold=self.settings.relevance_threshold,
            filter={"$and": [{"owner": owner}, {"version_id": {"$in": versions}}]})

    def cleanup(self):
        from chromadb.errors import NotFoundError
        from langchain_chroma import Chroma

        for doc in self.catalog.pending():
            try:
                try:
                    target = self.index if doc["fingerprint"] == self.fingerprint else Chroma(
                        collection_name=f"rag_v2_{doc['fingerprint']}", client=self.index._client,
                        create_collection_if_not_exists=False)
                    target.delete(where={"version_id": doc["active_version"]})
                except NotFoundError:
                    pass
                Path(doc["path"]).unlink(missing_ok=True)
                self.catalog.forget(doc["id"])
            except Exception:
                logging.getLogger("rag").warning("Document cleanup pending; retry on next operation or restart")
                return False
        return True

    def recover(self):
        if not self.cleanup():
            raise RuntimeError("Document cleanup could not finish")
        with self.catalog.connect() as db:
            paths = {r[0] for r in db.execute("SELECT path FROM files")}
        for path in (self.catalog.root / "uploads").iterdir():
            if path.is_file() and str(path) not in paths:
                path.unlink()
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
