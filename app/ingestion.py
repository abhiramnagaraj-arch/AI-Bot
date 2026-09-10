import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Event, Thread

from langchain_core.documents import Document

from .logger import log_event


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
                    "app.pdf_loader",
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


class IngestionWorker:
    def __init__(self, services):
        self.services = services
        self.stop_event, self.wake = Event(), Event()
        self.thread = Thread(target=self.run, name="rag-ingestion", daemon=True)

    def start(self):
        self.services.store.recover()
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.wake.set()
        # The lifespan retains its process lock until worker shutdown completes.
        self.thread.join(timeout=self.services.settings.parser_timeout + 120)
        if self.thread.is_alive():
            raise RuntimeError("Ingestion worker did not stop; process must exit before restarting")

    def run_one(self):
        store = self.services.store
        job = store.next_job()
        if not job:
            return False
        try:
            documents = parse_document(job["path"], self.services.settings)
            index, splitter, _ = self.services.ai()
            if job["fingerprint"] != index.fingerprint:
                raise ValueError("Index settings changed. Upload this document again.")
            chunks = splitter.split_documents(documents)
            if not chunks or len(chunks) > self.services.settings.max_chunks:
                raise ValueError("The document produced too many chunks, or no indexable text.")
            # Serialize mutations with chat per owner: no deleted evidence can be persisted later.
            with self.services.owner_lock(job["owner"]):
                if store.job(job["owner"], job["id"])["status"] != "processing":
                    return True
                index.add(chunks, job)
                store.publish(job["id"], len(chunks), len(documents))
            log_event("ingestion_complete", chunks=len(chunks), status="ready")
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Indexing failed. Check model availability and retry the upload."
            )
            store.fail_job(job["id"], message)
            # Incomplete vectors are inaccessible, but reclaim them if the index is available.
            if self.services._ai is not None:
                try:
                    self.services._ai[0].delete_version(job["id"])
                except Exception:
                    pass
            log_event("ingestion_failed", code=type(exc).__name__)
        return True

    def cleanup(self):
        for job in self.services.store.cleanup_jobs():
            try:
                with self.services.owner_lock(job["owner"]):
                    index, _, _ = self.services.ai()
                    index.delete_version(job["id"], job["fingerprint"])
                    Path(job["path"]).unlink(missing_ok=True)
                    self.services.store.cleaned(job["id"])
            except Exception:
                log_event("cleanup_pending", code="cleanup_retry")
                break

    def run(self):
        try:
            self.services.ai()
        except Exception:
            log_event("model_initialization_failed", code="embedding_unavailable")
        last_maintenance = 0.0
        while not self.stop_event.is_set():
            try:
                self.cleanup()
                if self.run_one():
                    continue
                if time.monotonic() - last_maintenance > 3600:
                    self.services.store.maintenance()
                    last_maintenance = time.monotonic()
            except Exception:
                log_event("worker_error", code="worker_retry")
            self.wake.wait(2)
            self.wake.clear()
