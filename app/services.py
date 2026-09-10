import hashlib
import json
from threading import BoundedSemaphore, Lock, RLock

from .config import Settings
from .ingestion import IngestionWorker
from .storage import Store


def index_fingerprint(settings):
    data = {
        "schema": 2,
        "loader": "preserve-v1",
        "embedding": settings.embedding_model,
        "revision": settings.embedding_revision,
        "chunk": settings.chunk_tokens,
        "overlap": settings.chunk_overlap,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


class Services:
    def __init__(self, settings: Settings):
        self.settings, self.store = settings, Store(settings)
        self._ai, self._ai_lock = None, Lock()
        self._owner_locks = [RLock() for _ in range(64)]
        self.chat_slots = BoundedSemaphore(settings.max_concurrent_chats)
        self.upload_slots = BoundedSemaphore(2)
        self.worker = IngestionWorker(self)
        self.fingerprint = index_fingerprint(settings)
        self._process_lock = None
        self.http_uploads = BoundedSemaphore(2)

    def owner_lock(self, owner):
        return self._owner_locks[int(hashlib.sha256(owner.encode()).hexdigest(), 16) % 64]

    def start(self):
        # Embedded Chroma + SQLite + this worker deliberately support exactly one app process.
        import fcntl

        self._process_lock = (self.store.root / "application.lock").open("a")
        try:
            fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._process_lock.close()
            raise RuntimeError(
                "Runtime directory is already in use. Run one application worker per data directory."
            ) from None
        self.worker.start()

    def close(self):
        self.worker.close()
        if self._process_lock:
            self._process_lock.close()

    def ai(self):
        with self._ai_lock:
            if self._ai is None:
                from .chunker import create_splitter
                from .embedder import create_embeddings
                from .llm import create_model
                from .rag import RagService
                from .vector_store import VectorIndex

                embeddings = create_embeddings(self.settings)
                splitter = create_splitter(self.settings, embeddings)
                index = VectorIndex(self.settings, embeddings)
                self._ai = (
                    index,
                    splitter,
                    RagService(self.settings, self.store, index, create_model(self.settings)),
                )
            return self._ai
