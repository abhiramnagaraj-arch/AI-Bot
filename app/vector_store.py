import hashlib
import json
from threading import RLock

from langchain_chroma import Chroma


def cosine_relevance(distance):
    # Cosine similarity can be negative; clamp irrelevant results to zero.
    # This also prevents LangChain's out-of-range warning from logging Documents.
    return max(0.0, min(1.0, 1.0 - float(distance)))


class VectorIndex:
    def __init__(self, settings, embeddings):
        self.settings = settings
        self.lock = RLock()
        fingerprint = json.dumps(
            {
                "schema": 2,
                "loader": "preserve-v1",
                "embedding": settings.embedding_model,
                "revision": settings.embedding_revision,
                "chunk": settings.chunk_tokens,
                "overlap": settings.chunk_overlap,
            },
            sort_keys=True,
        )
        self.fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
        self.store = Chroma(
            collection_name=f"rag_v2_{self.fingerprint}",
            embedding_function=embeddings,
            persist_directory=str(settings.data_dir / "chroma"),
            collection_metadata={"hnsw:space": "cosine"},
            relevance_score_fn=cosine_relevance,
        )

    def add(self, documents, job):
        self.delete_version(job["id"])
        for start in range(0, len(documents), 64):
            batch = documents[start : start + 64]
            ids = [f"{job['id']}:{start + i}" for i in range(len(batch))]
            for i, doc in enumerate(batch):
                doc.metadata = {
                    "owner": job["owner"],
                    "document_id": job["document_id"],
                    "version_id": job["id"],
                    "filename": job["filename"],
                    "page": int(doc.metadata.get("page", -1)),
                    "chunk_index": start + i,
                    "start_index": int(doc.metadata.get("start_index", -1)),
                }
            with self.lock:
                self.store.add_documents(batch, ids=ids)

    def delete_version(self, version_id, fingerprint=None):
        with self.lock:
            target = self.store
            if fingerprint and fingerprint != self.fingerprint:
                from chromadb.errors import NotFoundError

                try:
                    target = Chroma(
                        collection_name=f"rag_v2_{fingerprint}",
                        client=self.store._client,
                        create_collection_if_not_exists=False,
                    )
                except NotFoundError:
                    return
            target.delete(where={"version_id": version_id})

    def search(self, query, owner, versions):
        if not versions:
            return []
        scope = {"$and": [{"owner": owner}, {"version_id": {"$in": versions}}]}
        with self.lock:
            return self.store.similarity_search_with_relevance_scores(
                query,
                k=self.settings.fetch_k,
                filter=scope,
                score_threshold=self.settings.relevance_threshold,
            )

    def ready(self):
        with self.lock:
            self.store.get(limit=1)

    def overview(self, owner, version_id, samples=24):
        from langchain_core.documents import Document

        with self.lock:
            result = self.store.get(
                where={"$and": [{"owner": owner}, {"version_id": version_id}]},
                include=["documents", "metadatas"],
            )
        values = sorted(
            zip(result["documents"], result["metadatas"]), key=lambda item: item[1]["chunk_index"]
        )
        total = len(values)
        positions = (
            sorted({round(i * (total - 1) / (samples - 1)) for i in range(min(total, samples))})
            if total > samples
            else range(total)
        )
        return [
            Document(page_content=values[i][0], metadata=values[i][1]) for i in positions
        ], total
