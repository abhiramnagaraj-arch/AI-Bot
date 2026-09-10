from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.vector_store import VectorIndex


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0] if "leave" in text.lower() else [0.0, 1.0, 0.0]


def job(vid="version1", owner="alice"):
    return {"id": vid, "owner": owner, "document_id": "doc1", "filename": "policy.txt"}


def test_real_chroma_scope_idempotency_and_deletion(tmp_path):
    index = VectorIndex(Settings(_env_file=None, data_dir=tmp_path), TinyEmbeddings())
    doc = Document(page_content="Annual leave is 20 days", metadata={"page": 0})
    index.add([doc], job())
    index.add([doc], job())
    assert len(index.store.get()["ids"]) == 1
    assert (
        index.search("annual leave", "alice", ["version1"])[0][0].page_content == doc.page_content
    )
    assert index.search("annual leave", "bob", ["version1"]) == []
    assert index.search("annual leave", "alice", ["staged-version"]) == []
    index.delete_version("version1")
    assert index.store.get()["ids"] == []


def test_cleanup_can_remove_previous_fingerprint_collection(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    previous = VectorIndex(settings, TinyEmbeddings())
    previous.add([Document(page_content="Annual leave policy")], job())
    updated = VectorIndex(settings.model_copy(update={"chunk_tokens": 180}), TinyEmbeddings())
    updated.delete_version("version1", previous.fingerprint)
    assert previous.store.get()["ids"] == []


def test_cosine_score_bound_prevents_document_dump_warnings():
    from app.vector_store import cosine_relevance

    assert cosine_relevance(1.8) == 0.0
    assert cosine_relevance(-0.000001) == 1.0
    assert cosine_relevance(0.25) == 0.75
