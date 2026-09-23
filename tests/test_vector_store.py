import io

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.documents import Documents, cosine_relevance
from app.storage import Store
from tests.conftest import TinyEmbeddings


def test_real_chroma_scope_and_previous_index_cleanup(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    catalog = Store(settings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=30)
    previous = Documents(settings, catalog, TinyEmbeddings(), splitter)
    result = previous.ingest("alice", "policy.txt", io.BytesIO(b"Annual leave is 20 days"))
    doc = result["document_id"]
    assert len(previous.search("leave", "alice", [doc])) == 1
    assert previous.search("leave", "bob", [doc]) == []
    assert previous.search("leave", "alice", ["staged"]) == []
    catalog.delete_document("alice", doc)
    updated = Documents(
        settings.model_copy(update={"chunk_tokens": 180}), catalog, TinyEmbeddings(), splitter
    )
    assert updated.cleanup()
    assert previous.index.get()["ids"] == []


def test_cosine_score_bound_prevents_document_dump_warnings():
    assert cosine_relevance(1.8) == 0.0
    assert cosine_relevance(-0.000001) == 1.0
    assert cosine_relevance(0.25) == 0.75
