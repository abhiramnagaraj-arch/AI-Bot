import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.documents import Documents
from app.guardrails import AnswerDraft, Rewrite
from app.rag import RagService
from app.server import create_app
from app.storage import Store

KEYS = {"alice": "alice-access-key-0123456789", "bob": "bob-access-key-0123456789"}


class FakeModel:
    def __init__(self):
        self.calls = []
        self.answer = None
        self.error = None

    def with_structured_output(self, schema):
        def invoke(prompt):
            self.calls.append((schema.__name__, prompt))
            if self.error:
                raise self.error
            if schema is Rewrite:
                return Rewrite(question=str(prompt.messages[-1].content))
            if self.answer:
                return self.answer
            return AnswerDraft(
                status="answered",
                answer="Employees receive 20 days of annual leave. [S1]",
                citations=["S1"],
            )

        return RunnableLambda(invoke)


class TinyEmbeddings(Embeddings):
    """Stable embeddings for integration tests, without downloads or a model server."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def harness(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, users_json=json.dumps(KEYS))
    store = Store(settings)
    model = FakeModel()
    documents = Documents(
        settings,
        store,
        TinyEmbeddings(),
        RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=30),
    )
    rag = RagService(settings, store, documents, model)
    app = create_app(settings, store, documents, rag)
    with TestClient(app, headers={"X-RAG-Request": "1"}) as client:
        yield client, app.state, model, documents


def login(client, username="alice"):
    response = client.post(
        "/api/v1/auth/login", json={"username": username, "access_key": KEYS[username]}
    )
    assert response.status_code == 200, response.text


def upload(
    client,
    services,
    content=b"Employees receive 20 days of annual leave. Submit requests to your manager.",
    name="policy.txt",
):
    response = client.post("/api/v1/documents", files={"file": (name, content)})
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["status"] == "ready"
    return data


def session(client):
    response = client.post("/api/v1/sessions", json={"title": "Test conversation"})
    assert response.status_code == 201, response.text
    return response.json()["id"]
