import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.rag import RagService
from app.schemas import AnswerDraft, Rewrite
from app.server import create_app
from app.services import Services

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


class FakeIndex:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.docs = []
        self.queries = []
        self.ignore_filter = False

    def add(self, documents, job):
        self.delete_version(job["id"])
        for document in documents:
            self.docs.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "owner": job["owner"],
                        "document_id": job["document_id"],
                        "version_id": job["id"],
                        "filename": job["filename"],
                    },
                )
            )

    def search(self, query, owner, versions):
        self.queries.append(query)
        return [
            (d, 0.9)
            for d in self.docs
            if self.ignore_filter
            or (d.metadata["owner"] == owner and d.metadata["version_id"] in versions)
        ]

    def delete_version(self, vid, fingerprint=None):
        self.docs = [d for d in self.docs if d.metadata["version_id"] != vid]

    def ready(self):
        return True

    def overview(self, owner, version_id, samples=24):
        docs = [
            d
            for d in self.docs
            if d.metadata["owner"] == owner and d.metadata["version_id"] == version_id
        ]
        return docs[:samples], len(docs)


@pytest.fixture
def harness(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, users_json=json.dumps(KEYS))
    services = Services(settings)
    model = FakeModel()
    index = FakeIndex(services.fingerprint)
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=30)
    services._ai = (index, splitter, RagService(settings, services.store, index, model))
    app = create_app(settings, services, start_worker=False)
    with TestClient(app, headers={"X-RAG-Request": "1"}) as client:
        yield client, services, model, index


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
    process=True,
):
    response = client.post("/api/v1/documents", files={"file": (name, content)})
    assert response.status_code == 202, response.text
    data = response.json()
    if process:
        assert services.worker.run_one()
        job = client.get(f"/api/v1/jobs/{data['job_id']}").json()
        assert job["status"] == "ready", job
    return data


def session(client):
    response = client.post("/api/v1/sessions", json={"title": "Test conversation"})
    assert response.status_code == 201, response.text
    return response.json()["id"]
