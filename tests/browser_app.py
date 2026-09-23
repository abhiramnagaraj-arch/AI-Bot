"""Browser-test server: real API/storage/parser with deterministic model and retrieval."""

import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.documents import Documents
from app.rag import RagService
from app.server import create_app
from app.storage import Store
from tests.conftest import KEYS, FakeModel, TinyEmbeddings

settings = Settings(
    _env_file=None, data_dir=Path(os.environ["BROWSER_TEST_DATA"]), users_json=json.dumps(KEYS)
)
store = Store(settings)
documents = Documents(
    settings,
    store,
    TinyEmbeddings(),
    RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=30),
)
app = create_app(settings, store, documents, RagService(settings, store, documents, FakeModel()))
