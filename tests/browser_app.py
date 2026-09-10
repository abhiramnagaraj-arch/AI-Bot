"""Browser-test server: real API/storage/parser with deterministic model and retrieval."""

import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.rag import RagService
from app.server import create_app
from app.services import Services
from tests.conftest import KEYS, FakeIndex, FakeModel

settings = Settings(
    _env_file=None, data_dir=Path(os.environ["BROWSER_TEST_DATA"]), users_json=json.dumps(KEYS)
)
services = Services(settings)
index = FakeIndex(services.fingerprint)
services._ai = (
    index,
    RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=30),
    RagService(settings, services.store, index, FakeModel()),
)
app = create_app(settings, services)
