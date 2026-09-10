"""Validated configuration; importing this module never creates runtime state."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    environment: Literal["development", "production"] = "development"
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "[::1]", "testserver"]
    data_dir: Path = PROJECT_ROOT / "runtime" / "langchain-v2"
    users_json: SecretStr = SecretStr("{}")
    ollama_base_url: str = "http://localhost:11434"
    model: str = "gemma2:2b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_device: str = "cpu"
    allow_model_downloads: bool = False
    chunk_tokens: int = Field(default=220, ge=32, le=4096)
    chunk_overlap: int = Field(default=35, ge=0)
    top_k: int = Field(default=6, ge=1, le=20)
    fetch_k: int = Field(default=18, ge=1, le=100)
    relevance_threshold: float = Field(default=0.25, ge=0, le=1)
    context_tokens: int = Field(default=8192, ge=2048, le=131072)
    output_tokens: int = Field(default=768, ge=128, le=4096)
    history_chars: int = Field(default=6000, ge=0, le=30000)
    evidence_chars: int = Field(default=10000, ge=500, le=60000)
    history_turns: int = Field(default=8, ge=1, le=50)
    max_query_chars: int = Field(default=2000, ge=10, le=8000)
    model_timeout: float = Field(default=90, ge=1, le=300)
    max_upload_mb: int = Field(default=25, ge=1, le=100)
    max_storage_mb: int = Field(default=500, ge=1)
    max_documents: int = Field(default=100, ge=1, le=1000)
    max_pages: int = Field(default=1000, ge=1)
    max_extracted_chars: int = Field(default=2000000, ge=1000)
    max_chunks: int = Field(default=20000, ge=10)
    parser_timeout: int = Field(default=45, ge=1, le=300)
    requests_per_minute: int = Field(default=60, ge=1)
    session_hours: int = Field(default=24, ge=1, le=720)
    retention_days: int = Field(default=30, ge=1, le=365)
    redact_pii: bool = False

    @property
    def fingerprint(self):
        # Preserve compatibility with documents indexed by the previous implementation.
        values = {"schema": 2, "loader": "preserve-v1", "embedding": self.embedding_model,
                  "revision": self.embedding_revision, "chunk": self.chunk_tokens,
                  "overlap": self.chunk_overlap}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:24]

    @property
    def users(self) -> dict[str, str]:
        return json.loads(self.users_json.get_secret_value())

    @model_validator(mode="after")
    def validate_settings(self):
        try:
            users = self.users
            assert isinstance(users, dict)
            assert all(
                isinstance(k, str) and 1 <= len(k) <= 100 and isinstance(v, str) and len(v) >= 16
                for k, v in users.items()
            )
        except (ValueError, AssertionError, TypeError):
            raise ValueError(
                "RAG_USERS_JSON must map usernames to access keys of at least 16 characters"
            ) from None
        if self.environment == "production" and not users:
            raise ValueError("Production requires RAG_USERS_JSON with provisioned users")
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        if self.fetch_k < self.top_k:
            raise ValueError("Fetch count must be at least top_k")
        if self.output_tokens + 1024 >= self.context_tokens:
            raise ValueError("Context window must leave room for the prompt and evidence")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must use HTTP or HTTPS")
        return self
