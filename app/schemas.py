from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(StrictModel):
    username: str = Field(min_length=1, max_length=100)
    access_key: str = Field(default="", max_length=512, repr=False)


class SessionRequest(StrictModel):
    title: str = Field(default="New conversation", min_length=1, max_length=100)


class ChatRequest(StrictModel):
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    query: str = Field(min_length=1, max_length=8000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a question")
        return value

    @field_validator("document_ids")
    @classmethod
    def unique_ids(cls, values):
        import re

        if any(not re.fullmatch(r"[a-f0-9]{32}", v) for v in values):
            raise ValueError("Invalid document ID")
        return list(dict.fromkeys(values))


class Source(StrictModel):
    citation_id: str
    document_id: str
    version_id: str
    filename: str
    page: int | None = None
    excerpt: str
    url: str


class AnswerDraft(StrictModel):
    status: Literal["answered", "insufficient_evidence", "clarification_needed"]
    answer: str = Field(min_length=1)
    citations: list[Annotated[str, Field(pattern=r"^S[1-9][0-9]*$")]] = Field(
        default_factory=list,
        max_length=20,
        description="Plain source IDs used in the answer, e.g. S1. Never filenames.",
    )

    @field_validator("answer")
    @classmethod
    def bounded_answer(cls, value):
        # Large JSON-schema maxLength values crash some Ollama grammar engines.
        # Enforce the bound locally, while num_predict limits generation itself.
        if not value.strip() or len(value) > 12000:
            raise ValueError("Answer length is invalid")
        return value.strip()


class Rewrite(StrictModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def bounded_question(cls, value):
        if not value.strip() or len(value) > 2000:
            raise ValueError("Rewritten question length is invalid")
        return value.strip()


class ChatResponse(StrictModel):
    request_id: str
    session_id: str
    status: Literal["answered", "insufficient_evidence", "clarification_needed", "blocked"]
    answer: str
    sources: list[Source] = Field(default_factory=list)
    time_ms: int


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)
