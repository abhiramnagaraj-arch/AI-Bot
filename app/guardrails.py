"""Small explicit policies used at runnable boundaries; no claim of universal detection."""

import re
import unicodedata
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


INJECTION = re.compile(
    r"(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|earlier|prior|system)\s+(?:instructions|directions|rules|prompts)"
    r"|(?:reveal|print|show|repeat|expose)\s+(?:the\s+|your\s+)?(?:hidden\s+)?(?:system|developer)\s+prompt"
    r"|(?:you are now|act as)\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken)",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL_REDACTED]", text)
    return re.sub(
        r"(?<!\w)(?:\+\d{1,3}[ -])?\d{3}[- ]\d{3}[- ]\d{4}(?!\w)", "[PHONE_REDACTED]", text
    )


def detect_injection(text: str) -> bool:
    return bool(INJECTION.search(unicodedata.normalize("NFKC", text)))


def validate_query(query: str, settings) -> str:
    if len(query) > settings.max_query_chars:
        raise AppError(
            422,
            "query_too_long",
            f"Questions must be at most {settings.max_query_chars} characters.",
        )
    if len(query.encode("utf-8")) > settings.context_tokens - settings.output_tokens - 3000:
        raise AppError(
            422,
            "query_too_long",
            "Shorten the question to leave room for document evidence in the model context.",
        )
    if any(ord(c) < 32 and c not in "\n\t\r" for c in query):
        raise AppError(
            422, "invalid_query", "The question contains unsupported control characters."
        )
    return redact_pii(query) if settings.redact_pii else query


def validate_citations(draft, sources):
    available = {s.citation_id for s in sources}
    inline = set(re.findall(r"\[(S\d+)\]", draft.answer))
    declared = set(draft.citations)
    if draft.status == "answered" and inline and not declared:
        # An inline reference is still an explicit model citation. Recover only
        # its exact ID and validate it against evidence below.
        declared = inline
        draft.citations = [s.citation_id for s in sources if s.citation_id in inline]
    if draft.status == "answered":
        if not declared or not declared <= available or (inline and inline != declared):
            raise AppError(
                502,
                "invalid_answer",
                "The model could not produce a verifiable answer. Try a more specific question.",
            )
        if not inline:
            # Some small models return valid structured citations but omit inline
            # markers. Render the validated references; never invent a source ID.
            draft.answer += " " + " ".join(
                f"[{s.citation_id}]" for s in sources if s.citation_id in declared
            )
    elif declared or inline:
        if not (declared | inline) <= available:
            raise AppError(
                502,
                "invalid_answer",
                "The model returned an unknown evidence reference. Please retry.",
            )
        # A refusal/clarification makes no supported factual answer. Small models
        # may still attach a valid passage marker; omit it from the abstention.
        draft.answer = re.sub(r"\[(S\d+)\]", "", draft.answer).strip()
        draft.citations = []
        declared = set()
    return [s for s in sources if s.citation_id in declared]
