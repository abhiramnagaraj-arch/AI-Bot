# LangChain implementation and workflow

Folio separates the application boundary (FastAPI/authentication/storage) from the AI workflow (LangChain). It is a bounded retrieve-then-generate application, with an optional question-rewriting call for an existing conversation. It does not run an agent loop.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated environment settings; no import-time file writes or model downloads |
| `server.py` | App lifespan, HTTP routes, body limits, same-origin policy, safe errors and static UI |
| `auth.py` | Provisioned access-key login, opaque cookie sessions, rate limits, principal resolution |
| `storage.py` | SQLite document versions, jobs, sessions, messages, overview cache and retention |
| `services.py` | Dependency initialization, single-process lock, bounded chat/upload capacity and owner locks |
| `pdf_loader.py` | LangChain loader dispatch and resource-limited subprocess entry point |
| `ingestion.py` | Durable job execution, interrupted-job recovery and retryable physical deletion |
| `chunker.py` | LangChain tokenizer-aware splitting bounded by the embedding model's effective sequence length |
| `embedder.py` | HuggingFaceEmbeddings factory with normalized vectors and a pinned revision |
| `vector_store.py` | Chroma indexing, scoped search, version deletion and overview passage sampling |
| `retriever.py` | LangChain BaseRetriever with authorized version scope, deduplication and injection-pattern exclusion |
| `prompts.py` | Separate instructions for grounded answering and standalone-question rewriting |
| `rag.py` | LCEL history/rewrite/retrieve/generate/validate/persist workflow and cached overviews |
| `memory.py` | LangChain BaseChatMessageHistory on owner-checked SQLite; bounded paired history |
| `schemas.py` | API contracts, structured model responses and typed application errors |
| `guardrails.py` | Input limits, optional narrow PII redaction, injection signals and citation validation |
| `logger.py` | Allowlisted operational metadata; no raw prompts, documents, answers or credentials |
| `main.py` | Thin interactive API client, sharing the browser's backend behavior |

## Upload and indexing

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as FastAPI
    participant S as SQLite
    participant W as Ingestion worker
    participant L as LangChain loaders / splitter
    participant V as Chroma
    U->>A: Authenticated file upload
    A->>A: Bound body size, validate format, hash file
    A->>S: Record document/version/job as queued
    A-->>U: 202 with document and job IDs
    W->>S: Claim queued job as processing
    W->>L: Parse in limited subprocess and split text
    L-->>W: Documents with page metadata
    W->>V: Add normalized embeddings and versioned chunks
    W->>S: Publish active version as ready
    U->>A: Poll document status
    A-->>U: Ready / failed; previous active version if any
```

Files are stored under generated names outside the static directory. Original filenames are display metadata. Limits apply to upload bytes, file type/signature, pages, expanded DOCX archives, extracted text, chunk count, pending jobs, file storage, and document count. UTF-8 text is preserved rather than deleting lines that mention tables or figures.

The parser runs in a separate Linux process with CPU, address-space, output-file and elapsed-time limits. Scans require OCR. Layout-aware table extraction and OCR are future capabilities.

A job is persisted before work begins. On restart, `processing` jobs return to `queued`; unrecorded upload files are removed. Chunk IDs are stable within a version, and retrying a version first removes its partial chunks. Incomplete versions never become active retrieval scope.

SQL and Chroma do not share a transaction. Publication therefore follows staged indexing: write vectors first, then atomically change the active version in SQL. Failed replacements leave the previous active version available. Deletion revokes the metadata scope immediately and the worker retries file/vector cleanup. Old collection fingerprints are included in cleanup.

## Conversational RAG

```mermaid
flowchart LR
    Q[Question] --> Auth[User and source ownership]
    Auth --> History[LangChain message history]
    History --> Rewrite[Standalone question for retrieval]
    Rewrite --> Retrieval[Authorized active versions only]
    Retrieval --> Evidence[Deduplicate and fit evidence budget]
    Evidence --> Model[Grounded prompt and ChatOllama]
    Model --> Check[Structured response and citation checks]
    Check --> Save[Persist the successful turn]
    Save --> Reply[Answer, status and sources]
```

The chain is `RunnablePassthrough.assign(history=...) | prepare | retrieve | generate | save_history`. Stages use LangChain Documents, messages, prompt templates, a BaseRetriever, and ChatOllama structured output. SQLiteHistory implements the supported LangChain message-history interface. It avoids the deprecated history wrapper and keeps graph orchestration out of this branch.

New questions make one answer call. Existing conversations can add one rewrite call. There is no autonomous retrieval loop. Only connection-establishment failures on the answer chain retry, at most twice. Timeouts, malformed responses and invalid citations are explicit errors and do not enter chat history.

Generation uses the original question. The rewritten question is used for search only. History keeps complete user/assistant pairs with both a turn limit and a UTF-8 byte budget. Evidence and history share a conservative byte-based context allowance, reserving output and schema/chat overhead. This avoids an unrelated tokenizer download but can underuse the available context window.

Every retrieved passage is filtered by owner and authorized active version at the database boundary and checked again by the retriever. Selected source IDs are validated against the authenticated user. After generation, active versions are checked again. Owner-level locks serialize chat, publication and deletion to prevent a late answer from reintroducing removed evidence into history.

The model returns `status`, `answer`, and `citations`. Supplied source IDs are validated. Unknown IDs or conflicting inline/structured references fail closed. If a small model supplies valid structured citations but omits inline markers, the application renders those validated markers at the end. If only explicit inline IDs are present, the source list is recovered from those IDs. Known markers on an insufficient-evidence response are omitted; unknown IDs are always rejected. Source metadata and download links are generated by the application. Scores are never presented as confidence.

Large string limits are enforced by local Pydantic validators rather than transmitted as JSON-schema `maxLength`; live testing found that the installed Ollama grammar engine crashes for sufficiently large repetition bounds.

## Guardrails and their limits

- Authentication, ownership, selected-source validation, and filesystem separation enforce access control outside the model.
- Input/body/queue/concurrency/context limits constrain resource use.
- Retrieved evidence is placed in a human data message rather than concatenated into system instructions.
- Suspicious instruction patterns are a supplementary signal. They can miss paraphrased attacks or exclude benign security documentation; they are not a complete prompt-injection classifier.
- Prompt instructions restrict factual answers to retrieved evidence and distinguish missing evidence, ambiguity and conflicts.
- Output schema and source-ID checks prevent invented references, but do not establish semantic entailment for every sentence. Model quality needs corpus-specific evaluation and human review.
- Optional PII redaction covers email addresses and common formatted phone numbers only. It is not a general PII detector. Default operational logs never include message content, regardless of this setting.
- The application contains no shell, web browsing, database-query or action tools accessible to the model.

History and original documents still contain user data. They live in access-controlled persistent storage and require operator-managed disk encryption/backups where appropriate. Deletion/replacement clears all of that user's stored conversation messages. Information already downloaded or displayed cannot be recalled.

## Overviews

`POST /api/v1/documents/{id}/overview` obtains ordered passages, samples when necessary, generates a cited overview under the same context/output policy, and caches it for the active version and model/prompt configuration. The response includes `passages_used`, `total_passages` and `partial`. Partial coverage is prepended by application code, so a long-document overview is never silently presented as a complete summary.

This is a bounded overview, not an exhaustive map/reduce summarization of arbitrarily long documents. It deliberately avoids hundreds of uncontrolled model calls.

## HTTP contract

| Endpoint | Behavior |
| --- | --- |
| `POST /api/v1/auth/login`, `GET /api/v1/auth/me`, `POST /api/v1/auth/logout` | Login, identity and logout |
| `POST/GET /api/v1/documents` | Upload / list authorized documents |
| `GET/PUT/DELETE /api/v1/documents/{id}` | Inspect / stage replacement / delete |
| `GET /api/v1/documents/{id}/source` | Authorized attachment download; optional active version ID |
| `POST /api/v1/documents/{id}/overview` | Bounded cached overview with citations and coverage |
| `GET /api/v1/jobs/{id}` | Safe ingestion status |
| `POST/GET /api/v1/sessions` | Create / list conversations |
| `GET /api/v1/sessions/{id}/messages` | Bounded persistent history |
| `DELETE /api/v1/sessions/{id}` | Remove conversation |
| `POST /api/v1/chat` | Question, session ID and optional document IDs |
| `GET /health/live`, `GET /health/ready` | Process health / metadata, index, Ollama reachability and installed-model checks |

All mutation requests require `X-RAG-Request: 1` and same-origin browser requests. Production cookies are Secure, HttpOnly and SameSite=Strict. Readiness checks service reachability and model availability, not the factual quality of generated answers.

## Later LangGraph branch

Retain the document lifecycle, ownership boundaries, source contracts, model integrations and evaluation suite. Replace the bounded LCEL orchestration with graph state and checkpoints. Add narrowly scoped retrieval and other tools, routing, step/time budgets, and approval/resume handling for side effects. Framework changes must not bypass access enforcement in the application services.
