# Folio — LangChain Document Assistant

A document RAG application with a browser workspace, authenticated private libraries, persistent conversations, background ingestion, and grounded answers with source references. This branch uses LangChain Core runnable composition, LangChain loaders/splitters, Hugging Face embeddings, Chroma, and ChatOllama.

The supported deployment is **one application process on one host**. It includes production safeguards and release tooling; production answer quality and capacity still need to be validated on your documents and hardware. See [validation](docs/validation.md), [architecture](docs/architecture.md), and [operations](docs/operations.md).

## What is implemented

- PDF, UTF-8 TXT/Markdown and DOCX uploads; size, type, page, expansion and extraction limits.
- Durable ingestion jobs and recovery after restart; SHA-256 duplicate detection.
- Staged document replacement: the old version stays available until the replacement succeeds.
- Private source selection, downloads, deletion, and source/page citations.
- LangChain conversation history with a standalone-question rewrite stage; no astronomy-specific rules.
- Structured answers: `answered`, `insufficient_evidence`, `clarification_needed`, or `blocked`.
- Versioned, cached document overviews with explicit passage coverage. Long-document overviews can be partial.
- User authentication, HttpOnly cookies, same-origin write checks, ownership checks, limits and safe errors.
- Privacy-conscious operational logs, health checks, hashed dependency lock, Docker, CI, and offline backup/restore.

Scanned PDFs require OCR before upload. Advanced layout/table understanding, external source connectors, tools, and agentic behavior are not included in this branch. Source references establish which passages were supplied; they do not mathematically prove every claim is correct.

## Local setup (Linux, Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
cp .env.example .env
```

`requirements.lock` targets Linux x86-64/Python 3.12 with CPU PyTorch. `requirements.txt` contains the direct dependencies for other environments; resolve and test an appropriate lock before deploying elsewhere. This project's existing `venv` can also be used after installing the updated requirements.

Start Ollama, then explicitly provision the model artifacts:

```bash
ollama serve
# In another terminal, from the project directory:
python -m scripts.prepare_models
python -m app.server
```

Open **http://127.0.0.1:8000**. For loopback-only development with `RAG_USERS_JSON={}`, sign in as `local` with an empty access key. Local login is refused from remote clients. For provisioned users, set `RAG_USERS_JSON` in `.env` to a JSON object mapping each username to a unique random access key of at least 16 characters. Never commit the file.

Runtime loads cached embedding artifacts by default. `scripts.prepare_models` deliberately downloads the pinned embedding revision and the configured Ollama model. Model changes are configuration changes; embedding/splitter changes require re-uploading documents to build the new index.

## Use the application

1. Add a supported file in **Your library**. Wait for **Ready**; a failed job shows a safe explanation.
2. Use all ready documents or choose individual sources.
3. Ask a question. Expand **supporting passages** to inspect evidence and download the original source.
4. Ask a follow-up in the same conversation, or create a new chat for a new thread.
5. Use **Overview** for a cached, cited document overview. Its coverage is shown explicitly.
6. **Replace** stages a new version. **Delete** removes access immediately and schedules physical cleanup.

Replacing or deleting a document clears stored conversation messages for that user. This conservative privacy policy prevents old document content being reused from history. Existing browser content cannot be recalled from a user who already received it.

The browser's **Stop waiting** cancels the browser request; model execution may finish and persist the answer on the server. Reopen the conversation before retrying. Operations are serialized per user to preserve document/history consistency.

## Workflow

```text
Upload -> validate -> persist job -> LangChain loader -> token-aware splitting
       -> normalized embeddings -> staged Chroma version -> publish ready version

Question -> authenticate and authorize sources -> load bounded message history
         -> rewrite follow-up -> scoped LangChain retriever -> bounded evidence
         -> prompt + ChatOllama structured output -> validate source references
         -> persist successful turn -> answer, status and citations
```

`app/rag.py` expresses the chat workflow with LCEL (`|`, `RunnablePassthrough`, `RunnableLambda`). `app/memory.py` implements LangChain's `BaseChatMessageHistory` against owner-checked SQLite storage. The deprecated `RunnableWithMessageHistory` wrapper is deliberately not used; LangGraph persistence is reserved for the later branch.

The answer model receives the original question, bounded conversation history, and retrieved evidence. Rewriting affects retrieval only. Instructions remain separate from untrusted document text. Unanswerable retrieval does not invoke the answer model. Similarity thresholds are ranking cutoffs, not confidence probabilities.

## CLI

The CLI uses the same API, access policy, and document services as the browser:

```bash
python -m app.main --username local --upload documents/example.pdf
python -m app.main --username local
```

## Tests and evaluation

```bash
pip install pytest==8.4.2 ruff==0.12.12
ruff check app tests scripts
ruff format --check app tests scripts
pytest -q
node --check frontend/script.js
python -m scripts.evaluate --mode retrieval --output runtime/retrieval-results.json
python -m scripts.evaluate --mode generation --limit 5 --output runtime/generation-results.json
python -m scripts.smoke_live
```

Tests use synthetic documents and temporary runtime directories. Most use deterministic AI components; separate tests exercise real Chroma and document parsers. `smoke_live` uses the configured actual Ollama model and local embeddings. The 50-case synthetic evaluation set is a starting point, not a certification of general accuracy. See [evals/README.md](evals/README.md).

For the browser smoke test, install `playwright==1.55.0` and Google Chrome, then run `python -m scripts.browser_smoke`. It tests the browser and real API/parser with a deterministic model and writes screenshots to `/tmp`.

## Deployment and branch strategy

See [operations](docs/operations.md) for Docker provisioning, HTTPS, backups, restore, retention and troubleshooting. Do not expose development mode publicly or run multiple Uvicorn workers against the same runtime. Do not publish the Ollama endpoint.

- `main`: conventional Python RAG variant; unchanged by this implementation.
- `LangChain`: this deterministic conversational document RAG application.
- `LangGraph`: next phase; will build on the tested services and add controlled tools, checkpoints and bounded agent routing.

The new runtime is `runtime/langchain-v2`. Previous databases and documents are preserved and are not silently migrated. Re-upload source files through the new UI. The previous plan is retained in [docs/production_readiness_plan.md](docs/production_readiness_plan.md) as assessment history; current behavior is described here and in the architecture/operations guides.
