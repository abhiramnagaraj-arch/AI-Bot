# Validation record — LangChain implementation

Date: 2026-09-09. Branch: `LangChain`. Application/source changes are local and uncommitted. No deployment or publication was performed.

## Completed checks

| Check | Result |
| --- | --- |
| Final local regression suite | **41 passed** in 176.09 seconds |
| Additional derived-overview deletion regression | **1 passed**; cleanup removes cached source-derived overview content |
| Regression suite inside the hashed-dependency Docker image | **39 passed** in 203.58 seconds; final refusal/error-containment changes were subsequently covered by the local suite |
| Ruff lint and formatting | Passed |
| JavaScript syntax | Passed |
| Git whitespace checks | Passed |
| Docker Compose configuration | Validated using the example configuration; production credentials must still be provisioned |
| Docker build | Built `folio-rag:langchain` with the complete hashed Linux/CPU dependency lock |
| Production container startup | Non-root user `rag`, read-only root filesystem, temporary data storage and read-only model cache; `/health/ready` returned HTTP 200 |
| Browser workflow | Login, upload, background indexing, chat, citations, source download, history reload, mobile layout and deletion passed; no JavaScript errors |
| Backup/restore | Synthetic metadata/source round-trip passed; paths rebased and login tokens invalidated |
| Real Chroma behavior | Scope filtering, duplicate ingestion, version deletion and cleanup of prior collection fingerprints passed |
| Real PDF/DOCX/Markdown parsers | Text extraction, page metadata, content preservation, scan handling and extraction limits passed |

Browser verification uses the real frontend/API/parser with deterministic model/retrieval components. It is separate from the real-model checks below.

## Real-model conversation

Actual components: `gemma2:2b` through Ollama, pinned `all-MiniLM-L6-v2` embeddings, Chroma, LangChain LCEL, and SQLite message history. A separate temporary CPU-only Ollama service used the already installed model artifacts. Only synthetic documents and a temporary index were used.

| Question | Observed result | Time |
| --- | --- | --- |
| How many days of annual leave do employees receive? | 20 days per year, with validated source `S1` | 14.825 s |
| How far in advance must I request it? | Follow-up resolved to annual leave; 5 working days, with source `S1` | 25.443 s |
| What is the employee annual salary? | `insufficient_evidence`; no fabricated amount or sources | 24.680 s |
| What is the population of Tokyo? | `insufficient_evidence`; no outside-knowledge answer | 7.329 s |

[Recorded synthetic responses](../evals/live_smoke_results.json). These timings were obtained during concurrent development/testing and are not a capacity benchmark or SLA. IDs/links in the artifact refer to temporary test data, not a running document library.

Live testing exposed and fixed:

- A model-runner crash caused by large string `maxLength` constraints in Ollama's JSON grammar. Output length is now bounded by generation settings and local validation.
- Small-model inconsistency between explicit inline and structured citations. The application normalizes only valid supplied IDs and still rejects invented references.
- Valid source markers attached to insufficient-evidence responses. These are validated and removed from the abstention rather than creating a spurious API error.

## Retrieval evaluation

The 50-case synthetic set contains 45 answerable questions and 5 absent-answer questions across three fictional policies. The expected document appeared in top-k results for **44/45 answerable cases (97.8%)**, with `top_k=6` and cutoff `0.25`. Median measured search time was 27 ms and p95 was 51 ms in that run.

[Detailed retrieval results](../evals/retrieval_results.json). This small corpus makes source-level recall easier than realistic passage retrieval. The result does not measure answer correctness, entailment, hallucination rate, general adversarial resistance or multi-user capacity. One answerable retrieval miss remains in this synthetic evaluation; thresholds must be tuned on a representative corpus rather than optimized solely for this fixture.

The evaluation also exposed a privacy issue in LangChain's default out-of-range relevance warning: cosine similarity can be negative and the warning includes Document contents. The Chroma integration now supplies a bounded relevance conversion, preventing that warning from dumping private source text.

## Limits before a hosted rollout

- The supported architecture is a single-host, single-process authenticated app. It is not a horizontally scaled SaaS deployment.
- HTTPS, per-user provisioned keys, host configuration, disk/backup protection and resource capacity are deployment responsibilities.
- The parser requires text PDFs; scans need OCR, and PDFs with unreadable/blank pages are conservatively rejected with instructions. Complex table/layout understanding is not implemented.
- Long-document overviews report partial coverage. They are not exhaustive summaries of every passage.
- Citation checks establish source identity and consistency, not semantic proof of every statement. Injection patterns are supplementary and have known false-positive/false-negative limits.
- No full held-out evaluation on the user's real corpus, load test, external security audit, SSO, or live production restore exercise was performed.
- `main` was not changed and `LangGraph` was not started. Existing runtime data and source files were preserved; the new application uses `runtime/langchain-v2`.

Temporary browser/test servers and the separate Ollama verification service are stopped after checks. The built image remains local for review. Original source files were checkpointed before editing in `/tmp/ai-bot-before-langchain-20260909-130004.zip`.
