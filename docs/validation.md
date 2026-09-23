# Simplification validation · 10 September 2026

Application source decreased from **3,779 to 1,855 lines (50.9%)**, counting Python, HTML, JavaScript and CSS, including formatting/comments. The backend now has seven modules plus `__init__.py`, down from twenty Python files. Generated dependency locks and tests are excluded from this comparison.

| Check | Result |
| --- | --- |
| Ruff lint and formatting, JavaScript syntax, Git whitespace | Passed |
| Final automated regression suite | **55 passed** in 275 seconds; one upstream Starlette/AnyIO deprecation warning |
| Browser workflow | Passed: login, upload, indexing, answer, citations, download, reload, mobile layout and deletion; no JavaScript errors |
| Real embeddings + Chroma + Ollama (`gemma2:2b`) | All four live cases passed |
| Synthetic retrieval evaluation | Expected source retrieved in 44/45 answerable questions across 50 cases; top-k 6 |
| Docker | Build passed; temporary non-root/read-only container returned readiness 200 and served the frontend |
| Existing catalog migration | Verified on a temporary copy of the existing database; its ready document survived; original runtime unchanged |

Regression coverage includes authentication/CSRF, account and document isolation, supported document parsing, actual PDF page citations, malformed/encrypted/unreadable files, duplicates, upload/chunk/body limits, failed partial indexing, interrupted-upload cleanup, source deletion, conversation persistence/deletion, follow-ups, greetings, clarification responses, invalid citations, provider failures, private-error containment, rate limits, concurrency, index changes, migration and backup/restore.

Live model checks:

- Annual leave entitlement: correctly answered **20 days**, with a valid source.
- Follow-up asking how far in advance: correctly answered **5 working days**, with a valid source.
- Salary absent from the document: returned insufficient evidence.
- Unrelated Tokyo population question: returned insufficient evidence.

The deterministic regression suite checks application behavior, not universal model accuracy. The retrieval result uses a small synthetic corpus and is not an answer-accuracy score. Prompt-injection filters are heuristic; citations validate source identities but do not prove claim entailment. Test your own domain questions before deployment. Only the single-process operating profile was verified.

Temporary browser/container test services were removed. The local Ollama service was reused. No deployment or branch switch was performed.
