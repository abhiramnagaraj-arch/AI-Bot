# Operation and deployment

## Supported operating profile

Linux x86-64, Python 3.12, one Uvicorn process, one embedded Chroma runtime, and one durable ingestion worker. The application refuses a second running process using the same runtime lock. Model inference is provided by Ollama. The default is CPU embeddings with a small local generation model; latency depends strongly on hardware and concurrent activity.

No horizontal scaling is claimed. To run multiple replicas, move metadata/history to shared storage, use a networked vector service, and replace the process-local limits/locks/worker with shared coordination. Test those changes independently.

## Production configuration

1. Copy `.env.example` to `.env` and provision a unique random access key per user in `RAG_USERS_JSON`.
2. Set `RAG_ENVIRONMENT=production` and `RAG_ALLOWED_HOSTS` to your domain plus localhost health-check hosts.
3. Put the service behind a trusted HTTPS reverse proxy. Preserve the original Host header and configure trusted forwarding explicitly. Production cookies require HTTPS.
4. Keep Ollama private. Bind the backend only to a private/loopback interface behind the proxy.
5. Set upload/storage/document quotas, timeouts, retention and concurrency for available disk/RAM/CPU/GPU.
6. Provision model artifacts explicitly and verify readiness plus a real synthetic question before opening access.

Access keys are provisioned secrets in environment configuration; no self-service registration/password-reset or SSO integration is included. The database stores hashes of opaque session tokens, not the original access keys. Removing a user from configuration revokes access. Rotate user keys and clear that user's auth tokens when revoking a compromised login. Protect `.env` with filesystem permissions.

## Docker

```bash
# Fill .env with production users and allowed hosts first.
docker compose up -d ollama
docker compose --profile setup run --rm provision
docker compose up --build -d backend
```

The backend uses the hashed Linux/CPU dependency lock, runs as UID 10001 with a read-only root filesystem, and persists only its data/model volumes. There is no source bind mount in the production definition. It binds port 8000 to host loopback. An HTTPS reverse proxy is an external deployment prerequisite, not an implicitly exposed public endpoint.

The Ollama image tag is explicit. Model tags are configurable; record the actual model digest when promoting a release. For strict artifact reproducibility, pin base/Ollama images to approved digests and use a managed model-artifact process. Model cache volumes are provisioned by the setup service. Docker's resource/health limits are an initial profile to benchmark, not a throughput guarantee.

```bash
docker compose ps
docker compose logs --tail=100 backend
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

Readiness checks metadata, loaded embeddings/index access, Ollama reachability, and model presence. It is not an inference benchmark. Application errors carry an `X-Request-ID` and safe error body. Default logs contain allowlisted operational metadata only; provider/framework debug logs must not be enabled on private workloads without reviewing their content.

## Backups and restore

Stop the application before making a coordinated snapshot. The backup command checks the runtime lock and archives metadata/history, source files and Chroma together. Treat archives as private data. Use encrypted, access-controlled backup storage.

```bash
# Local deployment, after stopping the app:
python -m scripts.backup backup runtime/langchain-v2 /secure/backups/folio.backup.tar.gz
# Restore only into an empty/new directory:
python -m scripts.backup restore /secure/backups/folio.backup.tar.gz runtime/restored
# Set RAG_DATA_DIR=runtime/restored, then start the app and verify retrieval.
```

Restore rebases source-file paths and invalidates login sessions. It refuses path traversal, links and a nonempty destination. Only restore trusted operator-created archives. For Compose, stop `backend` and run the same command in a one-off container with the data volume and a dedicated backup mount. Verify the backup mount is writable by UID 10001. Do not copy only SQLite while leaving vector/source files behind.

An automated test covers metadata/source round-trip and session invalidation. Operators should additionally rehearse a complete restore of their actual corpus and vector store before rollout.

## Retention, deletion and recovery

- Authentication tokens expire after `RAG_SESSION_HOURS`.
- Conversation sessions inactive for `RAG_RETENTION_DAYS` are periodically removed; history is also bounded to `RAG_HISTORY_TURNS` pairs.
- Documents are retained until explicitly deleted. Quotas bound logical upload bytes and document counts; vector/model disk overhead requires host-level monitoring.
- Replacements are staged. The active version changes only after successful vector insertion.
- Deleted/replaced versions are cleaned by the worker. Cleanup failures retry while the application runs.
- Interrupted ingestion returns to the queue on restart. Invalid documents remain failed until a corrected upload is submitted.
- Changing the embedding revision or splitter configuration creates a new collection fingerprint. Queries against old active versions return `reindex_required`; re-upload those documents. Old indexes are not silently overwritten.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Sign-in fails locally | Use `127.0.0.1`, username `local`, development mode and empty user mapping; otherwise use a provisioned key. |
| Sign-in succeeds but hosted requests are unauthenticated | Production cookies require HTTPS; confirm proxy/domain settings. |
| Model not ready | Run model provisioning; check `RAG_OLLAMA_BASE_URL`, model tag, embedding revision/cache and filesystem permissions. |
| PDF has no readable text | Perform OCR before upload; encrypted PDFs must be unlocked. |
| Indexing fails | Inspect safe job error; check parser limits, embedding cache, runtime disk and available memory. Retry with the corrected file. |
| `documents_changed` / `conversation_busy` | Wait for the current operation, then reload history and retry. |
| `invalid_answer` | The model returned invalid structure/references. Try a focused question; evaluate a stronger configured model if persistent. |
| Unexpected refusals | Inspect source text and run corpus evaluations; calibrate retrieval cutoff/chunking rather than interpreting score as confidence. |
| Ollama runner crashes with a grammar error | Use the application's current schemas. Large string limits are locally validated to avoid unsupported grammar repetition bounds. |
| Second process refused | Run one worker per runtime; do not use `uvicorn --workers 2` or multiple replicas on the embedded store. |
| Browser timeout or Stop waiting | Backend inference can continue. Reload the conversation before sending the same question again. |

## Release checklist

Run lint, formatting checks, tests, browser smoke, locked Docker build, model provisioning, real-model smoke, corpus-specific held-out evaluation, a restore rehearsal, and capacity testing on the target hardware. Define p95 latency/error/capacity targets from those results. Record known model/guardrail limitations. Do not substitute the small synthetic evaluation for domain acceptance testing.
