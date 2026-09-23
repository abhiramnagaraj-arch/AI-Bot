# Deployment and maintenance

Run one Uvicorn process per data directory on Linux/Python 3.12. AI operations are serialized; an overlapping request receives HTTP 429. There is no background job queue. Set the reverse-proxy request timeout to cover document parsing/embedding and model inference, and limit upload bodies at the proxy as well as in the application.

## Hosted configuration

Set `RAG_ENVIRONMENT=production`, provision a unique random access key of at least 16 characters per user in `RAG_USERS_JSON`, and set `RAG_ALLOWED_HOSTS` to your domain plus localhost health-check hosts. Use HTTPS through a trusted reverse proxy; production cookies require it. Keep Ollama private. Preserve the original Host header and configure trusted forwarding explicitly.

```bash
docker compose up -d ollama
docker compose --profile setup run --rm provision
docker compose up --build -d backend
```

The backend binds to host loopback, runs without root, uses a read-only container filesystem and persists data/model volumes. HTTPS termination is configured outside Compose. Model provisioning is explicit; runtime does not download artifacts. `/health/live` checks the HTTP service; `/health/ready` checks SQLite, the index and model availability, not inference quality.

Use `.env.example` for common settings. Additional validated limits are in `app/config.py`. Changes to embeddings or splitting require deleting and re-uploading affected documents. Removing an account from configuration revokes its access; rotating a key should also invalidate that user's existing login tokens. Do not enable provider debug logging on private documents.

## Backup and upgrade

Stop the backend before a backup. Snapshot sources, metadata and Chroma together; protect backups as private data.

```bash
python -m scripts.backup backup runtime/langchain-v2 /secure/backups/folio.backup.tar.gz
python -m scripts.backup restore /secure/backups/folio.backup.tar.gz runtime/restored
```

Restore requires an empty destination, rebases file paths and invalidates login tokens. Set `RAG_DATA_DIR=runtime/restored` before starting the restored application. For Docker, run these commands in a one-off container with the data and backup mounts.

The first startup upgrades the old version/job catalog transactionally to one record per document. Ready documents, source URLs and conversations survive. Obsolete queued/failed uploads are cleaned rather than resumed, and cached overviews are discarded. Roll back code and the coordinated pre-upgrade backup together; old code cannot read the new catalog.

## Recovery and troubleshooting

- Failed uploads: correct the document or resolve model/disk availability, then upload again.
- Busy response: wait for the current upload or answer to finish, then retry.
- Interrupted uploads: restart cleans partial files and vectors before AI operations resume.
- Pending deletion: access is already removed; cleanup retries on the next AI operation or restart.
- Unavailable model: run `scripts.prepare_models`, check Ollama and the configured cache/model name.
- Invalid answer: the model returned invalid structure or citations. Retry a focused question and evaluate the model on your corpus.
- No PDF text: unlock the PDF, perform OCR or remove blank pages before upload.
- Login fails locally: use loopback, development mode and `local`; configured accounts require their keys.

Conversation history is bounded to the configured number of turns. Expired logins and inactive conversations are removed during maintenance. Documents remain until deleted. Upload quotas do not include vector/model overhead; monitor host disk space and benchmark memory/latency on your target machine before opening access.
