"""FastAPI boundary; one application process serializes index mutations and answers."""

import fcntl
import hmac
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from threading import Lock
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import FRONTEND_DIR, Settings
from .documents import Documents
from .guardrails import AppError, ChatRequest, ChatResponse, LoginRequest, SessionRequest
from .rag import RagService
from .storage import Store

LOG = logging.getLogger("rag")

COOKIE = "rag_session"


class RateLimiter:
    def __init__(self, capacity=10000):
        self.capacity, self.entries, self.lock = capacity, OrderedDict(), Lock()

    def check(self, key, limit):
        now = time.monotonic()
        with self.lock:
            while self.entries and next(iter(self.entries.values()))[0] < now - 60:
                self.entries.popitem(last=False)
            start, count = self.entries.get(key, (now, 0))
            if now - start >= 60:
                start, count = now, 0
            if count >= limit:
                raise AppError(
                    429, "rate_limit", "Too many requests. Please wait a minute and retry."
                )
            if key not in self.entries and len(self.entries) >= self.capacity:
                raise AppError(429, "rate_limit", "The server is busy. Please retry shortly.")
            self.entries[key] = (start, count + 1)


def principal(request: Request):
    if hasattr(request.state, "owner"):
        return request.state.owner
    services = request.app.state
    token = request.cookies.get(COOKIE, "")
    owner = services.store.login_owner(token) if token else None
    # Revoked users lose access immediately, even with a still-valid session cookie.
    configured = services.settings.users
    local_allowed = (
        not configured and services.settings.environment == "development" and owner == "local"
    )
    if not owner or (owner not in configured and not local_allowed):
        raise AppError(401, "unauthenticated", "Sign in to continue.")
    request.app.state.limiter.check("user:" + owner, services.settings.requests_per_minute)
    request.state.owner = owner
    return owner


def authenticate(request, username, access_key):
    settings = request.app.state.settings
    remote = request.client.host if request.client else "unknown"
    request.app.state.limiter.check("login:" + remote, 10)
    configured = settings.users
    if not configured and settings.environment == "development":
        if remote in {"127.0.0.1", "::1"} and username == "local":
            return "local"
    expected = configured.get(username, "invalid-key-placeholder")
    valid = hmac.compare_digest(expected.encode(), access_key.encode())
    if username in configured and valid:
        return username
    raise AppError(401, "invalid_credentials", "Invalid username or access key.")


class BodyLimitMiddleware:
    def __init__(self, app, upload_bytes):
        self.app, self.upload_bytes = app, upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        is_upload = scope["path"].startswith("/api/v1/documents") and scope["method"] in {
            "POST",
            "PUT",
        }
        limit = self.upload_bytes + 65536 if is_upload else 32768
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                # FastAPI preserves HTTPException during body parsing; arbitrary
                # exceptions would incorrectly become a generic HTTP 400.
                raise HTTPException(413, "The upload or request exceeds the configured size limit.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings=None, store=None, documents=None, rag=None):
    settings = settings or Settings()
    operation_lock = Lock()

    def initialize(state):
        state.store.maintenance()
        if state.documents is None:
            state.documents = Documents(settings, state.store)
        state.documents.recover()
        if state.rag is None:
            state.rag = RagService(settings, state.store, state.documents)

    @asynccontextmanager
    async def lifespan(app):
        settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (settings.data_dir / "application.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Run one application worker per data directory.") from None
            app.state.store = store or Store(settings)
            app.state.documents, app.state.rag = documents, rag
            try:
                initialize(app.state)
            except Exception:
                LOG.warning("AI initialization unavailable; retry on the next request")
            yield

    app = FastAPI(
        title="Document Assistant · LangChain",
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.limiter = RateLimiter()
    app.add_middleware(BodyLimitMiddleware, upload_bytes=settings.max_upload_mb * 1024**2)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @contextmanager
    def operation(request, needs_ai=True):
        if not operation_lock.acquire(blocking=False):
            raise AppError(
                429, "busy", "The assistant is processing another request. Please retry shortly."
            )
        try:
            state = request.app.state
            if needs_ai:
                try:
                    initialize(state)
                except Exception:
                    raise AppError(
                        503,
                        "index_unavailable",
                        "The embedding model or index is unavailable. Check server setup.",
                    ) from None
            yield state
        finally:
            operation_lock.release()

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": getattr(request.state, "request_id", ""),
                }
            },
            status_code=exc.status,
            headers={"Retry-After": "60"} if exc.status == 429 else None,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return await app_error(request, AppError(exc.status_code, "http_error", str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return await app_error(
            request,
            AppError(422, "invalid_request", "Check the request fields, file and question length."),
        )

    @app.middleware("http")
    async def request_context(request, call_next):
        request.state.request_id = uuid.uuid4().hex
        start = time.monotonic()
        try:
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                origin = request.headers.get("origin")
                if request.headers.get("x-rag-request") != "1" or (
                    origin and urlsplit(origin).netloc != request.headers.get("host")
                ):
                    raise AppError(
                        403,
                        "invalid_origin",
                        "Use the same-origin application to make this request.",
                    )
            # Authenticate uploads before multipart parsing allocates temporary files.
            if request.url.path == "/api/v1/documents" and request.method == "POST":
                principal(request)
            response = await call_next(request)
        except AppError as exc:
            response = await app_error(request, exc)
        except Exception as exc:
            LOG.error(
                "request_failed request_id=%s code=%s", request.state.request_id, type(exc).__name__
            )
            response = await app_error(
                request,
                AppError(
                    500,
                    "internal_error",
                    "The request could not be completed. Please retry or check server setup.",
                ),
            )
        response.headers.update(
            {
                "X-Request-ID": request.state.request_id,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
            }
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        LOG.info(
            "http_request request_id=%s status=%s duration_ms=%s",
            request.state.request_id,
            response.status_code,
            int((time.monotonic() - start) * 1000),
        )
        return response

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(request: Request):
        try:
            if request.app.state.rag is None:
                raise RuntimeError("index_not_ready")
            with request.app.state.store.connect() as db:
                db.execute("SELECT 1")
            request.app.state.documents.index.get(limit=1)
            response = httpx.get(settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=3)
            response.raise_for_status()
            models = {m["name"] for m in response.json().get("models", [])}
            if not {settings.model, settings.model + ":latest"} & models:
                raise RuntimeError("model_missing")
            return {"status": "ready"}
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)

    @app.post("/api/v1/auth/login")
    def login(payload: LoginRequest, request: Request):
        owner = authenticate(request, payload.username, payload.access_key)
        token = request.app.state.store.create_login(owner)
        response = JSONResponse({"username": owner})
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            secure=settings.environment == "production",
            samesite="strict",
            max_age=settings.session_hours * 3600,
            path="/",
        )
        return response

    @app.get("/api/v1/auth/me")
    def me(owner=Depends(principal)):
        return {
            "username": owner,
            "max_upload_mb": settings.max_upload_mb,
            "max_query_chars": settings.max_query_chars,
        }

    @app.post("/api/v1/auth/logout")
    def logout(request: Request, owner=Depends(principal)):
        request.app.state.store.logout(request.cookies.get(COOKIE, ""))
        response = JSONResponse({"status": "signed_out"})
        response.delete_cookie(COOKIE, path="/")
        return response

    @app.post("/api/v1/sessions", status_code=201)
    def new_session(payload: SessionRequest, request: Request, owner=Depends(principal)):
        return request.app.state.store.new_session(owner, payload.title)

    @app.get("/api/v1/sessions")
    def sessions(request: Request, owner=Depends(principal)):
        return request.app.state.store.sessions(owner)

    @app.get("/api/v1/sessions/{sid}/messages")
    def messages(sid: str, request: Request, owner=Depends(principal)):
        return [
            {
                "role": "user" if m["type"] == "human" else "assistant",
                "content": m["data"]["content"],
                **{
                    k: v
                    for k, v in m["data"].get("additional_kwargs", {}).items()
                    if k in {"sources", "status"}
                },
            }
            for m in request.app.state.store.messages(owner, sid)
        ]

    @app.delete("/api/v1/sessions/{sid}")
    def delete_session(sid: str, request: Request, owner=Depends(principal)):
        with operation(request, needs_ai=False) as state:
            state.store.delete_session(owner, sid)
        return {"status": "deleted"}

    @app.get("/api/v1/documents")
    def documents_list(request: Request, owner=Depends(principal)):
        return request.app.state.store.documents(owner)

    @app.post("/api/v1/documents", status_code=201)
    def upload(request: Request, file: UploadFile = File(...), owner=Depends(principal)):
        with operation(request) as state:
            return state.documents.ingest(owner, file.filename, file.file)

    @app.delete("/api/v1/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request, owner=Depends(principal)):
        with operation(request) as state:
            state.store.delete_document(owner, doc_id)
            if not state.documents.cleanup():
                raise AppError(
                    503,
                    "cleanup_pending",
                    "Document access was removed. Physical cleanup will retry on the next operation or restart.",
                )
        return {"status": "deleted", "history_cleared": True}

    @app.get("/api/v1/documents/{doc_id}/source")
    def source(
        doc_id: str, request: Request, version_id: str | None = None, owner=Depends(principal)
    ):
        path, filename = request.app.state.store.source(owner, doc_id, version_id)
        if not path.is_file():
            raise AppError(404, "source_missing", "The stored source file is unavailable.")
        return FileResponse(path, filename=filename, content_disposition_type="attachment")

    @app.post("/api/v1/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request, owner=Depends(principal)):
        with operation(request) as state:
            return state.rag.chat(owner, payload, request.state.request_id)

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, access_log=False)
