import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .auth import COOKIE, RateLimiter, authenticate, principal
from .config import FRONTEND_DIR, Settings
from .logger import log_event
from .schemas import AppError, ChatRequest, ChatResponse, LoginRequest, SessionRequest
from .services import Services


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
                raise AppError(
                    413,
                    "body_too_large",
                    "The upload or request exceeds the configured size limit.",
                )
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings=None, services=None, start_worker=True):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.services = services or Services(settings)
        if start_worker:
            app.state.services.start()
        try:
            yield
        finally:
            if start_worker:
                await run_in_threadpool(app.state.services.close)

    app = FastAPI(
        title="Document Assistant · LangChain",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url=None,
    )
    app.state.limiter = RateLimiter()
    app.add_middleware(BodyLimitMiddleware, upload_bytes=settings.max_upload_mb * 1024**2)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.middleware("http")
    async def request_context(request, call_next):
        request.state.request_id = uuid.uuid4().hex
        start = time.monotonic()
        upload_slot = False
        if request.url.path.startswith("/api/v1/documents") and request.method in {"POST", "PUT"}:
            svc = request.app.state.services
            token = request.cookies.get(COOKIE, "")
            if not token or not svc.store.login_owner(token):
                return JSONResponse(
                    {
                        "error": {
                            "code": "unauthenticated",
                            "message": "Sign in to upload documents.",
                        }
                    },
                    status_code=401,
                )
            upload_slot = svc.http_uploads.acquire(blocking=False)
            if not upload_slot:
                return JSONResponse(
                    {
                        "error": {
                            "code": "upload_busy",
                            "message": "Upload capacity reached. Please retry.",
                        }
                    },
                    status_code=429,
                )
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            valid_origin = not origin or urlsplit(origin).netloc == request.headers.get("host")
            if request.headers.get("x-rag-request") != "1" or not valid_origin:
                if upload_slot:
                    request.app.state.services.http_uploads.release()
                return JSONResponse(
                    {
                        "error": {
                            "code": "invalid_origin",
                            "message": "Use the same-origin application to make this request.",
                            "request_id": request.state.request_id,
                        }
                    },
                    status_code=403,
                )
        try:
            response = await call_next(request)
        except AppError as exc:
            response = await app_error(request, exc)
        except Exception as exc:
            # Contain unhandled exceptions here so Uvicorn cannot print provider
            # exception payloads (which can include private prompt/source text).
            log_event(
                "request_failed", request_id=request.state.request_id, code=type(exc).__name__
            )
            response = await app_error(
                request,
                AppError(
                    500,
                    "internal_error",
                    "The request could not be completed. Use the request ID when contacting support.",
                ),
            )
        finally:
            if upload_slot:
                request.app.state.services.http_uploads.release()
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        log_event(
            "http_request",
            request_id=request.state.request_id,
            status=response.status_code,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        return response

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

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic errors include submitted inputs; never echo them (especially credentials).
        return await app_error(
            request,
            AppError(422, "invalid_request", "Check the request fields, file and question length."),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        log_event(
            "request_failed",
            request_id=getattr(request.state, "request_id", ""),
            code=type(exc).__name__,
        )
        return await app_error(
            request,
            AppError(
                500,
                "internal_error",
                "The request could not be completed. Use the request ID when contacting support.",
            ),
        )

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(request: Request):
        svc = request.app.state.services
        try:
            with svc.store.connect() as db:
                db.execute("SELECT 1")
            if svc._ai is None:
                raise RuntimeError("embedding_not_ready")
            await run_in_threadpool(svc._ai[0].ready)
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(settings.ollama_base_url.rstrip("/") + "/api/tags")
                response.raise_for_status()
                models = {m["name"] for m in response.json().get("models", [])}
                if settings.model not in models and settings.model + ":latest" not in models:
                    raise RuntimeError("model_missing")
            return {"status": "ready"}
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)

    @app.post("/api/v1/auth/login")
    def login(payload: LoginRequest, request: Request):
        owner = authenticate(request, payload.username, payload.access_key)
        token = request.app.state.services.store.create_login(owner)
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
        request.app.state.services.store.logout(request.cookies.get(COOKIE, ""))
        response = JSONResponse({"status": "signed_out"})
        response.delete_cookie(COOKIE, path="/")
        return response

    @app.post("/api/v1/sessions", status_code=201)
    def new_session(payload: SessionRequest, request: Request, owner=Depends(principal)):
        return request.app.state.services.store.new_session(owner, payload.title)

    @app.get("/api/v1/sessions")
    def sessions(request: Request, owner=Depends(principal)):
        return request.app.state.services.store.sessions(owner)

    @app.get("/api/v1/sessions/{sid}/messages")
    def messages(sid: str, request: Request, owner=Depends(principal)):
        values = request.app.state.services.store.messages(owner, sid)
        return [
            {
                "role": "user" if m["type"] == "human" else "assistant",
                "content": m["data"]["content"],
                "sources": m["data"].get("additional_kwargs", {}).get("sources", []),
                "status": m["data"].get("additional_kwargs", {}).get("status"),
            }
            for m in values
        ]

    @app.delete("/api/v1/sessions/{sid}")
    def delete_session(sid: str, request: Request, owner=Depends(principal)):
        svc = request.app.state.services
        with svc.owner_lock(owner):
            svc.store.delete_session(owner, sid)
        return {"status": "deleted"}

    @app.get("/api/v1/documents")
    def documents(request: Request, owner=Depends(principal)):
        return request.app.state.services.store.documents(owner)

    @app.get("/api/v1/documents/{doc_id}")
    def document(doc_id: str, request: Request, owner=Depends(principal)):
        return request.app.state.services.store.document(owner, doc_id)

    async def accept_upload(request, owner, file, doc_id=None):
        svc = request.app.state.services
        if not svc.upload_slots.acquire(blocking=False):
            raise AppError(
                429, "upload_busy", "Another upload is in progress. Please retry shortly."
            )
        path = None
        keep = False
        try:
            filename = Path((file.filename or "").replace("\\", "/")).name
            if not filename or len(filename) > 180 or any(ord(c) < 32 for c in filename):
                raise AppError(422, "invalid_filename", "Choose a file with a valid filename.")
            suffix = Path(filename).suffix.lower()
            if suffix not in {".pdf", ".txt", ".md", ".docx"}:
                raise AppError(
                    415, "unsupported_type", "Supported formats: PDF, TXT, Markdown and DOCX."
                )
            path = svc.store.root / "uploads" / (uuid.uuid4().hex + suffix)
            size, digest, head = 0, hashlib.sha256(), b""
            with path.open("xb") as target:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > settings.max_upload_mb * 1024**2:
                        raise AppError(
                            413, "upload_too_large", "The file exceeds the upload size limit."
                        )
                    if not head:
                        head = block[:1024]
                    digest.update(block)
                    target.write(block)
            if not size:
                raise AppError(422, "empty_file", "The uploaded file is empty.")
            if suffix == ".pdf" and b"%PDF-" not in head:
                raise AppError(415, "invalid_file", "The file does not have a PDF signature.")
            if suffix == ".docx" and not head.startswith(b"PK"):
                raise AppError(415, "invalid_file", "The file does not have a DOCX signature.")
            if suffix in {".txt", ".md"} and b"\x00" in head:
                raise AppError(
                    415, "invalid_file", "Text documents must use UTF-8 without binary content."
                )

            def enqueue():
                with svc.owner_lock(owner):
                    return svc.store.enqueue(
                        owner, filename, path, digest.hexdigest(), size, svc.fingerprint, doc_id
                    )

            document_id, job_id, keep = await run_in_threadpool(enqueue)
            svc.worker.wake.set()
            return {"document_id": document_id, "job_id": job_id, "duplicate": not keep}
        finally:
            await file.close()
            if path and not keep:
                path.unlink(missing_ok=True)
            svc.upload_slots.release()

    @app.post("/api/v1/documents", status_code=202)
    async def upload(request: Request, file: UploadFile = File(...), owner=Depends(principal)):
        return await accept_upload(request, owner, file)

    @app.put("/api/v1/documents/{doc_id}", status_code=202)
    async def replace(
        doc_id: str, request: Request, file: UploadFile = File(...), owner=Depends(principal)
    ):
        return await accept_upload(request, owner, file, doc_id)

    @app.delete("/api/v1/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request, owner=Depends(principal)):
        svc = request.app.state.services
        with svc.owner_lock(owner):
            svc.store.delete_document(owner, doc_id)
        svc.worker.wake.set()
        return {"status": "deleted", "history_cleared": True}

    @app.get("/api/v1/documents/{doc_id}/source")
    def source(
        doc_id: str, request: Request, version_id: str | None = None, owner=Depends(principal)
    ):
        path, filename = request.app.state.services.store.source(owner, doc_id, version_id)
        if not path.is_file():
            raise AppError(404, "source_missing", "The stored source file is unavailable.")
        # Attachments never execute in the app's origin.
        return FileResponse(path, filename=filename, content_disposition_type="attachment")

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str, request: Request, owner=Depends(principal)):
        return request.app.state.services.store.job(owner, job_id)

    @app.post("/api/v1/documents/{doc_id}/overview")
    async def overview(doc_id: str, request: Request, owner=Depends(principal)):
        svc = request.app.state.services
        if not svc.chat_slots.acquire(blocking=False):
            raise AppError(429, "chat_busy", "The assistant is at capacity. Please retry shortly.")

        def invoke():
            with svc.owner_lock(owner):
                svc.store.document(owner, doc_id)
                try:
                    _, _, rag = svc.ai()
                except Exception:
                    raise AppError(
                        503, "index_unavailable", "The embedding model or index is unavailable."
                    ) from None
                return rag.overview(owner, doc_id, request.state.request_id)

        try:
            return await run_in_threadpool(invoke)
        finally:
            svc.chat_slots.release()

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request, owner=Depends(principal)):
        svc = request.app.state.services
        if not svc.chat_slots.acquire(blocking=False):
            raise AppError(429, "chat_busy", "The assistant is at capacity. Please retry shortly.")

        def invoke():
            lock = svc.owner_lock(owner)
            if not lock.acquire(blocking=False):
                raise AppError(
                    409,
                    "conversation_busy",
                    "A request is already running for your account. Please wait for it to finish.",
                )
            try:
                svc.store.session(owner, payload.session_id)
                svc.store.active(owner, payload.document_ids)
                try:
                    _, _, rag = svc.ai()
                except Exception:
                    raise AppError(
                        503,
                        "index_unavailable",
                        "The embedding model or index is unavailable. Check server setup.",
                    ) from None
                return rag.chat(owner, payload, request.state.request_id)
            finally:
                lock.release()

        try:
            return await run_in_threadpool(invoke)
        finally:
            svc.chat_slots.release()

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, access_log=False)
