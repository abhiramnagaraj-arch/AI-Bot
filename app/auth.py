import hmac
import time
from collections import OrderedDict
from threading import Lock

from fastapi import Request

from .schemas import AppError

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
    services = request.app.state.services
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
    return owner


def authenticate(request, username, access_key):
    settings = request.app.state.services.settings
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
