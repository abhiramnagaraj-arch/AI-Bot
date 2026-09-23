"""SQLite accounts, conversations and a single record per uploaded document."""

import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .guardrails import AppError


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "uploads").mkdir(exist_ok=True, mode=0o700)
        self.path = self.root / "metadata.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    digest TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id,seq);
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, filename TEXT NOT NULL,
                    active_version TEXT NOT NULL, path TEXT NOT NULL, sha TEXT NOT NULL,
                    size INTEGER NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                    chunks INTEGER NOT NULL DEFAULT 0, pages INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS files_owner ON files(owner,status);
            """)
            # One-time, transactional migration: keep indexed files, source URLs and history.
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='versions'").fetchone():
                db.execute("BEGIN IMMEDIATE")
                db.execute("""
                    INSERT INTO files
                    SELECT CASE WHEN v.id=d.active_version THEN d.id ELSE v.id END,
                        v.owner,v.filename,v.id,v.path,v.sha,v.size,v.fingerprint,
                        CASE WHEN v.id=d.active_version AND d.deleted=0 THEN 'ready' ELSE 'pending' END,
                        v.chunks,v.pages,v.created
                    FROM versions v JOIN documents d ON d.id=v.document_id
                    WHERE v.status!='cleaned'
                """)
                db.execute("DROP TABLE versions")
                db.execute("DROP TABLE documents")
            db.execute("PRAGMA user_version=3")
        self.maintenance()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def maintenance(self):
        with self.connect() as db:
            db.execute("DELETE FROM auth_tokens WHERE expires < ?", (time.time(),))
            db.execute(
                "DELETE FROM sessions WHERE updated < ?",
                (time.time() - self.settings.retention_days * 86400,),
            )

    def create_login(self, owner):
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM auth_tokens WHERE expires < ?", (time.time(),))
            db.execute(
                "INSERT INTO auth_tokens VALUES (?,?,?)",
                (
                    hashlib.sha256(token.encode()).hexdigest(),
                    owner,
                    time.time() + self.settings.session_hours * 3600,
                ),
            )
            # Bound active login sessions per account.
            db.execute(
                "DELETE FROM auth_tokens WHERE owner=? AND digest NOT IN "
                "(SELECT digest FROM auth_tokens WHERE owner=? ORDER BY expires DESC LIMIT 10)",
                (owner, owner),
            )
        return token

    def login_owner(self, token):
        with self.connect() as db:
            row = db.execute(
                "SELECT owner FROM auth_tokens WHERE digest=? AND expires>?",
                (hashlib.sha256(token.encode()).hexdigest(), time.time()),
            ).fetchone()
        return row["owner"] if row else None

    def logout(self, token):
        with self.connect() as db:
            db.execute(
                "DELETE FROM auth_tokens WHERE digest=?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            )

    def new_session(self, owner, title):
        sid, now = uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT COUNT(*) FROM sessions WHERE owner=?", (owner,)).fetchone()[0]
                >= 100
            ):
                raise AppError(
                    409, "session_limit", "Delete an old conversation before creating another."
                )
            db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, owner, title, now, now))
        return {"id": sid, "title": title, "created": now, "updated": now}

    def session(self, owner, sid):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE owner=? AND id=?", (owner, sid)
            ).fetchone()
        if not row:
            raise AppError(404, "session_not_found", "Conversation not found.")
        return dict(row)

    def sessions(self, owner):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,title,created,updated FROM sessions WHERE owner=? ORDER BY updated DESC",
                    (owner,),
                )
            ]

    def delete_session(self, owner, sid):
        self.session(owner, sid)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE id=? AND owner=?", (sid, owner))

    def messages(self, owner, sid):
        self.session(owner, sid)
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM messages WHERE session_id=? ORDER BY seq DESC LIMIT ?",
                (sid, self.settings.history_turns * 2),
            ).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def append_messages(self, owner, sid, messages):
        self.session(owner, sid)
        with self.connect() as db:
            db.executemany(
                "INSERT INTO messages(session_id,payload,created) VALUES (?,?,?)",
                [(sid, json.dumps(m), time.time()) for m in messages],
            )
            db.execute(
                "DELETE FROM messages WHERE session_id=? AND seq NOT IN "
                "(SELECT seq FROM messages WHERE session_id=? ORDER BY seq DESC LIMIT ?)",
                (sid, sid, self.settings.history_turns * 2),
            )
            db.execute("UPDATE sessions SET updated=? WHERE id=?", (time.time(), sid))

    def clear_messages(self, owner, sid):
        self.session(owner, sid)
        with self.connect() as db:
            db.execute("DELETE FROM messages WHERE session_id=?", (sid,))

    def documents(self, owner):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,filename,active_version,status,chunks,pages,created "
                    "FROM files WHERE owner=? AND status='ready' ORDER BY created DESC",
                    (owner,),
                )
            ]

    def document(self, owner, doc_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM files WHERE owner=? AND id=? AND status='ready'", (owner, doc_id)
            ).fetchone()
        if not row:
            raise AppError(404, "document_not_found", "Document not found.")
        return dict(row)

    def active(self, owner, selected=()):
        if selected:
            return [self.document(owner, doc_id) for doc_id in selected]
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM files WHERE owner=? AND status='ready'", (owner,)
                )
            ]

    def add_document(self, owner, filename, path, sha, fingerprint):
        size, doc_id = path.stat().st_size, uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute(
                "SELECT id FROM files WHERE owner=? AND sha=? AND fingerprint=? AND status='ready'",
                (owner, sha, fingerprint),
            ).fetchone()
            if duplicate:
                return duplicate[0], False
            count, used = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(size),0) FROM files WHERE owner=?", (owner,)
            ).fetchone()
            if count >= self.settings.max_documents:
                raise AppError(409, "document_limit", "Delete a document before uploading another.")
            if used + size > self.settings.max_storage_mb * 1024**2:
                raise AppError(
                    413, "storage_quota", "Your document storage quota has been reached."
                )
            db.execute(
                "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,0,0,?)",
                (
                    doc_id,
                    owner,
                    filename,
                    doc_id,
                    str(path),
                    sha,
                    size,
                    fingerprint,
                    "pending",
                    time.time(),
                ),
            )
        return doc_id, True

    def publish(self, doc_id, chunks, pages):
        with self.connect() as db:
            db.execute(
                "UPDATE files SET status='ready',chunks=?,pages=? WHERE id=?",
                (chunks, pages, doc_id),
            )

    def delete_document(self, owner, doc_id):
        self.document(owner, doc_id)
        with self.connect() as db:
            db.execute("UPDATE files SET status='pending' WHERE id=? AND owner=?", (doc_id, owner))
            # Old answers must not reintroduce deleted source content into prompts.
            db.execute(
                "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE owner=?)",
                (owner,),
            )

    def pending(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM files WHERE status='pending'")]

    def forget(self, doc_id):
        with self.connect() as db:
            db.execute("DELETE FROM files WHERE id=? AND status='pending'", (doc_id,))

    def source(self, owner, doc_id, version_id=None):
        doc = self.document(owner, doc_id)
        if version_id and version_id != doc["active_version"]:
            raise AppError(404, "source_not_found", "This source is no longer available.")
        return Path(doc["path"]), doc["filename"]
