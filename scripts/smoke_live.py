"""Exercise the real local embedding model, Chroma, Ollama and persistent history.
Uses synthetic documents in an isolated temporary runtime; never reads user documents.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path

from app.config import Settings
from app.documents import Documents
from app.guardrails import ChatRequest
from app.rag import RagService
from app.storage import Store


def main():
    with tempfile.TemporaryDirectory(prefix="folio-live-") as directory:
        root = Path(directory)
        settings = Settings(_env_file=None, data_dir=root, model_timeout=120)
        store = Store(settings)
        content = """Employee Leave Policy
Employees receive 20 days of annual leave per calendar year.
Requests must be submitted to the employee's manager at least 5 working days before the first day of leave.
Unused annual leave of up to 5 days may be carried into the next year.
This policy does not specify salary or bonus amounts.
"""
        print("Initializing actual embeddings and indexing synthetic policy...", flush=True)
        index = Documents(settings, store)
        doc = index.ingest("evaluation", "policy.txt", io.BytesIO(content.encode()))["document_id"]
        rag = RagService(settings, store, index)
        sid = store.new_session("evaluation", "Live smoke test")["id"]
        queries = [
            "How many days of annual leave do employees receive?",
            "How far in advance must I request it?",
            "What is the employee annual salary?",
            "What is the population of Tokyo?",
        ]
        results = []
        for query in queries:
            start = time.monotonic()
            print("Question:", query, flush=True)
            try:
                response = rag.chat(
                    "evaluation",
                    ChatRequest(session_id=sid, query=query, document_ids=[doc]),
                    "live-smoke",
                )
                result = response.model_dump()
                print(json.dumps(result), flush=True)
                results.append(result)
            except Exception as exc:
                result = {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "duration": time.monotonic() - start,
                }
                print(json.dumps(result), flush=True)
                results.append(result)
        output = os.environ.get("RAG_SMOKE_OUTPUT")
        if output:
            Path(output).write_text(
                json.dumps({"model": settings.model, "results": results}, indent=2)
            )
        if any("error" in result for result in results):
            raise SystemExit(1)
        expected = ["answered", "answered", "insufficient_evidence", "insufficient_evidence"]
        if [result["status"] for result in results] != expected:
            raise SystemExit("Unexpected answer/refusal behavior in the live smoke test")
        if "20" not in results[0]["answer"] or "5" not in results[1]["answer"]:
            raise SystemExit("The live answers missed the reference facts")
        if not all(result["sources"] for result in results[:2]):
            raise SystemExit("The live factual answers did not include sources")
        print("Live smoke passed: grounded answer, follow-up, missing fact and unrelated question.")


if __name__ == "__main__":
    main()
