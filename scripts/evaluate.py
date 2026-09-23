"""Evaluate synthetic RAG cases. Keyword matches are diagnostics, not semantic grades."""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from app.config import Settings
from app.documents import Documents
from app.guardrails import ChatRequest
from app.rag import RagService
from app.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "generation"], default="retrieval")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("runtime/evaluation-results.json"))
    args = parser.parse_args()
    cases = json.loads(Path("evals/cases.json").read_text())[: args.limit]
    with tempfile.TemporaryDirectory(prefix="folio-evaluation-") as directory:
        settings = Settings(data_dir=Path(directory))
        store = Store(settings)
        index = Documents(settings, store)
        for source in sorted(Path("evals/documents").glob("*.md")):
            with source.open("rb") as stream:
                index.ingest("evaluation", source.name, stream)
        active = store.active("evaluation")
        if len(active) != 3:
            raise RuntimeError("Evaluation documents failed to index")
        rag = RagService(settings, store, index)
        results = []
        for case in cases:
            start = time.monotonic()
            matches = index.search(
                case["question"], "evaluation", [d["active_version"] for d in active]
            )[: settings.top_k]
            names = [d.metadata["filename"] for d, _ in matches]
            result = {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case["source"],
                "retrieved_sources": names,
                "retrieval_hit": case["source"] in names if case["source"] else None,
            }
            if args.mode == "generation":
                sid = store.new_session("evaluation", case["id"])["id"]
                try:
                    answer = rag.chat(
                        "evaluation",
                        ChatRequest(session_id=sid, query=case["question"]),
                        case["id"],
                    )
                    result.update(
                        answer=answer.answer,
                        status=answer.status,
                        status_match=answer.status == case["expected_status"],
                        reference_term_match=all(
                            t.lower() in answer.answer.lower() for t in case["reference_terms"]
                        ),
                        sources=[s.model_dump() for s in answer.sources],
                    )
                except Exception as exc:
                    result["error"] = type(exc).__name__
            result["time_ms"] = int((time.monotonic() - start) * 1000)
            results.append(result)
        answerable = [r for r in results if r["retrieval_hit"] is not None]
        latencies = sorted(r["time_ms"] for r in results)
        report = {
            "mode": args.mode,
            "model": settings.model,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "top_k": settings.top_k,
            "cutoff": settings.relevance_threshold,
            "cases": len(results),
            "answerable_cases": len(answerable),
            "retrieval_recall_at_k": sum(r["retrieval_hit"] for r in answerable)
            / max(1, len(answerable)),
            "p50_ms": statistics.median(latencies),
            "p95_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)],
            "limitations": "Small synthetic corpus. Source-level retrieval only; generation requires human review for claim support and citation correctness. Not a production quality certification.",
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
