import hashlib
import json
import time

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from ollama import ResponseError
from pydantic import ValidationError

from .guardrails import detect_injection, redact_pii, validate_citations, validate_query
from .memory import SQLiteHistory, bounded_history
from .prompts import ANSWER_PROMPT, REWRITE_PROMPT
from .retriever import ScopedRetriever
from .schemas import AnswerDraft, AppError, ChatResponse, Rewrite, Source

NO_EVIDENCE = "I couldn't find enough evidence in the selected documents to answer that. Try selecting another document or asking a more specific question."


class RagService:
    def __init__(self, settings, store, index, model):
        self.settings, self.store, self.index = settings, store, index
        # Only transport errors retry; invalid answers never silently become successful text.
        answer_model = model.with_structured_output(AnswerDraft).with_retry(
            retry_if_exception_type=(httpx.ConnectError,), stop_after_attempt=2
        )
        self.answer_chain = ANSWER_PROMPT | answer_model
        self.rewrite_chain = REWRITE_PROMPT | model.with_structured_output(Rewrite)
        stages = (
            RunnableLambda(self.prepare, name="authorize_and_rewrite")
            | RunnableLambda(self.retrieve, name="retrieve_evidence")
            | RunnableLambda(self.generate, name="grounded_answer")
        )
        # The installed Core version deprecates RunnableWithMessageHistory in favor
        # of graph persistence. Keep this branch graph-free using explicit LCEL
        # load/save stages and the supported BaseChatMessageHistory interface.
        self.chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(
                    lambda data: SQLiteHistory(store, data["owner"], data["session_id"]).messages,
                    name="load_history",
                )
            )
            | stages
            | RunnableLambda(self.save_history, name="persist_validated_turn")
        )

    def save_history(self, data):
        SQLiteHistory(self.store, data["owner"], data["session_id"]).add_messages(
            [HumanMessage(data["question"]), data["history_message"]]
        )
        return data

    def prepare(self, data):
        data = dict(data)
        data["history"] = bounded_history(data["history"], self.settings.history_chars)
        active = self.store.active(data["owner"], data["document_ids"])
        if any(d["fingerprint"] != self.index.fingerprint for d in active):
            raise AppError(
                409,
                "reindex_required",
                "The index configuration changed. Re-upload the selected documents to rebuild them.",
            )
        data["active"] = active
        while data["history"]:
            cost = sum(
                len(str(m.content).encode())
                for m in REWRITE_PROMPT.invoke(
                    {"question": data["question"], "history": data["history"]}
                ).messages
            )
            if cost + self.settings.output_tokens + 1200 <= self.settings.context_tokens:
                break
            data["history"] = data["history"][2:]
        if data["history"] and active:
            rewritten = self.rewrite_chain.invoke(
                {"question": data["question"], "history": data["history"]}
            )
            data["search_question"] = rewritten.question
        else:
            data["search_question"] = data["question"]
        return data

    def retrieve(self, data):
        retriever = ScopedRetriever(
            index=self.index,
            owner=data["owner"],
            versions=[d["active_version"] for d in data["active"]],
            top_k=self.settings.top_k,
        )
        data["documents"] = retriever.invoke(data["search_question"])
        return data

    def generate(self, data):
        records, sources = [], []
        active = {d["id"]: d for d in data["active"]}
        for doc in data["documents"]:
            metadata = doc.metadata
            if metadata["document_id"] not in active:
                continue
            citation_id = f"S{len(sources) + 1}"
            page = metadata.get("page", -1)
            source = Source(
                citation_id=citation_id,
                document_id=metadata["document_id"],
                version_id=metadata["version_id"],
                filename=active[metadata["document_id"]]["filename"],
                page=page + 1 if page >= 0 else None,
                excerpt=doc.page_content[:800],
                url=f"/api/v1/documents/{metadata['document_id']}/source?version_id={metadata['version_id']}",
            )
            text = redact_pii(doc.page_content) if self.settings.redact_pii else doc.page_content
            record = {
                "id": citation_id,
                "filename": source.filename,
                "page": source.page,
                "text": text,
            }
            candidate = records + [record]
            if (
                len(json.dumps(candidate, ensure_ascii=False).encode())
                > self.settings.evidence_chars
            ):
                break
            inputs = {
                "question": data["question"],
                "history": data["history"],
                "context": json.dumps(candidate, ensure_ascii=False),
            }
            # A conservative UTF-8 byte ceiling avoids downloading an unrelated model tokenizer.
            # Reserve schema/chat framing overhead and configured output separately.
            prompt_bytes = sum(
                len(str(m.content).encode()) for m in ANSWER_PROMPT.invoke(inputs).messages
            )
            while (
                data["history"]
                and prompt_bytes + self.settings.output_tokens + 1200 > self.settings.context_tokens
            ):
                data["history"] = data["history"][2:]
                inputs["history"] = data["history"]
                prompt_bytes = sum(
                    len(str(m.content).encode()) for m in ANSWER_PROMPT.invoke(inputs).messages
                )
            if prompt_bytes + self.settings.output_tokens + 1200 > self.settings.context_tokens:
                break
            records.append(record)
            sources.append(source)
        if not records:
            draft = AnswerDraft(status="insufficient_evidence", answer=NO_EVIDENCE)
        else:
            draft = self.answer_chain.invoke(
                {
                    "question": data["question"],
                    "history": data["history"],
                    "context": json.dumps(records, ensure_ascii=False),
                }
            )
        if self.settings.redact_pii:
            draft.answer = redact_pii(draft.answer)
            for source in sources:
                source.excerpt = redact_pii(source.excerpt)
        sources = validate_citations(draft, sources)
        # Re-check active versions before returning or persisting anything.
        current = {
            d["active_version"] for d in self.store.active(data["owner"], data["document_ids"])
        }
        if any(s.version_id not in current for s in sources):
            raise AppError(
                409, "documents_changed", "Documents changed while answering. Please ask again."
            )
        payload = {
            "status": draft.status,
            "answer": draft.answer,
            "sources": [s.model_dump() for s in sources],
        }
        return {
            **payload,
            "evidence_count": len(records),
            "owner": data["owner"],
            "session_id": data["session_id"],
            "question": data["question"],
            "history_message": AIMessage(content=draft.answer, additional_kwargs=payload),
        }

    def overview(self, owner, doc_id, request_id):
        start = time.monotonic()
        active = self.store.active(owner, [doc_id])
        if not active:
            raise AppError(409, "document_not_ready", "Wait until this document finishes indexing.")
        if active[0]["fingerprint"] != self.index.fingerprint:
            raise AppError(
                409, "reindex_required", "Re-upload this document with the current index settings."
            )
        key = hashlib.sha256(
            (
                self.settings.model
                + str(ANSWER_PROMPT)
                + str(self.settings.redact_pii)
                + str(self.settings.context_tokens)
                + str(self.settings.evidence_chars)
            ).encode()
        ).hexdigest()
        cached = self.store.summary(owner, doc_id, key)
        if cached:
            return {**cached, "request_id": request_id, "time_ms": 0, "cached": True}
        documents, total = self.index.overview(owner, active[0]["active_version"])
        documents = [doc for doc in documents if not detect_injection(doc.page_content)]
        try:
            result = self.generate(
                {
                    "owner": owner,
                    "session_id": "",
                    "document_ids": [doc_id],
                    "active": active,
                    "documents": documents,
                    "history": [],
                    "question": "Give an overview of the topics, key facts, requirements and exceptions in these passages. Cite the evidence. Do not claim coverage beyond these passages.",
                }
            )
        except httpx.TimeoutException:
            raise AppError(
                504, "model_timeout", "The overview took too long. Please retry."
            ) from None
        except (httpx.HTTPError, ResponseError):
            raise AppError(
                503, "model_unavailable", "The language model is currently unavailable."
            ) from None
        except (ValueError, ValidationError):
            raise AppError(
                502, "invalid_answer", "The model could not produce a validated overview."
            ) from None
        partial = result["evidence_count"] < total
        answer = result["answer"]
        if partial:
            answer = (
                f"Partial overview: based on {result['evidence_count']} of {total} indexed passages.\n\n"
                + answer
            )
        payload = {
            "status": result["status"],
            "answer": answer,
            "sources": result["sources"],
            "partial": partial,
            "passages_used": result["evidence_count"],
            "total_passages": total,
        }
        self.store.summary(owner, doc_id, key, payload)
        return {
            **payload,
            "request_id": request_id,
            "time_ms": int((time.monotonic() - start) * 1000),
            "cached": False,
        }

    def chat(self, owner, request, request_id):
        start = time.monotonic()
        question = validate_query(request.query, self.settings)
        self.store.session(owner, request.session_id)
        # Check ownership even if a request will be blocked or greeted.
        self.store.active(owner, request.document_ids)
        if detect_injection(question):
            result = {
                "status": "blocked",
                "answer": "I can help with questions about your documents, but cannot follow requests to override my instructions.",
                "sources": [],
            }
        elif question.lower().strip("!. ") in {"hi", "hello", "hey", "help", "what can you do"}:
            result = {
                "status": "answered",
                "answer": "Upload documents, select the sources you want to use, and ask a question. I'll answer with references or explain when the documents don't contain enough information.",
                "sources": [],
            }
        else:
            try:
                result = self.chain.invoke(
                    {
                        "question": question,
                        "owner": owner,
                        "document_ids": request.document_ids,
                        "session_id": request.session_id,
                    }
                )
            except AppError:
                raise
            except httpx.TimeoutException:
                raise AppError(
                    504, "model_timeout", "The model took too long to respond. Please retry."
                ) from None
            except (httpx.HTTPError, ResponseError):
                raise AppError(
                    503,
                    "model_unavailable",
                    "The language model is unavailable. Please try again shortly.",
                ) from None
            except (ValidationError, ValueError):
                raise AppError(
                    502,
                    "invalid_answer",
                    "The model returned an invalid answer. Try a more specific question.",
                ) from None
        return ChatResponse(
            request_id=request_id,
            session_id=request.session_id,
            status=result["status"],
            answer=result["answer"],
            sources=result["sources"],
            time_ms=int((time.monotonic() - start) * 1000),
        )
