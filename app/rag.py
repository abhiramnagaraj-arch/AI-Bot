import json
import time

import httpx
from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import ValidationError

from .guardrails import (
    AnswerDraft,
    AppError,
    ChatResponse,
    Rewrite,
    Source,
    detect_injection,
    redact_pii,
    validate_citations,
    validate_query,
)
from .prompts import ANSWER_PROMPT, REWRITE_PROMPT

NO_EVIDENCE = "I couldn't find enough evidence in the selected documents to answer that. Try selecting another document or asking a more specific question."


def bounded_history(messages, char_budget):
    selected, used = [], 0
    # Keep complete user/assistant pairs and strip stored citation metadata from prompts.

    for i in range(len(messages) - 2, -1, -2):
        pair = messages[i : i + 2]
        if pair[1].additional_kwargs.get("status") == "blocked":
            continue
        cost = sum(len(str(m.content).encode("utf-8")) for m in pair)
        if used + cost > char_budget:
            break
        selected[0:0] = [HumanMessage(str(pair[0].content)), AIMessage(str(pair[1].content))]
        used += cost
    return selected


class RagService:
    def __init__(self, settings, store, index, model=None):
        model = model or ChatOllama(
            model=settings.model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=settings.context_tokens,
            num_predict=settings.output_tokens,
            client_kwargs={"timeout": settings.model_timeout},
        )
        self.settings, self.store, self.index = settings, store, index
        # Only transport errors retry; invalid answers never silently become successful text.
        answer_model = model.with_structured_output(AnswerDraft).with_retry(
            retry_if_exception_type=(httpx.ConnectError,), stop_after_attempt=2
        )
        self.answer_chain = ANSWER_PROMPT | answer_model
        self.rewrite_chain = REWRITE_PROMPT | model.with_structured_output(Rewrite)
        self.chain = (
            RunnableLambda(self.prepare, name="resolve_question")
            | RunnableLambda(self.retrieve, name="retrieve_evidence")
            | RunnableLambda(self.generate, name="grounded_answer")
        )

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
        versions = [d["active_version"] for d in data["active"]]
        matches = self.index.search(data["search_question"], data["owner"], versions)
        documents, seen = [], set()
        for doc, score in matches:
            text = doc.page_content.strip()
            if (
                doc.metadata.get("owner") != data["owner"]
                or doc.metadata.get("version_id") not in versions
                or text in seen
                or detect_injection(text)
            ):
                continue
            seen.add(text)
            documents.append(doc)
            if len(documents) == self.settings.top_k:
                break
        data["documents"] = documents
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
        return payload

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
        elif question.lower().strip("?!. ") in {"hi", "hello", "hey", "help", "what can you do"}:
            result = {
                "status": "answered",
                "answer": "Upload documents, select the sources you want to use, and ask a question. I'll answer with references or explain when the documents don't contain enough information.",
                "sources": [],
            }
        else:
            try:
                result = self.chain.invoke(
                    {
                        "history": messages_from_dict(
                            self.store.messages(owner, request.session_id)
                        ),
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
        self.store.append_messages(
            owner,
            request.session_id,
            messages_to_dict(
                [
                    HumanMessage(question),
                    AIMessage(
                        content=result["answer"],
                        additional_kwargs={
                            "status": result["status"],
                            "sources": result["sources"],
                        },
                    ),
                ]
            ),
        )
        return ChatResponse(
            request_id=request_id,
            session_id=request.session_id,
            status=result["status"],
            answer=result["answer"],
            sources=result["sources"],
            time_ms=int((time.monotonic() - start) * 1000),
        )
