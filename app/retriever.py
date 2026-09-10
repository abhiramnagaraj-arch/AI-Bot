from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .guardrails import detect_injection


class ScopedRetriever(BaseRetriever):
    """A LangChain retriever whose scope is supplied by trusted server state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: object
    owner: str
    versions: list[str]
    top_k: int

    def _get_relevant_documents(self, query, *, run_manager):
        matches = self.index.search(query, self.owner, self.versions)
        result, seen = [], set()
        for document, score in matches:
            metadata = document.metadata
            # Defense in depth even if a vector provider ignores its filter.
            if (
                metadata.get("owner") != self.owner
                or metadata.get("version_id") not in self.versions
            ):
                continue
            text = document.page_content.strip()
            if text in seen or detect_injection(text):
                continue
            seen.add(text)
            result.append(document)
            if len(result) >= self.top_k:
                break
        return result
