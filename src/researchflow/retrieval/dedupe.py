from typing import Any

from researchflow.state.models import Evidence


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    """
    Drop duplicate evidence (architecture Stage 2).

    Web: same URL. RAG: same document_id. Keeps first occurrence.
    """
    seen_urls: set[str] = set()
    seen_rag_chunks: set[tuple[str, str | None, str]] = set()
    unique: list[Evidence] = []

    for item in items:
        if item.source_type.value == "web" and item.url:
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
        if item.source_type.value == "rag" and item.document_id:
            page = item.metadata.get("page") if item.metadata else None
            snippet = item.content[:240]
            key = (item.document_id, str(page) if page is not None else None, snippet)
            if key in seen_rag_chunks:
                continue
            seen_rag_chunks.add(key)

        unique.append(item)

    return unique


def merge_evidence_state(
    existing: list[dict[str, Any]] | None,
    new_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    LangGraph reducer for `evidence`: concat then dedupe.

    Used when parallel `Send` branches each append one batch.
    """
    left = existing or []
    right = new_items or []
    if not right:
        return left
    parsed = [Evidence.model_validate(item) for item in left + right]
    return [item.model_dump(mode="json") for item in dedupe_evidence(parsed)]
