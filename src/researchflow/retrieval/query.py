from researchflow.retrieval.boost import normalize_preferred_documents
from researchflow.state.models import ResearchTask
from researchflow.state.research_state import ResearchState


def retrieval_query_for_task(state: ResearchState, task: ResearchTask) -> str:
    """
    Widen phrasing on retrieval retries (architecture Stage 2 retry behavior).
    """
    query = task.question
    attempts = state["retrieval_attempts"].get(task.id, 0)
    if attempts > 0:
        query = (
            f"{query} "
            "Use alternative keywords, synonyms, and related technical terms."
        )

    preferred = normalize_preferred_documents(state.get("preferred_rag_documents") or [])
    if preferred:
        labels = ", ".join(preferred)
        query = f"{query} Prefer internal document sources when relevant: {labels}."
    return query
