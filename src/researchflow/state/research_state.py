from typing import Annotated, Any, TypedDict

from researchflow.retrieval.dedupe import merge_evidence_state


class ResearchState(TypedDict):
    """
    Shared LangGraph state.

    Nested objects are stored as plain dicts (from Pydantic `.model_dump(mode="json")`)
    until we add more models in later phases.
    """

    user_question: str
    question_type: str
    requires_current_information: bool
    requires_internal_information: bool
    research_tasks: list[dict[str, Any]]
    evidence: Annotated[list[dict[str, Any]], merge_evidence_state]
    evaluated_evidence: list[dict[str, Any]]
    research_assessment: list[dict[str, Any]]
    draft_report: dict[str, Any]
    validation_status: str
    validation_issues: list[dict[str, Any]]
    final_report: dict[str, Any]
    retrieval_attempts: dict[str, int]
    synthesis_attempts: int
    preferred_rag_documents: list[str]
    run_id: str


def initial_research_state(
    user_question: str,
    *,
    preferred_rag_documents: list[str] | None = None,
) -> ResearchState:
    """Build a valid starting state before any node runs (Phase 0 — mostly empty)."""
    return ResearchState(
        user_question=user_question,
        preferred_rag_documents=list(preferred_rag_documents or []),
        question_type="",
        requires_current_information=False,
        requires_internal_information=False,
        research_tasks=[],
        evidence=[],
        evaluated_evidence=[],
        research_assessment=[],
        draft_report={},
        validation_status="",
        validation_issues=[],
        final_report={},
        retrieval_attempts={},
        synthesis_attempts=0,
        run_id="",
    )
