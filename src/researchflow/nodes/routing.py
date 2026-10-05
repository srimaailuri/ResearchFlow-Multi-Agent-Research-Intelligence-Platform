from langgraph.graph import END

from researchflow.config.settings import get_settings
from researchflow.state.models import AssessmentStatus, ResearchAssessment, ResearchTask
from researchflow.state.research_state import ResearchState


def _tasks_eligible_for_retrieval_retry(state: ResearchState) -> list[ResearchTask]:
    max_retries = get_settings().max_retrieval_attempts
    attempts = state["retrieval_attempts"]
    by_id = {a["research_id"]: a for a in state["research_assessment"]}
    eligible: list[ResearchTask] = []

    for task_data in state["research_tasks"]:
        task = ResearchTask.model_validate(task_data)
        assessment = by_id.get(task.id)
        if not assessment:
            continue
        if ResearchAssessment.model_validate(assessment).status != AssessmentStatus.INSUFFICIENT:
            continue
        if attempts.get(task.id, 0) < max_retries:
            eligible.append(task)
    return eligible


def route_after_evaluate(state: ResearchState) -> str:
    """Retry retrieval for insufficient tasks, else continue to synthesis."""
    if _tasks_eligible_for_retrieval_retry(state):
        return "schedule_retrieval_retry"
    return "synthesize_report"


def schedule_retrieval_retry(state: ResearchState) -> dict:
    """Bump per-task retrieval_attempts before re-fetching."""
    eligible = _tasks_eligible_for_retrieval_retry(state)
    updates = dict(state["retrieval_attempts"])
    for task in eligible:
        updates[task.id] = updates.get(task.id, 0) + 1
    return {"retrieval_attempts": updates}


def route_after_validate(state: ResearchState) -> str:
    """Retry synthesis with validation feedback, else finish."""
    if state["validation_status"] == "passed":
        return END
    max_retries = get_settings().max_synthesis_attempts
    if state["synthesis_attempts"] < max_retries:
        return "increment_synthesis_attempt"
    return END


def increment_synthesis_attempt(state: ResearchState) -> dict:
    return {"synthesis_attempts": state["synthesis_attempts"] + 1}
