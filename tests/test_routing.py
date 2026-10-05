from researchflow.nodes.routing import route_after_evaluate, route_after_validate, schedule_retrieval_retry
from researchflow.state import (
    AssessmentStatus,
    ResearchAssessment,
    ResearchTask,
    ScoreLevel,
    SourceType,
    initial_research_state,
)


def _state_with_rq1_insufficient() -> dict:
    s = initial_research_state("q")
    s["research_tasks"] = [
        ResearchTask(id="RQ1", question="q", sources=[SourceType.WEB]).model_dump(mode="json")
    ]
    s["research_assessment"] = [
        ResearchAssessment(
            research_id="RQ1",
            status=AssessmentStatus.INSUFFICIENT,
            confidence=ScoreLevel.LOW,
            evidence_gap="none",
        ).model_dump(mode="json")
    ]
    return s


def test_route_after_evaluate_retries_when_insufficient() -> None:
    assert route_after_evaluate(_state_with_rq1_insufficient()) == "schedule_retrieval_retry"


def test_route_after_evaluate_synthesizes_when_sufficient() -> None:
    s = _state_with_rq1_insufficient()
    s["research_assessment"] = [
        ResearchAssessment(
            research_id="RQ1",
            status=AssessmentStatus.SUFFICIENT,
            confidence=ScoreLevel.MEDIUM,
        ).model_dump(mode="json")
    ]
    assert route_after_evaluate(s) == "synthesize_report"


def test_schedule_retrieval_retry_increments_attempts() -> None:
    s = _state_with_rq1_insufficient()
    out = schedule_retrieval_retry(s)
    assert out["retrieval_attempts"]["RQ1"] == 1


def test_route_after_validate_passed_ends() -> None:
    from langgraph.graph import END

    s = initial_research_state("q")
    s["validation_status"] = "passed"
    assert route_after_validate(s) == END
