from researchflow.state.models import (
    AssessmentStatus,
    ConflictingEvidenceItem,
    DraftReport,
    EvaluatedEvidence,
    Evidence,
    EvidenceScores,
    FinalReport,
    KeyFinding,
    LimitationItem,
    ResearchAssessment,
    ResearchPlan,
    ResearchTask,
    ScoreLevel,
    SourceType,
    ValidationIssue,
)

__all__ = [
    "ResearchState",
    "AssessmentStatus",
    "EvaluatedEvidence",
    "Evidence",
    "EvidenceScores",
    "ConflictingEvidenceItem",
    "DraftReport",
    "FinalReport",
    "KeyFinding",
    "LimitationItem",
    "ResearchAssessment",
    "ValidationIssue",
    "ScoreLevel",
    "ResearchPlan",
    "ResearchTask",
    "SourceType",
    "initial_research_state",
]


def __getattr__(name: str):
    if name == "ResearchState":
        from researchflow.state.research_state import ResearchState

        return ResearchState
    if name == "initial_research_state":
        from researchflow.state.research_state import initial_research_state

        return initial_research_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
