from researchflow.nodes.validate import validate_report
from researchflow.state import (
    AssessmentStatus,
    DraftReport,
    Evidence,
    KeyFinding,
    ResearchAssessment,
    ScoreLevel,
    SourceType,
    initial_research_state,
)


def test_validate_passes_with_valid_citations() -> None:
    s = initial_research_state("What is RAG?")
    s["evidence"] = [
        Evidence(
            evidence_id="E1",
            research_id="RQ1",
            source_type=SourceType.RAG,
            title="doc",
            document_id="d1",
            content="Retrieval augmented generation combines search with LLMs.",
        ).model_dump(mode="json")
    ]
    s["research_assessment"] = [
        ResearchAssessment(
            research_id="RQ1",
            status=AssessmentStatus.SUFFICIENT,
            confidence=ScoreLevel.MEDIUM,
        ).model_dump(mode="json")
    ]
    s["draft_report"] = DraftReport(
        executive_summary="Summary",
        key_findings=[
            KeyFinding(
                finding="RAG uses retrieval",
                supporting_evidence=["E1"],
                research_id="RQ1",
            )
        ],
        conclusion="Done",
    ).model_dump(mode="json")

    result = validate_report(s)
    assert result["validation_status"] == "passed"
    assert result["final_report"]["research_question"] == "What is RAG?"


def test_validate_fails_on_invalid_citation() -> None:
    s = initial_research_state("q")
    s["evidence"] = []
    s["research_assessment"] = []
    s["draft_report"] = DraftReport(
        executive_summary="s",
        key_findings=[
            KeyFinding(finding="claim", supporting_evidence=["E99"], research_id="RQ1")
        ],
        conclusion="c",
    ).model_dump(mode="json")

    result = validate_report(s)
    assert result["validation_status"] == "failed"
    assert result["validation_issues"]
