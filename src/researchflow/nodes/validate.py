from researchflow.state.models import (
    AssessmentStatus,
    DraftReport,
    Evidence,
    FinalReport,
    KeyFinding,
    ResearchAssessment,
    ValidationIssue,
)
from researchflow.state.research_state import ResearchState


def _build_sources(state: ResearchState) -> list[dict]:
    sources: list[dict] = []
    for raw in state["evidence"]:
        item = Evidence.model_validate(raw)
        entry: dict = {
            "evidence_id": item.evidence_id,
            "title": item.title,
            "source_type": item.source_type.value,
        }
        if item.url:
            entry["url"] = item.url
        if item.document_id:
            entry["document_id"] = item.document_id
        sources.append(entry)
    return sources


def _build_detailed_analysis(
    findings: list[KeyFinding],
    evidence_by_id: dict[str, Evidence],
) -> list[dict]:
    analysis: list[dict] = []
    for finding in findings:
        snippets = []
        for eid in finding.supporting_evidence:
            item = evidence_by_id.get(eid)
            if item:
                snippets.append({"evidence_id": eid, "content": item.content[:500]})
        analysis.append(
            {
                "research_id": finding.research_id,
                "finding": finding.finding,
                "supporting_evidence": finding.supporting_evidence,
                "evidence_snippets": snippets,
            }
        )
    return analysis


def _validate_draft(
    draft: DraftReport,
    assessments: list[ResearchAssessment],
    valid_evidence_ids: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for finding in draft.key_findings:
        if not finding.supporting_evidence:
            issues.append(
                ValidationIssue(
                    type="missing_citation",
                    claim=finding.finding,
                    reason="Key finding has no supporting_evidence ids",
                )
            )
            continue
        for eid in finding.supporting_evidence:
            if eid not in valid_evidence_ids:
                issues.append(
                    ValidationIssue(
                        type="invalid_citation",
                        claim=finding.finding,
                        reason=f"Unknown evidence_id: {eid}",
                    )
                )

    limitation_ids = {lim.research_id for lim in draft.limitations}
    for assessment in assessments:
        if assessment.status in (
            AssessmentStatus.INSUFFICIENT,
            AssessmentStatus.PARTIAL,
        ):
            if assessment.research_id not in limitation_ids:
                issues.append(
                    ValidationIssue(
                        type="missing_gap_disclosure",
                        claim=assessment.research_id,
                        reason=(
                            f"Task {assessment.research_id} is {assessment.status.value} "
                            "but draft limitations omit it"
                        ),
                    )
                )

    return issues


def validate_report(state: ResearchState) -> dict:
    """Stage 4 sub-phase B: rule-based validation, then final_report if passed."""
    draft = DraftReport.model_validate(state["draft_report"])
    assessments = [
        ResearchAssessment.model_validate(a) for a in state["research_assessment"]
    ]
    evidence_by_id = {
        Evidence.model_validate(raw).evidence_id: Evidence.model_validate(raw)
        for raw in state["evidence"]
    }
    valid_ids = set(evidence_by_id.keys())

    issues = _validate_draft(draft, assessments, valid_ids)
    issue_payload = [i.model_dump(mode="json") for i in issues]

    if issues:
        return {
            "validation_status": "failed",
            "validation_issues": issue_payload,
            "final_report": {},
        }

    final = FinalReport(
        research_question=state["user_question"],
        executive_summary=draft.executive_summary,
        key_findings=[f.model_dump(mode="json") for f in draft.key_findings],
        detailed_analysis=_build_detailed_analysis(draft.key_findings, evidence_by_id),
        conflicting_evidence=[c.model_dump(mode="json") for c in draft.conflicting_evidence],
        limitations=[lim.model_dump(mode="json") for lim in draft.limitations],
        conclusion=draft.conclusion,
        sources=_build_sources(state),
    )
    return {
        "validation_status": "passed",
        "validation_issues": [],
        "final_report": final.model_dump(mode="json"),
    }
