from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Where a sub-question should be answered (architecture Stage 1 routing)."""

    RAG = "rag"
    WEB = "web"


class ResearchTask(BaseModel):
    """
    One answerable sub-question from Understand & Plan (architecture §3, Stage 1).

    Each task is routed independently to RAG, web, or both—not a separate 'objective' field.
    """

    id: str
    question: str
    sources: list[SourceType] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Structured output from Stage 1 — Understand & Plan """

    question_type: str
    requires_current_information: bool
    requires_internal_information: bool
    research_tasks: list[ResearchTask] = Field(default_factory=list)


class Evidence(BaseModel):
    """One normalized evidence item from RAG or web (architecture Stage 2)."""

    evidence_id: str
    research_id: str
    source_type: SourceType
    title: str
    content: str
    url: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AssessmentStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class EvidenceScores(BaseModel):
    relevance: ScoreLevel
    authority: ScoreLevel
    recency: ScoreLevel
    specificity: ScoreLevel
    corroboration: ScoreLevel


class EvaluatedEvidence(BaseModel):
    """Scored evidence item (architecture Stage 3)."""

    evidence_id: str
    research_id: str
    evaluation: EvidenceScores
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    confidence: ScoreLevel


class TaskEvaluatedEvidenceBatch(BaseModel):
    """Structured output: score all evidence for one research task in one LLM call."""

    items: list[EvaluatedEvidence] = Field(default_factory=list)


class ResearchAssessment(BaseModel):
    """Per-task sufficiency verdict (architecture Stage 3)."""

    research_id: str
    status: AssessmentStatus
    confidence: ScoreLevel
    evidence_gap: str | None = None


class KeyFinding(BaseModel):
    finding: str
    supporting_evidence: list[str] = Field(default_factory=list)
    research_id: str


class ConflictingEvidenceItem(BaseModel):
    claim: str
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)


class LimitationItem(BaseModel):
    research_id: str
    gap: str


class DraftReport(BaseModel):
    """Synthesis output (architecture Stage 4, sub-phase A)."""

    executive_summary: str
    key_findings: list[KeyFinding] = Field(default_factory=list)
    conflicting_evidence: list[ConflictingEvidenceItem] = Field(default_factory=list)
    limitations: list[LimitationItem] = Field(default_factory=list)
    conclusion: str


class ValidationIssue(BaseModel):
    type: str
    claim: str
    reason: str


class FinalReport(BaseModel):
    """User-facing report after validation passes (architecture Stage 4)."""

    research_question: str
    executive_summary: str
    key_findings: list[dict[str, Any]] = Field(default_factory=list)
    detailed_analysis: list[dict[str, Any]] = Field(default_factory=list)
    conflicting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[dict[str, Any]] = Field(default_factory=list)
    conclusion: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
