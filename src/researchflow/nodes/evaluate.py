import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage

from researchflow.config.settings import get_settings
from researchflow.llm import get_evaluate_chat_model, invoke_with_retry
from researchflow.observability import log_event
from researchflow.state.models import (
    AssessmentStatus,
    EvaluatedEvidence,
    Evidence,
    EvidenceScores,
    ResearchAssessment,
    ResearchTask,
    ScoreLevel,
    TaskEvaluatedEvidenceBatch,
)
from researchflow.state.research_state import ResearchState

EVALUATE_SYSTEM_PROMPT = """You are the evaluation stage of a research agent.

Score every evidence item for one research sub-question in a single response. Rules:
1. Return exactly one EvaluatedEvidence entry per input evidence item (same evidence_id and research_id).
2. Score relevance, authority, recency, specificity, corroboration as high, medium, or low.
3. List supports as short claim ids derived from the content (e.g. claim_<evidence_id>_1).
4. Use contradicts only when the content clearly conflicts with the sub-question.
5. Set confidence from how well the snippet answers the sub-question.
6. Do not invent facts beyond the evidence content."""

logger = logging.getLogger(__name__)


def _normalize_batch(
    batch: TaskEvaluatedEvidenceBatch,
    task: ResearchTask,
    items: list[Evidence],
) -> list[EvaluatedEvidence]:
    by_id = {row.evidence_id: row for row in items}
    normalized: list[EvaluatedEvidence] = []
    seen: set[str] = set()

    for entry in batch.items:
        source = by_id.get(entry.evidence_id)
        if not source:
            continue
        normalized.append(
            entry.model_copy(
                update={
                    "evidence_id": source.evidence_id,
                    "research_id": task.id,
                }
            )
        )
        seen.add(source.evidence_id)

    for item in items:
        if item.evidence_id in seen:
            continue
        normalized.append(
            EvaluatedEvidence(
                evidence_id=item.evidence_id,
                research_id=task.id,
                evaluation=EvidenceScores(
                    relevance=ScoreLevel.LOW,
                    authority=ScoreLevel.LOW,
                    recency=ScoreLevel.MEDIUM,
                    specificity=ScoreLevel.LOW,
                    corroboration=ScoreLevel.LOW,
                ),
                supports=[],
                contradicts=[],
                confidence=ScoreLevel.LOW,
            )
        )

    return normalized


def _score_task_evidence(task: ResearchTask, items: list[Evidence]) -> list[EvaluatedEvidence]:
    if not items:
        return []

    llm = get_evaluate_chat_model().with_structured_output(TaskEvaluatedEvidenceBatch)
    payload = {
        "research_id": task.id,
        "research_sub_question": task.question,
        "evidence": [item.model_dump(mode="json") for item in items],
    }
    batch: TaskEvaluatedEvidenceBatch = invoke_with_retry(
        llm,
        [
            SystemMessage(content=EVALUATE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ],
    )
    return _normalize_batch(batch, task, items)


def _assess_task(
    task: ResearchTask,
    evaluated_for_task: list[EvaluatedEvidence],
) -> ResearchAssessment:
    """Roll up LLM scores into one verdict per research task."""
    if not evaluated_for_task:
        return ResearchAssessment(
            research_id=task.id,
            status=AssessmentStatus.INSUFFICIENT,
            confidence=ScoreLevel.LOW,
            evidence_gap=f"No evidence collected for task {task.id}",
        )

    relevance = [entry.evaluation.relevance for entry in evaluated_for_task]
    if all(level == ScoreLevel.LOW for level in relevance):
        return ResearchAssessment(
            research_id=task.id,
            status=AssessmentStatus.INSUFFICIENT,
            confidence=ScoreLevel.LOW,
            evidence_gap=f"Evidence for task {task.id} is not relevant enough",
        )

    if any(level == ScoreLevel.HIGH for level in relevance):
        return ResearchAssessment(
            research_id=task.id,
            status=AssessmentStatus.SUFFICIENT,
            confidence=ScoreLevel.MEDIUM,
            evidence_gap=None,
        )

    return ResearchAssessment(
        research_id=task.id,
        status=AssessmentStatus.PARTIAL,
        confidence=ScoreLevel.MEDIUM,
        evidence_gap=f"Task {task.id} has limited relevant evidence",
    )


def evaluate_and_assess(state: ResearchState) -> dict:
    """
    Stage 3: batch LLM scoring per research task, tasks evaluated in parallel.

    One API call scores all evidence for a task (vs one call per evidence row).
    """
    log_event(
        logger,
        "node_start",
        node="evaluate_and_assess",
        evidence_count=len(state["evidence"]),
    )
    tasks = [ResearchTask.model_validate(t) for t in state["research_tasks"]]
    evidence_by_task: dict[str, list[Evidence]] = {task.id: [] for task in tasks}
    for raw in state["evidence"]:
        item = Evidence.model_validate(raw)
        if item.research_id in evidence_by_task:
            evidence_by_task[item.research_id].append(item)

    settings = get_settings()
    workers = max(1, min(settings.max_evaluate_workers, len(tasks) or 1))
    evaluated: list[EvaluatedEvidence] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_score_task_evidence, task, evidence_by_task[task.id]): task
            for task in tasks
            if evidence_by_task[task.id]
        }
        for future in as_completed(futures):
            evaluated.extend(future.result())

    by_task: dict[str, list[EvaluatedEvidence]] = {}
    for entry in evaluated:
        by_task.setdefault(entry.research_id, []).append(entry)

    assessments: list[ResearchAssessment] = []
    for task in tasks:
        assessments.append(_assess_task(task, by_task.get(task.id, [])))

    log_event(
        logger,
        "node_complete",
        node="evaluate_and_assess",
        evaluated_count=len(evaluated),
        assessment_count=len(assessments),
    )
    return {
        "evaluated_evidence": [e.model_dump(mode="json") for e in evaluated],
        "research_assessment": [a.model_dump(mode="json") for a in assessments],
    }
