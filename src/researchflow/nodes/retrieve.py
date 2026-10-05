import logging
from typing import Any, TypedDict

from langgraph.types import Send

logger = logging.getLogger(__name__)

from researchflow.retrieval import dedupe_evidence, retrieve_rag, retrieve_web
from researchflow.retrieval.cap import cap_evidence
from researchflow.retrieval.query import retrieval_query_for_task
from researchflow.state.models import (
    AssessmentStatus,
    Evidence,
    ResearchAssessment,
    ResearchTask,
    SourceType,
)
from researchflow.observability import log_event
from researchflow.state.research_state import ResearchState


class RetrieveJob(TypedDict):
    """Payload for one parallel retrieval (one task + one source)."""

    task: dict[str, Any]
    source: str
    sequence: int
    query: str
    preferred_rag_documents: list[str]


def _preferred_documents(state: ResearchState) -> list[str]:
    return list(state.get("preferred_rag_documents") or [])


def _build_retrieval_jobs(state: ResearchState) -> list[RetrieveJob]:
    jobs: list[RetrieveJob] = []
    sequence = 1
    preferred = _preferred_documents(state)
    for task_data in state["research_tasks"]:
        task = ResearchTask.model_validate(task_data)
        query = retrieval_query_for_task(state, task)
        for source in task.sources:
            jobs.append(
                RetrieveJob(
                    task=task.model_dump(mode="json"),
                    source=source.value,
                    sequence=sequence,
                    query=query,
                    preferred_rag_documents=preferred,
                )
            )
            sequence += 1
    return jobs


def _next_sequence(state: ResearchState) -> int:
    return len(state["evidence"]) + 1


def _jobs_for_tasks(
    state: ResearchState,
    tasks: list[ResearchTask],
    *,
    sequence_start: int,
) -> tuple[list[RetrieveJob], int]:
    jobs: list[RetrieveJob] = []
    sequence = sequence_start
    preferred = _preferred_documents(state)
    for task in tasks:
        query = retrieval_query_for_task(state, task)
        for source in task.sources:
            jobs.append(
                RetrieveJob(
                    task=task.model_dump(mode="json"),
                    source=source.value,
                    sequence=sequence,
                    query=query,
                    preferred_rag_documents=preferred,
                )
            )
            sequence += 1
    return jobs, sequence


def fan_out_retrieval(state: ResearchState) -> list[Send] | str:
    """
    Map each (task, source) to a `retrieve_one` worker (Phase 2b).

    Returns Send list for parallel runs, or routes to finalize when there is nothing to fetch.
    """
    jobs = _build_retrieval_jobs(state)
    if not jobs:
        return "finalize_retrieval"
    return [Send("retrieve_one", job) for job in jobs]


def fan_out_retrieval_retry(state: ResearchState) -> list[Send] | str:
    """
    Re-fetch only insufficient tasks that were scheduled (retrieval_attempts >= 1).
    """
    by_id = {a["research_id"]: a for a in state["research_assessment"]}
    retry_tasks: list[ResearchTask] = []
    for task_data in state["research_tasks"]:
        task = ResearchTask.model_validate(task_data)
        assessment = by_id.get(task.id)
        if not assessment:
            continue
        if ResearchAssessment.model_validate(assessment).status != AssessmentStatus.INSUFFICIENT:
            continue
        if state["retrieval_attempts"].get(task.id, 0) < 1:
            continue
        retry_tasks.append(task)

    jobs, _ = _jobs_for_tasks(state, retry_tasks, sequence_start=_next_sequence(state))
    if not jobs:
        return "synthesize_report"
    return [Send("retrieve_one", job) for job in jobs]


def retrieve_one(job: RetrieveJob) -> dict:
    """Worker: RAG or web retrieval for a single job."""
    task = ResearchTask.model_validate(job["task"])
    source = SourceType(job["source"])
    sequence = job["sequence"]
    query = job["query"]

    try:
        if source == SourceType.WEB:
            items = retrieve_web(task, sequence=sequence, query=query)
        else:
            items = retrieve_rag(
                task,
                sequence=sequence,
                query=query,
                preferred_documents=job.get("preferred_rag_documents") or [],
            )
    except Exception:
        logger.exception(
            "Retrieval failed for task=%s source=%s",
            task.id,
            source.value,
        )
        items = []

    return {
        "evidence": [item.model_dump(mode="json") for item in items],
    }


def finalize_retrieval(state: ResearchState) -> dict:
    """
    Sync point after parallel workers; ensures evidence is deduped once more.

    Safe no-op when there were zero jobs (empty research_tasks).
    """
    log_event(logger, "node_start", node="finalize_retrieval", raw_evidence=len(state["evidence"]))
    parsed = [Evidence.model_validate(item) for item in state["evidence"]]
    unique = dedupe_evidence(parsed)
    capped = cap_evidence(
        unique,
        preferred_documents=_preferred_documents(state),
    )
    log_event(
        logger,
        "node_complete",
        node="finalize_retrieval",
        deduped=len(unique),
        capped=len(capped),
    )
    return {
        "evidence": [item.model_dump(mode="json") for item in capped],
    }


def retrieve_evidence(state: ResearchState) -> dict:
    """
    Sequential retrieval (Phase 2 style). Kept for tests; the graph uses fan_out + Send.
    """
    collected: list[Evidence] = []
    sequence = 1
    preferred = _preferred_documents(state)

    for task_data in state["research_tasks"]:
        task = ResearchTask.model_validate(task_data)
        query = retrieval_query_for_task(state, task)
        for source in task.sources:
            if source == SourceType.WEB:
                collected.extend(retrieve_web(task, sequence=sequence, query=query))
            else:
                collected.extend(
                    retrieve_rag(
                        task,
                        sequence=sequence,
                        query=query,
                        preferred_documents=preferred,
                    )
                )
            sequence += 1

    unique = dedupe_evidence(collected)
    capped = cap_evidence(unique, preferred_documents=preferred)
    return {
        "evidence": [item.model_dump(mode="json") for item in capped],
    }
