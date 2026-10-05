"""HTTP API for the ResearchFlow LangGraph pipeline."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from researchflow.api.jobs import JobStore, job_to_payload
from researchflow.config.settings import get_settings
from researchflow.graph.bootstrap import setup_research_graph, shutdown_research_graph
from researchflow.graph.invoke import graph_invoke_config, invoke_research_graph
from researchflow.observability import get_request_id, log_context
from researchflow.state import initial_research_state

_graph = None
_run_semaphore: asyncio.Semaphore | None = None
_job_store = JobStore()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _graph, _run_semaphore
    settings = get_settings()
    _run_semaphore = asyncio.Semaphore(settings.max_concurrent_research_jobs)
    _graph = setup_research_graph(settings)

    async def _warm_rag_index() -> None:
        try:
            from researchflow.retrieval.rag import _get_vectorstore

            await asyncio.to_thread(_get_vectorstore)
        except Exception:
            pass

    await _warm_rag_index()
    yield
    shutdown_research_graph()


app = FastAPI(
    title="ResearchFlow API",
    description="Run the multi-agent research pipeline (plan → retrieve → evaluate → synthesize → validate).",
    version="0.1.0",
    lifespan=lifespan,
)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=8000)
    preferred_rag_documents: list[str] = Field(
        default_factory=list,
        description="Optional PDF filenames or stems to boost in RAG (e.g. benchmark source_documents).",
    )


class ResearchResponse(BaseModel):
    run_id: str
    question: str
    validation_status: str
    final_report: dict[str, Any]
    draft_report: dict[str, Any]
    research_tasks: list[dict[str, Any]]
    evidence_count: int
    evidence_by_source: dict[str, int]
    research_assessment: list[dict[str, Any]]
    retrieval_attempts: dict[str, int]
    synthesis_attempts: int
    validation_issues: list[dict[str, Any]]
    elapsed_seconds: float
    full_state: dict[str, Any] | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    question: str


class RunCheckpointResponse(BaseModel):
    run_id: str
    next: list[str]
    values: dict[str, Any]


class JobStatusResponse(BaseModel):
    job_id: str
    run_id: str | None = None
    question: str
    status: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    full_state: dict[str, Any] | None = None


def _summarize_state(
    question: str,
    state: dict[str, Any],
    elapsed: float,
    *,
    run_id: str,
) -> ResearchResponse:
    evidence = state.get("evidence") or []
    by_source: dict[str, int] = {}
    for row in evidence:
        st = str(row.get("source_type", "unknown"))
        by_source[st] = by_source.get(st, 0) + 1

    return ResearchResponse(
        run_id=run_id,
        question=question,
        validation_status=state.get("validation_status") or "",
        final_report=state.get("final_report") or {},
        draft_report=state.get("draft_report") or {},
        research_tasks=state.get("research_tasks") or [],
        evidence_count=len(evidence),
        evidence_by_source=by_source,
        research_assessment=state.get("research_assessment") or [],
        retrieval_attempts=state.get("retrieval_attempts") or {},
        synthesis_attempts=int(state.get("synthesis_attempts") or 0),
        validation_issues=state.get("validation_issues") or [],
        elapsed_seconds=round(elapsed, 2),
        full_state=None,
    )


async def _invoke_graph(
    question: str,
    *,
    preferred_rag_documents: list[str] | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    if _graph is None or _run_semaphore is None:
        raise RuntimeError("Research graph is not initialized")
    async with _run_semaphore:
        return await asyncio.to_thread(
            invoke_research_graph,
            _graph,
            initial_research_state(
                question,
                preferred_rag_documents=preferred_rag_documents,
            ),
            run_id=run_id,
            request_id=get_request_id(),
        )


async def _run_job(job_id: str) -> None:
    job = await _job_store.get(job_id)
    if job is None:
        return

    await _job_store.mark_running(job_id)
    started = time.perf_counter()
    try:
        state, run_id = await _invoke_graph(
            job.question,
            preferred_rag_documents=job.preferred_rag_documents,
            run_id=job.job_id,
        )
        elapsed = time.perf_counter() - started
        summary = _summarize_state(job.question, state, elapsed, run_id=run_id)
        result = summary.model_dump(mode="json")
        full_state = state if job.include_full_state else None
        await _job_store.mark_completed(
            job_id,
            result=result,
            full_state=full_state,
            elapsed=elapsed,
            run_id=run_id,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        await _job_store.mark_failed(job_id, str(exc), elapsed)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    with log_context(request_id=request_id):
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "graph_loaded": str(_graph is not None)}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ready" if _graph is not None else "starting",
        "graph_loaded": _graph is not None,
        "max_concurrent_research_jobs": settings.max_concurrent_research_jobs,
    }


@app.post("/research/jobs", response_model=JobCreateResponse, status_code=202)
async def create_research_job(
    body: ResearchRequest,
    background_tasks: BackgroundTasks,
    full_state: bool = Query(False, description="Include full LangGraph state when job completes"),
) -> JobCreateResponse:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Research graph is not initialized")

    job = await _job_store.create(
        body.question,
        include_full_state=full_state,
        preferred_rag_documents=body.preferred_rag_documents,
    )
    background_tasks.add_task(_run_job, job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status, question=job.question)


@app.get("/research/jobs/{job_id}", response_model=JobStatusResponse)
async def get_research_job(job_id: str) -> JobStatusResponse:
    job = await _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job_to_payload(job))


@app.get("/research/jobs")
async def list_research_jobs(limit: int = Query(20, ge=1, le=100)) -> list[JobStatusResponse]:
    jobs = await _job_store.list_recent(limit=limit)
    return [JobStatusResponse(**job_to_payload(job)) for job in jobs]


@app.get("/research/runs/{run_id}", response_model=RunCheckpointResponse)
async def get_research_run_checkpoint(run_id: str) -> RunCheckpointResponse:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Research graph is not initialized")
    settings = get_settings()
    if settings.checkpointer_mode.strip().lower() in {"none", "off", "disabled"}:
        raise HTTPException(status_code=501, detail="Checkpointer is disabled")

    snapshot = await asyncio.to_thread(_graph.get_state, graph_invoke_config(run_id))
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunCheckpointResponse(
        run_id=run_id,
        next=list(snapshot.next or ()),
        values=dict(snapshot.values),
    )


@app.post("/research/run", response_model=ResearchResponse)
async def run_research(
    body: ResearchRequest,
    full_state: bool = Query(False, description="Include complete LangGraph state in the response"),
) -> ResearchResponse:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Research graph is not initialized")

    started = time.perf_counter()
    try:
        state, run_id = await _invoke_graph(
            body.question,
            preferred_rag_documents=body.preferred_rag_documents,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - started
    response = _summarize_state(body.question, state, elapsed, run_id=run_id)
    if full_state:
        response = response.model_copy(update={"full_state": state})
    return response


def main() -> None:
    import uvicorn

    uvicorn.run(
        "researchflow.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
