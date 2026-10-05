"""In-process research job queue (async API)."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class ResearchJobRecord:
    job_id: str
    question: str
    status: JobStatus
    created_at: float
    include_full_state: bool = False
    preferred_rag_documents: list[str] = field(default_factory=list)
    started_at: float | None = None
    completed_at: float | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    full_state: dict[str, Any] | None = None
    run_id: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        question: str,
        *,
        include_full_state: bool,
        preferred_rag_documents: list[str] | None = None,
    ) -> ResearchJobRecord:
        job = ResearchJobRecord(
            job_id=str(uuid.uuid4()),
            question=question,
            status="queued",
            created_at=time.time(),
            include_full_state=include_full_state,
            preferred_rag_documents=list(preferred_rag_documents or []),
        )
        async with self._lock:
            self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: str) -> ResearchJobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_recent(self, limit: int = 20) -> list[ResearchJobRecord]:
        async with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()

    async def mark_completed(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        full_state: dict[str, Any] | None,
        elapsed: float,
        run_id: str | None = None,
    ) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.completed_at = time.time()
            job.elapsed_seconds = round(elapsed, 2)
            job.result = result
            job.full_state = full_state
            job.run_id = run_id

    async def mark_failed(self, job_id: str, error: str, elapsed: float) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.completed_at = time.time()
            job.elapsed_seconds = round(elapsed, 2)
            job.error = error


def job_to_payload(job: ResearchJobRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "run_id": job.run_id,
        "question": job.question,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "elapsed_seconds": job.elapsed_seconds,
        "error": job.error,
    }
    if job.result is not None:
        payload["result"] = job.result
    if job.include_full_state and job.full_state is not None:
        payload["full_state"] = job.full_state
    return payload
