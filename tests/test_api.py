import importlib
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    api_module = importlib.import_module("researchflow.api.app")

    async def fake_invoke(question: str, *, preferred_rag_documents=None, run_id=None):
        rid = run_id or "test-run-id"
        return (
            {
                "user_question": question,
                "run_id": rid,
                "validation_status": "passed",
                "final_report": {"executive_summary": "ok"},
                "draft_report": {},
                "research_tasks": [],
                "evidence": [],
                "research_assessment": [],
                "retrieval_attempts": {},
                "synthesis_attempts": 0,
                "validation_issues": [],
            },
            rid,
        )

    monkeypatch.setattr(api_module, "_invoke_graph", fake_invoke)

    with TestClient(api_module.app) as client:
        api_module._graph = object()  # noqa: SLF001
        api_module._run_semaphore = __import__("asyncio").Semaphore(2)  # noqa: SLF001
        yield client


def test_health(api_client: TestClient) -> None:
    response = api_client.get("/health", headers={"X-Request-ID": "client-req-99"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers.get("X-Request-ID") == "client-req-99"


def test_async_job_flow(api_client: TestClient) -> None:
    create = api_client.post(
        "/research/jobs",
        json={"question": "What is RAG?"},
    )
    assert create.status_code == 202
    job_id = create.json()["job_id"]

    body = {}
    for _ in range(50):
        status = api_client.get(f"/research/jobs/{job_id}")
        assert status.status_code == 200
        body = status.json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert body["status"] == "completed"
    assert body["result"]["validation_status"] == "passed"
