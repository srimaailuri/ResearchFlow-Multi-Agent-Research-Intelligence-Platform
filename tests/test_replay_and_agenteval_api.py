import importlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client_with_checkpoint(monkeypatch):
    api_module = importlib.import_module("researchflow.api.app")

    class FakeGraph:
        def get_state(self, _config):
            return SimpleNamespace(
                values={"run_id": "run-1", "question": "Q", "evidence": [], "validation_status": "passed"},
                next=("synthesize_report",),
            )

        def invoke(self, input_state, _config):
            assert input_state is None
            return {
                "run_id": "run-1",
                "user_question": "Q",
                "validation_status": "passed",
                "final_report": {"executive_summary": "done"},
                "evidence": [],
            }

    fake_graph = FakeGraph()
    monkeypatch.setattr(api_module, "setup_research_graph", lambda _settings: fake_graph)
    real_settings = api_module.get_settings()
    monkeypatch.setattr(
        api_module,
        "get_settings",
        lambda: SimpleNamespace(
            **{**real_settings.model_dump(), "checkpointer_mode": "memory"},
        ),
    )

    with TestClient(api_module.app) as client:
        yield client


def test_agenteval_trace_from_checkpoint(api_client_with_checkpoint: TestClient) -> None:
    response = api_client_with_checkpoint.get("/research/runs/run-1/agenteval-trace")
    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "researchflow"
    assert body["run_id"] == "run-1"


def test_replay_when_pending_steps(api_client_with_checkpoint: TestClient) -> None:
    response = api_client_with_checkpoint.post("/research/runs/run-1/replay")
    assert response.status_code == 200
    assert response.json()["validation_status"] == "passed"
    assert response.json()["agenteval_trace"]["system"] == "researchflow"
