import json
import logging
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from researchflow.graph.invoke import graph_invoke_config, invoke_research_graph
from researchflow.observability.context import log_context
from researchflow.observability.logging import ContextFilter, JsonLogFormatter, log_event
from researchflow.state import initial_research_state


class _MiniState(TypedDict):
    user_question: str
    run_id: str


def _noop(_state: _MiniState) -> dict:
    return {}


def test_invoke_sets_run_id_and_checkpoint() -> None:
    builder = StateGraph(_MiniState)
    builder.add_node("noop", _noop)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    graph = builder.compile(checkpointer=MemorySaver())

    state, run_id = invoke_research_graph(
        graph,
        {"user_question": "ping", "run_id": ""},
        run_id="fixed-run-123",
        request_id="req-abc",
    )
    assert run_id == "fixed-run-123"
    assert state["run_id"] == "fixed-run-123"
    snapshot = graph.get_state(graph_invoke_config(run_id))
    assert snapshot.values.get("user_question") == "ping"


def test_invoke_research_state_run_id() -> None:
    class FakeGraph:
        def invoke(self, state, config):
            assert config == graph_invoke_config("fixed-run-123")
            return {**state, "validation_status": "passed"}

    state, run_id = invoke_research_graph(
        FakeGraph(),
        initial_research_state("What is RAG?"),
        run_id="fixed-run-123",
    )
    assert run_id == "fixed-run-123"
    assert state["run_id"] == "fixed-run-123"


def test_json_log_includes_request_and_run_ids() -> None:
    logger = logging.getLogger("researchflow.test.observability")
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.log_fields = {"event": "unit_test_event", "foo": "bar"}
    with log_context(request_id="req-1", run_id="run-2"):
        ContextFilter().filter(record)
    payload = json.loads(JsonLogFormatter().format(record))
    assert payload["request_id"] == "req-1"
    assert payload["run_id"] == "run-2"
    assert payload["event"] == "unit_test_event"
    assert payload["foo"] == "bar"
