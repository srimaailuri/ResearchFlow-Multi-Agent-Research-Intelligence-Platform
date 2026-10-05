from __future__ import annotations

import logging
import uuid
from typing import Any

from researchflow.observability import bind_context, get_request_id, log_context, log_event
from researchflow.state.research_state import ResearchState

logger = logging.getLogger(__name__)


def new_run_id() -> str:
    return str(uuid.uuid4())


def graph_invoke_config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def invoke_research_graph(
    graph: Any,
    state: ResearchState,
    *,
    run_id: str | None = None,
    request_id: str | None = None,
) -> tuple[ResearchState, str]:
    """
    Invoke the compiled graph with checkpoint thread_id = run_id.

    Returns final state and the run_id used (for debugging / get_state).
    """
    run_id = run_id or new_run_id()
    req_id = request_id or get_request_id() or run_id

    with log_context(request_id=req_id, run_id=run_id):
        bind_context(request_id=req_id, run_id=run_id)
        state_with_run: ResearchState = {**state, "run_id": run_id}
        log_event(
            logger,
            "graph_invoke_start",
            question=(state_with_run.get("user_question") or "")[:500],
        )
        try:
            result: ResearchState = graph.invoke(state_with_run, graph_invoke_config(run_id))
        except Exception:
            log_event(logger, "graph_invoke_failed")
            raise
        log_event(
            logger,
            "graph_invoke_complete",
            validation_status=result.get("validation_status"),
            evidence_count=len(result.get("evidence") or []),
        )
        return result, run_id
