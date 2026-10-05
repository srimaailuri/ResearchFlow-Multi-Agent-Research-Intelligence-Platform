from typing import Any

from langgraph.graph import END, START, StateGraph

from researchflow.nodes.evaluate import evaluate_and_assess
from researchflow.nodes.plan import understand_and_plan
from researchflow.nodes.retrieve import (
    fan_out_retrieval,
    fan_out_retrieval_retry,
    finalize_retrieval,
    retrieve_one,
)
from researchflow.nodes.routing import (
    increment_synthesis_attempt,
    route_after_evaluate,
    route_after_validate,
    schedule_retrieval_retry,
)
from researchflow.nodes.synthesize import synthesize_report
from researchflow.nodes.validate import validate_report
from researchflow.state.research_state import ResearchState


def build_research_graph(*, checkpointer: Any | None = None):
    """Compile the research pipeline graph (retrieval, evaluate, synthesize, retries)."""
    builder = StateGraph(ResearchState)
    builder.add_node("understand_and_plan", understand_and_plan)
    builder.add_node("retrieve_one", retrieve_one)
    builder.add_node("finalize_retrieval", finalize_retrieval)
    builder.add_node("evaluate_and_assess", evaluate_and_assess)
    builder.add_node("schedule_retrieval_retry", schedule_retrieval_retry)
    builder.add_node("synthesize_report", synthesize_report)
    builder.add_node("validate_report", validate_report)
    builder.add_node("increment_synthesis_attempt", increment_synthesis_attempt)

    builder.add_edge(START, "understand_and_plan")
    builder.add_conditional_edges(
        "understand_and_plan",
        fan_out_retrieval,
        ["retrieve_one", "finalize_retrieval"],
    )
    builder.add_edge("retrieve_one", "finalize_retrieval")
    builder.add_edge("finalize_retrieval", "evaluate_and_assess")

    builder.add_conditional_edges(
        "evaluate_and_assess",
        route_after_evaluate,
        ["schedule_retrieval_retry", "synthesize_report"],
    )
    builder.add_conditional_edges(
        "schedule_retrieval_retry",
        fan_out_retrieval_retry,
        ["retrieve_one", "synthesize_report"],
    )

    builder.add_edge("synthesize_report", "validate_report")
    builder.add_conditional_edges(
        "validate_report",
        route_after_validate,
        ["increment_synthesis_attempt", END],
    )
    builder.add_edge("increment_synthesis_attempt", "synthesize_report")

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
