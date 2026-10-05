import logging

from langchain_core.messages import HumanMessage, SystemMessage

from researchflow.llm import get_chat_model, invoke_with_retry
from researchflow.observability import log_event
from researchflow.state.models import ResearchPlan
from researchflow.state.research_state import ResearchState

PLAN_SYSTEM_PROMPT = """You are the planning stage of a research agent.

Given a user research question:
1. Classify question_type (e.g. factual, comparison, analytical, why_how, multi_hop, conflicting).
2. Set requires_current_information if the answer needs recent or external web information.
3. Set requires_internal_information if the answer needs a curated internal knowledge base (RAG).
4. Decompose the question into independent research_tasks. Each task must be a clear sub-question.
5. For each task, assign sources: "rag", "web", or both when comparing internal vs external info.
6. A curated internal PDF library is available via source "rag" (research papers on LLMs, RAG, agents).
   Prefer "rag" (alone or with "web") for definitions, surveys, and technical mechanisms answerable from papers.
   Use "web" when the user needs very recent news or when no internal doc is likely to help.
7. Use task ids like RQ1, RQ2, RQ3. Do not retrieve data or write answers; plan only."""

logger = logging.getLogger(__name__)


def understand_and_plan(state: ResearchState) -> dict:
    """Stage 1: turn user_question into research_tasks and routing flags."""
    log_event(logger, "node_start", node="understand_and_plan")
    llm = get_chat_model().with_structured_output(ResearchPlan)
    plan: ResearchPlan = invoke_with_retry(
        llm,
        [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=state["user_question"]),
        ],
    )
    tasks = [task.model_dump(mode="json") for task in plan.research_tasks]
    log_event(logger, "node_complete", node="understand_and_plan", task_count=len(tasks))
    return {
        "question_type": plan.question_type,
        "requires_current_information": plan.requires_current_information,
        "requires_internal_information": plan.requires_internal_information,
        "research_tasks": tasks,
    }
