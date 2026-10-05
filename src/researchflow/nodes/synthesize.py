import json

from langchain_core.messages import HumanMessage, SystemMessage

from researchflow.llm import get_chat_model, invoke_with_retry
from researchflow.state.models import DraftReport
from researchflow.state.research_state import ResearchState

SYNTHESIS_SYSTEM_PROMPT = """You are the synthesis stage of a research agent.

Write a draft report using ONLY the evaluated evidence provided. Rules:
1. Answer the original research question using the research tasks as structure.
2. Every key finding MUST cite supporting_evidence as evidence_id strings that exist in the input.
3. Do not invent statistics or facts not supported by the evidence content.
4. Add limitations entries for tasks marked insufficient or partial in research_assessment.
5. Use conflicting_evidence only when the input supports a real disagreement.
6. If evidence is thin, say so in executive_summary and conclusion."""

RETRY_SYNTHESIS_APPENDIX = """

If validation_issues is non-empty, this is a correction pass. Fix every issue:
remove or rewrite unsupported claims, fix citations, and disclose evidence gaps."""


def synthesize_report(state: ResearchState) -> dict:
    """Stage 4 sub-phase A: LLM draft report grounded on evaluated evidence."""
    context = {
        "user_question": state["user_question"],
        "research_tasks": state["research_tasks"],
        "evaluated_evidence": state["evaluated_evidence"],
        "research_assessment": state["research_assessment"],
        "evidence_content": state["evidence"],
        "validation_issues": state["validation_issues"],
        "synthesis_attempts": state["synthesis_attempts"],
    }
    system_prompt = SYNTHESIS_SYSTEM_PROMPT
    if state["validation_issues"]:
        system_prompt += RETRY_SYNTHESIS_APPENDIX
    llm = get_chat_model().with_structured_output(DraftReport)
    draft: DraftReport = invoke_with_retry(
        llm,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(context, indent=2)),
        ],
    )
    return {"draft_report": draft.model_dump(mode="json")}
