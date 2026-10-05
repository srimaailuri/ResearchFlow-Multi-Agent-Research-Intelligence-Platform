from researchflow.retrieval.query import retrieval_query_for_task
from researchflow.state import ResearchTask, SourceType, initial_research_state


def test_retrieval_query_widens_on_retry() -> None:
    state = initial_research_state("q")
    state["retrieval_attempts"] = {"RQ1": 1}
    task = ResearchTask(id="RQ1", question="core question", sources=[SourceType.WEB])
    q = retrieval_query_for_task(state, task)
    assert "alternative keywords" in q.lower()
    assert "core question" in q
