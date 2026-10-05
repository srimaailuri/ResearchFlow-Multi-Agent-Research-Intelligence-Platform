from researchflow.graph import build_research_graph
from researchflow.state import Evidence, EvaluatedEvidence, EvidenceScores, ScoreLevel, SourceType, initial_research_state
from researchflow.state.models import TaskEvaluatedEvidenceBatch


def test_graph_compiles() -> None:
    graph = build_research_graph()
    assert graph is not None


def test_retrieve_and_evaluate_path(monkeypatch) -> None:
    from researchflow.nodes.evaluate import evaluate_and_assess
    from researchflow.nodes.retrieve import fan_out_retrieval, finalize_retrieval, retrieve_one
    from researchflow.retrieval import merge_evidence_state

    def fake_web(task, *, sequence, query=None):
        return [
            Evidence(
                evidence_id=f"E{sequence}",
                research_id=task.id,
                source_type=SourceType.WEB,
                title="web",
                url="https://example.com",
                content="RAG combines retrieval with generation.",
            )
        ]

    def fake_rag(task, *, sequence, query=None, preferred_documents=None):
        return [
            Evidence(
                evidence_id=f"E{sequence}",
                research_id=task.id,
                source_type=SourceType.RAG,
                title="paper",
                document_id="RAG",
                content="Retrieval-augmented generation augments LLMs with retrieved docs.",
            )
        ]

    class FakeEvalLLM:
        def with_structured_output(self, schema):
            class Chain:
                def invoke(self, messages):
                    return TaskEvaluatedEvidenceBatch(
                        items=[
                            EvaluatedEvidence(
                                evidence_id="E1",
                                research_id="RQ1",
                                evaluation=EvidenceScores(
                                    relevance=ScoreLevel.HIGH,
                                    authority=ScoreLevel.MEDIUM,
                                    recency=ScoreLevel.MEDIUM,
                                    specificity=ScoreLevel.HIGH,
                                    corroboration=ScoreLevel.LOW,
                                ),
                                supports=["claim_E1"],
                                contradicts=[],
                                confidence=ScoreLevel.MEDIUM,
                            ),
                            EvaluatedEvidence(
                                evidence_id="E2",
                                research_id="RQ1",
                                evaluation=EvidenceScores(
                                    relevance=ScoreLevel.HIGH,
                                    authority=ScoreLevel.MEDIUM,
                                    recency=ScoreLevel.MEDIUM,
                                    specificity=ScoreLevel.HIGH,
                                    corroboration=ScoreLevel.LOW,
                                ),
                                supports=["claim_E2"],
                                contradicts=[],
                                confidence=ScoreLevel.MEDIUM,
                            ),
                        ]
                    )

            return Chain()

    monkeypatch.setattr("researchflow.nodes.retrieve.retrieve_web", fake_web)
    monkeypatch.setattr("researchflow.nodes.retrieve.retrieve_rag", fake_rag)
    monkeypatch.setattr("researchflow.nodes.evaluate.get_evaluate_chat_model", lambda: FakeEvalLLM())

    state = initial_research_state("What is RAG?")
    state["research_tasks"] = [
        {
            "id": "RQ1",
            "question": "What is RAG?",
            "sources": ["web", "rag"],
        }
    ]

    route = fan_out_retrieval(state)
    assert isinstance(route, list)
    for send in route:
        patch = retrieve_one(send.arg)
        state["evidence"] = merge_evidence_state(state["evidence"], patch["evidence"])

    state.update(finalize_retrieval(state))
    state.update(evaluate_and_assess(state))
    assert state["research_assessment"]
    assert len(state["evaluated_evidence"]) == 2
