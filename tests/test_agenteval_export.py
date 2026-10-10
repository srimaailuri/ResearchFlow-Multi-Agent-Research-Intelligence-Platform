from researchflow.observability.agenteval_export import to_agenteval_trace


def test_to_agenteval_trace_from_state() -> None:
    state = {
        "run_id": "rf-1",
        "question": "What is RAG?",
        "evidence": [{"document_id": "RAG.pdf", "score": 0.9}],
        "final_report": {"executive_summary": "Retrieval augments generation."},
        "validation_status": "passed",
    }
    payload = to_agenteval_trace(state, system_version="0.1.0")
    assert payload["system"] == "researchflow"
    assert payload["retrievals"][0]["source"] == "RAG.pdf"
    assert "Retrieval" in payload["final_answer"]
