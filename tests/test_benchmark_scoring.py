from researchflow.eval.scoring import (
    aggregate_benchmark_scores,
    collect_report_text,
    score_answer_similarity,
    score_benchmark_run,
    topic_is_present,
)


def test_topic_matching_handles_hyphens_and_phrases() -> None:
    corpus = "The pipeline includes pre retrieval, retrieval, post retrieval, and generation."
    assert topic_is_present(corpus, "pre-retrieval")
    assert topic_is_present(corpus, "post-retrieval")
    assert not topic_is_present(corpus, "toolformer pretraining")


def test_score_benchmark_run_from_state() -> None:
    item = {
        "expected_answer": "RAG combines parametric memory with a non-parametric retrieval index.",
        "expected_topics": ["parametric memory", "non-parametric memory", "hallucination"],
        "source_documents": ["RAG.pdf"],
    }
    state = {
        "validation_status": "passed",
        "evidence": [{"source_type": "rag", "document_id": "RAG"}],
        "final_report": {
            "executive_summary": (
                "RAG uses parametric memory in the generator and non-parametric memory "
                "via a retriever to reduce hallucination."
            ),
            "conclusion": "Retrieval augments generation.",
            "key_findings": [{"finding": "External index supplies non-parametric knowledge."}],
        },
    }
    scores = score_benchmark_run(item, state)
    assert scores["topic_recall"] >= 0.66
    assert scores["answer_similarity"] > 0.3
    assert scores["source_document_recall"] == 1.0
    assert scores["validation_passed"] is True
    assert scores["overall_score"] > 0.5


def test_collect_report_text_prefers_final_report() -> None:
    state = {
        "draft_report": {"executive_summary": "draft only"},
        "final_report": {"executive_summary": "final version", "conclusion": "done"},
    }
    text = collect_report_text(state)
    assert "final version" in text
    assert "draft only" in text


def test_aggregate_benchmark_scores() -> None:
    runs = [
        {"scores": {"overall_score": 0.8, "topic_recall": 1.0, "answer_similarity": 0.7, "source_document_recall": 1.0, "validation_passed": True, "auto_pass": True}},
        {"scores": {"overall_score": 0.4, "topic_recall": 0.5, "answer_similarity": 0.2, "source_document_recall": 0.0, "validation_passed": False, "auto_pass": False}},
    ]
    agg = aggregate_benchmark_scores(runs)
    assert agg["run_count"] == 2
    assert agg["auto_pass_count"] == 1
    assert agg["mean_overall_score"] == 0.6


def test_answer_similarity_empty_generated() -> None:
    assert score_answer_similarity("some expected answer here", "") == 0.0
