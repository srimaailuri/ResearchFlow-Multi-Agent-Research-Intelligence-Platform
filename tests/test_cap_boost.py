from researchflow.retrieval.boost import normalize_document_stem, normalize_preferred_documents
from researchflow.retrieval.cap import cap_evidence
from researchflow.retrieval.query import retrieval_query_for_task
from researchflow.state import ResearchTask, SourceType, initial_research_state
from researchflow.state.models import Evidence


def _evidence(
    eid: str,
    task_id: str,
    *,
    score: float,
    doc_id: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=eid,
        research_id=task_id,
        source_type=SourceType.RAG,
        title=doc_id or "paper",
        document_id=doc_id,
        content="text",
        metadata={"retrieval_score": score},
    )


def test_normalize_preferred_documents() -> None:
    assert normalize_document_stem("RAG.pdf") == "RAG"
    assert normalize_preferred_documents(["RAG.pdf", "RAG.pdf", " Toolformer.pdf "]) == [
        "RAG",
        "Toolformer",
    ]


def test_cap_evidence_per_task_and_total() -> None:
    items = [
        _evidence("E1", "RQ1", score=0.9),
        _evidence("E2", "RQ1", score=0.5),
        _evidence("E3", "RQ1", score=0.1),
        _evidence("E4", "RQ2", score=0.8),
        _evidence("E5", "RQ2", score=0.7),
        _evidence("E6", "RQ2", score=0.6),
    ]
    capped = cap_evidence(items, max_per_task=2, max_total=3)
    assert len(capped) == 3
    assert {row.evidence_id for row in capped} == {"E1", "E4", "E5"}


def test_cap_prefers_boosted_preferred_documents_on_global_tie() -> None:
    items = [
        _evidence("E1", "RQ1", score=0.5, doc_id="Other"),
        Evidence(
            evidence_id="E2",
            research_id="RQ1",
            source_type=SourceType.RAG,
            title="RAG",
            document_id="RAG",
            content="text",
            metadata={"retrieval_score": 0.5, "preferred_document_boost": 0.2},
        ),
    ]
    capped = cap_evidence(
        items,
        max_per_task=1,
        max_total=1,
        preferred_documents=["RAG.pdf"],
    )
    assert len(capped) == 1
    assert capped[0].evidence_id == "E2"


def test_retrieval_query_includes_preferred_docs() -> None:
    state = initial_research_state("q", preferred_rag_documents=["RAGChecker.pdf"])
    task = ResearchTask(id="RQ1", question="How is RAG evaluated?", sources=[SourceType.RAG])
    query = retrieval_query_for_task(state, task)
    assert "RAGChecker" in query
