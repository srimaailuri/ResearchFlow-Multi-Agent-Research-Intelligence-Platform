from researchflow.retrieval.dedupe import dedupe_evidence, merge_evidence_state
from researchflow.state.models import Evidence, SourceType


def test_dedupe_web_by_url() -> None:
    items = [
        Evidence(
            evidence_id="E1",
            research_id="RQ1",
            source_type=SourceType.WEB,
            title="a",
            url="https://example.com/a",
            content="one",
        ),
        Evidence(
            evidence_id="E2",
            research_id="RQ1",
            source_type=SourceType.WEB,
            title="b",
            url="https://example.com/a",
            content="dup",
        ),
    ]
    assert len(dedupe_evidence(items)) == 1


def test_merge_evidence_state_dedupes() -> None:
    left = [
        Evidence(
            evidence_id="E1",
            research_id="RQ1",
            source_type=SourceType.RAG,
            title="t",
            document_id="doc-a",
            content="x",
        ).model_dump(mode="json")
    ]
    right = [
        Evidence(
            evidence_id="E2",
            research_id="RQ1",
            source_type=SourceType.RAG,
            title="t2",
            document_id="doc-a",
            content="y",
        ).model_dump(mode="json")
    ]
    merged = merge_evidence_state(left, right)
    assert len(merged) == 2


def test_dedupe_rag_same_chunk_only() -> None:
    chunk = "same chunk text"
    items = [
        Evidence(
            evidence_id="E1",
            research_id="RQ1",
            source_type=SourceType.RAG,
            title="t",
            document_id="doc-a",
            content=chunk,
            metadata={"page": 1},
        ),
        Evidence(
            evidence_id="E2",
            research_id="RQ1",
            source_type=SourceType.RAG,
            title="t",
            document_id="doc-a",
            content=chunk,
            metadata={"page": 1},
        ),
    ]
    assert len(dedupe_evidence(items)) == 1
