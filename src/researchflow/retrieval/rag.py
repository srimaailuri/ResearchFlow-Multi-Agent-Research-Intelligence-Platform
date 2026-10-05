import shutil
import threading
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from researchflow.config.settings import get_settings
from researchflow.retrieval.boost import document_matches_preferred, normalize_preferred_documents
from researchflow.state.models import Evidence, ResearchTask, SourceType

_vectorstore: Chroma | None = None
_vectorstore_lock = threading.Lock()

_INDEX_READY = ".index_ready"
_EMBEDDING_MODEL_TAG = ".embedding_model"


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def _index_is_ready(persist_dir: Path, embedding_model: str) -> bool:
    if not (persist_dir / _INDEX_READY).is_file():
        return False
    tag_path = persist_dir / _EMBEDDING_MODEL_TAG
    if not tag_path.is_file():
        return False
    return tag_path.read_text(encoding="utf-8").strip() == embedding_model


def _mark_index_ready(persist_dir: Path, embedding_model: str) -> None:
    (persist_dir / _EMBEDDING_MODEL_TAG).write_text(embedding_model, encoding="utf-8")
    (persist_dir / _INDEX_READY).write_text("ok", encoding="utf-8")


def _build_vectorstore(persist_dir: Path, embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    doc_root = Path(get_settings().rag_documents_dir)
    raw_docs = []
    for pdf_path in sorted(doc_root.glob("**/*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["document_id"] = pdf_path.stem
            page.metadata["document_type"] = "research_paper"
            page.metadata["source_file"] = pdf_path.name
        raw_docs.extend(pages)

    if not raw_docs:
        store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        _mark_index_ready(persist_dir, get_settings().embedding_model)
        return store

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    ).split_documents(raw_docs)

    batch_size = 8
    store: Chroma | None = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        while True:
            try:
                if store is None:
                    store = Chroma.from_documents(
                        batch,
                        embedding=embeddings,
                        persist_directory=str(persist_dir),
                    )
                else:
                    store.add_documents(batch)
                break
            except Exception as exc:
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    time.sleep(25)
                    continue
                raise
        time.sleep(1.0)

    assert store is not None
    _mark_index_ready(persist_dir, get_settings().embedding_model)
    return store


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore

        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        embeddings = _embeddings()

        if _index_is_ready(persist_dir, settings.embedding_model):
            _vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
            )
            return _vectorstore

        if persist_dir.exists() and any(persist_dir.iterdir()):
            shutil.rmtree(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

        _vectorstore = _build_vectorstore(persist_dir, embeddings)
        return _vectorstore


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(0.0, distance))


def _collect_rag_hits(
    store: Chroma,
    search_query: str,
    *,
    k: int,
    preferred_stems: set[str],
) -> list[tuple[object, float]]:
    merged: dict[tuple[str, str], tuple[object, float]] = {}

    def add_hit(doc: object, distance: float) -> None:
        page_key = str(getattr(doc, "page_content", "")[:240])
        doc_id = str(doc.metadata.get("document_id", ""))
        key = (doc_id, page_key)
        score = _distance_to_score(distance)
        source_file = str(doc.metadata.get("source_file") or "")
        if document_matches_preferred(doc_id, source_file, preferred_stems):
            score += get_settings().rag_preferred_document_boost
        existing = merged.get(key)
        if existing is None or score > existing[1]:
            merged[key] = (doc, score)

    for doc, distance in store.similarity_search_with_score(search_query, k=k):
        add_hit(doc, distance)

    for stem in preferred_stems:
        try:
            focused = store.similarity_search_with_score(
                search_query,
                k=max(2, k // 2),
                filter={"document_id": stem},
            )
        except Exception:
            continue
        for doc, distance in focused:
            add_hit(doc, distance)

    ranked = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def retrieve_rag(
    task: ResearchTask,
    *,
    sequence: int,
    query: str | None = None,
    preferred_documents: list[str] | None = None,
) -> list[Evidence]:
    settings = get_settings()
    search_query = query or task.question
    preferred_stems = set(normalize_preferred_documents(preferred_documents or []))
    store = _get_vectorstore()
    candidate_k = max(settings.retrieval_top_k, settings.retrieval_rag_candidate_k)
    hits = _collect_rag_hits(
        store,
        search_query,
        k=candidate_k,
        preferred_stems=preferred_stems,
    )

    items: list[Evidence] = []
    for index, (doc, score) in enumerate(hits[: settings.retrieval_top_k]):
        doc_id = str(doc.metadata.get("document_id") or doc.metadata.get("source_file") or "unknown")
        source_file = str(doc.metadata.get("source_file") or doc_id)
        suffix = "" if index == 0 else f"_{index}"
        boosted = document_matches_preferred(doc_id, source_file, preferred_stems)
        items.append(
            Evidence(
                evidence_id=f"E{sequence}{suffix}",
                research_id=task.id,
                source_type=SourceType.RAG,
                title=source_file,
                document_id=doc_id,
                content=doc.page_content[:4000],
                metadata={
                    "document_type": doc.metadata.get("document_type", "research_paper"),
                    "page": doc.metadata.get("page"),
                    "retrieval_score": round(score, 4),
                    "preferred_document_boost": settings.rag_preferred_document_boost if boosted else 0.0,
                    "preferred_document_match": boosted,
                },
            )
        )
    return items
