from researchflow.config.settings import get_settings
from researchflow.retrieval.boost import document_matches_preferred, normalize_preferred_documents
from researchflow.state.models import Evidence


def _rank_score(item: Evidence, preferred_stems: set[str]) -> float:
    meta = item.metadata or {}
    base = float(meta.get("retrieval_score", 0.0))
    if document_matches_preferred(item.document_id, item.title, preferred_stems):
        base += float(meta.get("preferred_document_boost", 0.0))
    return base


def cap_evidence(
    items: list[Evidence],
    *,
    max_per_task: int | None = None,
    max_total: int | None = None,
    preferred_documents: list[str] | None = None,
) -> list[Evidence]:
    """
    Limit evidence volume before evaluate/synthesize (cost control).

    Keeps highest ``retrieval_score`` rows per task, then applies a global cap.
    """
    settings = get_settings()
    per_task = max_per_task if max_per_task is not None else settings.max_evidence_per_task
    total = max_total if max_total is not None else settings.max_total_evidence
    preferred_stems = set(normalize_preferred_documents(preferred_documents or []))

    by_task: dict[str, list[Evidence]] = {}
    for item in items:
        by_task.setdefault(item.research_id, []).append(item)

    capped: list[Evidence] = []
    for group in by_task.values():
        ranked = sorted(group, key=lambda row: _rank_score(row, preferred_stems), reverse=True)
        capped.extend(ranked[: max(0, per_task)])

    if len(capped) > total:
        capped = sorted(capped, key=lambda row: _rank_score(row, preferred_stems), reverse=True)
        capped = capped[: max(0, total)]

    return capped
