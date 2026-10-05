from pathlib import Path


def normalize_document_stem(name: str) -> str:
    """Map benchmark/API names like ``RAG.pdf`` to index stems like ``RAG``."""
    cleaned = name.strip()
    if not cleaned:
        return ""
    return Path(cleaned).stem


def normalize_preferred_documents(names: list[str]) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for name in names:
        stem = normalize_document_stem(name)
        if stem and stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


def document_matches_preferred(document_id: str | None, source_file: str | None, preferred_stems: set[str]) -> bool:
    if not preferred_stems:
        return False
    candidates = {normalize_document_stem(document_id or ""), normalize_document_stem(source_file or "")}
    return bool(candidates & preferred_stems)
