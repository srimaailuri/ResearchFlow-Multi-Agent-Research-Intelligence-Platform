"""Deterministic benchmark scoring vs expected_answer / expected_topics."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from researchflow.retrieval.boost import normalize_document_stem

_TOPIC_STOPWORDS = frozenset({"the", "and", "for", "with", "from", "that", "this"})


def _normalize_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def collect_report_text(state: dict[str, Any]) -> str:
    """Merge final (or draft) report fields into one searchable corpus."""
    parts: list[str] = []
    for key in ("final_report", "draft_report"):
        report = state.get(key) or {}
        if not isinstance(report, dict):
            continue
        for field in ("executive_summary", "conclusion"):
            value = report.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for finding in report.get("key_findings") or []:
            if isinstance(finding, dict):
                text = finding.get("finding") or finding.get("text") or ""
            else:
                text = str(finding)
            if text.strip():
                parts.append(text.strip())
        for section in report.get("detailed_analysis") or []:
            if isinstance(section, dict):
                for field in ("summary", "analysis", "content"):
                    value = section.get(field)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
    return " ".join(parts)


def topic_is_present(corpus: str, topic: str) -> bool:
    corpus_norm = _normalize_text(corpus.replace("-", " "))
    topic_norm = _normalize_text(topic.replace("-", " "))
    if not topic_norm:
        return False
    if topic_norm in corpus_norm:
        return True
    if _normalize_text(topic) in _normalize_text(corpus):
        return True
    tokens = [t for t in topic_norm.split() if len(t) > 2 and t not in _TOPIC_STOPWORDS]
    if len(tokens) >= 2:
        return all(token in corpus_norm for token in tokens)
    return topic_norm in corpus_norm


def score_topics(
    expected_topics: list[str],
    generated_text: str,
) -> dict[str, Any]:
    hits: list[str] = []
    misses: list[str] = []
    for topic in expected_topics:
        if topic_is_present(generated_text, topic):
            hits.append(topic)
        else:
            misses.append(topic)
    total = len(expected_topics)
    recall = (len(hits) / total) if total else 1.0
    return {
        "expected_count": total,
        "hit_count": len(hits),
        "recall": round(recall, 4),
        "hits": hits,
        "misses": misses,
    }


def score_answer_similarity(expected_answer: str, generated_text: str) -> float:
    if not expected_answer.strip():
        return 1.0
    if not generated_text.strip():
        return 0.0
    expected = _normalize_text(expected_answer)
    generated = _normalize_text(generated_text)
    sequence_ratio = SequenceMatcher(None, expected, generated).ratio()
    expected_tokens = {t for t in expected.split() if len(t) > 2}
    generated_tokens = {t for t in generated.split() if len(t) > 2}
    if not expected_tokens:
        return round(sequence_ratio, 4)
    token_recall = len(expected_tokens & generated_tokens) / len(expected_tokens)
    return round(0.45 * sequence_ratio + 0.55 * token_recall, 4)


def score_source_documents(
    expected_documents: list[str],
    rag_document_ids: list[str],
) -> dict[str, Any]:
    expected_stems = {normalize_document_stem(name) for name in expected_documents if name.strip()}
    seen_stems = {normalize_document_stem(name) for name in rag_document_ids if name.strip()}
    if not expected_stems:
        return {"expected_count": 0, "hit_count": 0, "recall": 1.0, "hits": [], "misses": []}
    hits = sorted(expected_stems & seen_stems)
    misses = sorted(expected_stems - seen_stems)
    recall = len(hits) / len(expected_stems)
    return {
        "expected_count": len(expected_stems),
        "hit_count": len(hits),
        "recall": round(recall, 4),
        "hits": hits,
        "misses": misses,
    }


def score_benchmark_run(
    item: dict[str, Any],
    state: dict[str, Any],
    *,
    topic_pass_recall: float = 0.5,
    answer_pass_similarity: float = 0.35,
) -> dict[str, Any]:
    generated_text = collect_report_text(state)
    expected_answer = str(item.get("expected_answer") or "")
    expected_topics = list(item.get("expected_topics") or [])
    expected_docs = list(item.get("source_documents") or [])

    rag_ids: list[str] = []
    for row in state.get("evidence") or []:
        if row.get("source_type") == "rag" and row.get("document_id"):
            rag_ids.append(str(row["document_id"]))

    topics = score_topics(expected_topics, generated_text)
    answer_similarity = score_answer_similarity(expected_answer, generated_text)
    sources = score_source_documents(expected_docs, rag_ids)
    validation_passed = (state.get("validation_status") or "").lower() == "passed"

    overall = (
        0.35 * topics["recall"]
        + 0.35 * answer_similarity
        + 0.15 * sources["recall"]
        + (0.15 if validation_passed else 0.0)
    )
    auto_pass = (
        validation_passed
        and topics["recall"] >= topic_pass_recall
        and answer_similarity >= answer_pass_similarity
    )

    return {
        "generated_text_length": len(generated_text),
        "topic_recall": topics["recall"],
        "topics_hit": topics["hits"],
        "topics_missed": topics["misses"],
        "answer_similarity": answer_similarity,
        "source_document_recall": sources["recall"],
        "source_documents_hit": sources["hits"],
        "source_documents_missed": sources["misses"],
        "validation_passed": validation_passed,
        "overall_score": round(overall, 4),
        "auto_pass": auto_pass,
        "thresholds": {
            "topic_recall": topic_pass_recall,
            "answer_similarity": answer_pass_similarity,
        },
    }


def aggregate_benchmark_scores(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [run["scores"] for run in runs if run.get("scores")]
    if not scored:
        return {"run_count": 0}

    n = len(scored)
    return {
        "run_count": n,
        "mean_overall_score": round(sum(s["overall_score"] for s in scored) / n, 4),
        "mean_topic_recall": round(sum(s["topic_recall"] for s in scored) / n, 4),
        "mean_answer_similarity": round(sum(s["answer_similarity"] for s in scored) / n, 4),
        "mean_source_document_recall": round(
            sum(s["source_document_recall"] for s in scored) / n,
            4,
        ),
        "validation_pass_rate": round(
            sum(1 for s in scored if s["validation_passed"]) / n,
            4,
        ),
        "auto_pass_rate": round(sum(1 for s in scored if s["auto_pass"]) / n, 4),
        "auto_pass_count": sum(1 for s in scored if s["auto_pass"]),
    }
