"""Export ResearchFlow pipeline state to AgentEval execution trace JSON."""

from __future__ import annotations

from typing import Any


def to_agenteval_trace(state: dict[str, Any], *, system_version: str | None = None) -> dict[str, Any]:
    run_id = str(state.get("run_id") or "unknown")
    retrievals: list[dict[str, Any]] = []
    for rank, item in enumerate(state.get("evidence") or [], start=1):
        if not isinstance(item, dict):
            continue
        doc = item.get("document_id") or item.get("source") or ""
        retrievals.append(
            {
                "source": doc,
                "rank": rank,
                "relevance": item.get("score"),
            }
        )
    return {
        "run_id": run_id,
        "system": "researchflow",
        "system_version": system_version,
        "input": str(state.get("question") or ""),
        "steps": [],
        "retrievals": retrievals,
        "final_answer": _report_text(state),
        "usage": {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "llm_calls": 0},
        "status": "completed" if state.get("validation_status") != "failed" else "failed",
    }


def _report_text(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("final_report", "draft_report"):
        report = state.get(key) or {}
        if not isinstance(report, dict):
            continue
        for field in ("executive_summary", "conclusion"):
            value = report.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " ".join(parts)
