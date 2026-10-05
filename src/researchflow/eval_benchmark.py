"""Run the research graph against benchmark questions from JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from researchflow.config.settings import get_settings
from researchflow.eval import aggregate_benchmark_scores, score_benchmark_run
from researchflow.graph.bootstrap import setup_research_graph, shutdown_research_graph
from researchflow.graph.invoke import invoke_research_graph
from researchflow.state import initial_research_state


def _load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark file must be a JSON array of question objects")
    return data


def _filter_questions(
    items: list[dict[str, Any]],
    *,
    ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if ids:
        by_id = {item["id"]: item for item in items}
        missing = [qid for qid in ids if qid not in by_id]
        if missing:
            raise ValueError(f"Unknown question ids: {', '.join(missing)}")
        return [by_id[qid] for qid in ids]
    if limit is not None:
        return items[:limit]
    return items


def _summarize_state(
    item: dict[str, Any],
    state: dict[str, Any],
    elapsed_s: float,
    *,
    run_id: str,
) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    by_source: dict[str, int] = {}
    rag_docs: set[str] = set()
    for row in evidence:
        st = row.get("source_type", "unknown")
        by_source[st] = by_source.get(st, 0) + 1
        if st == "rag" and row.get("document_id"):
            rag_docs.add(str(row["document_id"]))

    final = state.get("final_report") or {}
    assessments = state.get("research_assessment") or []

    return {
        "id": item.get("id"),
        "run_id": run_id,
        "question": item.get("question"),
        "type": item.get("type"),
        "expected_source_documents": item.get("source_documents"),
        "expected_answer": item.get("expected_answer"),
        "elapsed_seconds": round(elapsed_s, 1),
        "question_type": state.get("question_type"),
        "research_task_count": len(state.get("research_tasks") or []),
        "research_tasks": state.get("research_tasks"),
        "evidence_count": len(evidence),
        "evidence_by_source": by_source,
        "rag_document_ids_seen": sorted(rag_docs),
        "research_assessment": assessments,
        "retrieval_attempts": state.get("retrieval_attempts"),
        "synthesis_attempts": state.get("synthesis_attempts"),
        "validation_status": state.get("validation_status"),
        "validation_issues": state.get("validation_issues"),
        "generated_executive_summary": final.get("executive_summary"),
        "generated_conclusion": final.get("conclusion"),
        "generated_key_findings": final.get("key_findings"),
        "scores": None,
    }


def _print_run_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"[{summary['id']}] {summary['question'][:100]}...")
    print(f"  run_id: {summary.get('run_id')} | Time: {summary['elapsed_seconds']}s | validation: {summary['validation_status']}")
    print(f"  Plan tasks: {summary['research_task_count']} | evidence: {summary['evidence_count']} {summary['evidence_by_source']}")
    print(f"  RAG docs retrieved: {summary['rag_document_ids_seen']}")
    print(f"  Expected PDFs: {summary['expected_source_documents']}")
    for a in summary.get("research_assessment") or []:
        print(f"    {a.get('research_id')}: {a.get('status')} (gap={a.get('evidence_gap')})")
    if summary.get("generated_executive_summary"):
        text = summary["generated_executive_summary"].replace("\n", " ")
        print(f"  Summary: {text[:280]}...")
    scores = summary.get("scores")
    if scores:
        print(
            f"  Scores: overall={scores['overall_score']} | "
            f"topics={scores['topic_recall']} | answer={scores['answer_similarity']} | "
            f"sources={scores['source_document_recall']} | auto_pass={scores['auto_pass']}"
        )
        if scores.get("topics_missed"):
            print(f"  Topics missed: {scores['topics_missed'][:8]}")
    print("=" * 72)


def run_benchmark(
    questions_path: Path,
    *,
    output_dir: Path,
    ids: list[str] | None = None,
    limit: int | None = None,
    save_full_state: bool = False,
    score_runs: bool = True,
) -> Path:
    items = _filter_questions(_load_questions(questions_path), ids=ids, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    graph = setup_research_graph()
    run_summaries: list[dict[str, Any]] = []

    try:
        for item in items:
            qid = item["id"]
            question = item["question"]
            print(f"\n>>> Running graph for {qid}...", flush=True)
            started = time.perf_counter()
            preferred = item.get("source_documents") or []
            state, run_id = invoke_research_graph(
                graph,
                initial_research_state(question, preferred_rag_documents=preferred),
            )
            elapsed = time.perf_counter() - started

            summary = _summarize_state(item, state, elapsed, run_id=run_id)
            if score_runs:
                summary["scores"] = score_benchmark_run(
                    item,
                    state,
                    topic_pass_recall=settings.benchmark_topic_pass_recall,
                    answer_pass_similarity=settings.benchmark_answer_pass_similarity,
                )
            run_summaries.append(summary)
            _print_run_summary(summary)

            if save_full_state:
                out = output_dir / f"{qid}_full_state.json"
                out.write_text(json.dumps({"run_id": run_id, "state": state}, indent=2), encoding="utf-8")
    finally:
        shutdown_research_graph()

    report: dict[str, Any] = {
        "benchmark_file": str(questions_path),
        "run_count": len(run_summaries),
        "runs": run_summaries,
    }
    if score_runs:
        report["aggregate_scores"] = aggregate_benchmark_scores(run_summaries)
        agg = report["aggregate_scores"]
        print(
            f"\nAggregate: auto_pass {agg.get('auto_pass_count', 0)}/{agg.get('run_count', 0)} | "
            f"mean overall={agg.get('mean_overall_score')} | "
            f"topics={agg.get('mean_topic_recall')} | answer={agg.get('mean_answer_similarity')}"
        )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResearchFlow on benchmark JSON questions.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("Data/evaluate_json/research_questions_memory_ragchecker.json"),
        help="Path to benchmark questions JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/evaluate_json/runs"),
        help="Directory for benchmark reports",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N questions")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated ids, e.g. Q001,Q002")
    parser.add_argument(
        "--full-state",
        action="store_true",
        help="Also write full graph state JSON per question",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override MODEL_NAME for this run (e.g. gemini-3-flash-preview if 3.8-flash quota is exhausted)",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip deterministic auto-scoring vs expected_answer / expected_topics",
    )
    args = parser.parse_args()
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    if args.model:
        import os

        os.environ["MODEL_NAME"] = args.model
        from researchflow.config.settings import get_settings

        get_settings.cache_clear()

    try:
        run_benchmark(
            args.file,
            output_dir=args.output_dir,
            ids=ids,
            limit=args.limit,
            save_full_state=args.full_state,
            score_runs=not args.no_score,
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
