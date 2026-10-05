"""Run the research graph from the command line: python -m researchflow [question]."""

import argparse
import json
import sys

from researchflow.graph.bootstrap import setup_research_graph, shutdown_research_graph
from researchflow.graph.invoke import invoke_research_graph
from researchflow.state import initial_research_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ResearchFlow LangGraph pipeline (RAG + web retrieval, retries).",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What is RAG?",
        help="Research question (default: %(default)s)",
    )
    args = parser.parse_args()

    graph = setup_research_graph()
    try:
        final_state, run_id = invoke_research_graph(
            graph,
            initial_research_state(args.question),
        )
    finally:
        shutdown_research_graph()
    json.dump({"run_id": run_id, "state": final_state}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
