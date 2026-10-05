from __future__ import annotations

from typing import Any

from researchflow.config.settings import Settings, get_settings
from researchflow.graph.builder import build_research_graph
from researchflow.graph.checkpoint import get_checkpointer_manager
from researchflow.observability import configure_logging


def setup_research_graph(settings: Settings | None = None) -> Any:
    """Configure logging, checkpoint backend, and compile the graph."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    manager = get_checkpointer_manager()
    checkpointer = manager.setup(settings)
    return build_research_graph(checkpointer=checkpointer)


def shutdown_research_graph() -> None:
    get_checkpointer_manager().close()
