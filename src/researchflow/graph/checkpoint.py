from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from researchflow.config.settings import Settings

_manager: "CheckpointerManager | None" = None


class CheckpointerManager:
    """Lifecycle wrapper for LangGraph checkpoint backends."""

    def __init__(self) -> None:
        self._saver: Any | None = None
        self._context: Any | None = None

    @property
    def saver(self) -> Any | None:
        return self._saver

    def setup(self, settings: Settings) -> Any | None:
        mode = settings.checkpointer_mode.strip().lower()
        if mode in {"none", "off", "disabled"}:
            self._saver = None
            return None
        if mode == "memory":
            self._saver = MemorySaver()
            return self._saver
        if mode == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = Path(settings.checkpoint_sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._context = SqliteSaver.from_conn_string(str(db_path))
            self._saver = self._context.__enter__()
            return self._saver
        raise ValueError(f"Unknown CHECKPOINTER_MODE: {settings.checkpointer_mode!r}")

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
        self._saver = None


def get_checkpointer_manager() -> CheckpointerManager:
    global _manager
    if _manager is None:
        _manager = CheckpointerManager()
    return _manager


def reset_checkpointer_manager() -> None:
    """Test helper: tear down singleton checkpointer state."""
    global _manager
    if _manager is not None:
        _manager.close()
        _manager = None
