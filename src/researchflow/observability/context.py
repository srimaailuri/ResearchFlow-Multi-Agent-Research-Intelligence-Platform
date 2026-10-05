from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


def get_run_id() -> str | None:
    return _run_id.get()


def bind_context(*, request_id: str | None = None, run_id: str | None = None) -> None:
    if request_id is not None:
        _request_id.set(request_id)
    if run_id is not None:
        _run_id.set(run_id)


@contextmanager
def log_context(
    *,
    request_id: str | None = None,
    run_id: str | None = None,
) -> Iterator[None]:
    tokens: list[tuple[ContextVar[str | None], object]] = []
    if request_id is not None:
        tokens.append((_request_id, _request_id.set(request_id)))
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
