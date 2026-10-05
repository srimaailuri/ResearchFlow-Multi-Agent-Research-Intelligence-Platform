"""Retryable LLM invocation for transient Gemini errors (429/503)."""

from __future__ import annotations

from typing import TypeVar

from langchain_core.runnables import Runnable
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from researchflow.config.settings import get_settings

T = TypeVar("T")


def _is_retryable_llm_error(exc: BaseException) -> bool:
    message = str(exc).upper()
    markers = (
        "429",
        "503",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "HIGH DEMAND",
        "QUOTA",
    )
    return any(marker in message for marker in markers)


def invoke_with_retry(runnable: Runnable, messages: list) -> T:
    """Invoke a LangChain runnable with exponential backoff on rate-limit errors."""
    settings = get_settings()
    max_attempts = max(1, settings.llm_max_retries)

    @retry(
        retry=retry_if_exception(_is_retryable_llm_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=settings.llm_retry_min_wait_seconds,
            max=60,
            jitter=1.0,
        ),
        reraise=True,
    )
    def _call() -> T:
        return runnable.invoke(messages)

    return _call()
