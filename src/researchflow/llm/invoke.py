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
from researchflow.llm.throttle import wait_before_chat_call

T = TypeVar("T")

_AUTH_MARKERS = (
    "API_KEY_INVALID",
    "API KEY EXPIRED",
    "API KEY INVALID",
    "INVALID API KEY",
    "401 UNAUTHORIZED",
    "UNAUTHENTICATED",
    "PERMISSION DENIED",
)


def _is_auth_error(exc: BaseException) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in _AUTH_MARKERS)


def _is_retryable_llm_error(exc: BaseException) -> bool:
    if _is_auth_error(exc):
        return False
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


def _auth_error_message() -> str:
    return (
        "Google API key is invalid or expired. Update GOOGLE_API_KEY in .env and "
        "create a new key at https://aistudio.google.com/apikey"
    )


def invoke_with_retry(runnable: Runnable, messages: list) -> T:
    """Invoke a LangChain runnable with exponential backoff on rate-limit errors."""
    settings = get_settings()
    max_attempts = max(1, settings.llm_max_retries)

    @retry(
        retry=retry_if_exception(_is_retryable_llm_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=settings.llm_retry_min_wait_seconds,
            max=max(60.0, float(settings.llm_retry_max_wait_seconds)),
            jitter=2.0,
        ),
        reraise=True,
    )
    def _call() -> T:
        wait_before_chat_call()
        return runnable.invoke(messages)

    try:
        return _call()
    except Exception as exc:
        if _is_auth_error(exc):
            raise RuntimeError(_auth_error_message()) from exc
        raise
