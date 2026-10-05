from unittest.mock import MagicMock

import pytest

from researchflow.llm.invoke import _is_retryable_llm_error, invoke_with_retry


def test_retryable_error_detection() -> None:
    assert _is_retryable_llm_error(Exception("503 UNAVAILABLE high demand"))
    assert _is_retryable_llm_error(Exception("429 RESOURCE_EXHAUSTED quota"))
    assert not _is_retryable_llm_error(Exception("400 invalid argument"))


def test_invoke_with_retry_recovers(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    from researchflow.config.settings import get_settings
    from researchflow.llm import clear_llm_cache

    get_settings.cache_clear()
    clear_llm_cache()

    calls = {"n": 0}

    def flaky_invoke(_messages):
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception("503 UNAVAILABLE temporary")
        return "ok"

    runnable = MagicMock()
    runnable.invoke.side_effect = flaky_invoke

    assert invoke_with_retry(runnable, []) == "ok"
    assert calls["n"] == 2


def test_invoke_with_retry_not_retryable(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    from researchflow.config.settings import get_settings
    from researchflow.llm import clear_llm_cache

    get_settings.cache_clear()
    clear_llm_cache()

    runnable = MagicMock()
    runnable.invoke.side_effect = Exception("400 bad request")

    with pytest.raises(Exception, match="400"):
        invoke_with_retry(runnable, [])
