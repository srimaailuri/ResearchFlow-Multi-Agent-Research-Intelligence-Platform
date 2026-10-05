import pytest

from researchflow.config.settings import get_settings


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used-in-unit-tests")
    monkeypatch.setenv("CHECKPOINTER_MODE", "none")
    monkeypatch.setenv("LOG_JSON", "false")
    get_settings.cache_clear()
    from researchflow.llm import clear_llm_cache

    clear_llm_cache()
    yield
    get_settings.cache_clear()
    clear_llm_cache()
