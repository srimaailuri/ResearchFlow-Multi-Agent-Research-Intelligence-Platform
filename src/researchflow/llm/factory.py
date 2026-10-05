from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from researchflow.config.settings import get_settings


@lru_cache
def _cached_chat_model(model_name: str) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Primary chat model (plan, synthesize). Cached per model name."""
    return _cached_chat_model(get_settings().model_name)


def get_evaluate_chat_model() -> ChatGoogleGenerativeAI:
    """Evaluate-stage model; defaults to primary when EVALUATE_MODEL_NAME is unset."""
    settings = get_settings()
    model_name = settings.evaluate_model_name or settings.model_name
    return _cached_chat_model(model_name)


def clear_llm_cache() -> None:
    _cached_chat_model.cache_clear()
