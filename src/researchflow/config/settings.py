from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: str
    model_name: str = "gemini-3.8-flash"
    evaluate_model_name: str | None = None
    embedding_model: str = "gemini-embedding-001"
    llm_max_retries: int = 4
    llm_retry_min_wait_seconds: float = 2.0
    max_retrieval_attempts: int = 2
    max_synthesis_attempts: int = 2
    max_evaluate_workers: int = 4
    max_concurrent_research_jobs: int = 2

    rag_documents_dir: str = "Data/test_documents"
    chroma_persist_dir: str = ".chroma"
    retrieval_top_k: int = 3
    retrieval_rag_candidate_k: int = 8
    rag_preferred_document_boost: float = 0.2
    max_evidence_per_task: int = 5
    max_total_evidence: int = 15

    checkpointer_mode: str = "memory"
    checkpoint_sqlite_path: str = ".checkpoints/researchflow.db"
    log_level: str = "INFO"
    log_json: bool = True

    benchmark_topic_pass_recall: float = 0.5
    benchmark_answer_pass_similarity: float = 0.35

    tavily_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Load settings once at first use (avoids requiring .env on import)."""
    return Settings()
