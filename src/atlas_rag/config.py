"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for AtlasRAG. Values are loaded from environment
    variables and .env file, with sensible defaults for local development."""

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "AtlasRAG"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas_rag"

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Qdrant (vector store) ────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "atlas_chunks"

    # ── Elasticsearch (BM25 keyword search) ──────────────────────────────
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "atlas_chunks"

    # ── S3 / MinIO (document storage) ────────────────────────────────────
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "atlas-documents"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # ── Authentication ───────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ── LLM ──────────────────────────────────────────────────────────────
    default_llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Celery ───────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Observability ────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
