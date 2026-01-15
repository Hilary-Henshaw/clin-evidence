"""Application settings loaded from environment variables."""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All configuration for ClinEvidence, sourced from env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Azure OpenAI LLM ──────────────────────────────────────────
    llm_deployment: str = Field(alias="deployment_name")
    llm_model: str = Field(default="gpt-4o", alias="model_name")
    azure_endpoint: str
    openai_api_key: SecretStr
    openai_api_version: str = "2024-02-15"

    # ── Azure OpenAI Embeddings ───────────────────────────────────
    embedding_deployment: str = Field(alias="embedding_deployment_name")
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        alias="embedding_model_name",
    )
    embedding_azure_endpoint: str
    embedding_openai_api_key: SecretStr
    embedding_openai_api_version: str = "2024-02-15"

    # ── Eleven Labs ───────────────────────────────────────────────
    elevenlabs_api_key: SecretStr = Field(alias="ELEVEN_LABS_API_KEY")
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_monolingual_v1"

    # ── Tavily ────────────────────────────────────────────────────
    tavily_api_key: SecretStr = Field(alias="TAVILY_API_KEY")
    search_max_results: int = 5

    # ── HuggingFace ───────────────────────────────────────────────
    huggingface_token: SecretStr = Field(
        default=SecretStr(""), alias="HUGGINGFACE_TOKEN"
    )

    # ── Qdrant ────────────────────────────────────────────────────
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(
        default=None, alias="QDRANT_API_KEY"
    )

    # ── Session ───────────────────────────────────────────────────
    session_secret: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        alias="SESSION_SECRET_KEY",
    )

    # ── Knowledge Base ────────────────────────────────────────────
    kb_chunk_size: int = 512
    kb_chunk_overlap: int = 50
    kb_embedding_dim: int = 1536
    kb_retrieval_top_k: int = 5
    kb_reranker_top_k: int = 3
    kb_min_confidence: float = 0.40
    kb_max_context_tokens: int = 8192
    kb_qdrant_path: str = "./data/qdrant_db"
    kb_docs_path: str = "./data/docs_db"
    kb_collection_name: str = "clinevidence_kb"
    kb_parsed_docs_path: str = "./data/parsed_docs"

    # ── Imaging ───────────────────────────────────────────────────
    brain_mri_model_path: str = "./models/brain_mri.pth"
    chest_xray_model_path: str = "./models/chest_xray.pth"
    skin_lesion_model_path: str = "./models/skin_lesion.pth.tar"
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB
    allowed_image_formats: list[str] = [
        "png",
        "jpg",
        "jpeg",
    ]

    # ── Server ────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    rate_limit_per_minute: int = 30
    upload_dir: str = "./uploads"

    # ── Agent Temperatures ────────────────────────────────────────
    decision_temperature: float = 0.1
    conversation_temperature: float = 0.7
    rag_temperature: float = 0.3
    search_temperature: float = 0.3
    chunking_temperature: float = 0.0

    # ── Routing ───────────────────────────────────────────────────
    routing_confidence_threshold: float = 0.85
    max_conversation_messages: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()  # type: ignore[call-arg]
    logger.info(
        "Settings loaded",
        extra={
            "llm_model": settings.llm_model,
            "kb_collection": settings.kb_collection_name,
        },
    )
    return settings
