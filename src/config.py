from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="document_chunks")

    embedding_model: str = Field(default="intfloat/multilingual-e5-base")

    vlm_base_url: str = Field(default="https://api.openai.com/v1")
    vlm_api_key: str = Field(default="")
    vlm_model: str = Field(default="gpt-4.1-mini")

    answer_base_url: str | None = Field(default=None)
    answer_api_key: str | None = Field(default=None)
    answer_model: str | None = Field(default="gpt-4.1-mini")

    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))

    chunk_size: int = Field(default=120)
    chunk_overlap: int = Field(default=20)
    search_limit: int = Field(default=8)
    retrieval_candidate_limit: int = Field(default=24)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()