from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="OpenAI's API key")

    MODEL_EMBEDDING: str = "text-embedding-3-small"
    MODEL_CHAT: str = "gpt-4o-mini"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    TOP_K: int = 8
    PERSIST_DIR: str = ".chroma"
    COHERE_API_KEY: str | None = None
    RERANK_MODEL: str = "rerank-english=v3.0"
    RERANK_TOP_N: int = 5

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra="ignore",
    )

settings = Settings()