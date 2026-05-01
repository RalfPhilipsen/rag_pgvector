from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"

    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_dim: int = 768
    chat_model: str = "openai/gpt-oss-20b"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5


settings = Settings()
