"""Validated environment-driven configuration."""
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    llm_mode: str = "mock"
    groq_api_key: str | None = None
    groq_model: str = ""
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None
    log_level: str = "INFO"
    batch_size: int = Field(default=5, ge=1, le=50)
    max_articles: int = Field(default=20, ge=1, le=500)
    max_refinement_rounds: int = Field(default=3, ge=0, le=10)
    max_llm_calls: int = Field(default=30, ge=1, le=500)
    output_dir: Path = Path("data/results")
    request_timeout: float = Field(default=30, gt=0, le=180)
    @model_validator(mode="after")
    def provider_config(self):
        if self.llm_mode not in {"mock", "groq"}:
            raise ValueError("LLM_MODE must be mock or groq")
        if self.llm_mode == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required in Groq mode")
        return self

settings = Settings()
