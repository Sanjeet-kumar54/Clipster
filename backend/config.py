"""
Application configuration — all secrets & tunables loaded from env.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────────────────
    app_name: str = "ClipSkari API"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
    )

    # ── Supabase ────────────────────────────────────────────────
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_anon_key: str = Field(default="", description="Public anon key (frontend-safe)")
    supabase_service_key: str = Field(default="", description="Service role key (server-only!)")
    supabase_jwt_secret: str = Field(default="", description="Supabase JWT secret for token verification")

    # ── Modal ───────────────────────────────────────────────────
    modal_app_name: str = "clipskari-reframer"
    modal_token_id: str = ""
    modal_token_secret: str = ""
    modal_deployed: bool = Field(
        default=False,
        description="If true, call .remote() on deployed functions. If false, use local stub.",
    )

    # ── Job scheduling ──────────────────────────────────────────
    job_poll_interval_sec: int = 5
    job_max_runtime_sec: int = 60 * 60  # 1 hour
    job_max_concurrent_per_user: int = 1

    # ── Storage ─────────────────────────────────────────────────
    storage_bucket_clips: str = "clips"
    storage_bucket_qc: str = "qc-grids"
    signed_url_expiry_sec: int = 60 * 60 * 24 * 7  # 7 days

    # ── LLM (Groq) ──────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3-32b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ── Pexels (optional, for B-roll) ───────────────────────────
    pexels_api_key: str = ""

    # ── Misc ────────────────────────────────────────────────────
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def is_configured(self) -> bool:
        """True if all critical secrets are present."""
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def is_modal_configured(self) -> bool:
        return bool(self.modal_token_id and self.modal_token_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
