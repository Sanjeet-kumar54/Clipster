"""
Pydantic schemas for request/response bodies.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# ── Auth ────────────────────────────────────────────────────────────────
class MagicLinkRequest(BaseModel):
    email: str = Field(..., description="User email for magic link")
    redirect_to: Optional[str] = Field(default=None, description="Post-login redirect URL")


class MagicLinkResponse(BaseModel):
    success: bool
    message: str


class SessionResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    user: dict


# ── Jobs ────────────────────────────────────────────────────────────────
BatchConfigOverrides = dict[str, Any]
PipelineConfig = dict[str, Any]


class CreateAutomationJobRequest(BaseModel):
    """YouTube URL → auto-selected reframed clips."""
    source_url: HttpUrl = Field(..., description="YouTube video URL")
    title: Optional[str] = None
    min_clips: int = Field(default=5, ge=1, le=20)
    max_clips: int = Field(default=10, ge=1, le=20)
    min_clip_sec: int = Field(default=30, ge=10, le=120)
    max_clip_sec: int = Field(default=45, ge=10, le=180)
    whisper_model: str = Field(default="large-v3")
    language: Optional[str] = Field(default=None, description="ISO code e.g. 'en' or 'hi'")
    caption_language: Literal["hinglish", "english"] = "hinglish"
    batch_config: BatchConfigOverrides = Field(default_factory=dict)


class CreateManifestJobRequest(BaseModel):
    """Manual mode: caller supplies explicit clip definitions."""
    title: Optional[str] = None
    manifest: dict = Field(..., description="Manifest dict with 'batch' and 'clips' keys")


class JobResponse(BaseModel):
    id: str
    status: str
    mode: str
    source_url: Optional[str]
    title: Optional[str]
    clips_count: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    elapsed_sec: Optional[float]
    error_message: Optional[str]
    batch_config: dict
    qc_grid_url: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class JobDetailResponse(JobResponse):
    manifest: Optional[dict]
    pipeline_config: Optional[dict]
    clips: list[dict]
    logs: list[dict]


# ── Clips ───────────────────────────────────────────────────────────────
class ClipResponse(BaseModel):
    id: str
    job_id: str
    index_in_job: int
    caption: Optional[str]
    subtext: Optional[str]
    duration_sec: Optional[float]
    score: Optional[float]
    storage_path: str
    signed_url: Optional[str]
    file_size_bytes: Optional[int]
    theme: Optional[str]
    color_grading: Optional[str]
    status: str
    created_at: datetime


class ClipDownloadResponse(BaseModel):
    url: str
    expires_in_sec: int


# ── Health ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    environment: str
    services: dict[str, Any]


# ── User stats ──────────────────────────────────────────────────────────
class UserStatsResponse(BaseModel):
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_clips: int
    credits_remaining: int
    last_job_at: Optional[datetime]
