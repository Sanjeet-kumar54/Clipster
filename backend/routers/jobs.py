"""Jobs router — submit, list, get, cancel, delete reframing jobs."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from config import Settings
from deps import CurrentUser, SettingsDep, SupabaseAdmin
from schemas import (
    CreateAutomationJobRequest,
    CreateManifestJobRequest,
    JobDetailResponse,
    JobListResponse,
    JobResponse,
)
from services.modal_client import get_modal_client
from services.supabase_db import (
    consume_credit,
    create_job,
    delete_job,
    get_job,
    get_job_with_clips,
    list_jobs,
    update_job,
)
from supabase import Client

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)


def _row_to_job(row: dict, qc_grid_url: Optional[str] = None) -> JobResponse:
    return JobResponse(
        id=row["id"],
        status=row["status"],
        mode=row["mode"],
        source_url=row.get("source_url"),
        title=row.get("title"),
        clips_count=row.get("clips_count", 0),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        elapsed_sec=row.get("elapsed_sec"),
        error_message=row.get("error_message"),
        batch_config=row.get("batch_config", {}) or {},
        qc_grid_url=qc_grid_url,
    )


@router.post("/automation", response_model=JobResponse, status_code=201)
async def create_automation_job(
    req: CreateAutomationJobRequest,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    settings: SettingsDep,
):
    """Submit a new automation job: YouTube URL → auto-selected reframed clips."""
    # Consume a credit first
    if not consume_credit(admin, user_id):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="No credits remaining. Upgrade your plan or wait for refills.",
        )

    pipeline_config = {
        "source_url": str(req.source_url),
        "min_clips": req.min_clips,
        "max_clips": req.max_clips,
        "min_clip_sec": req.min_clip_sec,
        "max_clip_sec": req.max_clip_sec,
        "whisper_model": req.whisper_model,
        "language": req.language,
        "caption_language": req.caption_language,
        "llm_config": {
            "api_key": settings.groq_api_key,
            "base_url": settings.groq_base_url,
            "model": settings.groq_model,
        },
    }

    # Insert job row
    job = create_job(
        admin,
        user_id=user_id,
        mode="automation",
        source_url=str(req.source_url),
        manifest=None,
        batch_config=req.batch_config,
        pipeline_config=pipeline_config,
        title=req.title,
    )

    # Spawn Modal function
    modal_client = get_modal_client(settings)
    call_id = modal_client.spawn_automation(
        job_id=job["id"],
        pipeline_config=pipeline_config,
        batch_overrides=req.batch_config,
    )

    # Update job with Modal call ID & mark as running
    updated = update_job(
        admin,
        job["id"],
        modal_call_id=call_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    job.update(updated)
    return _row_to_job(job)


@router.post("/manifest", response_model=JobResponse, status_code=201)
async def create_manifest_job(
    req: CreateManifestJobRequest,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    settings: SettingsDep,
):
    """Submit a new manifest job: caller supplies explicit clip definitions."""
    if not req.manifest.get("clips"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Manifest must include at least one clip",
        )

    if not consume_credit(admin, user_id):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="No credits remaining.",
        )

    batch_config = req.manifest.get("batch", {})
    job = create_job(
        admin,
        user_id=user_id,
        mode="manifest",
        source_url=None,
        manifest=req.manifest,
        batch_config=batch_config,
        pipeline_config=None,
        title=req.title,
    )

    modal_client = get_modal_client(settings)
    call_id = modal_client.spawn_manifest(
        job_id=job["id"],
        manifest=req.manifest,
    )

    updated = update_job(
        admin,
        job["id"],
        modal_call_id=call_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    job.update(updated)
    return _row_to_job(job)


@router.get("", response_model=JobListResponse)
async def list_user_jobs(
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List the current user's jobs, newest first."""
    rows = list_jobs(
        admin, user_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    jobs = [_row_to_job(r) for r in rows]
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job_detail(
    job_id: str,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    settings: SettingsDep,
):
    """Get full job detail including clips and recent logs."""
    # Use the RPC for efficient single-call fetch
    data = get_job_with_clips(admin, job_id)
    if not data or not data.get("job"):
        raise HTTPException(status_code=404, detail="Job not found")

    job_row = data["job"]
    if job_row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Refresh signed URLs for clips
    clips = data.get("clips", [])
    for clip in clips:
        if clip.get("storage_path"):
            try:
                from services.supabase_db import create_signed_url
                url = create_signed_url(
                    admin, "clips", clip["storage_path"],
                    expiry_sec=settings.signed_url_expiry_sec,
                )
                clip["signed_url"] = url
            except Exception as e:
                logger.warning("Failed to refresh signed URL for clip %s: %s",
                               clip.get("id"), e)

    # Generate QC grid URL if present
    qc_grid_url = None
    if job_row.get("qc_grid_path"):
        try:
            from services.supabase_db import create_signed_url
            qc_grid_url = create_signed_url(
                admin, "qc-grids", job_row["qc_grid_path"],
                expiry_sec=settings.signed_url_expiry_sec,
            )
        except Exception:
            pass

    return JobDetailResponse(
        **_row_to_job(job_row, qc_grid_url=qc_grid_url).model_dump(),
        manifest=job_row.get("manifest"),
        pipeline_config=job_row.get("pipeline_config"),
        clips=clips,
        logs=data.get("logs", []),
    )


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    settings: SettingsDep,
):
    """Cancel a running job."""
    job = get_job(admin, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("queued", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status '{job['status']}'",
        )

    modal_client = get_modal_client(settings)
    if job.get("modal_call_id"):
        modal_client.cancel_call(job["modal_call_id"])

    updated = update_job(
        admin,
        job_id,
        status="cancelled",
        completed_at=datetime.now(timezone.utc).isoformat(),
        error_message="Cancelled by user",
    )
    return _row_to_job({**job, **updated})


@router.delete("/{job_id}", status_code=204)
async def remove_job(
    job_id: str,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
):
    """Delete a job and its clips (storage + DB)."""
    if not delete_job(admin, job_id, user_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return None
