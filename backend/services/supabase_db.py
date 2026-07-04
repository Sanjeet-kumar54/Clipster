"""
Supabase helpers — DB queries, storage operations, RPCs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from supabase import Client

logger = logging.getLogger(__name__)


# ── Jobs ────────────────────────────────────────────────────────────────
def create_job(
    sb: Client,
    *,
    user_id: str,
    mode: str,
    source_url: Optional[str],
    manifest: Optional[dict],
    batch_config: dict,
    pipeline_config: Optional[dict],
    title: Optional[str] = None,
) -> dict:
    """Insert a new job row. Returns the inserted job dict."""
    payload = {
        "user_id": user_id,
        "mode": mode,
        "source_url": source_url,
        "manifest": manifest,
        "batch_config": batch_config,
        "pipeline_config": pipeline_config,
        "title": title,
        "status": "queued",
    }
    resp = sb.table("jobs").insert(payload).execute()
    if not resp.data:
        raise RuntimeError(f"Failed to insert job: {resp}")
    return resp.data[0]


def get_job(sb: Client, job_id: str, user_id: str) -> Optional[dict]:
    """Fetch a single job. RLS ensures user_id filter is enforced."""
    resp = (
        sb.table("jobs")
        .select("*")
        .eq("id", job_id)
        .eq("user_id", user_id)
        .execute()
    )
    return resp.data[0] if resp.data else None


def list_jobs(
    sb: Client,
    user_id: str,
    *,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List jobs for a user, newest first."""
    q = sb.table("jobs").select("*").eq("user_id", user_id)
    if status:
        q = q.eq("status", status)
    q = q.order("created_at", desc=True).range(offset, offset + limit - 1)
    resp = q.execute()
    return resp.data


def update_job(sb: Client, job_id: str, **fields) -> dict:
    """Update job fields. Returns the updated row."""
    # Always touch updated_at
    fields.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    resp = sb.table("jobs").update(fields).eq("id", job_id).execute()
    return resp.data[0] if resp.data else {}


def delete_job(sb: Client, job_id: str, user_id: str) -> bool:
    """Delete a job and its clips (cascade). Returns True if deleted."""
    # First delete storage objects for associated clips
    clips_resp = (
        sb.table("clips")
        .select("storage_path")
        .eq("job_id", job_id)
        .execute()
    )
    for clip in clips_resp.data:
        path = clip.get("storage_path")
        if path:
            try:
                sb.storage.from_("clips").remove([path])
            except Exception as e:
                logger.warning("Failed to delete storage object %s: %s", path, e)
    # Delete the QC grid if exists
    job = get_job(sb, job_id, user_id)
    if job and job.get("qc_grid_path"):
        try:
            sb.storage.from_("qc-grids").remove([job["qc_grid_path"]])
        except Exception:
            pass
    # Delete DB rows (cascade handles clips & logs)
    resp = sb.table("jobs").delete().eq("id", job_id).eq("user_id", user_id).execute()
    return bool(resp.data)


# ── Clips ───────────────────────────────────────────────────────────────
def insert_clips(sb: Client, job_id: str, user_id: str, clips: list[dict]) -> list[dict]:
    """Bulk-insert clip rows for a job."""
    if not clips:
        return []
    payload = [
        {
            "job_id": job_id,
            "user_id": user_id,
            "index_in_job": idx,
            "caption": c.get("caption"),
            "subtext": c.get("subtext"),
            "source_start_sec": c.get("start_sec") or c.get("source_start_sec"),
            "source_end_sec": c.get("end_sec") or c.get("source_end_sec"),
            "duration_sec": c.get("duration_sec"),
            "score": c.get("score"),
            "storage_path": c.get("storage_path") or c.get("output_path", ""),
            "signed_url": c.get("signed_url"),
            "file_size_bytes": c.get("size_bytes") or c.get("file_size_bytes"),
            "theme": c.get("theme"),
            "color_grading": c.get("color_grading"),
            "status": "ready" if c.get("storage_path") else "pending",
        }
        for idx, c in enumerate(clips)
    ]
    resp = sb.table("clips").insert(payload).execute()
    return resp.data


def list_clips(sb: Client, job_id: str) -> list[dict]:
    resp = (
        sb.table("clips")
        .select("*")
        .eq("job_id", job_id)
        .order("index_in_job")
        .execute()
    )
    return resp.data


def get_clip(sb: Client, clip_id: str, user_id: str) -> Optional[dict]:
    resp = (
        sb.table("clips")
        .select("*")
        .eq("id", clip_id)
        .eq("user_id", user_id)
        .execute()
    )
    return resp.data[0] if resp.data else None


def create_signed_url(sb: Client, bucket: str, path: str, expiry_sec: int) -> str:
    """Generate a fresh signed URL for a storage object."""
    resp = sb.storage.from_(bucket).create_signed_url(path, expires_in=expiry_sec)
    if isinstance(resp, dict) and "signedURL" in resp:
        return resp["signedURL"]
    if isinstance(resp, dict) and "signedUrl" in resp:
        return resp["signedUrl"]
    if isinstance(resp, str):
        return resp
    raise RuntimeError(f"Unexpected signed URL response: {resp}")


# ── Job logs ────────────────────────────────────────────────────────────
def append_log(
    sb: Client,
    job_id: str,
    *,
    level: str = "info",
    message: str,
    phase: Optional[str] = None,
    progress: Optional[float] = None,
) -> dict:
    """Append a single log line."""
    payload = {
        "job_id": job_id,
        "level": level,
        "message": message,
        "phase": phase,
        "progress": progress,
    }
    resp = sb.table("job_logs").insert(payload).execute()
    return resp.data[0] if resp.data else {}


def list_logs(sb: Client, job_id: str, limit: int = 200) -> list[dict]:
    resp = (
        sb.table("job_logs")
        .select("*")
        .eq("job_id", job_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return resp.data


# ── RPCs ────────────────────────────────────────────────────────────────
def get_user_stats(sb: Client, user_id: str) -> dict:
    resp = sb.rpc("get_user_stats", {"p_user_id": user_id}).execute()
    return resp.data if resp.data else {}


def get_job_with_clips(sb: Client, job_id: str) -> dict:
    resp = sb.rpc("get_job_with_clips", {"p_job_id": job_id}).execute()
    return resp.data if resp.data else {}


def consume_credit(sb: Client, user_id: str) -> bool:
    resp = sb.rpc("consume_credit", {"p_user_id": user_id}).execute()
    return bool(resp.data)
