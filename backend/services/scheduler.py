"""
Job scheduler — background task that polls Modal for completion and
writes results back to Supabase. Runs as an asyncio task started on
FastAPI startup.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from config import Settings
from services.modal_client import ModalClient, get_modal_client
from services.supabase_db import (
    append_log,
    create_signed_url,
    insert_clips,
    list_jobs,
    update_job,
)
from supabase import create_client

logger = logging.getLogger(__name__)


class JobScheduler:
    """Background poller that finalizes Modal jobs in Supabase.

    Runs a single async loop that:
      1. Lists all jobs in 'running' state from Supabase (admin client)
      2. For each, polls its Modal call for completion
      3. On completion: writes clips to Supabase, updates job status
      4. On failure: writes error_message, sets status='failed'
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.modal: ModalClient = get_modal_client(settings)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._admin = None  # lazy init in loop

    def _get_admin(self):
        if self._admin is None:
            self._admin = create_client(
                self.settings.supabase_url, self.settings.supabase_service_key
            )
        return self._admin

    async def start(self):
        if self._task and not self._task.done():
            logger.warning("JobScheduler already running")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("JobScheduler started (interval=%ss)", self.settings.job_poll_interval_sec)

    async def stop(self):
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        logger.info("JobScheduler stopped")

    async def _run_loop(self):
        while not self._stop.is_set():
            try:
                await self._poll_once()
            except Exception as e:
                logger.exception("JobScheduler poll failed: %s", e)
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.settings.job_poll_interval_sec,
                )
            except TimeoutError:
                pass  # expected — interval elapsed, poll again

    async def _poll_once(self):
        """Single poll iteration — runs in a thread to avoid blocking."""
        await asyncio.to_thread(self._poll_sync)

    def _poll_sync(self):
        sb = self._get_admin()
        # Find all running jobs
        resp = (
            sb.table("jobs")
            .select("*")
            .eq("status", "running")
            .not_.is_("modal_call_id", "null")
            .execute()
        )
        if not resp.data:
            return

        for job in resp.data:
            call_id = job.get("modal_call_id")
            if not call_id or call_id.startswith("stub-call-"):
                # Skip stub calls — they won't return results
                continue
            try:
                result = self.modal.get_call_result(call_id, timeout=0.1)
                if result is None:
                    # Still running — check timeout
                    self._check_timeout(sb, job)
                    continue
                self._finalize_job(sb, job, result)
            except Exception as e:
                logger.error("Polling job %s failed: %s", job.get("id"), e)
                self._fail_job(sb, job, error=f"Modal call error: {e}")

    def _check_timeout(self, sb, job: dict):
        """Mark job as failed if it exceeds max runtime."""
        started_at = job.get("started_at")
        if not started_at:
            return
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        if elapsed > self.settings.job_max_runtime_sec:
            self._fail_job(
                sb,
                job,
                error=f"Job timed out after {elapsed/60:.1f} minutes",
            )

    def _finalize_job(self, sb, job: dict, result: dict):
        """Write Modal result back to Supabase."""
        job_id = job["id"]
        user_id = job["user_id"]
        status = result.get("status", "error")

        if status == "ok":
            clips_data = result.get("clips", [])
            # Insert clip rows
            if clips_data:
                insert_clips(sb, job_id, user_id, clips_data)
            # Update job status
            update_job(
                sb,
                job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                elapsed_sec=result.get("elapsed_sec"),
                clips_count=len(clips_data),
                qc_grid_path=f"{job_id}/QC_PREVIEW.png" if result.get("qc_grid_path") else None,
            )
            append_log(sb, job_id, level="info", message=f"Job completed: {len(clips_data)} clips",
                       phase="done", progress=100.0)
            logger.info("Job %s completed with %d clips", job_id, len(clips_data))
        else:
            self._fail_job(sb, job, error=result.get("error", "Unknown error"))

    def _fail_job(self, sb, job: dict, error: str):
        """Mark a job as failed."""
        job_id = job["id"]
        update_job(
            sb,
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error_message=error,
        )
        append_log(sb, job_id, level="error", message=f"Job failed: {error}", phase="done")
        logger.warning("Job %s failed: %s", job_id, error)

    def _refresh_signed_urls(self, sb, job: dict):
        """Refresh signed URLs for clips that are about to expire."""
        # TODO: implement periodic refresh for long-lived clips
        pass


# Singleton
_scheduler: Optional[JobScheduler] = None


def get_scheduler(settings: Settings = None) -> JobScheduler:
    global _scheduler
    if _scheduler is None:
        from config import get_settings
        settings = settings or get_settings()
        _scheduler = JobScheduler(settings)
    return _scheduler
