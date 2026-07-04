"""Clips router — list, download, delete individual output clips."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from deps import CurrentUser, SettingsDep, SupabaseAdmin
from schemas import ClipDownloadResponse
from services.supabase_db import create_signed_url, get_clip

router = APIRouter(prefix="/clips", tags=["clips"])
logger = logging.getLogger(__name__)


@router.get("/{clip_id}/download")
async def download_clip(
    clip_id: str,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    settings: SettingsDep,
):
    """Generate a fresh signed URL and redirect to it.

    Returns a 307 redirect to the Supabase Storage signed URL.
    The frontend can also fetch the URL via the /url endpoint below.
    """
    clip = get_clip(admin, clip_id, user_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.get("storage_path"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Clip is not yet ready for download",
        )
    url = create_signed_url(
        admin, "clips", clip["storage_path"],
        expiry_sec=settings.signed_url_expiry_sec,
    )
    return RedirectResponse(url=url, status_code=307)


@router.get("/{clip_id}/url", response_model=ClipDownloadResponse)
async def get_clip_url(
    clip_id: str,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
    settings: SettingsDep,
):
    """Get a fresh signed URL for downloading the clip."""
    clip = get_clip(admin, clip_id, user_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.get("storage_path"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Clip is not yet ready for download",
        )
    url = create_signed_url(
        admin, "clips", clip["storage_path"],
        expiry_sec=settings.signed_url_expiry_sec,
    )
    return ClipDownloadResponse(
        url=url,
        expires_in_sec=settings.signed_url_expiry_sec,
    )


@router.delete("/{clip_id}", status_code=204)
async def delete_clip(
    clip_id: str,
    user_id: CurrentUser,
    admin: SupabaseAdmin,
):
    """Delete a single clip (storage + DB row)."""
    clip = get_clip(admin, clip_id, user_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Delete storage object
    if clip.get("storage_path"):
        try:
            admin.storage.from_("clips").remove([clip["storage_path"]])
        except Exception as e:
            logger.warning("Failed to delete storage object %s: %s",
                           clip["storage_path"], e)

    # Delete DB row
    admin.table("clips").delete().eq("id", clip_id).eq("user_id", user_id).execute()
    return None
