"""Auth router — bridges Supabase Magic Link auth to the API."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from config import Settings, get_settings
from deps import CurrentUser, SupabaseAdmin, get_supabase_anon
from schemas import MagicLinkRequest, MagicLinkResponse
from services.supabase_db import get_user_stats

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/magic-link", response_model=MagicLinkResponse)
async def send_magic_link(
    req: MagicLinkRequest,
    settings: Settings = Depends(get_settings),
):
    """Send a Supabase Magic Link to the user's email.

    The frontend should call this, then poll for the session after the
    user clicks the link and is redirected back.
    """
    try:
        anon = get_supabase_anon(settings)
        resp = anon.auth.sign_in_with_otp(
            {"email": req.email, "options": {"email_redirect_to": req.redirect_to}}
        )
        # gotrue v2 returns response with .error attribute; newer versions raise
        err = getattr(resp, "error", None)
        if err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err.message,
            )
        return MagicLinkResponse(
            success=True,
            message="Magic link sent. Check your inbox.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Magic link send failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_me(user_id: CurrentUser, admin: SupabaseAdmin):
    """Get the current user's profile + stats."""
    profile_resp = (
        admin.table("profiles").select("*").eq("id", user_id).execute()
    )
    if not profile_resp.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    stats = get_user_stats(admin, user_id)
    return {"profile": profile_resp.data[0], "stats": stats}


@router.post("/logout")
async def logout(user_id: CurrentUser):
    """Logout — frontend should discard tokens. Supabase JWTs are stateless."""
    return {"success": True, "message": "Logged out. Discard tokens on client."}
