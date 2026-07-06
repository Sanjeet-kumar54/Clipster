"""
Dependency injection — auth, db, modal clients.
"""
from __future__ import annotations

from typing import Annotated, Optional

import jwt
from fastapi import Depends, Header, HTTPException, status
from supabase import Client, create_client

from config import Settings, get_settings


# ── Supabase clients ────────────────────────────────────────────────────
def get_supabase_anon(settings: Settings = Depends(get_settings)) -> Client:
    """Anon client — acts as the requesting user (RLS enforced)."""
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase anon credentials not configured",
        )
    return create_client(settings.supabase_url, settings.supabase_anon_key)


def get_supabase_admin(settings: Settings = Depends(get_settings)) -> Client:
    """Admin client — uses service role key, bypasses RLS. SERVER-ONLY."""
    if not settings.supabase_url or not settings.supabase_service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase service credentials not configured",
        )
    return create_client(settings.supabase_url, settings.supabase_service_key)


# ── Auth ────────────────────────────────────────────────────────────────
def get_current_user_id(
    authorization: Annotated[Optional[str], Header()] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    """Extract & verify the Supabase JWT from the Authorization header.

    Returns the user's UUID. Raises 401 if missing/invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1]

    # 1. Try local verification (Fast Path)
    if settings.supabase_jwt_secret:
        try:
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
            user_id = payload.get("sub")
            if user_id:
                return user_id
        except Exception as e:
            logger.info("Local JWT verification failed (falling back to network check): %s", e)

    # 2. Fallback: verify against Supabase auth.getUser (Network Round-Trip)
    try:
        anon = get_supabase_anon(settings)
        anon.auth.set_session(token, "")  # Set session for this call
        resp = anon.auth.get_user(token)
        if not resp.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return resp.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


# Common dependency aliases
CurrentUser = Annotated[str, Depends(get_current_user_id)]
SupabaseAdmin = Annotated[Client, Depends(get_supabase_admin)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
