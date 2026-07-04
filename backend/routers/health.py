"""Health check router."""
from __future__ import annotations

from fastapi import APIRouter

from config import Settings, get_settings
from deps import SettingsDep
from schemas import HealthResponse
from services.modal_client import get_modal_client

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep):
    """Liveness + readiness probe."""
    services = {
        "supabase": "ok" if settings.is_configured else "missing",
        "modal": "ok" if settings.is_modal_configured else "stub",
        "groq": "ok" if settings.groq_api_key else "missing",
    }
    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        services=services,
    )


@router.get("/health/modal")
async def modal_health():
    """Deep Modal container check (spawns a GPU container — slow!).
    Use sparingly, e.g. on Settings page refresh.
    """
    client = get_modal_client()
    return client.health_check()
