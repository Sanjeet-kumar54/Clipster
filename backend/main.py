"""
ClipSkari FastAPI entrypoint.

Run locally:
    uvicorn main:app --reload --port 8000

Run in production:
    gunicorn -k uvicorn.workers.UvicornWorker -w 4 main:app
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import Settings, get_settings
from routers import auth, clips, config_ref, health, jobs
from services.scheduler import get_scheduler


# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clipskari")


# ── App lifecycle ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background scheduler on app startup, stop on shutdown."""
    settings = get_settings()
    logger.info("Starting %s v%s (%s)",
                settings.app_name, settings.app_version, settings.environment)
    if not settings.is_configured:
        logger.warning("⚠ Supabase not configured — API will fail on protected routes")
    if not settings.is_modal_configured:
        logger.warning("⚠ Modal not configured — running in STUB mode (no real GPU calls)")

    # Start the job scheduler (skipped if not configured)
    if settings.is_configured:
        try:
            scheduler = get_scheduler(settings)
            await scheduler.start()
        except Exception as e:
            logger.error("Failed to start scheduler: %s", e)

    yield

    # Shutdown
    if settings.is_configured:
        try:
            scheduler = get_scheduler(settings)
            await scheduler.stop()
        except Exception:
            pass
    logger.info("Shutdown complete")


# ── App ─────────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Long-video to short-video converter API — converts podcasts & long videos into vertical social-ready clips.",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ─────────────────────────────────────────────────────────────
API_PREFIX = settings.api_prefix  # /api/v1
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(clips.router, prefix=API_PREFIX)
app.include_router(config_ref.router, prefix=API_PREFIX)


# ── Root ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": f"{API_PREFIX}/docs" if settings.environment != "production" else None,
        "health": f"{API_PREFIX}/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level="info",
    )
