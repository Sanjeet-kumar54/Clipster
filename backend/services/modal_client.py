"""
Modal client — wraps calls to the deployed GPU functions.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from config import Settings

logger = logging.getLogger(__name__)


class ModalClient:
    """Thin wrapper around Modal remote function calls.

    Falls back to a stub implementation if Modal isn't configured
    (useful for local FastAPI development without spinning up GPUs).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._app = None
        self._run_automation = None
        self._run_manifest = None
        self._fetch_clip = None
        self._health = None
        if settings.is_modal_configured:
            try:
                import modal

                self._run_automation = modal.Function.from_name(
                    settings.modal_app_name, "run_automation"
                )
                self._run_manifest = modal.Function.from_name(
                    settings.modal_app_name, "run_manifest"
                )
                self._fetch_clip = modal.Function.from_name(
                    settings.modal_app_name, "fetch_clip_output"
                )
                self._health = modal.Function.from_name(
                    settings.modal_app_name, "health"
                )
                logger.info("Modal client connected to app: %s", settings.modal_app_name)
            except Exception as e:
                logger.warning("Modal client init failed: %s — using stub mode", e)
                self._app = None

    @property
    def is_live(self) -> bool:
        return self._app is not None

    # ── Public API ──────────────────────────────────────────────

    def spawn_automation(
        self,
        job_id: str,
        pipeline_config: dict,
        batch_overrides: Optional[dict] = None,
    ) -> str:
        """Spawn the automation pipeline asynchronously. Returns Modal call ID."""
        pipeline_config = {**pipeline_config, "job_id": job_id}
        if self.is_live:
            call = self._run_automation.spawn(pipeline_config, batch_overrides or {})
            return call.object_id
        # Stub mode — log and return fake call ID
        logger.warning(
            "[STUB] Would spawn automation: job=%s url=%s",
            job_id,
            pipeline_config.get("source_url"),
        )
        return f"stub-call-{job_id}"

    def spawn_manifest(self, job_id: str, manifest: dict) -> str:
        """Spawn the manifest pipeline asynchronously. Returns Modal call ID."""
        if self.is_live:
            call = self._run_manifest.spawn(manifest)
            return call.object_id
        logger.warning(
            "[STUB] Would spawn manifest: job=%s clips=%d",
            job_id,
            len(manifest.get("clips", [])),
        )
        return f"stub-call-{job_id}"

    def get_call_result(self, call_id: str, timeout: float = 0.1) -> Optional[Any]:
        """Poll for a spawned call's result. Returns None if still running."""
        if not self.is_live:
            return None
        try:
            import modal

            call = modal.call_helpers.FunctionCall.from_id(call_id)
            try:
                return call.get(timeout=timeout)
            except TimeoutError:
                return None
        except Exception as e:
            logger.error("Modal get_call_result failed: %s", e)
            raise

    def is_call_running(self, call_id: str) -> bool:
        if not self.is_live:
            return False
        try:
            import modal

            call = modal.call_helpers.FunctionCall.from_id(call_id)
            return not call.is_completed()
        except Exception:
            return False

    def cancel_call(self, call_id: str) -> bool:
        if not self.is_live:
            return False
        try:
            import modal

            call = modal.call_helpers.FunctionCall.from_id(call_id)
            call.cancel()
            return True
        except Exception as e:
            logger.error("Cancel failed: %s", e)
            return False

    def health_check(self) -> dict:
        if not self.is_live:
            return {"ok": False, "error": "Modal not configured (stub mode)"}
        try:
            return self._health.remote()
        except Exception as e:
            return {"ok": False, "error": str(e)}


# Singleton
_client: Optional[ModalClient] = None


def get_modal_client(settings: Settings = None) -> ModalClient:
    global _client
    if _client is None:
        from config import get_settings
        settings = settings or get_settings()
        _client = ModalClient(settings)
    return _client
