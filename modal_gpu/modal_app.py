"""
Modal GPU application — exposes the v8 reframer as remote-callable functions.

Deployment:
    modal deploy modal_app.py          # deploy to Modal cloud
    modal serve modal_app.py           # live-reload for local dev

Functions exposed:
    run_automation.remote(...)  -> dict  (YouTube URL → auto-selected reframed clips)
    run_manifest.remote(...)    -> dict  (Manual clip definitions)
    upload_clip.remote(...)     -> bytes (Fetch a single output clip's bytes)
    health.remote()             -> dict  (Container sanity check)

Secrets (defined on Modal dashboard or via `modal secret create`):
    GROQ_API_KEY       — required for LLM scoring + caption generation
    PEXELS_API_KEY     — optional, for B-roll insertion
    SUPABASE_URL       — required, to write job status back to DB
    SUPABASE_SERVICE_KEY — required, server-side Supabase access
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path

import modal

# ── Stub/app definition ────────────────────────────────────────────────
app = modal.App("clipskari-reframer")

# ── Image: CUDA + Python + all heavy deps ──────────────────────────────
# The image is cached by Modal so subsequent cold starts are fast.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ffmpeg", "git", "wget", "curl",
        "fonts-noto-color-emoji", "fonts-noto-core",
        "fonts-noto-cjk",  # Devanagari + CJK fallback
    )
    .pip_install(
        # Core ML
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        "ultralytics==8.2.78",
        "filterpy==1.4.5",
        "opencv-python-headless==4.10.0.84",
        "Pillow==10.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "python_speech_features==0.6",
        # Whisper + ASR
        "faster-whisper==1.0.3",
        "ctranslate2==4.3.1",
        # YouTube
        "yt-dlp==2024.8.6",
        # HTTP / LLM
        "requests==2.32.3",
        # Supabase
        "supabase==2.7.4",
        # Logging / utils
        "tqdm==4.66.5",
    )
    # ── Clone TalkNet-ASD repo into /opt/talknet ──
    .run_commands(
        "git clone --depth 1 https://github.com/TaoRuijie/TalkNet-ASD.git /opt/talknet",
    )
    # ── Download TalkNet pretrained weights ──
    .run_commands(
        'pip install --quiet gdown==6.0.0',
        'gdown --id 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O /opt/talknet/pretrain_TalkSet.model || echo "TalkNet weights download failed"',
    )
    # ── Download YOLO face models ──
    .run_commands(
        'wget -q -O /opt/yolov8n-face.pt "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt" || true',
        'wget -q -O /opt/yolov11n-face.pt "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt" || true',
    )
    # ── Copy the reframer script into the image ──
    .add_local_file("reframer.py", "/root/reframer.py")
    .run_commands("mkdir -p /opt/models /tmp/working")
    .env({
        "TALKNET_REPO": "/opt/talknet",
        "YOLO_MODELS_DIR": "/opt",
        "HF_HOME": "/opt/hf_cache",
        "XDG_CACHE_HOME": "/opt/cache",
    })
)

# ── Modal volume for model caching across cold starts ──────────────────
model_cache = modal.Volume.from_name("clipskari-models", create_if_missing=True)
working_dir = modal.Volume.from_name("clipskari-working", create_if_missing=True)


# ── Helper: patch reframer paths before importing ─────────────────────
def _bootstrap_env():
    """Set up env vars so the reframer finds models in the right places."""
    os.environ["TALKNET_REPO"] = "/opt/talknet"
    # The reframer's `download_face_model` will look for these paths first
    os.environ["YOLO_FACE_DIR"] = "/opt"
    # Working directory (replaces /kaggle/working)
    os.makedirs("/tmp/working", exist_ok=True)
    os.chdir("/tmp/working")


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=60 * 60,           # 1 hour max per job
    volumes={
        "/opt/models": model_cache,
        "/tmp/working": working_dir,
    },
    secrets=[
        modal.Secret.from_name("groq-api-key"),
        modal.Secret.from_name("pexels-api-key", required=False),
        modal.Secret.from_name("supabase-credentials", required=False),
    ],
)
def run_automation(
    pipeline_config: dict,
    batch_overrides: dict | None = None,
) -> dict:
    """Run full automation pipeline: YouTube URL → reframed clips.

    Args (mirrors reframer.run_automation):
        pipeline_config: must include `source_url`.
        batch_overrides: BatchConfig field overrides (theme, FX, etc.)

    Returns:
        dict with keys: status, clips, qc_grid_path, manifest, elapsed_sec, error?
    """
    _bootstrap_env()
    sys.path.insert(0, "/root")
    import reframer

    # Pull secrets from Modal-injected env
    secrets = {
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
        "pexels_api_key": os.environ.get("PEXELS_API_KEY", ""),
    }

    # The reframer hard-codes /kaggle/working in some places. Create a symlink
    # so /kaggle/working → /tmp/working transparently.
    os.makedirs("/kaggle", exist_ok=True)
    if not os.path.exists("/kaggle/working"):
        os.symlink("/tmp/working", "/kaggle/working")

    result = reframer.run_automation(pipeline_config, batch_overrides, secrets)

    # Upload each output clip to Supabase Storage (if creds available)
    sb_url = os.environ.get("SUPABASE_URL")
    sb_key = os.environ.get("SUPABASE_SERVICE_KEY")
    job_id = pipeline_config.get("job_id")

    if sb_url and sb_key and job_id and result.get("status") == "ok":
        try:
            from supabase import create_client
            sb = create_client(sb_url, sb_key)
            for clip in result["clips"]:
                local_path = clip["output_path"]
                if not os.path.exists(local_path):
                    continue
                object_key = f"{job_id}/{os.path.basename(local_path)}"
                with open(local_path, "rb") as f:
                    sb.storage.from_("clips").upload(
                        path=object_key,
                        file=f,
                        file_options={"content-type": "video/mp4", "upsert": "true"},
                    )
                # Generate a signed URL valid for 7 days
                signed = sb.storage.from_("clips").create_signed_url(
                    object_key, expires_in=60 * 60 * 24 * 7
                )
                clip["storage_path"] = object_key
                clip["signed_url"] = signed.get("signedURL")
                # Upload QC grid too
            if result.get("qc_grid_path") and os.path.exists(result["qc_grid_path"]):
                with open(result["qc_grid_path"], "rb") as f:
                    sb.storage.from_("clips").upload(
                        path=f"{job_id}/QC_PREVIEW.png",
                        file=f,
                        file_options={"content-type": "image/png", "upsert": "true"},
                    )
        except Exception as e:
            result["storage_warning"] = f"Supabase upload failed: {e}"

    # Commit working volume
    working_dir.commit()
    return result


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={
        "/opt/models": model_cache,
        "/tmp/working": working_dir,
    },
    secrets=[
        modal.Secret.from_name("groq-api-key"),
        modal.Secret.from_name("pexels-api-key", required=False),
    ],
)
def run_manifest(manifest: dict) -> dict:
    """Run manual mode: caller supplies explicit clip definitions."""
    _bootstrap_env()
    sys.path.insert(0, "/root")
    import reframer

    os.makedirs("/kaggle", exist_ok=True)
    if not os.path.exists("/kaggle/working"):
        os.symlink("/tmp/working", "/kaggle/working")

    secrets = {
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
        "pexels_api_key": os.environ.get("PEXELS_API_KEY", ""),
    }
    result = reframer.run_manifest(manifest, secrets)
    working_dir.commit()
    return result


@app.function(image=image, volumes={"/tmp/working": working_dir})
def fetch_clip_output(relative_path: str) -> bytes:
    """Fetch the bytes of a previously generated clip output.

    Args:
        relative_path: Path relative to /tmp/working, e.g. "output_01.mp4"

    Returns:
        bytes: The file contents.
    """
    full = os.path.join("/tmp/working", relative_path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Clip not found: {relative_path}")
    with open(full, "rb") as f:
        return f.read()


@app.function(image=image)
def health() -> dict:
    """Sanity check — verify CUDA, dependencies, and TalkNet weights."""
    info = {"ok": True, "checks": {}}
    try:
        import torch
        info["checks"]["cuda_available"] = torch.cuda.is_available()
        info["checks"]["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        info["checks"]["torch"] = f"error: {e}"
        info["ok"] = False
    try:
        import cv2
        info["checks"]["opencv"] = cv2.__version__
    except Exception as e:
        info["checks"]["opencv"] = f"error: {e}"
        info["ok"] = False
    try:
        import faster_whisper
        info["checks"]["faster_whisper"] = faster_whisper.__version__
    except Exception as e:
        info["checks"]["faster_whisper"] = f"error: {e}"
        info["ok"] = False
    info["checks"]["talknet_weights"] = os.path.exists(
        "/opt/talknet/pretrain_TalkSet.model"
    )
    info["checks"]["yolov8n_face"] = os.path.exists("/opt/yolov8n-face.pt")
    return info


# ── Local dev entrypoint ───────────────────────────────────────────────
@app.local_entrypoint()
def main(
    url: str = "",
    manifest_path: str = "",
):
    """Local testing entrypoint.

    Usage:
        modal run modal_app.py --url "https://youtube.com/watch?v=..."
        modal run modal_app.py --manifest-path /path/to/manifest.json
    """
    if manifest_path:
        with open(manifest_path) as f:
            manifest = json.load(f)
        result = run_manifest.remote(manifest)
    else:
        if not url:
            print("❌ Either --url or --manifest-path must be provided")
            return
        result = run_automation.remote({
            "source_url": url,
            "min_clips": 3,
            "max_clips": 5,
        })
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
