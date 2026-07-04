# Modal GPU Container

This directory contains the GPU video reframer wrapped as a Modal function.

## Files

| File | Purpose |
|------|---------|
| `reframer.py` | The full v8 reframer script (4900+ lines), refactored to expose `run_automation()` and `run_manifest()` callable entry points instead of a `__main__` block. |
| `modal_app.py` | Modal application definition — builds the CUDA image, downloads TalkNet/YOLO weights at build time, exposes 4 remote functions. |
| `requirements.txt` | Python deps for the GPU container (mirrors Modal image for local reference). |

## Deploy

```bash
# Install Modal CLI
pip install modal
modal token new --token-id "..." --token-secret "..."  # from modal.com settings

# Create required secrets (one-time)
modal secret create groq-api-key GROQ_API_KEY=gsk_xxx
modal secret create pexels-api-key PEXELS_API_KEY=xxx   # optional
modal secret create supabase-credentials \
    SUPABASE_URL=https://xxx.supabase.co \
    SUPABASE_SERVICE_KEY=eyJhbG...

# Deploy
cd modal_gpu/
modal deploy modal_app.py
```

After deploy, Modal prints the function URLs. The FastAPI backend calls these via the `modal-client` Python package — no HTTP needed.

## Local Testing

```bash
# Run the automation pipeline locally on Modal
modal run modal_app.py --url "https://youtube.com/watch?v=..."

# Run a manual manifest
modal run modal_app.py --manifest-path /path/to/manifest.json

# Health check (no GPU spawn)
modal run modal_app.py::health
```

## Cost Control

- Default GPU: **A10G** (~$0.000752/sec)
- Cold start: ~60–90s (model downloads cached in image)
- Typical 5-clip batch: 8–15 min runtime → ~$0.40–$0.70 per batch
- Set `min_containers=0` (default) so container scales to zero when idle
- The 1-hour `timeout` is a safety ceiling, not billed unless hit

## Model Cache Volumes

- `clipskari-models` (mounted at `/opt/models`) — caches Whisper/HF model weights across cold starts
- `clipskari-working` (mounted at `/tmp/working`) — ephemeral clip storage during processing

Both are persistent Modal Volumes — first cold start populates them, subsequent starts are fast.
