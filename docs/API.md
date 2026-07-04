# REST API Reference

**Base URL**: `https://clipskari-api.onrender.com/api/v1` (production)
**Local**: `http://localhost:8000/api/v1`
**Auth**: Bearer JWT in `Authorization` header (except `/auth/magic-link` and `/health`)

---

## Authentication

### POST /auth/magic-link
Send a Supabase Magic Link to the user's email.

**Request**
```json
{
  "email": "user@example.com",
  "redirect_to": "https://clipskari.vercel.app/auth/callback"
}
```

**Response 200**
```json
{
  "success": true,
  "message": "Magic link sent. Check your inbox."
}
```

### GET /auth/me
Get the current user's profile + stats.

**Response 200**
```json
{
  "profile": {
    "id": "uuid",
    "email": "user@example.com",
    "full_name": "User Name",
    "plan": "free",
    "credits": 4,
    "created_at": "2025-01-01T00:00:00Z"
  },
  "stats": {
    "total_jobs": 5,
    "completed_jobs": 4,
    "failed_jobs": 1,
    "total_clips": 23,
    "credits_remaining": 4,
    "last_job_at": "2025-01-15T10:30:00Z"
  }
}
```

### POST /auth/logout
Logout (stateless — frontend discards tokens).

---

## Jobs

### POST /jobs/automation
Submit a new automation job: YouTube URL → auto-selected reframed clips.

**Request**
```json
{
  "source_url": "https://www.youtube.com/watch?v=...",
  "title": "My podcast episode",
  "min_clips": 5,
  "max_clips": 10,
  "caption_language": "hinglish",
  "batch_config": {
    "card_theme": "neon_void",
    "target_width": 1080,
    "target_height": 1920,
    "color_grading_preset": "vibrant",
    "watermark_enabled": true,
    "watermark_path": "@clipskari",
    "punch_zoom_enabled": true,
    "speaker_glow_enabled": true
  }
}
```

**Response 201**
```json
{
  "id": "uuid",
  "status": "running",
  "mode": "automation",
  "source_url": "https://www.youtube.com/watch?v=...",
  "title": "My podcast episode",
  "clips_count": 0,
  "created_at": "2025-01-15T10:00:00Z",
  "started_at": "2025-01-15T10:00:00Z",
  "completed_at": null,
  "elapsed_sec": null,
  "error_message": null,
  "batch_config": { ... },
  "qc_grid_url": null
}
```

**Errors**
- `402 Payment Required` — no credits remaining
- `422 Unprocessable Entity` — invalid URL or batch_config

### POST /jobs/manifest
Submit a manual manifest job.

**Request**
```json
{
  "title": "Manual batch",
  "manifest": {
    "batch": {
      "card_theme": "neon_void",
      "caption_language": "hinglish"
    },
    "clips": [
      {
        "url": "https://www.youtube.com/watch?v=...",
        "start": 60,
        "end": 120,
        "caption": "🔥 Example caption"
      }
    ]
  }
}
```

### GET /jobs
List the current user's jobs.

**Query params**
- `status` (optional): `queued | running | completed | failed | cancelled`
- `limit` (default 50, max 200)
- `offset` (default 0)

**Response 200**
```json
{
  "jobs": [
    { "id": "...", "status": "completed", ... }
  ],
  "total": 5
}
```

### GET /jobs/{id}
Get full job detail including clips and recent logs.

**Response 200**
```json
{
  "id": "uuid",
  "status": "completed",
  "mode": "automation",
  "source_url": "...",
  "title": "...",
  "clips_count": 5,
  "created_at": "...",
  "started_at": "...",
  "completed_at": "...",
  "elapsed_sec": 720.5,
  "error_message": null,
  "batch_config": { ... },
  "qc_grid_url": "https://abc.supabase.co/storage/v1/object/sign/qc-grids/uuid/QC_PREVIEW.png?token=...",
  "manifest": null,
  "pipeline_config": { ... },
  "clips": [
    {
      "id": "uuid",
      "index_in_job": 0,
      "caption": "🔥 Example caption",
      "duration_sec": 42.5,
      "score": 87.3,
      "storage_path": "uuid/clip_001.mp4",
      "signed_url": "https://...",
      "file_size_bytes": 12345678,
      "theme": "neon_void",
      "color_grading": "vibrant",
      "status": "ready",
      "created_at": "..."
    }
  ],
  "logs": [
    {
      "level": "info",
      "message": "Whisper loaded",
      "phase": "whisper",
      "progress": null,
      "created_at": "..."
    }
  ]
}
```

### POST /jobs/{id}/cancel
Cancel a running job.

**Response 200**
```json
{ "id": "...", "status": "cancelled", ... }
```

**Errors**
- `400 Bad Request` — job not in `queued` or `running` state
- `404 Not Found` — job doesn't exist or doesn't belong to user

### DELETE /jobs/{id}
Delete a job and all its clips (storage + DB).

**Response 204** (no content)

---

## Clips

### GET /clips/{id}/url
Get a fresh signed URL for downloading the clip.

**Response 200**
```json
{
  "url": "https://abc.supabase.co/storage/v1/object/sign/clips/uuid/clip_001.mp4?token=...",
  "expires_in_sec": 604800
}
```

### GET /clips/{id}/download
Redirect (307) to a fresh signed URL. Useful for direct download links.

### DELETE /clips/{id}
Delete a single clip (storage + DB row).

**Response 204**

---

## Config

### GET /config/all
Get all configuration catalogs in one call.

**Response 200**
```json
{
  "themes": [
    { "id": "classic_white", "name": "Classic White", "description": "...", "swatch": { "bg": "#FFFFFF", "text": "#0A0A0A", "accent": "#FF2828" } },
    ...
  ],
  "color_grading_presets": [
    { "id": "off", "name": "Off" },
    { "id": "cinematic", "name": "Cinematic" },
    ...
  ],
  "watermark_positions": [ ... ],
  "visual_fx": [
    { "id": "punch_zoom", "name": "Punch Zoom on Speaker Switch", "default": true },
    ...
  ],
  "caption_languages": [
    { "id": "hinglish", "name": "Hinglish (Hindi+English mix)" },
    { "id": "english", "name": "English" }
  ],
  "output_resolutions": [ ... ]
}
```

---

## Health

### GET /health
Liveness + readiness probe.

**Response 200**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "production",
  "services": {
    "supabase": "ok",
    "modal": "ok",
    "groq": "ok"
  }
}
```

### GET /health/modal
Deep Modal container check (spawns a GPU — slow, ~30s). Use sparingly.

**Response 200**
```json
{
  "ok": true,
  "checks": {
    "cuda_available": true,
    "cuda_device": "NVIDIA A10G",
    "opencv": "4.10.0",
    "faster_whisper": "1.0.3",
    "talknet_weights": true,
    "yolov8n_face": true
  }
}
```

---

## Error responses

All errors follow this format:

```json
{
  "detail": "Human-readable error message"
}
```

Common status codes:
- `400 Bad Request` — invalid input
- `401 Unauthorized` — missing/invalid JWT
- `402 Payment Required` — no credits remaining
- `403 Forbidden` — RLS denied access
- `404 Not Found` — resource doesn't exist or doesn't belong to user
- `409 Conflict` — clip not yet ready for download
- `422 Unprocessable Entity` — Pydantic validation error
- `500 Internal Server Error` — unexpected server error

---

## Rate limits

- **No explicit rate limit** in the FastAPI app — relies on Supabase + Modal quotas
- Supabase: 60 req/s per user (more than enough)
- Modal: limited by GPU container concurrency (default 1 per user)
- Groq: 30 req/min on free tier

---

## SDK examples

### cURL
```bash
# Get auth token (after Magic Link flow)
TOKEN="eyJhbGc..."

# Submit a job
curl -X POST https://clipskari-api.onrender.com/api/v1/jobs/automation \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://www.youtube.com/watch?v=...",
    "batch_config": {"card_theme": "neon_void"}
  }'

# Poll for status
curl https://clipskari-api.onrender.com/api/v1/jobs/{job_id} \
  -H "Authorization: Bearer $TOKEN"
```

### JavaScript
```typescript
const API = "https://clipskari-api.onrender.com/api/v1";

async function submitJob(token: string, url: string) {
  const resp = await fetch(`${API}/jobs/automation`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_url: url,
      batch_config: { card_theme: "neon_void" },
    }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}
```

### Python
```python
import httpx

API = "https://clipskari-api.onrender.com/api/v1"

def submit_job(token: str, url: str) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{API}/jobs/automation",
            headers={"Authorization": f"Bearer {token}"},
            json={"source_url": url, "batch_config": {"card_theme": "neon_void"}},
        )
        resp.raise_for_status()
        return resp.json()
```
