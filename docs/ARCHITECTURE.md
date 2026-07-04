# Architecture

## System overview

ClipSkari is a **polyglot monorepo** with four independently deployable components:

```
                          ┌────────────────────────────────────────────────┐
                          │              USER (browser)                    │
                          └───────────────┬────────────────────────────────┘
                                          │ HTTPS
                                          ▼
                          ┌────────────────────────────────────────────────┐
                          │           VERCEL — React SPA                  │
                          │  • Vite build, served as static files          │
                          │  • Client-side routing (React Router 6)        │
                          │  • TanStack Query for server state             │
                          └───────┬───────────────────────┬────────────────┘
                                  │                       │
                  REST /api/v1/*  │                       │  Magic Link OAuth
                                  ▼                       ▼
                ┌─────────────────────────┐  ┌─────────────────────────────┐
                │  RENDER — FastAPI        │  │  SUPABASE — Auth            │
                │  • Pydantic validation   │  │  • Magic Link email OTP     │
                │  • JWT verification      │  │  • JWT issue/refresh        │
                │  • Background scheduler  │  └─────────────────────────────┘
                │  • Modal client wrapper  │
                └──────┬──────────┬────────┘
                       │          │
        Service role   │          │  .spawn() / .get()
        key (RLS bypass)│         │
                       ▼          ▼
  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐
  │  SUPABASE — Postgres    │  │  MODAL — GPU container (A10G 24GB)      │
  │  • Tables: profiles,    │  │  • TalkNet-ASD + YOLOv8-face weights    │
  │    jobs, clips, logs    │  │  • faster-whisper large-v3              │
  │  • RLS policies enforce │  │  • Reframer script (4900 lines)         │
  │    per-user isolation   │  │  • Scales to zero when idle             │
  │  • Storage: clips       │  │  • ~$0.40–$0.70 per batch (5–10 clips)  │
  │    (private), QC grids  │  └─────────────────────────────────────────┘
  │    (public)             │
  └─────────────────────────┘
```

## Why this architecture?

### Why polyglot (not Next.js monolith)?

The original reframer is a Python GPU script with heavy deps (PyTorch, faster-whisper, ultralytics, TalkNet). Bundling it into a Next.js app would require either:
1. Running a Python subprocess from Node.js (fragile, hard to debug)
2. Re-writing 4800 lines in TypeScript (infeasible for a college project)

A polyglot architecture lets each piece use the right tool: Python for ML, TypeScript for UI, SQL for storage.

### Why Modal instead of EC2/GCP?

| Concern            | Modal                              | EC2 / GCP                       |
|--------------------|------------------------------------|---------------------------------|
| Cold start         | ~60s (cached image)                | N/A (always-on)                 |
| Idle cost          | $0 (scales to zero)                | $0.50–$2/hr (24/7 GPU)          |
| Per-job cost       | ~$0.40–$0.70 for 5-clip batch      | Same compute, but 24/7 billing  |
| Cold-start deploys | Just `modal deploy`                | AMI bake + ECS config           |
| GPU options        | T4, A10G, A100, H100               | Same                            |

For a college project (irregular usage, no budget for always-on GPU), Modal's scale-to-zero is essential.

### Why Supabase instead of custom Postgres + S3?

- **Auth**: Magic Link OTP built-in, no password storage to worry about
- **Storage**: S3-compatible buckets with RLS-aware signed URLs
- **DB**: Vanilla Postgres + RLS = security model that "just works"
- **Free tier**: 500MB DB, 1GB storage, 50K MAU — plenty for a demo

### Why FastAPI instead of Next.js API routes?

- **Async-first**: Background scheduler needs `asyncio`, not Node's event loop
- **Type-safe**: Pydantic v2 validators = runtime type checking
- **Python ecosystem**: Direct `import modal` and `from supabase import create_client`
- **No CORS pain**: Clean separation between frontend and API

## Data flow

### Submitting a job

```
1. User pastes YouTube URL on /jobs/new
2. Frontend POSTs to /api/v1/jobs/automation with Bearer JWT
3. FastAPI verifies JWT against Supabase JWT secret
4. FastAPI inserts jobs row (status=queued), consumes 1 credit
5. FastAPI calls modal_client.spawn_automation() — returns immediately with call_id
6. FastAPI updates job row (status=running, modal_call_id=...)
7. FastAPI returns job to frontend
8. Frontend polls GET /api/v1/jobs/{id} every 3s
```

### Background scheduler

```
Every 5s:
  1. SELECT * FROM jobs WHERE status='running' AND modal_call_id IS NOT NULL
  2. For each job:
     - Check if Modal call completed
     - If completed:
       • INSERT clips rows (one per output video)
       • UPDATE job SET status='completed', elapsed_sec=...
     - If failed:
       • UPDATE job SET status='failed', error_message=...
     - If still running & elapsed > 1hr:
       • UPDATE job SET status='failed', error_message='timeout'
```

### GPU job (Modal)

```
1. Modal receives pipeline_config + batch_overrides
2. reframer.run_automation() executes:
   a. ClipDownloader — yt-dlp audio + section downloads
   b. WhisperTranscriber — faster-whisper large-v3 → word-level timestamps
   c. ClipScorer — rule-based + Groq LLM scoring on top-50 candidates
   d. AutoClipGenerator — select top-N non-overlapping clips
   e. BatchReframer — GPU reframe each clip with YOLO + TalkNet + Visual FX
   f. QCGen — 3×3 thumbnail contact sheet
3. Upload each output to Supabase Storage at clips/{job_id}/clip_N.mp4
4. Upload QC grid to qc-grids/{job_id}/QC_PREVIEW.png
5. Return summary dict
6. FastAPI scheduler picks up the result on next poll
```

## Security model

### Authentication
- Supabase Magic Link — user enters email → receives OTP link → exchanges for JWT
- JWT signed with HS256 + Supabase JWT secret (passed to FastAPI as env var)
- FastAPI verifies signature locally (no network round-trip per request)
- Frontend includes `Authorization: Bearer <jwt>` on every API call

### Authorization (RLS)
- Every table has `enable row level security`
- Policies enforce `auth.uid() = user_id` on SELECT/INSERT/UPDATE/DELETE
- Storage `clips` bucket: users can only access files in their folder (`{user_id}/{job_id}/...`)
- Storage `qc-grids` and `avatars`: public read, authenticated write

### Service role key
- FastAPI backend uses **service role key** to bypass RLS when:
  - Updating job status from the GPU worker (no user session available)
  - Inserting log lines on behalf of the GPU job
  - Refreshing signed URLs
- **Never exposed to frontend** — only in FastAPI env vars
- Frontend uses **anon key** only — RLS-enforced

## Cost control

| Resource             | Free tier                      | Used per batch           |
|----------------------|--------------------------------|--------------------------|
| Supabase             | 500MB DB, 1GB storage, 50K MAU | ~50MB DB, ~500MB storage |
| Modal                | $30 free credit                | ~$0.50 per batch         |
| Groq                 | Free tier (rate-limited)       | 50 LLM calls per batch   |
| Vercel               | Free hobby tier                | Static hosting only      |
| Render (FastAPI)     | Free instance (sleeps)         | Always-on API            |

Estimated cost for full college demo (50 batches): **$25 on Modal, $0 elsewhere**.

## Failure modes & handling

| Failure                    | Detection                         | Recovery                                |
|----------------------------|-----------------------------------|-----------------------------------------|
| Modal cold start timeout   | Scheduler timeout (1hr)           | Job marked `failed`, user gets refund?  |
| TalkNet weights missing    | `health_check()` returns `false`  | Fallback to lip-keypoints scoring       |
| YOLO model download fails  | Same as above                     | Fallback to YOLOv8 body detection       |
| Whisper OOM                | Exception in Modal container      | Caught, returned as `error` in result   |
| Supabase RLS violation     | 403 from Supabase client          | Propagated to user as 403               |
| Expired signed URL         | 400 on Storage GET                | Frontend re-fetches via `/clips/{id}/url`|
| yt-dlp download fails      | Exception in `ClipDownloader`     | Logged, that clip skipped               |

## Performance characteristics

| Operation                  | Time            | Cost         |
|----------------------------|-----------------|--------------|
| Modal cold start (first)   | 60–90s          | $0           |
| Modal warm start           | 5–10s           | $0           |
| Whisper transcription (1hr audio) | 90–120s  | $0.07–$0.10  |
| LLM scoring (50 candidates)| 30–60s          | $0 (Groq free)|
| GPU reframe (per 30s clip) | 60–120s         | $0.05–$0.10  |
| QC grid generation         | 5–10s           | $0           |
| Supabase upload (per clip) | 2–5s            | $0           |
| **Total per 5-clip batch** | **8–15 min**    | **~$0.40–0.70** |
