# Clipster

> Convert long videos (podcasts, interviews, lectures) into vertical short clips for Reels, Shorts & TikTok — fully automated, GPU-powered, production-ready.

[![Architecture](https://img.shields.io/badge/architecture-polyglot-blue)](docs/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Clipster is a full-stack college project that takes a YouTube URL and produces 5–10 ready-to-post vertical clips with AI-selected moments, smart reframing, captions, themes, and 13 visual effects.

## What it does

1. **Phase 1 — Auto-select**: Paste a YouTube URL. Whisper large-v3 transcribes word-by-word, then a rule + Groq/Qwen LLM scorer ranks 30–45s windows for hook strength and shareability.
2. **Phase 2 — GPU reframe**: YOLO face detection + TalkNet Active Speaker Detection track the speaker. A bulletproof camera reframes to 9:16 with split-screen, punch zoom, speaker glow, depth-of-field, and 13 visual FX.
3. **Phase 3 — Deliver**: 9 card color themes (Classic White, Neon Void, Brat Summer, etc.), Hinglish/English captions, audio-reactive waveform border, QC contact sheet, and one-click download.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────┐
│  React SPA  │────▶│  FastAPI     │────▶│  Modal GPU     │────▶│ Supabase │
│  (Vercel)   │     │  Backend     │     │  (A10G 24GB)   │     │ Postgres │
│             │◀────│              │◀────│                │◀────│ + Storage│
└─────────────┘     └──────────────┘     └────────────────┘     └──────────┘
     ▲                    ▲                       │                   ▲
     │ Magic Link         │ Service role key      │                   │
     │ auth via Supabase  │                       │                   │
     └────────────────────┴───────────────────────┴───────────────────┘
```

| Component      | Tech                                      | Hosting          |
|----------------|-------------------------------------------|------------------|
| Frontend       | React 18 + Vite 5 + TS + Tailwind + shadcn| Vercel           |
| Backend        | FastAPI + Python 3.11 + Pydantic 2        | Render / Fly.io  |
| GPU compute    | Modal functions on NVIDIA A10G            | Modal cloud      |
| Database       | Supabase Postgres + RLS policies          | Supabase         |
| Auth           | Supabase Magic Link (passwordless)        | Supabase         |
| Storage        | Supabase buckets (private clips + public QC) | Supabase      |
| LLM scoring    | Groq + Qwen3-32B (free tier)              | Groq cloud       |

## Project structure

```
Clipster/
├── README.md                ← you are here
├── docker-compose.yml       ← one-command local dev stack
├── .env.example             ← all env vars documented
├── frontend/                ← React SPA (Vite + TS + Tailwind)
│   ├── src/{pages,components,hooks,lib,types}
│   └── vercel.json
├── backend/                 ← FastAPI app
│   ├── main.py              ← entry, routers, lifespan
│   ├── routers/             ← auth, jobs, clips, health, config_ref
│   ├── services/            ← modal_client, supabase_db, scheduler
│   └── Dockerfile
├── modal_gpu/               ← GPU container
│   ├── reframer.py          ← full v8 reframer (4900+ lines, callable)
│   ├── modal_app.py         ← @stub.function wrappers (A10G)
│   └── requirements.txt
├── supabase/                ← DB migrations
│   └── migrations/
│       ├── 001_init.sql     ← tables, RLS, storage buckets
│       └── 002_rpcs.sql     ← helper RPCs
└── docs/
    ├── ARCHITECTURE.md      ← system design deep-dive
    ├── DEPLOYMENT.md        ← step-by-step prod setup
    └── API.md               ← REST API reference
```

## Quick start (local dev)

### Prerequisites
- Node 20+, npm 10+
- Python 3.11+
- Docker & docker-compose (for one-command setup)
- Accounts: Supabase (free), Modal (free $30 credit), Groq (free)

### 1. Clone & configure
```bash
git clone https://github.com/Sanjeet-kumar54/Clipster.git Clipster
cd Clipster
cp .env.example .env        # fill in real values
```

### 2. Start backend + DB (Docker)
```bash
docker-compose up -d
# FastAPI on http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 3. Start frontend
```bash
cd frontend
npm install
cp .env.example .env.local  # fill in Supabase URL + anon key
npm run dev
# → http://localhost:5173
```

### 4. Apply Supabase migrations
See [`supabase/README.md`](supabase/README.md) — paste the SQL into Supabase SQL Editor.

### 5. Deploy Modal (one-time)
```bash
cd modal_gpu
pip install modal
modal token new
modal secret create groq-api-key GROQ_API_KEY=gsk_xxx
modal deploy modal_app.py
```

## Deploy to production

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for the full guide:
1. Vercel: deploy `frontend/` directory
2. Render/Fly: deploy `backend/` from Dockerfile
3. Modal: `modal deploy modal_gpu/modal_app.py`
4. Supabase: link project, push migrations

## Key features

- **Magic Link auth** — passwordless, no signup friction
- **Two submission modes** — full auto (YouTube URL) or manual manifest
- **9 card themes** + 8 color grading presets + 15 toggleable visual FX
- **Live job status** — 3-second polling, streaming logs from GPU
- **One-click clip download** — fresh signed URLs, 7-day expiry
- **QC contact sheet** — 3×3 thumbnail grid for visual review
- **RLS-enforced** — users can only see their own jobs & clips

## Tech notes

- The original v8 reframer script (4800+ lines) is preserved **verbatim** in `modal_gpu/reframer.py` — only the `__main__` block was refactored into callable `run_automation()` / `run_manifest()` entry points.
- All Kaggle-specific paths (`/kaggle/working`) are redirected to `/tmp/working` at runtime via `_patch_kaggle_paths()` and a symlink.
- Modal caches the heavy CUDA image + TalkNet weights + YOLO models — cold starts after first deploy are ~60s, warm starts ~5s.
- Supabase RLS policies are strict: users can only access their own `jobs`, `clips`, and `job_logs` rows. Storage buckets enforce per-user folder isolation.

## College submission checklist

- [x] End-to-end runnable locally with `docker-compose up`
- [x] Production deployment guides for all 3 cloud services
- [x] Architecture diagram + design rationale in `docs/ARCHITECTURE.md`
- [x] REST API reference in `docs/API.md`
- [x] Environment variable documentation in `.env.example`
- [x] Database schema with RLS policies
- [x] Auth flow (Magic Link)
- [x] Background job scheduler with timeout & error handling
- [x] Cost control (Modal scales to zero)
- [x] Type-safe frontend (TypeScript strict mode)
- [x] Type-safe backend (Pydantic v2 validation)

## License

MIT — see [LICENSE](LICENSE).
