# Deployment Guide

This guide walks you through deploying ClipSkari to production from scratch. Total time: ~30 minutes (excluding wait time for service provisioning).

## Prerequisites

| Account         | Free tier                | Why needed                |
|-----------------|--------------------------|---------------------------|
| [Supabase](https://supabase.com) | 500MB DB, 1GB storage | DB, auth, file storage    |
| [Modal](https://modal.com)       | $30 credit            | GPU container hosting     |
| [Groq](https://console.groq.com) | Free (rate-limited)   | LLM scoring + captions    |
| [Vercel](https://vercel.com)     | Hobby (free)          | Frontend hosting          |
| [Render](https://render.com)     | Free instance         | FastAPI backend hosting   |
| [GitHub](https://github.com)     | Free                  | Source code repo          |

---

## Step 1: Push to GitHub

```bash
cd clipskari
git init
git add .
git commit -m "Initial commit: ClipSkari full-stack"
git branch -M main
git remote add origin https://github.com/<your-username>/clipskari.git
git push -u origin main
```

---

## Step 2: Supabase setup

### 2.1 Create project
1. Go to [supabase.com](https://supabase.com) → New Project
2. Pick a name (`clipskari-prod`), strong password, region closest to users
3. Wait ~2 min for provisioning

### 2.2 Apply migrations
1. Open Supabase Dashboard → SQL Editor → New query
2. Paste the contents of `supabase/migrations/001_init.sql` → Run
3. Paste the contents of `supabase/migrations/002_rpcs.sql` → Run

### 2.3 Configure auth
1. Authentication → Providers → Email → **Enable**
2. Authentication → Email Templates → Magic Link → customize if desired
3. Authentication → URL Configuration:
   - Site URL: `https://<your-app>.vercel.app`
   - Redirect URLs:
     - `http://localhost:5173/auth/callback`
     - `https://<your-app>.vercel.app/auth/callback`

### 2.4 Grab credentials
1. Project Settings → API
2. Note down:
   - **Project URL** (e.g. `https://abc123.supabase.co`)
   - **anon public** key
   - **service_role** key (KEEP SECRET!)
   - **JWT Secret** (under "JWT Settings")

---

## Step 3: Modal setup

### 3.1 Install & authenticate
```bash
pip install modal
modal token new
# Opens browser → log in → token saved to ~/.modal.toml
```

### 3.2 Create secrets
```bash
cd modal_gpu

# Required: Groq API key (from console.groq.com → API Keys)
modal secret create groq-api-key GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

# Optional: Pexels API key for B-roll (from pexels.com/api)
modal secret create pexels-api-key PEXELS_API_KEY=xxxxxxxxxxxxxxxx

# Required for Supabase uploads from GPU container
modal secret create supabase-credentials \
    SUPABASE_URL=https://abc123.supabase.co \
    SUPABASE_SERVICE_KEY=eyJhbGciOiJI...your-service-role-key
```

### 3.3 Deploy the GPU app
```bash
modal deploy modal_app.py
```

First deploy takes ~10 minutes (downloads CUDA image + TalkNet weights + YOLO models). Subsequent deploys: 30s.

After deploy, you'll see:
```
✓ Created app 'clipskari-reframer'
✓ Created functions: run_automation, run_manifest, fetch_clip_output, health
```

### 3.4 Grab Modal credentials
1. Go to [modal.com](https://modal.com) → Settings → API Tokens
2. Create a new token → note **Token ID** and **Token Secret**

### 3.5 Test the deployment
```bash
modal run modal_app.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```
You should see live logs from the GPU container, then a JSON result with clips.

---

## Step 4: Backend deployment (Render)

### 4.1 Create Render service
1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Settings:
   - **Name**: `clipskari-api`
   - **Root Directory**: `backend/`
   - **Runtime**: Docker
   - **Instance Type**: Free (or Starter for always-on)
4. Environment Variables (add all of these):

   | Key | Value |
   |-----|-------|
   | `ENVIRONMENT` | `production` |
   | `CORS_ORIGINS` | `https://<your-app>.vercel.app` |
   | `SUPABASE_URL` | `https://abc123.supabase.co` |
   | `SUPABASE_ANON_KEY` | `eyJhbG...` (anon key) |
   | `SUPABASE_SERVICE_KEY` | `eyJhbG...` (service role key) |
   | `SUPABASE_JWT_SECRET` | your JWT secret |
   | `MODAL_APP_NAME` | `clipskari-reframer` |
   | `MODAL_TOKEN_ID` | `ak-xxxx` |
   | `MODAL_TOKEN_SECRET` | `as-xxxx` |
   | `MODAL_DEPLOYED` | `true` |
   | `GROQ_API_KEY` | `gsk_xxx` |
   | `GROQ_MODEL` | `qwen/qwen3-32b` |

5. Deploy → wait ~3 min for Docker build
6. Test: `curl https://clipskari-api.onrender.com/api/v1/health` should return JSON

⚠️ **Render free tier sleeps after 15 min of inactivity**. For a smoother demo, upgrade to Starter ($7/mo).

---

## Step 5: Frontend deployment (Vercel)

### 5.1 Create Vercel project
1. Go to [vercel.com](https://vercel.com) → New Project
2. Import your GitHub repo
3. Settings:
   - **Root Directory**: `frontend/`
   - **Framework Preset**: Vite (auto-detected)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist/`
4. Environment Variables:

   | Key | Value |
   |-----|-------|
   | `VITE_SUPABASE_URL` | `https://abc123.supabase.co` |
   | `VITE_SUPABASE_ANON_KEY` | `eyJhbG...` (anon key) |
   | `VITE_API_BASE_URL` | `https://clipskari-api.onrender.com/api/v1` |

5. Deploy → wait ~1 min
6. Note your URL (e.g. `https://clipskari.vercel.app`)

### 5.2 Update Supabase redirect URLs
Go back to Supabase Dashboard → Authentication → URL Configuration → add your Vercel URL to Redirect URLs (replace the placeholder you used in Step 2.3).

### 5.3 Test the full flow
1. Visit `https://<your-app>.vercel.app`
2. Click "Get started" → enter email → check inbox → click magic link
3. You should land on `/dashboard` (empty)
4. Click "New job" → paste a YouTube URL → submit
5. Watch the job status update every 3s
6. After ~10 min, your clips should appear with download buttons

---

## Step 6: Post-deploy verification

### Health checks
```bash
# Backend liveness
curl https://clipskari-api.onrender.com/api/v1/health

# Modal container health (slow — spawns a GPU)
curl https://clipskari-api.onrender.com/api/v1/health/modal

# Frontend
curl https://<your-app>.vercel.app/ | head
```

### End-to-end test
1. Sign in via Magic Link
2. Submit a 10-min YouTube video as an automation job
3. Wait ~10 min — clips should auto-appear
4. Download a clip — should redirect to Supabase Storage signed URL
5. Delete the job — should remove clips from Storage + DB

---

## Cost monitoring

### Modal
- Dashboard → Usage → see real-time GPU seconds used
- Set billing alerts at $5, $10, $20
- Default: container scales to zero when idle (no idle charges)

### Supabase
- Dashboard → Project Settings → Usage
- Free tier: 500MB DB, 1GB storage, 50K MAU
- Set up email alerts at 80% capacity

### Render
- Free tier: 750 hr/mo (sleeps after 15 min idle)
- Upgrade to Starter ($7/mo) for always-on

### Vercel
- Hobby: 100GB bandwidth/mo (plenty for SPA)
- Upgrade to Pro ($20/mo) for team features

---

## Troubleshooting

### "Modal not configured — running in STUB mode"
Backend can't find `MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET`. Verify env vars in Render dashboard.

### "Job stuck in running forever"
- Check Render logs for scheduler errors
- Verify Modal function deployed: `modal app list`
- Check Modal logs: `modal app logs clipskari-reframer`
- Job has 1-hour timeout — wait or manually cancel via `POST /api/v1/jobs/{id}/cancel`

### "Magic link doesn't redirect"
- Verify Redirect URLs in Supabase Auth settings include your Vercel domain
- Check browser console for OAuth errors

### "CORS error in browser"
- Verify `CORS_ORIGINS` env var in Render includes your exact Vercel URL (no trailing slash)
- Restart the Render service after changing env vars

### "Clip download fails with 400"
- Signed URLs expire after 7 days
- Frontend calls `GET /api/v1/clips/{id}/url` to get a fresh URL on demand
- If still failing, verify Supabase Storage bucket exists and has RLS policies applied

---

## Rollback

If a deploy goes wrong:

1. **Vercel**: Instant Rollback button on each deployment
2. **Render**: Re-deploy from a previous commit hash
3. **Modal**: `modal deploy` from a previous git commit
4. **Supabase**: Database → Backups → restore from daily snapshot
