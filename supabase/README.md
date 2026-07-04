# Supabase Setup

## Apply migrations

### Option A: Supabase Dashboard
1. Open your Supabase project → SQL Editor
2. Paste `migrations/001_init.sql` → Run
3. Paste `migrations/002_rpcs.sql` → Run
4. (Optional) After signing up via the frontend, customize and run `seed.sql`

### Option B: Supabase CLI (recommended)
```bash
npm install -g supabase

# Link to your project
supabase link --project-ref YOUR_PROJECT_REF

# Push migrations
supabase db push

# Or apply individually
supabase db execute --file supabase/migrations/001_init.sql
supabase db execute --file supabase/migrations/002_rpcs.sql
```

## What gets created

| Object | Purpose |
|--------|---------|
| `public.profiles` | User profile (1:1 with `auth.users`) — stores plan, credits |
| `public.jobs` | Job submissions (queued, running, completed, failed) |
| `public.clips` | Output clips (many per job) — stores caption, score, storage path |
| `public.job_logs` | Streaming log lines from the GPU job |
| `storage.buckets: clips` | Private bucket for output videos (512MB limit) |
| `storage.buckets: qc-grids` | Public bucket for QC preview images |
| `storage.buckets: avatars` | Public bucket for user avatars |

## Auth setup

1. **Authentication → Providers → Email** → Enable
2. **Authentication → Email Templates → Magic Link** → Customize the template if desired
3. **Authentication → URL Configuration** → Set Site URL to your Vercel domain (e.g. `https://clipskari.vercel.app`)
4. **Authentication → URL Configuration** → Add Redirect URLs:
   - `http://localhost:5173/auth/callback` (local dev)
   - `https://clipskari.vercel.app/auth/callback` (production)

## RLS policies summary

Every table has Row-Level Security enabled:
- Users can only SELECT/INSERT/UPDATE/DELETE their own rows (`user_id = auth.uid()`)
- Storage `clips` bucket: users can only access files in their own folder (`{user_id}/...`)
- Storage `qc-grids` and `avatars`: public read, authenticated write

## Service Role Key

The FastAPI backend uses the **service role key** (not the anon key) to:
- Update job status from the GPU worker (no user session available)
- Insert log lines on behalf of the GPU job
- Bypass RLS when needed (e.g., admin operations)

⚠️ **Never expose the service role key in the frontend.** Keep it only in the FastAPI backend's env vars.
