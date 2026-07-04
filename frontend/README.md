# ClipSkari Frontend

React + Vite + TypeScript SPA for the ClipSkari long-to-short video converter.

## Stack
- **Vite 5** — fast dev server & build
- **React 18** + **TypeScript 5** — type-safe UI
- **Tailwind CSS 3** + **shadcn/ui patterns** — modern dark theme
- **TanStack Query 5** — server state & caching
- **React Router 6** — client-side routing
- **Supabase JS** — auth & storage client
- **Sonner** — toast notifications
- **lucide-react** — icon set

## Local dev

```bash
# 1. Install deps
npm install

# 2. Configure env
cp .env.example .env.local
# Edit .env.local with your Supabase URL + anon key

# 3. Start dev server (proxies /api to localhost:8000)
npm run dev
```

The dev server runs at `http://localhost:5173`. Make sure the FastAPI backend is running on port 8000 — the Vite proxy will forward all `/api/*` requests.

## Build

```bash
npm run build       # outputs to dist/
npm run preview     # preview the production build locally
```

## Project structure

```
src/
├── main.tsx              # App entry
├── App.tsx               # Routes
├── index.css            # Tailwind + theme tokens
├── components/
│   ├── ui/              # shadcn primitives (button, card, input, etc.)
│   ├── Navbar.tsx
│   ├── JobCard.tsx
│   └── StatusBadge.tsx
├── hooks/
│   ├── useAuth.ts       # Supabase session
│   └── usePolling.ts    # Live job status updates
├── lib/
│   ├── api.ts           # FastAPI client
│   ├── supabase.ts      # Supabase client
│   └── utils.ts         # cn(), formatters
├── pages/
│   ├── Landing.tsx      # Public marketing page
│   ├── Login.tsx        # Magic link auth
│   ├── AuthCallback.tsx # OAuth redirect target
│   ├── Dashboard.tsx    # Job list + stats
│   ├── NewJob.tsx       # Submit automation/manifest
│   ├── JobDetail.tsx    # Status, clips, logs, download
│   └── Settings.tsx     # Profile + system health
└── types/
    └── index.ts         # Shared TS types
```

## Deploy to Vercel

1. Push this folder to a GitHub repo
2. In Vercel dashboard: New Project → Import the repo
3. Set **Root Directory** to `frontend/`
4. Set the env vars (Project Settings → Environment Variables):
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`
   - `VITE_API_BASE_URL` (your FastAPI URL, e.g. `https://clipskari-api.onrender.com/api/v1`)
5. Deploy — Vercel auto-detects Vite and runs `npm run build`

## Auth flow

1. User enters email on `/login`
2. Frontend calls `POST /api/v1/auth/magic-link`
3. Supabase sends a magic link email
4. User clicks → redirects to `/auth/callback?code=...`
5. Supabase JS client exchanges code for session
6. Frontend redirects to `/dashboard`
7. All API calls include `Authorization: Bearer <jwt>` header
