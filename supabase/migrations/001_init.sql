-- ============================================================
-- ClipSkari — Initial Schema Migration
-- ============================================================
-- Tables: profiles, jobs, clips, job_logs
-- Storage buckets: clips, qc-grids, avatars
-- RLS policies: users can only see/modify their own data
-- ============================================================

-- ── Profiles table (extends Supabase auth.users) ──────────────
create table if not exists public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    email       text unique not null,
    full_name   text,
    avatar_url  text,
    plan        text not null default 'free' check (plan in ('free', 'pro', 'admin')),
    credits     integer not null default 5,  -- free users start with 5 jobs
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

comment on table public.profiles is 'User profile, 1:1 with auth.users';

-- ── Jobs table (one per submission) ───────────────────────────
create table if not exists public.jobs (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references public.profiles(id) on delete cascade,
    status          text not null default 'queued'
                    check (status in ('queued', 'running', 'completed', 'failed', 'cancelled')),
    mode            text not null check (mode in ('automation', 'manifest')),
    source_url      text,                 -- YouTube URL (automation mode)
    title           text,                 -- video title (fetched via yt-dlp)
    manifest        jsonb,                -- full manifest (manifest mode) or null
    batch_config    jsonb not null,       -- BatchConfig overrides (theme, FX, etc.)
    pipeline_config jsonb,                -- Pipeline config (automation mode)
    modal_call_id   text,                 -- Modal function call ID for tracking
    error_message   text,
    started_at      timestamptz,
    completed_at    timestamptz,
    elapsed_sec     numeric,
    clips_count     integer not null default 0,
    qc_grid_path    text,                 -- storage path for QC preview image
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists idx_jobs_user_id on public.jobs(user_id);
create index if not exists idx_jobs_status on public.jobs(status);
create index if not exists idx_jobs_created_at on public.jobs(created_at desc);

comment on table public.jobs is 'A single clip-reframing job submission';

-- ── Clips table (one per output video, many per job) ──────────
create table if not exists public.clips (
    id              uuid primary key default gen_random_uuid(),
    job_id          uuid not null references public.jobs(id) on delete cascade,
    user_id         uuid not null references public.profiles(id) on delete cascade,
    index_in_job    integer not null default 0,  -- ordering within the job
    caption         text,
    subtext         text,
    source_start_sec numeric,
    source_end_sec   numeric,
    duration_sec    numeric,
    score           numeric,              -- LLM+rule score 0-100 (automation mode)
    storage_path    text not null,        -- e.g. "{job_id}/clip_001.mp4"
    signed_url      text,                 -- time-limited Supabase Storage signed URL
    file_size_bytes bigint,
    theme           text,                 -- card theme used
    color_grading   text,                 -- color grading preset used
    status          text not null default 'pending'
                    check (status in ('pending', 'processing', 'ready', 'failed')),
    created_at      timestamptz not null default now()
);

create index if not exists idx_clips_job_id on public.clips(job_id);
create index if not exists idx_clips_user_id on public.clips(user_id);

comment on table public.clips is 'Individual output clip metadata, many per job';

-- ── Job logs table (streaming progress) ───────────────────────
create table if not exists public.job_logs (
    id          uuid primary key default gen_random_uuid(),
    job_id      uuid not null references public.jobs(id) on delete cascade,
    level       text not null default 'info' check (level in ('debug', 'info', 'warning', 'error')),
    message     text not null,
    phase       text,                     -- 'download' | 'whisper' | 'scoring' | 'reframe' | 'qc' | 'upload'
    progress    numeric,                  -- 0-100 within current phase
    created_at  timestamptz not null default now()
);

create index if not exists idx_job_logs_job_id on public.job_logs(job_id);
create index if not exists idx_job_logs_created_at on public.job_logs(created_at desc);

comment on table public.job_logs is 'Streaming log lines from the GPU job';

-- ── updated_at triggers ───────────────────────────────────────
create or replace function public.touch_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trg_profiles_touch on public.profiles;
create trigger trg_profiles_touch before update on public.profiles
    for each row execute function public.touch_updated_at();

drop trigger if exists trg_jobs_touch on public.jobs;
create trigger trg_jobs_touch before update on public.jobs
    for each row execute function public.touch_updated_at();

-- ── Auto-create profile on user signup ────────────────────────
create or replace function public.handle_new_user()
returns trigger as $$
begin
    insert into public.profiles (id, email, full_name, avatar_url)
    values (
        new.id,
        new.email,
        coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email, '@', 1)),
        new.raw_user_meta_data->>'avatar_url'
    )
    on conflict (id) do nothing;
    return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- ── Storage buckets ───────────────────────────────────────────
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
    ('clips', 'clips', false, 536870912,  -- 512MB private
     array['video/mp4', 'video/quicktime', 'video/x-matroska']),
    ('qc-grids', 'qc-grids', true, 10485760,  -- 10MB public
     array['image/png', 'image/jpeg']),
    ('avatars', 'avatars', true, 2097152,  -- 2MB public
     array['image/png', 'image/jpeg', 'image/webp'])
on conflict (id) do nothing;

-- ── Row-Level Security (RLS) policies ─────────────────────────
alter table public.profiles enable row level security;
alter table public.jobs enable row level security;
alter table public.clips enable row level security;
alter table public.job_logs enable row level security;

-- ── Profiles policies ─────────────────────────────────────────
drop policy if exists "Profiles are viewable by owner" on public.profiles;
create policy "Profiles are viewable by owner"
    on public.profiles for select
    using (auth.uid() = id);

drop policy if exists "Profiles are updatable by owner" on public.profiles;
create policy "Profiles are updatable by owner"
    on public.profiles for update
    using (auth.uid() = id);

drop policy if exists "Profiles are insertable by owner" on public.profiles;
create policy "Profiles are insertable by owner"
    on public.profiles for insert
    with check (auth.uid() = id);

-- ── Jobs policies ─────────────────────────────────────────────
drop policy if exists "Jobs are viewable by owner" on public.jobs;
create policy "Jobs are viewable by owner"
    on public.jobs for select
    using (auth.uid() = user_id);

drop policy if exists "Jobs are insertable by owner" on public.jobs;
create policy "Jobs are insertable by owner"
    on public.jobs for insert
    with check (auth.uid() = user_id);

drop policy if exists "Jobs are updatable by owner" on public.jobs;
create policy "Jobs are updatable by owner"
    on public.jobs for update
    using (auth.uid() = user_id);

drop policy if exists "Jobs are deletable by owner" on public.jobs;
create policy "Jobs are deletable by owner"
    on public.jobs for delete
    using (auth.uid() = user_id);

-- ── Clips policies ────────────────────────────────────────────
drop policy if exists "Clips are viewable by owner" on public.clips;
create policy "Clips are viewable by owner"
    on public.clips for select
    using (auth.uid() = user_id);

drop policy if exists "Clips are insertable by owner" on public.clips;
create policy "Clips are insertable by owner"
    on public.clips for insert
    with check (auth.uid() = user_id);

drop policy if exists "Clips are updatable by owner" on public.clips;
create policy "Clips are updatable by owner"
    on public.clips for update
    using (auth.uid() = user_id);

drop policy if exists "Clips are deletable by owner" on public.clips;
create policy "Clips are deletable by owner"
    on public.clips for delete
    using (auth.uid() = user_id);

-- ── Job logs policies ─────────────────────────────────────────
drop policy if exists "Logs are viewable by job owner" on public.job_logs;
create policy "Logs are viewable by job owner"
    on public.job_logs for select
    using (
        auth.uid() in (
            select user_id from public.jobs where id = job_id
        )
    );

drop policy if exists "Logs are insertable by job owner" on public.job_logs;
create policy "Logs are insertable by job owner"
    on public.job_logs for insert
    with check (
        auth.uid() in (
            select user_id from public.jobs where id = job_id
        )
    );

-- ── Storage policies ──────────────────────────────────────────
-- Private `clips` bucket: only owner can read/write their folder
drop policy if exists "Clips bucket: owner read" on storage.objects;
create policy "Clips bucket: owner read"
    on storage.objects for select
    using (
        bucket_id = 'clips'
        and auth.uid()::text = (storage.foldername(name))[1]
    );

drop policy if exists "Clips bucket: owner write" on storage.objects;
create policy "Clips bucket: owner write"
    on storage.objects for insert
    with check (
        bucket_id = 'clips'
        and auth.uid()::text = (storage.foldername(name))[1]
    );

drop policy if exists "Clips bucket: owner update" on storage.objects;
create policy "Clips bucket: owner update"
    on storage.objects for update
    using (
        bucket_id = 'clips'
        and auth.uid()::text = (storage.foldername(name))[1]
    );

drop policy if exists "Clips bucket: owner delete" on storage.objects;
create policy "Clips bucket: owner delete"
    on storage.objects for delete
    using (
        bucket_id = 'clips'
        and auth.uid()::text = (storage.foldername(name))[1]
    );

-- Public `qc-grids` bucket
drop policy if exists "QC grids: public read" on storage.objects;
create policy "QC grids: public read"
    on storage.objects for select
    using (bucket_id = 'qc-grids');

drop policy if exists "QC grids: authenticated write" on storage.objects;
create policy "QC grids: authenticated write"
    on storage.objects for insert
    with check (bucket_id = 'qc-grids' and auth.role() = 'authenticated');

-- Public `avatars` bucket
drop policy if exists "Avatars: public read" on storage.objects;
create policy "Avatars: public read"
    on storage.objects for select
    using (bucket_id = 'avatars');

drop policy if exists "Avatars: owner write" on storage.objects;
create policy "Avatars: owner write"
    on storage.objects for insert
    with check (
        bucket_id = 'avatars'
        and auth.uid()::text = (storage.foldername(name))[1]
    );

-- ── Useful views ──────────────────────────────────────────────
create or replace view public.v_jobs_with_clips as
select
    j.*,
    count(c.id) as clips_ready_count,
    max(c.created_at) as last_clip_at
from public.jobs j
left join public.clips c on c.job_id = j.id
group by j.id;

comment on view public.v_jobs_with_clips is 'Jobs enriched with clip counts';
