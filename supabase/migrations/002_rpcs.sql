-- ============================================================
-- ClipSkari — Helper RPC functions
-- ============================================================
-- Called from FastAPI to avoid round-trips for common operations
-- ============================================================

-- ── get_job_with_clips: fetch job + all its clips in one call ─
create or replace function public.get_job_with_clips(p_job_id uuid)
returns json as $$
declare
    result json;
begin
    select json_build_object(
        'job', to_jsonb(j.*),
        'clips', coalesce((
            select json_agg(json_build_object(
                'id', c.id,
                'caption', c.caption,
                'subtext', c.subtext,
                'source_start_sec', c.source_start_sec,
                'source_end_sec', c.source_end_sec,
                'duration_sec', c.duration_sec,
                'score', c.score,
                'storage_path', c.storage_path,
                'file_size_bytes', c.file_size_bytes,
                'theme', c.theme,
                'color_grading', c.color_grading,
                'status', c.status,
                'created_at', c.created_at
            ) order by c.index_in_job)
            from public.clips c where c.job_id = p_job_id
        ), '[]'::json),
        'logs', coalesce((
            select json_agg(json_build_object(
                'level', l.level,
                'message', l.message,
                'phase', l.phase,
                'progress', l.progress,
                'created_at', l.created_at
            ) order by l.created_at)
            from public.job_logs l where l.job_id = p_job_id limit 200
        ), '[]'::json)
    ) into result
    from public.jobs j
    where j.id = p_job_id and j.user_id = auth.uid();

    return result;
end;
$$ language plpgsql security definer;

-- ── decrement_credits: atomically consume a credit ───────────
create or replace function public.consume_credit(p_user_id uuid)
returns boolean as $$
declare
    remaining integer;
begin
    update public.profiles
    set credits = credits - 1
    where id = p_user_id and credits > 0
    returning credits into remaining;

    return found;
end;
$$ language plpgsql security definer;

-- ── get_user_stats: dashboard stats ───────────────────────────
create or replace function public.get_user_stats(p_user_id uuid)
returns json as $$
declare
    result json;
begin
    select json_build_object(
        'total_jobs', count(*),
        'completed_jobs', count(*) filter (where status = 'completed'),
        'failed_jobs', count(*) filter (where status = 'failed'),
        'total_clips', (
            select count(*) from public.clips where user_id = p_user_id
        ),
        'credits_remaining', (select credits from public.profiles where id = p_user_id),
        'last_job_at', max(created_at)
    ) into result
    from public.jobs
    where user_id = p_user_id;

    return result;
end;
$$ language plpgsql security definer;
