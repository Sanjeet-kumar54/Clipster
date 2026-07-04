import { useEffect, useState } from "react";

/**
 * Poll an async function on an interval. Useful for live-updating job status.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled: boolean = true
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const result = await fetcher();
        if (!cancelled) {
          setData(result);
          setError(null);
          setLoading(false);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e as Error);
          setLoading(false);
        }
      } finally {
        if (!cancelled) {
          timer = setTimeout(poll, intervalMs);
        }
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, enabled]);

  return { data, error, loading };
}
