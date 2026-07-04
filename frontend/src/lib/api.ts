import { getAuthToken } from "./supabase";
import type { AppConfig, Job, JobDetail, UserStats } from "@/types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const resp = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const errBody = await resp.json();
      detail = errBody.detail || errBody.message || detail;
    } catch {
      // body not JSON
    }
    throw new ApiError(detail, resp.status);
  }

  if (resp.status === 204) return null as T;
  if (resp.headers.get("content-type")?.includes("application/json")) {
    return resp.json() as Promise<T>;
  }
  return null as T;
}

// ── Auth ───────────────────────────────────────────────────────────────
export const api = {
  async sendMagicLink(email: string, redirectTo?: string) {
    return request<{ success: boolean; message: string }>("/auth/magic-link", {
      method: "POST",
      body: JSON.stringify({ email, redirect_to: redirectTo }),
    });
  },

  async getMe() {
    return request<{ profile: unknown; stats: UserStats }>("/auth/me");
  },

  // ── Jobs ─────────────────────────────────────────────────────────────
  async listJobs(statusFilter?: string) {
    const q = statusFilter ? `?status=${statusFilter}` : "";
    return request<{ jobs: Job[]; total: number }>(`/jobs${q}`);
  },

  async getJob(jobId: string) {
    return request<JobDetail>(`/jobs/${jobId}`);
  },

  async createAutomationJob(payload: {
    source_url: string;
    title?: string;
    min_clips?: number;
    max_clips?: number;
    min_clip_sec?: number;
    max_clip_sec?: number;
    whisper_model?: string;
    language?: string;
    caption_language?: "hinglish" | "english";
    batch_config?: Record<string, unknown>;
  }) {
    return request<Job>("/jobs/automation", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async createManifestJob(payload: {
    title?: string;
    manifest: Record<string, unknown>;
  }) {
    return request<Job>("/jobs/manifest", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async cancelJob(jobId: string) {
    return request<Job>(`/jobs/${jobId}/cancel`, { method: "POST" });
  },

  async deleteJob(jobId: string) {
    return request<void>(`/jobs/${jobId}`, { method: "DELETE" });
  },

  // ── Clips ────────────────────────────────────────────────────────────
  async getClipUrl(clipId: string) {
    return request<{ url: string; expires_in_sec: number }>(
      `/clips/${clipId}/url`
    );
  },

  // ── Config ───────────────────────────────────────────────────────────
  async getConfig() {
    return request<AppConfig>("/config/all");
  },

  // ── Health ───────────────────────────────────────────────────────────
  async getHealth() {
    return request<{
      status: string;
      version: string;
      environment: string;
      services: Record<string, string>;
    }>("/health");
  },
};

export { ApiError };
