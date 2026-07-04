/**
 * Core type definitions for the ClipSkari frontend.
 */

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type JobMode = "automation" | "manifest";
export type ClipStatus = "pending" | "processing" | "ready" | "failed";

export interface Job {
  id: string;
  status: JobStatus;
  mode: JobMode;
  source_url: string | null;
  title: string | null;
  clips_count: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  elapsed_sec: number | null;
  error_message: string | null;
  batch_config: Record<string, unknown>;
  qc_grid_url: string | null;
}

export interface JobDetail extends Job {
  manifest: Record<string, unknown> | null;
  pipeline_config: Record<string, unknown> | null;
  clips: Clip[];
  logs: JobLog[];
}

export interface Clip {
  id: string;
  job_id: string;
  index_in_job: number;
  caption: string | null;
  subtext: string | null;
  duration_sec: number | null;
  score: number | null;
  storage_path: string;
  signed_url: string | null;
  file_size_bytes: number | null;
  theme: string | null;
  color_grading: string | null;
  status: ClipStatus;
  created_at: string;
}

export interface JobLog {
  level: "debug" | "info" | "warning" | "error";
  message: string;
  phase: string | null;
  progress: number | null;
  created_at: string;
}

export interface UserProfile {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  plan: "free" | "pro" | "admin";
  credits: number;
  created_at: string;
}

export interface UserStats {
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_clips: number;
  credits_remaining: number;
  last_job_at: string | null;
}

export interface Theme {
  id: string;
  name: string;
  description: string;
  swatch: { bg: string; text: string; accent: string };
}

export interface ColorGradingPreset {
  id: string;
  name: string;
}

export interface VisualEffect {
  id: string;
  name: string;
  default: boolean;
}

export interface AppConfig {
  themes: Theme[];
  color_grading_presets: ColorGradingPreset[];
  watermark_positions: { id: string; name: string }[];
  visual_fx: VisualEffect[];
  caption_languages: { id: string; name: string }[];
  output_resolutions: { id: string; name: string; default: boolean }[];
}
