import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds < 0) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function formatRelativeTime(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const date = new Date(dateStr);
  const now = new Date();
  const diff = (now.getTime() - date.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return date.toLocaleDateString();
}

export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined) return "—";
  return score.toFixed(1);
}

export const STATUS_COLORS: Record<string, string> = {
  queued: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  running: "bg-amber-500/20 text-amber-300 border-amber-500/30 animate-pulse",
  completed: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  failed: "bg-red-500/20 text-red-300 border-red-500/30",
  cancelled: "bg-zinc-500/20 text-zinc-400 border-zinc-500/30",
  pending: "bg-zinc-500/20 text-zinc-400 border-zinc-500/30",
  processing: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  ready: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
};
