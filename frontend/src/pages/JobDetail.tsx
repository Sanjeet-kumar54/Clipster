import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { usePolling } from "@/hooks/usePolling";
import {
  formatRelativeTime,
  formatDuration,
  formatBytes,
  formatScore,
  cn,
} from "@/lib/utils";
import type { JobDetail } from "@/types";
import {
  ArrowLeft,
  Download,
  Trash2,
  X,
  Loader2,
  AlertCircle,
  Film,
} from "lucide-react";

export function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuth();
  const queryClient = useQueryClient();
  const [downloading, setDownloading] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !user) navigate("/login");
  }, [user, authLoading, navigate]);

  // Use polling for live updates when job is running
  const { data: job, error } = usePolling<JobDetail>(
    async () => {
      if (!id) throw new Error("Missing job ID");
      return api.getJob(id);
    },
    3000,
    !!user && !!id
  );

  const isRunning = job?.status === "running" || job?.status === "queued";

  const handleCancel = async () => {
    if (!id) return;
    try {
      await api.cancelJob(id);
      toast.success("Job cancelled");
      queryClient.invalidateQueries({ queryKey: ["job", id] });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Cancel failed");
    }
  };

  const handleDelete = async () => {
    if (!id) return;
    if (!confirm("Delete this job and all its clips? This cannot be undone.")) return;
    try {
      await api.deleteJob(id);
      toast.success("Job deleted");
      navigate("/dashboard");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const handleDownload = async (clipId: string) => {
    setDownloading(clipId);
    try {
      const { url } = await api.getClipUrl(clipId);
      window.open(url, "_blank");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(null);
    }
  };

  if (authLoading) return null;
  if (!user) return null;

  if (error) {
    return (
      <div className="container max-w-4xl py-8">
        <Card className="border-red-500/30">
          <CardContent className="flex flex-col items-center gap-4 p-12 text-center">
            <AlertCircle className="h-12 w-12 text-red-400" />
            <div>
              <h2 className="text-xl font-semibold">Job not found</h2>
              <p className="text-sm text-muted-foreground">
                {error instanceof Error ? error.message : "Unknown error"}
              </p>
            </div>
            <Button asChild>
              <a href="/dashboard">Back to dashboard</a>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="container max-w-4xl py-8">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading job...
        </div>
      </div>
    );
  }

  const completedClips = job.clips.filter((c) => c.status === "ready");
  const totalDuration = completedClips.reduce((acc, c) => acc + (c.duration_sec ?? 0), 0);

  return (
    <div className="container max-w-6xl py-8">
      {/* Header */}
      <Button variant="ghost" size="sm" asChild className="mb-4">
        <a href="/dashboard">
          <ArrowLeft className="h-4 w-4" />
          Back to dashboard
        </a>
      </Button>

      <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-3">
            <StatusBadge status={job.status} />
            <Badge variant="outline" className="capitalize">{job.mode}</Badge>
            <span className="text-sm text-muted-foreground">
              {formatRelativeTime(job.created_at)}
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            {job.title || (job.mode === "automation" ? job.source_url : "Manual manifest")}
          </h1>
          {job.source_url && (
            <a
              href={job.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 inline-block text-sm text-primary hover:underline"
            >
              {job.source_url}
            </a>
          )}
        </div>
        <div className="flex gap-2">
          {isRunning && (
            <Button variant="outline" onClick={handleCancel}>
              <X className="h-4 w-4" />
              Cancel
            </Button>
          )}
          <Button variant="outline" onClick={handleDelete}>
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>
      </div>

      {/* Running state banner */}
      {isRunning && (
        <Card className="mb-6 border-amber-500/30 bg-amber-500/5">
          <CardContent className="flex items-center gap-4 p-4">
            <Loader2 className="h-5 w-5 animate-spin text-amber-400" />
            <div className="flex-1">
              <div className="text-sm font-medium text-amber-300">
                Job is running on GPU
              </div>
              <div className="text-xs text-muted-foreground">
                Started {formatRelativeTime(job.started_at)} • Auto-refreshing every 3s
              </div>
            </div>
            <Progress value={50} className="w-32" />
          </CardContent>
        </Card>
      )}

      {/* Error banner */}
      {job.status === "failed" && job.error_message && (
        <Card className="mb-6 border-red-500/30 bg-red-500/5">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium text-red-300">Job failed</div>
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs text-red-200/80">
                  {job.error_message}
                </pre>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 md:grid-cols-3">
        {/* Stats sidebar */}
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-sm">Stats</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Stat label="Clips" value={`${completedClips.length}/${job.clips.length}`} />
              <Stat label="Runtime" value={formatDuration(job.elapsed_sec)} />
              <Stat
                label="Total clip length"
                value={formatDuration(totalDuration)}
              />
              <Stat
                label="Avg score"
                value={formatScore(
                  completedClips.reduce((a, c) => a + (c.score ?? 0), 0) /
                    Math.max(1, completedClips.length)
                )}
              />
            </CardContent>
          </Card>

          {job.qc_grid_url && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">QC preview</CardTitle>
                <CardDescription>3×3 contact sheet of all clips</CardDescription>
              </CardHeader>
              <CardContent>
                <img
                  src={job.qc_grid_url}
                  alt="QC preview grid"
                  className="w-full rounded-md"
                />
              </CardContent>
            </Card>
          )}
        </div>

        {/* Clips grid */}
        <div className="md:col-span-2">
          <h2 className="mb-4 text-lg font-semibold">
            Clips ({job.clips.length})
          </h2>
          {job.clips.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="flex flex-col items-center justify-center gap-3 py-12 text-center">
                <Film className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  {isRunning ? "Clips will appear here as they're generated..." : "No clips generated."}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              {job.clips.map((clip) => (
                <Card key={clip.id} className="overflow-hidden">
                  <div className="relative aspect-[9/16] bg-black">
                    {clip.signed_url ? (
                      <video
                        src={clip.signed_url}
                        className="h-full w-full object-cover"
                        controls
                        preload="metadata"
                      />
                    ) : (
                      <div className="flex h-full w-full items-center justify-center">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                      </div>
                    )}
                    <div className="absolute left-2 top-2 flex gap-1">
                      <Badge variant="secondary" className="bg-black/60 backdrop-blur">
                        #{clip.index_in_job + 1}
                      </Badge>
                      {clip.score !== null && (
                        <Badge className="bg-primary/80 backdrop-blur">
                          Score: {formatScore(clip.score)}
                        </Badge>
                      )}
                    </div>
                  </div>
                  <CardContent className="p-3">
                    <p className="mb-2 line-clamp-2 text-sm font-medium">
                      {clip.caption}
                    </p>
                    <div className="mb-3 flex items-center gap-3 text-xs text-muted-foreground">
                      <span>{formatDuration(clip.duration_sec)}</span>
                      <span>•</span>
                      <span>{formatBytes(clip.file_size_bytes)}</span>
                      {clip.theme && (
                        <>
                          <span>•</span>
                          <span className="capitalize">{clip.theme}</span>
                        </>
                      )}
                    </div>
                    <Button
                      size="sm"
                      className="w-full"
                      onClick={() => handleDownload(clip.id)}
                      disabled={downloading === clip.id || !clip.signed_url}
                    >
                      {downloading === clip.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="h-4 w-4" />
                      )}
                      Download
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Logs */}
      {job.logs.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="text-sm">Logs ({job.logs.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-96 overflow-y-auto rounded-md bg-zinc-950 p-4 font-mono text-xs">
              {job.logs.slice(-100).map((log, i) => (
                <div
                  key={i}
                  className={cn(
                    "flex gap-2 py-0.5",
                    log.level === "error" && "text-red-400",
                    log.level === "warning" && "text-amber-400",
                    log.level === "info" && "text-zinc-300",
                    log.level === "debug" && "text-zinc-500"
                  )}
                >
                  <span className="shrink-0 text-zinc-600">
                    {new Date(log.created_at).toLocaleTimeString()}
                  </span>
                  {log.phase && (
                    <span className="shrink-0 text-primary">[{log.phase}]</span>
                  )}
                  <span className="whitespace-pre-wrap">{log.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
