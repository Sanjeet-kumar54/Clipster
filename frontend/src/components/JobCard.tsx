import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { StatusBadge } from "./StatusBadge";
import { Badge } from "@/components/ui/badge";
import {
  formatRelativeTime,
  formatDuration,
} from "@/lib/utils";
import type { Job } from "@/types";
import { Film, Youtube, AlertCircle } from "lucide-react";

interface JobCardProps {
  job: Job;
}

export function JobCard({ job }: JobCardProps) {
  const isAutomation = job.mode === "automation";

  return (
    <Link to={`/jobs/${job.id}`}>
      <Card className="group cursor-pointer transition-all hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5">
        <CardContent className="p-5">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="mb-2 flex items-center gap-2">
                {isAutomation ? (
                  <Youtube className="h-4 w-4 shrink-0 text-red-500" />
                ) : (
                  <Film className="h-4 w-4 shrink-0 text-primary" />
                )}
                <Badge variant="outline" className="text-[10px] uppercase">
                  {job.mode}
                </Badge>
                <StatusBadge status={job.status} />
              </div>
              <h3 className="truncate font-medium text-foreground">
                {job.title || (isAutomation ? job.source_url : "Manual manifest")}
              </h3>
              <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
                <span>{formatRelativeTime(job.created_at)}</span>
                <span>•</span>
                <span>{job.clips_count} clips</span>
                {job.elapsed_sec && (
                  <>
                    <span>•</span>
                    <span>{formatDuration(job.elapsed_sec)} runtime</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {job.status === "failed" && job.error_message && (
            <div className="mt-3 flex items-start gap-2 rounded-md bg-red-500/10 p-2 text-xs text-red-300">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="line-clamp-2">{job.error_message}</span>
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
