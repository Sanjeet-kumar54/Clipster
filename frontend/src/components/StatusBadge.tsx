import { Badge } from "@/components/ui/badge";
import { cn, STATUS_COLORS } from "@/lib/utils";
import type { JobStatus } from "@/types";

interface StatusBadgeProps {
  status: JobStatus | string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <Badge
      className={cn("border", STATUS_COLORS[status] || STATUS_COLORS.pending, className)}
    >
      <span className="capitalize">{status}</span>
    </Badge>
  );
}
