import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { JobCard } from "@/components/JobCard";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { Plus, Film, CheckCircle, XCircle, Zap } from "lucide-react";

export function Dashboard() {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    if (!authLoading && !user) {
      navigate("/login");
    }
  }, [user, authLoading, navigate]);

  const { data: jobsData, isLoading } = useQuery({
    queryKey: ["jobs", filter],
    queryFn: () => api.listJobs(filter === "all" ? undefined : filter),
    enabled: !!user,
    refetchInterval: (query) => {
      // Refetch every 5s if there are running jobs
      const jobs = query.state.data?.jobs ?? [];
      const hasRunning = jobs.some((j) => j.status === "running" || j.status === "queued");
      return hasRunning ? 5000 : false;
    },
  });

  const { data: meData } = useQuery({
    queryKey: ["me"],
    queryFn: () => api.getMe(),
    enabled: !!user,
  });

  if (authLoading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="text-sm text-muted-foreground">Loading...</div>
      </div>
    );
  }

  if (!user) return null;

  const stats = meData?.stats;
  const jobs = jobsData?.jobs ?? [];

  const filters = ["all", "running", "completed", "failed", "cancelled"];

  return (
    <div className="container max-w-6xl py-8">
      {/* Header */}
      <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Welcome back, {user.email}
          </p>
        </div>
        <Button size="lg" asChild>
          <Link to="/jobs/new">
            <Plus className="h-4 w-4" />
            New job
          </Link>
        </Button>
      </div>

      {/* Stats */}
      <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          icon={<Film className="h-4 w-4" />}
          label="Total jobs"
          value={stats?.total_jobs ?? 0}
        />
        <StatCard
          icon={<CheckCircle className="h-4 w-4" />}
          label="Completed"
          value={stats?.completed_jobs ?? 0}
        />
        <StatCard
          icon={<XCircle className="h-4 w-4" />}
          label="Failed"
          value={stats?.failed_jobs ?? 0}
        />
        <StatCard
          icon={<Zap className="h-4 w-4" />}
          label="Credits left"
          value={stats?.credits_remaining ?? 0}
          highlight
        />
      </div>

      {/* Filters */}
      <div className="mb-4 flex gap-2 overflow-x-auto pb-2">
        {filters.map((f) => (
          <Button
            key={f}
            variant={filter === f ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter(f)}
            className="capitalize"
          >
            {f}
          </Button>
        ))}
      </div>

      {/* Jobs list */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="h-32 animate-pulse">
              <CardContent className="p-5">
                <div className="h-4 w-20 rounded bg-muted" />
                <div className="mt-3 h-5 w-3/4 rounded bg-muted" />
                <div className="mt-3 h-3 w-1/2 rounded bg-muted" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : jobs.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center gap-4 py-16 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <Film className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">No jobs yet</h3>
              <p className="text-sm text-muted-foreground">
                Submit your first YouTube URL to get reframed clips in minutes.
              </p>
            </div>
            <Button asChild>
              <Link to="/jobs/new">
                <Plus className="h-4 w-4" />
                Create your first job
              </Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} />
          ))}
        </div>
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  highlight = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  highlight?: boolean;
}) {
  return (
    <Card className={highlight ? "border-primary/30 bg-primary/5" : ""}>
      <CardContent className="p-4">
        <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
          {icon}
          <span>{label}</span>
        </div>
        <div className="text-2xl font-bold">{value}</div>
      </CardContent>
    </Card>
  );
}
