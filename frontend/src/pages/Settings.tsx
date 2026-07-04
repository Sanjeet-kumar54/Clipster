import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/lib/supabase";
import { CheckCircle, XCircle, Loader2, Cpu } from "lucide-react";

export function Settings() {
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuth();

  useEffect(() => {
    if (!authLoading && !user) navigate("/login");
  }, [user, authLoading, navigate]);

  const { data: meData } = useQuery({
    queryKey: ["me"],
    queryFn: () => api.getMe(),
    enabled: !!user,
  });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.getHealth(),
    enabled: !!user,
  });

  if (authLoading || !user) return null;

  const profile = meData?.profile as Record<string, unknown> | undefined;
  const stats = meData?.stats;

  return (
    <div className="container max-w-4xl py-8">
      <h1 className="mb-8 text-3xl font-bold tracking-tight">Settings</h1>

      <div className="grid gap-6">
        {/* Profile */}
        <Card>
          <CardHeader>
            <CardTitle>Profile</CardTitle>
            <CardDescription>Your account info & plan</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <Row label="Email" value={user.email ?? "—"} />
            <Row label="User ID" value={user.id} mono />
            <Row label="Plan" value={
              <Badge variant="outline" className="capitalize">
                {(profile?.plan as string) ?? "free"}
              </Badge>
            } />
            <Row label="Credits remaining" value={stats?.credits_remaining ?? 0} />
            <Row label="Joined" value={new Date(user.created_at ?? "").toLocaleDateString()} />
          </CardContent>
        </Card>

        {/* Account stats */}
        <Card>
          <CardHeader>
            <CardTitle>Account statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatBox label="Total jobs" value={stats?.total_jobs ?? 0} />
              <StatBox label="Completed" value={stats?.completed_jobs ?? 0} />
              <StatBox label="Failed" value={stats?.failed_jobs ?? 0} />
              <StatBox label="Total clips" value={stats?.total_clips ?? 0} />
            </div>
          </CardContent>
        </Card>

        {/* System health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              System health
            </CardTitle>
            <CardDescription>Backend service status</CardDescription>
          </CardHeader>
          <CardContent>
            {!health ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Checking...
              </div>
            ) : (
              <div className="space-y-3 text-sm">
                <Row
                  label="Overall"
                  value={
                    <Badge variant={health.status === "ok" ? "success" : "warning"}>
                      {health.status}
                    </Badge>
                  }
                />
                <Row label="Version" value={health.version} />
                <Row label="Environment" value={health.environment} />
                <div className="border-t border-border/40 pt-3">
                  <div className="mb-2 text-xs uppercase text-muted-foreground">
                    Services
                  </div>
                  {Object.entries(health.services).map(([name, status]) => (
                    <div key={name} className="flex items-center justify-between py-1">
                      <span className="capitalize">{name}</span>
                      {status === "ok" ? (
                        <CheckCircle className="h-4 w-4 text-emerald-400" />
                      ) : status === "stub" ? (
                        <Loader2 className="h-4 w-4 text-amber-400" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-400" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Danger zone */}
        <Card className="border-red-500/30">
          <CardHeader>
            <CardTitle className="text-red-300">Danger zone</CardTitle>
            <CardDescription>
              Sign out of your account
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="outline"
              onClick={async () => {
                await supabase.auth.signOut();
                navigate("/");
              }}
            >
              Sign out
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Row({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span className={mono ? "font-mono text-xs" : "font-medium"}>{value}</span>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-border/40 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}
