import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { ArrowLeft, Mail, Zap, CheckCircle2 } from "lucide-react";

export function Login() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const { user } = useAuth();
  const navigate = useNavigate();

  // If already logged in, redirect to dashboard
  if (user) {
    navigate("/dashboard");
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    setLoading(true);
    try {
      const redirectTo = `${window.location.origin}/auth/callback`;
      await api.sendMagicLink(email, redirectTo);
      setSent(true);
      toast.success("Magic link sent!", {
        description: "Check your inbox — the link expires in 10 minutes.",
      });
    } catch (err) {
      toast.error("Failed to send magic link", {
        description: err instanceof Error ? err.message : "Unknown error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-grid p-4">
      <div className="w-full max-w-md">
        <Link to="/" className="mb-6 flex items-center justify-center gap-2 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>

        <Card>
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary">
              <Zap className="h-6 w-6 text-white" fill="white" />
            </div>
            <CardTitle className="text-2xl">Sign in to ClipSkari</CardTitle>
            <CardDescription>
              {sent
                ? "Check your inbox for the magic link"
                : "We'll email you a secure one-time sign-in link"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {sent ? (
              <div className="space-y-4 text-center">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/10">
                  <CheckCircle2 className="h-8 w-8 text-emerald-400" />
                </div>
                <p className="text-sm text-muted-foreground">
                  We sent a magic link to <strong className="text-foreground">{email}</strong>.
                  Click the link in your email to sign in. The link expires in 10 minutes.
                </p>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => {
                    setSent(false);
                    setEmail("");
                  }}
                >
                  Use a different email
                </Button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email address</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-9"
                      required
                      autoFocus
                    />
                  </div>
                </div>
                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? "Sending..." : "Send magic link"}
                </Button>
                <p className="text-center text-xs text-muted-foreground">
                  By signing in, you agree to our Terms & Privacy Policy.
                  <br />
                  New users get 5 free credits.
                </p>
              </form>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
