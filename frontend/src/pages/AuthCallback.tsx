import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/lib/supabase";
import { Card, CardContent } from "@/components/ui/card";
import { Zap } from "lucide-react";

/**
 * OAuth callback handler.
 * Supabase redirects here with the auth code in the URL hash.
 * This component exchanges the code for a session, then redirects.
 */
export function AuthCallback() {
  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    // Supabase auto-detects the session from the URL hash on init.
    // We just need to wait for it to be set, then redirect.
    const timer = setTimeout(() => {
      if (user) {
        navigate("/dashboard");
      } else {
        // Wait a bit more — the auth listener may not have fired yet
        supabase.auth.getSession().then(({ data }) => {
          if (data.session) {
            navigate("/dashboard");
          } else {
            navigate("/login");
          }
        });
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [user, navigate]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-grid p-4">
      <Card className="max-w-md">
        <CardContent className="flex flex-col items-center gap-4 p-8 text-center">
          <div className="flex h-12 w-12 animate-pulse items-center justify-center rounded-xl bg-primary">
            <Zap className="h-6 w-6 text-white" fill="white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Signing you in...</h2>
            <p className="text-sm text-muted-foreground">
              Verifying your magic link
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
