import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { LogOut, Zap, Settings, LayoutDashboard, Plus } from "lucide-react";

export function Navbar() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate("/");
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
      <div className="container flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Zap className="h-4 w-4 text-white" fill="white" />
          </div>
          <span className="text-lg font-bold tracking-tight">ClipSkari</span>
          <Badge variant="outline" className="ml-1 hidden sm:inline-flex">
            v1.0
          </Badge>
        </Link>

        {user ? (
          <nav className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
              <Link to="/dashboard">
                <LayoutDashboard className="h-4 w-4" />
                <span className="hidden sm:inline">Dashboard</span>
              </Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/jobs/new">
                <Plus className="h-4 w-4" />
                <span className="hidden sm:inline">New Job</span>
              </Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/settings">
                <Settings className="h-4 w-4" />
                <span className="hidden sm:inline">Settings</span>
              </Link>
            </Button>
            <Button variant="outline" size="sm" onClick={handleSignOut}>
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Sign out</span>
            </Button>
          </nav>
        ) : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
              <Link to="/login">Sign in</Link>
            </Button>
            <Button size="sm" asChild>
              <Link to="/login">Get started</Link>
            </Button>
          </div>
        )}
      </div>
    </header>
  );
}
