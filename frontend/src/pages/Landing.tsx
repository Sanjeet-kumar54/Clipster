import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Zap,
  Youtube,
  Wand2,
  Sparkles,
  Download,
  ArrowRight,
  Cpu,
  Shield,
  Palette,
} from "lucide-react";

export function Landing() {
  return (
    <div className="min-h-screen bg-grid">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="container py-24 md:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-1.5 text-sm text-primary">
              <Sparkles className="h-4 w-4" />
              <span>Powered by Whisper large-v3 + TalkNet ASD + YOLO</span>
            </div>
            <h1 className="mb-6 text-5xl font-bold tracking-tight md:text-7xl">
              Turn long videos into
              <br />
              <span className="gradient-text">viral short clips</span>
            </h1>
            <p className="mb-10 text-lg text-muted-foreground md:text-xl">
              Paste a YouTube URL. Our GPU pipeline auto-selects the most engaging
              moments, reframes them to 9:16, adds captions, themes, and 13 visual
              effects — ready for Reels, Shorts & TikTok.
            </p>
            <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
              <Button size="lg" asChild className="w-full sm:w-auto">
                <Link to="/login">
                  Start free — 5 credits
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild className="w-full sm:w-auto">
                <Link to="/dashboard">View demo dashboard</Link>
              </Button>
            </div>
            <p className="mt-4 text-xs text-muted-foreground">
              No credit card required • Free tier includes 5 jobs
            </p>
          </div>
        </div>

        {/* Floating preview */}
        <div className="container pb-12">
          <div className="mx-auto max-w-4xl">
            <div className="relative aspect-video overflow-hidden rounded-xl border border-border/40 bg-card/50 shadow-2xl shadow-primary/10">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-fuchsia-500/20" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/20 backdrop-blur">
                    <Wand2 className="h-8 w-8 text-primary" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Live preview of your reframed clips
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="container py-16">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold md:text-4xl">
            Three-phase AI pipeline
          </h2>
          <p className="text-muted-foreground">
            From raw YouTube upload to ready-to-post short in minutes, fully automated.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <Card className="border-primary/20 bg-card/50">
            <CardContent className="p-6">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                <Youtube className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">1. Auto-clip selection</h3>
              <p className="text-sm text-muted-foreground">
                Whisper large-v3 transcribes word-by-word, then a rule + LLM
                scorer (Groq/Qwen) ranks 30–60s windows for hook strength,
                emotional impact, and shareability.
              </p>
            </CardContent>
          </Card>

          <Card className="border-primary/20 bg-card/50">
            <CardContent className="p-6">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                <Cpu className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">2. GPU reframing</h3>
              <p className="text-sm text-muted-foreground">
                YOLO face detection + TalkNet active speaker detection track the
                speaker. Bulletproof camera reframes to 9:16 with split-screen,
                punch zoom, speaker glow, and 13 visual FX.
              </p>
            </CardContent>
          </Card>

          <Card className="border-primary/20 bg-card/50">
            <CardContent className="p-6">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                <Download className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">3. Download & post</h3>
              <p className="text-sm text-muted-foreground">
                9 card color themes, Hinglish/English captions, audio-reactive
                waveform border, and QC contact sheet. One click to download
                every clip.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Tech stack */}
      <section className="container py-16">
        <div className="rounded-2xl border border-border/40 bg-card/30 p-8 md:p-12">
          <div className="mb-8 text-center">
            <h2 className="mb-3 text-2xl font-bold">Built on modern stack</h2>
            <p className="text-sm text-muted-foreground">
              Production-ready architecture, ready for your college submission.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
            {[
              { label: "Frontend", value: "React + Vite + TS", icon: "⚡" },
              { label: "Backend", value: "FastAPI + Python", icon: "🐍" },
              { label: "GPU compute", value: "Modal A10G", icon: "🎮" },
              { label: "Database", value: "Supabase + RLS", icon: "🗄️" },
              { label: "Auth", value: "Magic Link OTP", icon: "🔐" },
              { label: "Storage", value: "Supabase buckets", icon: "📦" },
              { label: "LLM scoring", value: "Groq + Qwen3-32B", icon: "🧠" },
              { label: "Hosting", value: "Vercel + Modal", icon: "🚀" },
            ].map((item) => (
              <div key={item.label} className="text-center">
                <div className="mb-2 text-3xl">{item.icon}</div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {item.label}
                </div>
                <div className="text-sm font-medium">{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="container py-16">
        <Card className="overflow-hidden border-primary/30 bg-gradient-to-br from-primary/10 via-card to-fuchsia-500/10">
          <CardContent className="p-8 text-center md:p-12">
            <Palette className="mx-auto mb-4 h-10 w-10 text-primary" />
            <h2 className="mb-3 text-3xl font-bold">Ready to create your first clip?</h2>
            <p className="mb-6 text-muted-foreground">
              Join the beta — get 5 free credits, no credit card needed.
            </p>
            <Button size="lg" asChild>
              <Link to="/login">
                <Zap className="h-4 w-4" fill="currentColor" />
                Get started
              </Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/40 py-8">
        <div className="container flex flex-col items-center justify-between gap-4 text-sm text-muted-foreground md:flex-row">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            <span>ClipSkari • College project submission</span>
          </div>
          <div className="flex items-center gap-4">
            <a href="https://github.com" className="hover:text-foreground">GitHub</a>
            <a href="/docs" className="hover:text-foreground">API docs</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
