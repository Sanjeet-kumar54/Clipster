import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, ApiError } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import type { AppConfig } from "@/types";
import { Youtube, Film, Sparkles, Loader2 } from "lucide-react";

export function NewJob() {
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuth();
  const [submitting, setSubmitting] = useState(false);

  // Automation form state
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [minClips, setMinClips] = useState(5);
  const [maxClips, setMaxClips] = useState(10);
  const [captionLanguage, setCaptionLanguage] = useState<"hinglish" | "english">("hinglish");
  const [theme, setTheme] = useState("classic_white");
  const [colorGrading, setColorGrading] = useState("vibrant");
  const [watermark, setWatermark] = useState("@clipskari");
  const [resolution, setResolution] = useState("1080x1920");

  // Visual FX toggles
  const [fx, setFx] = useState<Record<string, boolean>>({});

  // Manifest form state
  const [manifestText, setManifestText] = useState(`{
  "batch": {
    "card_theme": "neon_void",
    "caption_language": "hinglish",
    "watermark_path": "@clipskari"
  },
  "clips": [
    {
      "url": "https://www.youtube.com/watch?v=...",
      "start": 60,
      "end": 120,
      "caption": "🔥 Example caption"
    }
  ]
}`);

  useEffect(() => {
    if (!authLoading && !user) navigate("/login");
  }, [user, authLoading, navigate]);

  const { data: config } = useQuery<AppConfig>({
    queryKey: ["app-config"],
    queryFn: async () => {
      const r = await api.getConfig();
      // Initialize FX defaults
      const initialFx: Record<string, boolean> = {};
      r.visual_fx.forEach((v) => (initialFx[v.id] = v.default));
      setFx(initialFx);
      return r;
    },
    enabled: !!user,
  });

  const buildBatchConfig = (): Record<string, unknown> => {
    const [w, h] = resolution.split("x").map(Number);
    const cfg: Record<string, unknown> = {
      card_theme: theme,
      caption_language: captionLanguage,
      target_width: w,
      target_height: h,
      color_grading_preset: colorGrading,
      color_grading_intensity: 0.85,
      watermark_enabled: !!watermark,
      watermark_path: watermark || "@clipskari",
      watermark_opacity: 0.4,
      watermark_position: "top_left",
    };
    // Apply FX toggles — map IDs to BatchConfig field names
    if ("punch_zoom" in fx) cfg.punch_zoom_enabled = fx.punch_zoom;
    if ("speaker_glow" in fx) cfg.speaker_glow_enabled = fx.speaker_glow;
    if ("film_grain" in fx) cfg.film_grain_enabled = fx.film_grain;
    if ("border_waveform" in fx) cfg.border_waveform_enabled = fx.border_waveform;
    if ("split_panel_rounded_corners" in fx)
      cfg.split_panel_rounded_corners = fx.split_panel_rounded_corners;
    if ("face_beautify" in fx) cfg.face_beautify_enabled = fx.face_beautify;
    if ("border_glow" in fx) cfg.border_glow_enabled = fx.border_glow;
    if ("letterbox" in fx) cfg.letterbox_enabled = fx.letterbox;
    if ("card_animated_reveal" in fx) cfg.card_animated_reveal = fx.card_animated_reveal;
    if ("dynamic_color_grading" in fx) cfg.dynamic_color_grading = fx.dynamic_color_grading;
    if ("dof" in fx) cfg.dof_enabled = fx.dof;
    if ("ken_burns" in fx) cfg.ken_burns_enabled = fx.ken_burns;
    if ("live_caption" in fx) cfg.live_caption_enabled = fx.live_caption;
    if ("video_bulge" in fx) cfg.video_bulge_enabled = fx.video_bulge;
    return cfg;
  };

  const submitAutomation = async () => {
    if (!url) {
      toast.error("Please enter a YouTube URL");
      return;
    }
    setSubmitting(true);
    try {
      const job = await api.createAutomationJob({
        source_url: url,
        title: title || undefined,
        min_clips: minClips,
        max_clips: maxClips,
        caption_language: captionLanguage,
        batch_config: buildBatchConfig(),
      });
      toast.success("Job submitted!", {
        description: "Your clips will be ready in a few minutes.",
      });
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Submission failed";
      toast.error(msg);
    } finally {
      setSubmitting(false);
    }
  };

  const submitManifest = async () => {
    let manifest: unknown;
    try {
      manifest = JSON.parse(manifestText);
    } catch {
      toast.error("Invalid JSON in manifest");
      return;
    }
    setSubmitting(true);
    try {
      const job = await api.createManifestJob({
        title: title || undefined,
        manifest: manifest as Record<string, unknown>,
      });
      toast.success("Job submitted!");
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Submission failed";
      toast.error(msg);
    } finally {
      setSubmitting(false);
    }
  };

  if (authLoading) return null;
  if (!user) return null;

  return (
    <div className="container max-w-4xl py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">New reframing job</h1>
        <p className="text-muted-foreground">
          Paste a YouTube URL for full automation, or define clips manually.
        </p>
      </div>

      <Tabs defaultValue="automation" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="automation">
            <Youtube className="mr-2 h-4 w-4" />
            Auto mode
          </TabsTrigger>
          <TabsTrigger value="manifest">
            <Film className="mr-2 h-4 w-4" />
            Manual mode
          </TabsTrigger>
        </TabsList>

        {/* Automation tab */}
        <TabsContent value="automation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" />
                Source video
              </CardTitle>
              <CardDescription>
                We'll transcribe, score, and auto-select the best {minClips}–{maxClips} clips.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="url">YouTube URL</Label>
                <Input
                  id="url"
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="title">Title (optional)</Label>
                <Input
                  id="title"
                  placeholder="My podcast episode"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Min clips</Label>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={minClips}
                    onChange={(e) => setMinClips(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max clips</Label>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={maxClips}
                    onChange={(e) => setMaxClips(Number(e.target.value))}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Style & FX</CardTitle>
              <CardDescription>
                Choose your card theme, caption language, and visual effects.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Card theme</Label>
                  <Select value={theme} onValueChange={setTheme}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {config?.themes.map((t) => (
                        <SelectItem key={t.id} value={t.id}>
                          <div className="flex items-center gap-2">
                            <div
                              className="h-3 w-3 rounded-full border"
                              style={{ background: t.swatch.bg, borderColor: t.swatch.accent }}
                            />
                            {t.name}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Caption language</Label>
                  <Select value={captionLanguage} onValueChange={(v) => setCaptionLanguage(v as "hinglish" | "english")}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {config?.caption_languages.map((l) => (
                        <SelectItem key={l.id} value={l.id}>{l.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Color grading</Label>
                  <Select value={colorGrading} onValueChange={setColorGrading}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {config?.color_grading_presets.map((p) => (
                        <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Output resolution</Label>
                  <Select value={resolution} onValueChange={setResolution}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {config?.output_resolutions.map((r) => (
                        <SelectItem key={r.id} value={r.id}>{r.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="watermark">Watermark text</Label>
                <Input
                  id="watermark"
                  placeholder="@yourhandle"
                  value={watermark}
                  onChange={(e) => setWatermark(e.target.value)}
                />
              </div>

              <div className="space-y-3">
                <Label>Visual effects</Label>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {config?.visual_fx.map((eff) => (
                    <label
                      key={eff.id}
                      className="flex cursor-pointer items-center justify-between rounded-md border border-border/50 px-3 py-2 hover:bg-accent/30"
                    >
                      <span className="text-sm">{eff.name}</span>
                      <Switch
                        checked={fx[eff.id] ?? eff.default}
                        onCheckedChange={(v) => setFx((p) => ({ ...p, [eff.id]: v }))}
                      />
                    </label>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              Cancel
            </Button>
            <Button onClick={submitAutomation} disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Submit job
                </>
              )}
            </Button>
          </div>
        </TabsContent>

        {/* Manifest tab */}
        <TabsContent value="manifest" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Manual manifest</CardTitle>
              <CardDescription>
                Define exact clips with start/end timestamps and captions.
                Useful when you already know which segments to extract.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="title2">Title (optional)</Label>
                <Input
                  id="title2"
                  placeholder="My manual batch"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="manifest">Manifest JSON</Label>
                <textarea
                  id="manifest"
                  className="flex min-h-[300px] w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  value={manifestText}
                  onChange={(e) => setManifestText(e.target.value)}
                />
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              Cancel
            </Button>
            <Button onClick={submitManifest} disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Film className="h-4 w-4" />
                  Submit manifest
                </>
              )}
            </Button>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
