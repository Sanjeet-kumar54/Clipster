#!/usr/bin/env python3
"""
GPU Video Reframer - Batch Edition v8  (Automation + TalkNet ASD + Visual FX + Themes)
=====================================================

Uses akanametov/yolo-face for face detection  AND  TaoRuijie/TalkNet-ASD
for Active Speaker Detection (audio-visual synchronisation).

Key upgrades vs v7 — CARD THEMES + CAPTION LANGUAGE + VISUAL ENHANCEMENTS:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ v8 CHANGES:                                                            │
  │  • 9 Card Color Themes (classic_white, neon_void, brat_summer, etc.)  │
  │  • Caption Language option (hinglish / english)                       │
  │  • Caption Glow + Bulge text effect                                   │
  │  • Bold Waveform Border (audio-reactive sine animation)               │
  │  • Video Bulge Effect (subtle barrel distortion)                      │
  │  • Reduced Zoom + Faster Settle (1.04 peak, 8 frames out)            │
  │  • Seamless Split View (gap=0, no accent line by default)             │
  │  • Caption Animation Start-Only (reveal once, not per scene)          │
  │  • English caption templates + smart char limits                       │
  └──────────────────────────────────────────────────────────────────────────┘
  All v7 features preserved (Automation + TalkNet ASD + 13 Visual FX):
  FULL AUTOMATION PIPELINE (from v7):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ AUTOMATION: YouTube URL in → Reframed clips out (zero manual work)     │
  │  • ClipDownloader — yt-dlp audio + section download                    │
  │  • WhisperTranscriber — faster-whisper large-v3 word-level timestamps  │
  │  • ClipScorer — rule-based + LLM scoring for clip selection            │
  │  • AutoClipGenerator — sliding windows → top-N non-overlapping clips   │
  │  • QCGen — 3x3 thumbnail contact sheet for visual QC                   │
  │  • ManifestLoader — JSON manifest for manual clip definitions          │
  │  • PipelineOrchestrator — 3-phase: Whisper → GPU Reframe → QC Grid    │
  └──────────────────────────────────────────────────────────────────────────┘

Processes multiple videos sequentially on Kaggle free GPU (T4, 15 GB VRAM).

TalkNet-ASD model:
  Repo  : https://github.com/TaoRuijie/TalkNet-ASD
  Weights: Google Drive 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea
           (pretrain_TalkSet.model — trained on TalkSet, best for wild video)
  Paper : "TalkNet: A Model for Active Speaker Detection in the Wild"
          (Ruijie Tao et al., Interspeech 2021)
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies & download models                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# !pip install ultralytics filterpy opencv-python-headless torchaudio \
#             python_speech_features gdown faster-whisper yt-dlp -q
#
# # ── YOLO face models (akanametov/yolo-face) ──────────────────────────
# !wget -q -O /kaggle/working/yolov8n-face.pt \
#     "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt" \
#     || echo "yolov8n-face download failed"
# !wget -q -O /kaggle/working/yolov11n-face.pt \
#     "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt" \
#     || echo "yolov11n-face download failed"
#
# # ── TalkNet-ASD ──────────────────────────────────────────────────────
# !git clone https://github.com/TaoRuijie/TalkNet-ASD.git \
#     /kaggle/working/TalkNet-ASD 2>/dev/null || true
# !gdown --id 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea \
#     -O /kaggle/working/TalkNet-ASD/pretrain_TalkSet.model \
#     || echo "TalkNet weights download failed (will fall back to lip-score)"
#
# import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# # Verify face models
# for path, name in [("/kaggle/working/yolov8n-face.pt","yolov8n-face"),
#                     ("/kaggle/working/yolov11n-face.pt","yolov11n-face")]:
#     if os.path.exists(path):
#         mb = os.path.getsize(path)/1024/1024
#         if mb < 1.0: os.remove(path); print(f"⚠ {name} too small ({mb:.1f}MB)")
#         else: print(f"✓ {name}.pt ready ({mb:.1f}MB)")
#     else: print(f"⚠ {name}.pt not downloaded")
#
# # Verify TalkNet
# tn = "/kaggle/working/TalkNet-ASD/pretrain_TalkSet.model"
# if os.path.exists(tn):
#     mb = os.path.getsize(tn)/1024/1024
#     if mb < 5: os.remove(tn); print(f"⚠ TalkNet weights too small ({mb:.1f}MB)")
#     else: print(f"✓ TalkNet weights ready ({mb:.1f}MB)")
# else: print("⚠ TalkNet weights not found — will use lip-score fallback")
#
# # Fonts
# !apt-get install -y fonts-noto-color-emoji fonts-noto-core > /dev/null 2>&1
# print("✓ Fonts installed (Noto Sans Devanagari + Color Emoji)")
# print("\n✓ All dependencies ready!")
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Main script                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

import cv2
import numpy as np
import torch
import torchaudio
import subprocess
import time
import warnings
import os
import re
import shutil
import sys
import math
import json
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from io import BytesIO
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

# ── TalkNet imports (from cloned repo) ────────────────────────────────
TALKNET_REPO = "/kaggle/working/TalkNet-ASD"
if os.path.isdir(TALKNET_REPO):
    sys.path.insert(0, TALKNET_REPO)


# ============================================================
# BATCH JOB DEFINITION
# ============================================================
@dataclass
class VideoJob:
    """One video to process with its own caption."""
    input_video:  str
    output_video: str
    card_text:    str
    card_subtext: str = ""
    color_grading_preset: Optional[str] = None
    color_grading_intensity: float = 0.85
    delete_input_after: bool = False



# ============================================================
# v8: CARD COLOR THEMES
# ============================================================
CARD_THEMES = {
    "classic_white": {
        "card_bg_color": (255, 255, 255),
        "card_text_color": (10, 10, 10),
        "card_subtext_color": (110, 110, 110),
        "card_accent_color": (255, 40, 40),
        "card_shadow_blur": 32,
        "card_shadow_offset": (0, 10),
        "card_shadow_opacity": 0.30,
        "caption_glow_color": (255, 60, 60),
        "caption_glow_enabled": True,
        "border_wave_color": (255, 40, 40),
    },
    "neon_void": {
        "card_bg_color": (13, 13, 26),
        "card_text_color": (240, 238, 255),
        "card_subtext_color": (155, 143, 204),
        "card_accent_color": (177, 79, 255),
        "card_shadow_blur": 40,
        "card_shadow_offset": (0, 14),
        "card_shadow_opacity": 0.38,
        "caption_glow_color": (177, 79, 255),
        "caption_glow_enabled": True,
        "border_wave_color": (140, 50, 255),
    },
    "brat_summer": {
        "card_bg_color": (202, 255, 58),
        "card_text_color": (13, 13, 13),
        "card_subtext_color": (61, 82, 0),
        "card_accent_color": (13, 13, 13),
        "card_shadow_blur": 28,
        "card_shadow_offset": (0, 10),
        "card_shadow_opacity": 0.20,
        "caption_glow_color": (13, 13, 13),
        "caption_glow_enabled": True,
        "border_wave_color": (180, 255, 0),
    },
    "glazed_donut": {
        "card_bg_color": (253, 238, 231),
        "card_text_color": (59, 26, 16),
        "card_subtext_color": (160, 96, 80),
        "card_accent_color": (232, 132, 90),
        "card_shadow_blur": 32,
        "card_shadow_offset": (0, 10),
        "card_shadow_opacity": 0.22,
        "caption_glow_color": (232, 132, 90),
        "caption_glow_enabled": True,
        "border_wave_color": (232, 132, 90),
    },
    "digital_dusk": {
        "card_bg_color": (6, 15, 46),
        "card_text_color": (224, 242, 255),
        "card_subtext_color": (96, 123, 158),
        "card_accent_color": (0, 212, 255),
        "card_shadow_blur": 44,
        "card_shadow_offset": (0, 14),
        "card_shadow_opacity": 0.30,
        "caption_glow_color": (0, 212, 255),
        "caption_glow_enabled": True,
        "border_wave_color": (0, 180, 255),
    },
    "pink_pill": {
        "card_bg_color": (255, 45, 120),
        "card_text_color": (255, 255, 255),
        "card_subtext_color": (255, 200, 225),
        "card_accent_color": (255, 255, 255),
        "card_shadow_blur": 36,
        "card_shadow_offset": (0, 12),
        "card_shadow_opacity": 0.42,
        "caption_glow_color": (255, 200, 220),
        "caption_glow_enabled": True,
        "border_wave_color": (255, 45, 120),
    },
    "matcha_latte": {
        "card_bg_color": (239, 245, 233),
        "card_text_color": (26, 46, 16),
        "card_subtext_color": (107, 138, 85),
        "card_accent_color": (74, 124, 48),
        "card_shadow_blur": 26,
        "card_shadow_offset": (0, 8),
        "card_shadow_opacity": 0.16,
        "caption_glow_color": (74, 124, 48),
        "caption_glow_enabled": True,
        "border_wave_color": (74, 160, 40),
    },
    "midnight_chrome": {
        "card_bg_color": (17, 17, 17),
        "card_text_color": (245, 245, 245),
        "card_subtext_color": (119, 119, 119),
        "card_accent_color": (200, 200, 200),
        "card_shadow_blur": 40,
        "card_shadow_offset": (0, 14),
        "card_shadow_opacity": 0.50,
        "caption_glow_color": (180, 180, 255),
        "caption_glow_enabled": True,
        "border_wave_color": (150, 150, 200),
    },
    "burnt_orange": {
        "card_bg_color": (26, 3, 0),
        "card_text_color": (255, 237, 204),
        "card_subtext_color": (160, 90, 48),
        "card_accent_color": (255, 107, 0),
        "card_shadow_blur": 40,
        "card_shadow_offset": (0, 12),
        "card_shadow_opacity": 0.30,
        "caption_glow_color": (255, 120, 0),
        "caption_glow_enabled": True,
        "border_wave_color": (255, 80, 0),
    },
}

@dataclass
class BatchConfig:
    """Global settings shared across all jobs."""
    # ── Output ────────────────────────────────────────────────────────
    target_width:  int = 1080
    target_height: int = 1920

    # ── Storage safety ────────────────────────────────────────────────
    min_free_gb: float = 2.0

    # ── v8: Caption language ────────────────────────────────────────
    caption_language: str = "hinglish"  # v8: "english" or "hinglish"

    # ── v8: Card theme ──────────────────────────────────────────────────
    card_theme: str = "classic_white"  # v8: CARD_THEMES key

    # ── YOLO face model settings ──────────────────────────────────────
    yolo_imgsz: int = 1280
    yolo_face_variant: str = "auto"

    # ── TalkNet ASD settings ──────────────────────────────────────────
    talknet_enabled: bool = True
    talknet_infer_every: int = 5
    talknet_min_frames: int = 25
    talknet_smooth_window: int = 5
    talknet_speaking_threshold: float = 0.0
    talknet_durations: Tuple = (1, 2, 3)

    # ── Card defaults ─────────────────────────────────────────────────
    card_enabled: bool = True
    card_bg_color: Tuple = (255, 255, 255)
    card_padding_sides: float = 0.04
    card_padding_top: float = 0.22
    card_padding_bottom: float = 0.03
    card_corner_radius: float = 0.05
    card_text_size: float = 0.22
    card_text_color: Tuple = (10, 10, 10)
    card_text_weight: int = 800
    card_subtext_color: Tuple = (110, 110, 110)
    card_accent_color: Tuple = (255, 40, 40)
    card_shadow: bool = True
    card_shadow_blur: int = 32
    card_shadow_offset: Tuple = (0, 10)
    card_shadow_opacity: float = 0.30

    # ── Default color grading ──────────────────────────────────────────
    color_grading_preset:    str   = "vibrant"
    color_grading_intensity: float = 0.85

    # ── Tracking ──────────────────────────────────────────────────────
    tracking_lerp: float = 0.06
    dead_zone_px:  int = 18
    kalman_noise:  float = 0.0003
    transition_frames: int = 25
    hysteresis_frames: int = 15
    face_padding_ratio: float = 2.5
    face_vertical_pos:  float = 0.35
    face_center_bias:           float = 0.40
    face_stickiness_frames:     int   = 10
    face_stickiness_size_bonus: float = 0.35

    # ── Gates ─────────────────────────────────────────────────────────
    yolo_confidence:       float = 0.50
    min_face_size_ratio:   float = 0.07
    size_dominance_ratio:  float = 0.65
    max_live_faces:        int   = 2
    face_edge_reject_ratio: float = 0.08
    face_track_min_frames: int   = 8

    # ── Zoom ──────────────────────────────────────────────────────────
    zoom_enabled:        bool  = True
    zoom_rms_window:     int   = 15
    zoom_rms_threshold:  float = 0.04
    zoom_rms_peak:       float = 0.12
    zoom_max_factor:     float = 1.18
    zoom_in_lerp:        float = 0.06
    zoom_out_lerp:       float = 0.08  # v8: faster from 0.03

    # ── Split screen ──────────────────────────────────────────────────
    split_enabled:               bool  = True
    split_both_speaking_frames:  int   = 10
    split_exit_frames:           int   = 25
    split_transition_frames:     int   = 18
    split_gap_px:                int   = 0  # v8: seamless split (was 6)
    split_mode:                  str   = "top_bottom"
    split_padding_ratio:         float = 3.5
    split_overlap_bias:          float = 0.25
    split_overlap_check:         bool  = True
    split_face_confidence_frames: int  = 25
    split_cold_start_frames:     int   = 40
    split_panel_freeze_enabled:    bool = True
    split_panel_freeze_max_frames: int  = 60
    split_panel_reassign_frames:   int  = 8
    split_face_vertical_pos:    float = 0.45
    panel_min_face_coverage: float = 0.10
    panel_max_vert_drift:    float = 1.5

    # ── Re-ID ─────────────────────────────────────────────────────────
    reid_enabled:        bool  = True
    reid_gallery_size:   int   = 5
    reid_match_threshold: float = 0.55
    reid_max_lost_frames: int  = 90

    # ── Scene cut ─────────────────────────────────────────────────────
    scene_cut_enabled:   bool  = True
    scene_cut_threshold: float = 28.0
    scene_cut_cooldown:  int   = 15

    portrait_stabilize_only: bool = False

    # ── v6 Visual Effects ──────────────────────────────────────────────
    # 1. Split Gap Styling
    split_gap_style: str = "gradient"
    split_gap_gradient_px: int = 8
    split_gap_accent_line: bool = False  # v8: no accent line (was True)

    # 2. Punch Zoom on Speaker Switch
    punch_zoom_enabled: bool = True
    punch_zoom_peak: float = 1.04  # v8: reduced from 1.08
    punch_zoom_in_frames: int = 5
    punch_zoom_out_frames: int = 8  # v8: faster settle from 15

    # 3. Speaker Glow Ring (TalkNet-driven)
    speaker_glow_enabled: bool = True
    speaker_glow_color: Tuple = (255, 40, 40)
    speaker_glow_max_thickness: int = 4
    speaker_glow_pulse_speed: float = 0.15
    speaker_glow_min_score: float = 0.5

    # 4. Film Grain Overlay
    film_grain_enabled: bool = True
    film_grain_intensity: float = 0.03
    film_grain_cache_size: int = 8

    # 5. Panel Rounded Corners + Shadow
    split_panel_rounded_corners: bool = True
    split_panel_corner_radius: float = 0.03
    split_panel_shadow_enabled: bool = True
    split_panel_shadow_blur: int = 12
    split_panel_shadow_opacity: float = 0.25

    # 6. Watermark / Branding Stamp
    watermark_enabled: bool = False
    watermark_path: str = ""
    watermark_opacity: float = 0.4
    watermark_position: str = "bottom_right"
    watermark_size_ratio: float = 0.08
    watermark_fade_in_frames: int = 20

    # 7. Face Beautification
    face_beautify_enabled: bool = True
    face_beautify_strength: float = 0.3
    face_beautify_d: int = 9
    face_beautify_sigma: int = 30

    # 8. Border Glow (Audio-reactive)
    border_glow_enabled: bool = True
    border_glow_width: int = 5
    border_glow_max_opacity: float = 0.6
    border_glow_color: Tuple = (255, 180, 50)

    # 9. Cinematic Letterbox Bars
    letterbox_enabled: bool = True
    letterbox_bar_ratio: float = 0.04

    # 10. Animated Text Reveal
    card_animated_reveal: bool = True
    card_reveal_frames_per_line: int = 8
    card_reveal_slide_px: int = 15

    # 11. Dynamic Color Grading
    dynamic_color_grading: bool = False
    dynamic_cg_speech_high_preset: str = "vibrant"
    dynamic_cg_speech_low_preset: str = "cinematic"
    dynamic_cg_silence_preset: str = "moody"
    dynamic_cg_rms_threshold: float = 0.06
    dynamic_cg_crossfade_frames: int = 20

    # 12. Depth-of-Field Simulation
    dof_enabled: bool = True
    dof_blur_strength: float = 0.55
    dof_zoom_blur_boost: float = 0.25
    dof_bg_kernel_size: int = 99

    # 13. Ken Burns Effect
    ken_burns_enabled: bool = True
    ken_burns_min_stable_frames: int = 50
    ken_burns_max_zoom: float = 1.008
    ken_burns_drift_speed: float = 0.0003

    # ── v8: Caption Glow ─────────────────────────────────────────────
    caption_glow_enabled: bool = True  # v8
    caption_glow_color: Tuple = (255, 60, 60)  # v8

    # ── v8: Border Waveform ──────────────────────────────────────────
    border_waveform_enabled: bool = True  # v8
    border_wave_color: Tuple = (255, 40, 40)  # v8
    border_wave_width: int = 18  # v8: bolder (was 6 → 12, now 18 for clear visibility)
    border_wave_max_opacity: float = 0.85  # v8: raised for more visibility
    border_wave_speed: float = 0.12  # v8
    border_wave_frequency: float = 4.0  # v8

    # ── v8: Video Bulge Effect ───────────────────────────────────────
    video_bulge_enabled: bool = True  # v8
    video_bulge_strength: float = 0.02  # v8

    # ── v8: Live Glowing Caption (karaoke-style word sync) ──────────
    live_caption_enabled: bool = False  # v8: show word-by-word caption
    live_caption_correct_whisper: bool = False  # v8: manually edit Whisper output
    live_caption_max_words: int = 4  # v8: 3-5 words visible at once
    live_caption_font_size: float = 0.055  # v8: relative to frame height
    live_caption_position: str = "bottom_center"  # v8: bottom_center / mid_center
    live_caption_glow_color: Tuple = (0, 220, 255)  # v8: cyan glow (RGB)
    live_caption_text_color: Tuple = (255, 255, 255)  # v8: white text (RGB)
    live_caption_highlight_color: Tuple = (255, 255, 50)  # v8: yellow highlight for current word (RGB)
    live_caption_bg_opacity: float = 0.72  # v8: darker gradient for clear caption visibility
    live_caption_word_gap_sec: float = 0.05  # v8: gap between word groups
    live_caption_neon_cycle: bool = True  # v8: cycle neon colors for highlight words
    live_caption_attention_elements: bool = True  # v8: emoji pulse attention elements

    # ── v8: Auto B-Roll (Pexels Free API) ────────────────────────────
    broll_enabled: bool = False  # v8: auto B-roll insertion
    broll_pexels_api_key: str = "_pexel_api_key"  # v8: Pexels API key (free at pexels.com/api)
    broll_duration_sec: float = 3.0  # v8: B-roll clip duration
    broll_max_per_clip: int = 3  # v8: max B-roll triggers per keyword
    broll_style: str = "dissolve"  # v8: dissolve / pip / split


# ── Thin Config shim ──────────────────────────────────────────────────
@dataclass
class Config:
    input_video:  str = ""
    output_video: str = ""
    target_width:  int = 1080
    target_height: int = 1920
    transition_frames: int = 25
    hysteresis_frames: int = 15
    face_padding_ratio: float = 2.5
    face_vertical_pos:  float = 0.35
    card_enabled: bool = True
    card_bg_color: Tuple = (255, 255, 255)
    card_padding_sides: float = 0.04
    card_padding_top: float = 0.22
    card_padding_bottom: float = 0.03
    card_corner_radius: float = 0.05
    card_text: str = ""
    card_text_size: float = 0.22
    card_text_color: Tuple = (10, 10, 10)
    card_text_weight: int = 800
    card_subtext: str = ""
    card_subtext_color: Tuple = (110, 110, 110)
    card_accent_color: Tuple = (255, 40, 40)
    card_shadow: bool = True
    card_shadow_blur: int = 32
    card_shadow_offset: Tuple = (0, 10)
    card_shadow_opacity: float = 0.30
    tracking_lerp: float = 0.06
    dead_zone_px:  int = 18
    kalman_noise:  float = 0.0003
    yolo_confidence: float = 0.50
    yolo_imgsz: int = 1280
    min_face_size_ratio: float = 0.07
    size_dominance_ratio: float = 0.65
    max_live_faces: int = 2
    face_edge_reject_ratio: float = 0.08
    face_track_min_frames: int = 8
    face_center_bias:           float = 0.40
    face_stickiness_frames:     int   = 10
    face_stickiness_size_bonus: float = 0.35
    zoom_enabled:        bool  = True
    zoom_rms_window:     int   = 15
    zoom_rms_threshold:  float = 0.04
    zoom_rms_peak:       float = 0.12
    zoom_max_factor:     float = 1.18
    zoom_in_lerp:        float = 0.06
    zoom_out_lerp:       float = 0.08  # v8: faster from 0.03
    split_enabled:               bool  = True
    split_both_speaking_frames:  int   = 10
    split_exit_frames:           int   = 25
    split_transition_frames:     int   = 18
    split_gap_px:                int   = 0  # v8: seamless (was 6)
    split_mode:                  str   = "top_bottom"
    split_padding_ratio:         float = 3.5
    split_overlap_bias:          float = 0.25
    split_overlap_check:         bool  = True
    split_face_confidence_frames: int  = 25
    split_cold_start_frames:     int   = 40
    split_panel_freeze_enabled:    bool = True
    split_panel_freeze_max_frames: int  = 60
    split_panel_reassign_frames:   int  = 8
    split_face_vertical_pos:    float = 0.45
    panel_min_face_coverage: float = 0.10
    panel_max_vert_drift:    float = 1.5
    reid_enabled:        bool  = True
    reid_gallery_size:   int   = 5
    reid_match_threshold: float = 0.55
    reid_max_lost_frames: int  = 90
    scene_cut_enabled:   bool  = True
    scene_cut_threshold: float = 28.0
    scene_cut_cooldown:  int   = 15
    portrait_stabilize_only: bool = False
    color_grading: object = None
    # TalkNet
    talknet_enabled: bool = True
    talknet_infer_every: int = 5
    talknet_min_frames: int = 25
    talknet_smooth_window: int = 5
    talknet_speaking_threshold: float = 0.0
    talknet_durations: Tuple = (1, 2, 3)
    # ── v6 Visual Effects ──
    split_gap_style: str = "gradient"
    split_gap_gradient_px: int = 8
    split_gap_accent_line: bool = False  # v8: no accent (was True)
    punch_zoom_enabled: bool = True
    punch_zoom_peak: float = 1.04  # v8: reduced from 1.08
    punch_zoom_in_frames: int = 5
    punch_zoom_out_frames: int = 8  # v8: faster from 15
    speaker_glow_enabled: bool = True
    speaker_glow_color: Tuple = (255, 40, 40)
    speaker_glow_max_thickness: int = 4
    speaker_glow_pulse_speed: float = 0.15
    speaker_glow_min_score: float = 0.5
    film_grain_enabled: bool = True
    film_grain_intensity: float = 0.03
    film_grain_cache_size: int = 8
    split_panel_rounded_corners: bool = True
    split_panel_corner_radius: float = 0.03
    split_panel_shadow_enabled: bool = True
    split_panel_shadow_blur: int = 12
    split_panel_shadow_opacity: float = 0.25
    watermark_enabled: bool = False
    watermark_path: str = ""
    watermark_opacity: float = 0.4
    watermark_position: str = "bottom_right"
    watermark_size_ratio: float = 0.08
    watermark_fade_in_frames: int = 20
    face_beautify_enabled: bool = True
    face_beautify_strength: float = 0.3
    face_beautify_d: int = 9
    face_beautify_sigma: int = 30
    border_glow_enabled: bool = True
    border_glow_width: int = 5
    border_glow_max_opacity: float = 0.6
    border_glow_color: Tuple = (255, 180, 50)
    letterbox_enabled: bool = True
    letterbox_bar_ratio: float = 0.04
    card_animated_reveal: bool = True
    card_reveal_frames_per_line: int = 8
    card_reveal_slide_px: int = 15
    dynamic_color_grading: bool = False
    dynamic_cg_speech_high_preset: str = "vibrant"
    dynamic_cg_speech_low_preset: str = "cinematic"
    dynamic_cg_silence_preset: str = "moody"
    dynamic_cg_rms_threshold: float = 0.06
    dynamic_cg_crossfade_frames: int = 20
    dof_enabled: bool = True
    dof_blur_strength: float = 0.55
    dof_zoom_blur_boost: float = 0.25
    dof_bg_kernel_size: int = 99
    ken_burns_enabled: bool = True
    ken_burns_min_stable_frames: int = 50
    ken_burns_max_zoom: float = 1.008
    ken_burns_drift_speed: float = 0.0003
    # ── v8 new fields ──
    caption_language: str = "hinglish"  # v8
    card_theme: str = "classic_white"  # v8
    caption_glow_enabled: bool = True  # v8
    caption_glow_color: Tuple = (255, 60, 60)  # v8
    border_waveform_enabled: bool = True  # v8
    border_wave_color: Tuple = (255, 40, 40)  # v8
    border_wave_width: int = 18  # v8: bolder (was 6 → 12, now 18)
    border_wave_max_opacity: float = 0.85  # v8: raised for more visibility
    border_wave_speed: float = 0.12  # v8
    border_wave_frequency: float = 4.0  # v8
    video_bulge_enabled: bool = True  # v8
    video_bulge_strength: float = 0.02  # v8
    # ── v8: Live Glowing Caption ──
    live_caption_enabled: bool = False  # v8
    live_caption_correct_whisper: bool = False  # v8
    live_caption_max_words: int = 4  # v8
    live_caption_font_size: float = 0.055  # v8
    live_caption_position: str = "bottom_center"  # v8
    live_caption_glow_color: Tuple = (0, 220, 255)  # v8
    live_caption_text_color: Tuple = (255, 255, 255)  # v8
    live_caption_highlight_color: Tuple = (255, 255, 50)  # v8
    live_caption_bg_opacity: float = 0.72  # v8: darker gradient for clear visibility
    live_caption_word_gap_sec: float = 0.05  # v8
    live_caption_neon_cycle: bool = True  # v8: cycle neon colors for highlight words
    live_caption_attention_elements: bool = True  # v8: emoji pulse attention elements
    # ── v8: Auto B-Roll (Pexels Free API) ──
    broll_enabled: bool = False  # v8
    broll_pexels_api_key: str = "_pexel_api_key"  # v8
    broll_duration_sec: float = 3.0  # v8
    broll_max_per_clip: int = 3  # v8
    broll_style: str = "dissolve"  # v8


def _make_config(batch: BatchConfig, job: VideoJob) -> Config:
    """Merge BatchConfig + VideoJob into a single Config."""
    preset  = job.color_grading_preset or batch.color_grading_preset
    intense = job.color_grading_intensity or batch.color_grading_intensity
    cg = ColorGradingConfig(mode="preset" if preset else "off",
                             preset_name=preset or "cinematic",
                             intensity=intense)
    cfg = Config(color_grading=cg)
    for f in BatchConfig.__dataclass_fields__:
        if hasattr(cfg, f):
            setattr(cfg, f, getattr(batch, f))
    cfg.input_video  = job.input_video
    cfg.output_video = job.output_video
    cfg.card_text    = job.card_text
    cfg.card_subtext = job.card_subtext
    cfg.color_grading = cg

    # ── v8 FIX: Apply theme FIRST, then convert RGB→BGR ──
    # Theme colors in CARD_THEMES are defined as RGB tuples.
    # OpenCV functions (np.full, cv2.rectangle, etc.) need BGR.
    # PIL functions (draw.text) need RGB — those are NOT converted.
    # Previously the BGR conversion happened BEFORE theme override,
    # so theme RGB values were never converted → wrong colors on screen.

    # Step 1: Apply card theme (overrides individual card_* fields) — still RGB
    theme_name = getattr(batch, 'card_theme', 'classic_white')
    if theme_name and theme_name in CARD_THEMES:
        theme = CARD_THEMES[theme_name]
        for key, value in theme.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # Step 2: Convert only OpenCV-used colors from RGB→BGR
    # (PIL-used colors like card_text_color, card_subtext_color,
    #  caption_glow_color stay as RGB)
    # BUG FIX #5: Also handle lists (JSON parses [255,40,40] as list,
    # not tuple). The isinstance(c, tuple) check would skip lists from
    # manifest JSON, so colors from manifest overrides were never converted.
    _OPENCV_COLOR_KEYS = [
        'card_bg_color', 'card_accent_color', 'border_glow_color',
        'border_wave_color', 'speaker_glow_color',
    ]
    for key in _OPENCV_COLOR_KEYS:
        c = getattr(cfg, key, None)
        if isinstance(c, (tuple, list)) and len(c) == 3:
            c = tuple(c)  # normalize list → tuple
            setattr(cfg, key, (c[2], c[1], c[0]))  # RGB → BGR

    return cfg


# ============================================================
# SCRIPT-AWARE TEXT HELPERS
# ============================================================
def _is_devanagari(ch):
    """Check if character should be rendered with Devanagari font.

    Includes Devanagari block + common Indian currency/symbol characters
    that are present in NotoSansDevanagari but missing from Latin fonts
    like Montserrat (e.g. ₹ U+20B9 Indian Rupee Sign).
    """
    cp = ord(ch)
    return ('\u0900' <= ch <= '\u097F' or   # Devanagari
            '\u0980' <= ch <= '\u09FF' or   # Bengali
            cp == 0x20B9 or                  # ₹ Indian Rupee Sign
            cp == 0xA838 or                  # ₹ Devanagari Extended Rupee
            cp == 0x0964 or                  # । Devanagari Danda
            cp == 0x0965)                    # ॥ Double Danda

def _segment_by_script(text):
    if not text: return []
    segments = []; current = text[0]; current_deva = _is_devanagari(text[0])
    for ch in text[1:]:
        ch_deva = _is_devanagari(ch)
        if ch_deva == current_deva: current += ch
        else: segments.append((current, current_deva)); current = ch; current_deva = ch_deva
    segments.append((current, current_deva))
    return segments


# ============================================================
# COLOR GRADING ENGINE
# ============================================================
class ColorGradingMode:
    OFF = "off"; PRESET = "preset"; CUBE = "cube"

@dataclass
class ColorGradingConfig:
    mode: str = ColorGradingMode.OFF
    preset_name: str = "cinematic"
    cube_path: str = ""
    intensity: float = 1.0

class ColorGrader:
    PRESETS = {
        "cinematic":    {"shadows_lift":0.05,"highlights_roll":0.92,"contrast":1.15,"saturation":0.85,"temp_shift":(8,0,-8)},
        "moody":        {"shadows_lift":0.0, "highlights_roll":0.88,"contrast":1.25,"saturation":0.70,"temp_shift":(-5,-2,10)},
        "vibrant":      {"shadows_lift":0.02,"highlights_roll":0.95,"contrast":1.10,"saturation":1.30,"temp_shift":(3,2,-3)},
        "bleach_bypass":{"shadows_lift":0.0, "highlights_roll":0.85,"contrast":1.40,"saturation":0.45,"temp_shift":(0,0,0)},
        "golden_hour":  {"shadows_lift":0.06,"highlights_roll":0.90,"contrast":1.08,"saturation":1.10,"temp_shift":(15,3,-12)},
        "teal_orange":  {"shadows_lift":0.03,"highlights_roll":0.91,"contrast":1.20,"saturation":0.95,"temp_shift":(5,0,-5)},
        "matte":        {"shadows_lift":0.10,"highlights_roll":0.82,"contrast":0.95,"saturation":0.80,"temp_shift":(4,2,-2)},
    }
    def __init__(self, cg: ColorGradingConfig):
        self.cg = cg; self._lut = None
        self._intensity = np.clip(cg.intensity, 0, 1); self._ready = False
        if cg.mode == ColorGradingMode.OFF: return
        try:
            if cg.mode == ColorGradingMode.PRESET: self._build_preset()
            self._ready = True
        except Exception as e: print(f"  ⚠ Color grading failed ({e})")
    def _build_preset(self):
        p = self.PRESETS[self.cg.preset_name]; inp = np.linspace(0,1,256,dtype=np.float64)
        curve = inp*(1-p["shadows_lift"])+p["shadows_lift"]
        roll = p["highlights_roll"]; mask = curve>roll; over = curve[mask]-roll; r = 1-roll+1e-9
        curve[mask] = roll+(1-roll)*np.sin(np.pi/2*np.clip(over/r,0,1))
        curve = np.clip((curve-0.5)*p["contrast"]+0.5,0,1); sh = p["temp_shift"]
        lr = np.clip(curve*255+sh[0],0,255).astype(np.uint8)
        lg = np.clip(curve*255+sh[1],0,255).astype(np.uint8)
        lb = np.clip(curve*255+sh[2],0,255).astype(np.uint8)
        self._lut = np.stack([lb,lg,lr],axis=1); self._sat = p["saturation"]
    def apply(self, frame):
        if not self._ready: return frame
        try:
            b,g,r = cv2.split(frame)
            out = cv2.merge([cv2.LUT(b,self._lut[:,0]),cv2.LUT(g,self._lut[:,1]),cv2.LUT(r,self._lut[:,2])])
            if abs(self._sat-1)>0.01:
                hsv = cv2.cvtColor(out,cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,1] = np.clip(hsv[:,:,1]*self._sat,0,255)
                out = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            if self._intensity<1.0: out = cv2.addWeighted(frame,1-self._intensity,out,self._intensity,0)
            return out
        except Exception: return frame


# ============================================================
# VISUAL FX ENGINE  (v8 - 13 effects + waveform border)
# ============================================================
class VisualFX:
    """Applies all v6 visual effects in the correct order."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._grain_cache = []
        self._grain_idx = 0
        self._punch_zoom = 1.0
        self._punch_decay = 0
        self._kb_offset_x = 0.0
        self._kb_offset_y = 0.0
        self._kb_zoom = 1.0
        self._kb_stable_cnt = 0
        self._kb_last_spk = None
        self._dcg_preset = None
        self._watermark_img = None
        self._watermark_loaded = False
        self._reveal_frame = 0
        self._glow_frame = 0
        self._reveal_done = False  # v8: caption animation start-only

    def reset_scene(self):
        self._kb_offset_x = 0.0; self._kb_offset_y = 0.0; self._kb_zoom = 1.0
        self._kb_stable_cnt = 0; self._kb_last_spk = None
        # v8 FIX: Fast-forward reveal so ALL text lines are immediately visible
        # after a scene cut. The old code set _reveal_done=True which froze
        # the reveal frame counter — if a scene cut happened before Line 2
        # faded in (within ~16 frames), Line 2 would NEVER appear.
        # Now we set the frame high enough that every line is fully visible.
        self._reveal_done = True
        self._reveal_frame = 9999  # all lines past their reveal start
        self._glow_frame = 0

    # ── 1. Split Gap Styling ─────────────────────────────────────────
    @staticmethod
    def draw_split_gap(canvas, cfg, is_top_bottom=True):
        gap = cfg.split_gap_px
        if gap <= 0: return
        h, w = canvas.shape[:2]
        accent = cfg.card_accent_color
        if is_top_bottom:
            mid_y = h // 2; g_y1 = mid_y - gap // 2; g_y2 = g_y1 + gap
            if cfg.split_gap_style == "solid":
                canvas[g_y1:g_y2, :] = 0
            elif cfg.split_gap_style == "gradient":
                canvas[g_y1:g_y2, :] = 0
                grad_px = min(cfg.split_gap_gradient_px, gap, g_y1, h - g_y2)
                if grad_px > 1:
                    for dy in range(grad_px):
                        alpha = (dy + 1) / grad_px * 0.5
                        r_above = g_y1 - grad_px + dy
                        if 0 <= r_above < h:
                            canvas[r_above, :] = (canvas[r_above, :].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                        r_below = g_y2 + grad_px - 1 - dy
                        if 0 <= r_below < h:
                            canvas[r_below, :] = (canvas[r_below, :].astype(np.float32) * (1 - alpha)).astype(np.uint8)
            elif cfg.split_gap_style == "glow":
                canvas[g_y1:g_y2, :] = 0
                blur_r = (gap + 4) | 1
                gm = np.zeros((h, w), dtype=np.float32); gm[g_y1:g_y2, :] = 1.0
                gm = cv2.GaussianBlur(gm, (blur_r, blur_r), 0)
                for c in range(3):
                    canvas[:, :, c] = np.clip(canvas[:, :, c].astype(np.float32) * (1 - gm * 0.3), 0, 255).astype(np.uint8)
            if cfg.split_gap_accent_line:
                line_y = (g_y1 + g_y2) // 2
                cv2.line(canvas, (int(w * 0.15), line_y), (int(w * 0.85), line_y), accent, 2, cv2.LINE_AA)
        else:
            mid_x = w // 2; g_x1 = mid_x - gap // 2; g_x2 = g_x1 + gap
            if cfg.split_gap_style == "solid":
                canvas[:, g_x1:g_x2] = 0
            elif cfg.split_gap_style in ("gradient", "glow"):
                canvas[:, g_x1:g_x2] = 0
                if cfg.split_gap_style == "glow":
                    blur_r = (gap + 4) | 1
                    gm = np.zeros((h, w), dtype=np.float32); gm[:, g_x1:g_x2] = 1.0
                    gm = cv2.GaussianBlur(gm, (blur_r, blur_r), 0)
                    for c in range(3):
                        canvas[:, :, c] = np.clip(canvas[:, :, c].astype(np.float32) * (1 - gm * 0.3), 0, 255).astype(np.uint8)
            if cfg.split_gap_accent_line:
                line_x = (g_x1 + g_x2) // 2
                cv2.line(canvas, (line_x, int(h * 0.15)), (line_x, int(h * 0.85)), accent, 2, cv2.LINE_AA)

    # ── 2. Punch Zoom ────────────────────────────────────────────────
    def update_punch_zoom(self, speaker_changed):
        if not self.cfg.punch_zoom_enabled: return 1.0
        if speaker_changed:
            self._punch_zoom = self.cfg.punch_zoom_peak
            self._punch_decay = self.cfg.punch_zoom_in_frames + self.cfg.punch_zoom_out_frames
        if self._punch_decay > 0:
            self._punch_decay -= 1
            if self._punch_decay <= self.cfg.punch_zoom_out_frames:
                t = self._punch_decay / max(1, self.cfg.punch_zoom_out_frames)
                self._punch_zoom = 1.0 + (self.cfg.punch_zoom_peak - 1.0) * t
        else:
            self._punch_zoom = 1.0
        return self._punch_zoom

    # ── 3. Speaker Glow Ring ─────────────────────────────────────────
    def draw_speaker_glow(self, canvas, faces, current_speaker_id, frame_idx):
        if not self.cfg.speaker_glow_enabled: return
        if current_speaker_id is None: return
        face = faces.get(current_speaker_id) if faces else None
        if face is None: return
        spk_score = face.talknet_score if face.talknet_score != 0.0 else face.lip_score
        if spk_score < self.cfg.speaker_glow_min_score: return
        h, w = canvas.shape[:2]
        fx1 = int(max(0, face.cx - face.w * 0.55))
        fy1 = int(max(0, face.cy - face.h * 0.55))
        fx2 = int(min(w, face.cx + face.w * 0.55))
        fy2 = int(min(h, face.cy + face.h * 0.55))
        pulse = math.sin(frame_idx * self.cfg.speaker_glow_pulse_speed) * 2
        thickness = max(1, int(self.cfg.speaker_glow_max_thickness + pulse))
        alpha = min(1.0, spk_score / 3.0)
        color = tuple(int(c * alpha) for c in self.cfg.speaker_glow_color)
        cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), color, thickness, cv2.LINE_AA)
        if alpha > 0.3:
            inner_color = tuple(int(c * alpha * 0.4) for c in self.cfg.speaker_glow_color)
            cv2.rectangle(canvas, (fx1 - 2, fy1 - 2), (fx2 + 2, fy2 + 2), inner_color, thickness + 2, cv2.LINE_AA)

    # ── 4. Film Grain Overlay ────────────────────────────────────────
    def apply_film_grain(self, frame, frame_idx):
        if not self.cfg.film_grain_enabled: return frame
        h, w = frame.shape[:2]
        if not self._grain_cache:
            for _ in range(self.cfg.film_grain_cache_size):
                noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
                self._grain_cache.append(noise)
        grain = self._grain_cache[frame_idx % len(self._grain_cache)]
        return cv2.addWeighted(frame, 1.0, grain, self.cfg.film_grain_intensity, 0)

    # ── 5. Panel Rounded Corners + Shadow ────────────────────────────
    @staticmethod
    def apply_panel_rounded(panel, corner_radius_frac):
        ph, pw = panel.shape[:2]
        cr = int(min(pw, ph) * corner_radius_frac)
        if cr < 2: return panel
        mask = np.ones((ph, pw), dtype=np.float32)
        cv2.rectangle(mask, (0, cr), (cr, ph), 1, -1)
        cv2.rectangle(mask, (cr, 0), (pw, ph), 1, -1)
        cv2.ellipse(mask, (cr, cr), (cr, cr), 180, 0, 90, 0, -1)
        cv2.ellipse(mask, (pw - cr, cr), (cr, cr), 270, 0, 90, 0, -1)
        cv2.ellipse(mask, (cr, ph - cr), (cr, cr), 90, 0, 90, 0, -1)
        cv2.ellipse(mask, (pw - cr, ph - cr), (cr, cr), 0, 0, 90, 0, -1)
        m3 = mask[:, :, np.newaxis]
        return (panel.astype(np.float32) * m3).astype(np.uint8)

    # ── 6. Watermark / Branding Stamp ────────────────────────────────
    def _render_text_watermark(self, text, canvas_w, canvas_h):
        """Render a text handle (e.g. '@clipskari') into an RGBA numpy array."""
        font_size = max(20, int(canvas_w * self.cfg.watermark_size_ratio * 0.45))
        font = None
        for font_path in [
            os.path.join(FontManager.FONT_DIR, FontManager.FONTS["bold"]["file"]),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if os.path.exists(font_path) and os.path.getsize(font_path) > 10000:
                try: font = ImageFont.truetype(font_path, font_size); break
                except Exception: continue
        if font is None: font = ImageFont.load_default()
        tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0] + 16
        th = bbox[3] - bbox[1] + 10
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        ox, oy = 8 - bbox[0], 5 - bbox[1]
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
            draw.text((ox+dx, oy+dy), text, font=font, fill=(0, 0, 0, 140))
        draw.text((ox, oy), text, font=font, fill=(255, 255, 255, 230))
        return np.array(img)

    def apply_watermark(self, canvas, frame_idx, card_y_offset=None, card_y_end=None):
        """Apply watermark to canvas, optionally constrained to the video region.

        Args:
            canvas: BGR output frame.
            frame_idx: Current frame index (for fade-in).
            card_y_offset: Y coordinate where video region starts (below card text).
                           When card is enabled, this is int(canvas_h * card_padding_top).
                           When card is disabled, this is None.
            card_y_end: Y coordinate where video region ends (above bottom padding).
                        When card is enabled, this is canvas_h - int(canvas_h * card_padding_bottom).
                        When card is disabled, this is None.
        """
        if not self.cfg.watermark_enabled or not self.cfg.watermark_path: return canvas
        ch, cw = canvas.shape[:2]
        # Video region boundaries (when card is on, constrain to video part)
        vid_top = card_y_offset if card_y_offset is not None else 0
        vid_bottom = card_y_end if card_y_end is not None else ch
        if not self._watermark_loaded:
            self._watermark_loaded = True
            wp = self.cfg.watermark_path
            is_file = os.path.exists(wp) and os.path.isfile(wp)
            if is_file:
                try:
                    wm = cv2.imread(wp, cv2.IMREAD_UNCHANGED)
                    if wm is not None:
                        wm_w = int(cw * self.cfg.watermark_size_ratio)
                        wm_h = int(wm.shape[0] * wm_w / max(1, wm.shape[1]))
                        self._watermark_img = cv2.resize(wm, (wm_w, wm_h), interpolation=cv2.INTER_LANCZOS4)
                except Exception: self._watermark_img = None
            else:
                try:
                    self._watermark_img = self._render_text_watermark(wp, cw, ch)
                except Exception: self._watermark_img = None
            if self._watermark_img is not None:
                wm_h, wm_w = self._watermark_img.shape[:2]
                print(f"  ✓ Watermark loaded: {'image' if is_file else 'text'} ({wm_w}x{wm_h}px)")
        if self._watermark_img is None: return canvas
        wm = self._watermark_img; wh, ww = wm.shape[:2]
        pad = int(cw * 0.03)
        pos = self.cfg.watermark_position
        # Position constrained to video region (vid_top..vid_bottom)
        # Extra vertical margin when card is on — card shadow/border extends
        # below card_padding_top, so we add clearance to avoid overlap
        card_extra = int(ch * 0.008) if card_y_offset is not None else 0
        if pos == "bottom_right": x, y = cw - ww - pad, vid_bottom - wh - pad
        elif pos == "bottom_left": x, y = pad, vid_bottom - wh - pad
        elif pos == "top_right": x, y = cw - ww - pad, vid_top + pad + card_extra
        else: x, y = pad, vid_top + pad + card_extra
        alpha = self.cfg.watermark_opacity
        if frame_idx < self.cfg.watermark_fade_in_frames:
            alpha *= frame_idx / max(1, self.cfg.watermark_fade_in_frames)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(cw, x + ww), min(ch, y + wh)
        if x2 <= x1 or y2 <= y1: return canvas
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        roi = canvas[y1:y2, x1:x2].astype(np.float32)
        if wm.ndim == 3 and wm.shape[2] == 4:
            wm_a = wm[sy1:sy2, sx1:sx2, 3:4].astype(np.float32) / 255.0 * alpha
            wm_rgb = wm[sy1:sy2, sx1:sx2, :3][:, :, ::-1].astype(np.float32)
            canvas[y1:y2, x1:x2] = np.clip(roi * (1 - wm_a) + wm_rgb * wm_a, 0, 255).astype(np.uint8)
        elif wm.ndim == 3:
            wm_crop = wm[sy1:sy2, sx1:sx2][:, :, ::-1].astype(np.float32)
            canvas[y1:y2, x1:x2] = np.clip(roi * (1 - alpha) + wm_crop * alpha, 0, 255).astype(np.uint8)
        return canvas

    # ── 7. Face Beautification ───────────────────────────────────────
    @staticmethod
    def beautify_face(frame, faces, cfg):
        if not cfg.face_beautify_enabled: return frame
        result = frame.copy()
        for tid, face in faces.items():
            x1, y1 = int(max(0, face.bbox[0])), int(max(0, face.bbox[1]))
            x2, y2 = int(min(frame.shape[1], face.bbox[2])), int(min(frame.shape[0], face.bbox[3]))
            if x2 <= x1 or y2 <= y1: continue
            roi = result[y1:y2, x1:x2]
            smooth = cv2.bilateralFilter(roi, cfg.face_beautify_d, cfg.face_beautify_sigma, cfg.face_beautify_sigma)
            result[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - cfg.face_beautify_strength, smooth, cfg.face_beautify_strength, 0)
        return result

    # ── 8. Border Glow ───────────────────────────────────────────────
    @staticmethod
    def apply_border_glow(canvas, rms, cfg):
        if not cfg.border_glow_enabled: return canvas
        h, w = canvas.shape[:2]; bw = cfg.border_glow_width
        alpha = min(cfg.border_glow_max_opacity, rms / 0.12 * cfg.border_glow_max_opacity)
        if alpha < 0.02: return canvas
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), cfg.border_glow_color, bw, cv2.LINE_AA)
        return cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)


    # ── v8: Bold Waveform Border ──────────────────────────────────────
    @staticmethod
    def apply_waveform_border(canvas, rms, frame_idx, cfg):
        """v8: Bold glowing border with audio-reactive waveform animation.

        Triple-line bold effect (outermost glow + mid glow + inner bright core),
        wider default border, and higher minimum visibility.
        """
        if not getattr(cfg, 'border_waveform_enabled', True):
            # Fall back to simple border glow if waveform disabled
            return VisualFX.apply_border_glow(canvas, rms, cfg)
        h, w = canvas.shape[:2]
        color = getattr(cfg, 'border_wave_color', cfg.border_glow_color if hasattr(cfg, 'border_glow_color') else (255, 40, 40))
        base_w = getattr(cfg, 'border_wave_width', 18)
        max_op = getattr(cfg, 'border_wave_max_opacity', 0.85)
        wave_speed = getattr(cfg, 'border_wave_speed', 0.12)
        wave_freq = getattr(cfg, 'border_wave_frequency', 4.0)

        # Audio-reactive intensity — v8: minimum raised to 0.55 for always-visible border
        intensity = min(max_op, max(0.55, rms / 0.10 * max_op))

        # ── Triple-line bold effect ──
        # Outermost glow line (widest, dimmest — halo)
        outer_w = base_w + 8
        overlay_outer = canvas.copy()
        cv2.rectangle(overlay_outer, (0, 0), (w - 1, h - 1), color, outer_w, cv2.LINE_AA)

        # Mid glow line (medium width, medium brightness)
        mid_w = base_w + 3
        overlay_mid = canvas.copy()
        cv2.rectangle(overlay_mid, (0, 0), (w - 1, h - 1), color, mid_w, cv2.LINE_AA)

        # Inner bright core line (narrowest, brightest)
        overlay_inner = canvas.copy()
        cv2.rectangle(overlay_inner, (0, 0), (w - 1, h - 1), color, base_w, cv2.LINE_AA)

        # Create brightness modulation mask (sine wave along perimeter)
        mask = np.zeros((h, w), dtype=np.float32)
        phase = frame_idx * wave_speed

        # Top + Bottom edges: sine varies with x
        x_arr = np.arange(w, dtype=np.float32)
        top_wave = 0.5 + 0.5 * np.sin(2 * np.pi * wave_freq * x_arr / w + phase)
        bot_wave = 0.5 + 0.5 * np.sin(2 * np.pi * wave_freq * x_arr / w + phase + np.pi * 0.5)
        band = max(1, outer_w + 3)
        mask[:band, :] = np.maximum(mask[:band, :], top_wave[np.newaxis, :])
        mask[h - band:, :] = np.maximum(mask[h - band:, :], bot_wave[np.newaxis, :])

        # Left + Right edges: sine varies with y
        y_arr = np.arange(h, dtype=np.float32)
        left_wave = 0.5 + 0.5 * np.sin(2 * np.pi * wave_freq * y_arr / h + phase + np.pi)
        right_wave = 0.5 + 0.5 * np.sin(2 * np.pi * wave_freq * y_arr / h + phase + np.pi * 1.5)
        mask[:, :band] = np.maximum(mask[:, :band], left_wave[:, np.newaxis])
        mask[:, w - band:] = np.maximum(mask[:, w - band:], right_wave[:, np.newaxis])

        # Scale mask by intensity
        mask = mask * intensity

        # Blend: outermost glow (softest), then mid glow, then inner bright core
        m3 = mask[:, :, np.newaxis]
        result = canvas.astype(np.float32)
        # Outermost halo — softest blend
        result = result * (1 - m3 * 0.4) + overlay_outer.astype(np.float32) * m3 * 0.4
        # Mid glow — moderate blend
        result = result * (1 - m3 * 0.35) + overlay_mid.astype(np.float32) * m3 * 0.35
        # Inner bright core — strongest blend
        result = result * (1 - m3 * 0.3) + overlay_inner.astype(np.float32) * m3 * 0.3
        return np.clip(result, 0, 255).astype(np.uint8)

    # ── 9. Cinematic Letterbox ───────────────────────────────────────
    @staticmethod
    def apply_letterbox(canvas, cfg):
        if not cfg.letterbox_enabled: return canvas
        h, w = canvas.shape[:2]; bar_h = int(h * cfg.letterbox_bar_ratio)
        if bar_h < 2: return canvas
        canvas[:bar_h, :] = 0; canvas[h - bar_h:, :] = 0
        return canvas

    # ── 10. Animated Text Reveal ─────────────────────────────────────
    def get_reveal_alpha(self, line_idx, frame_idx):
        if not self.cfg.card_animated_reveal: return 1.0
        start = line_idx * self.cfg.card_reveal_frames_per_line
        if frame_idx < start: return 0.0
        elapsed = frame_idx - start
        if elapsed >= self.cfg.card_reveal_frames_per_line: return 1.0
        return elapsed / self.cfg.card_reveal_frames_per_line

    def get_reveal_offset(self, line_idx, frame_idx):
        if not self.cfg.card_animated_reveal: return 0
        start = line_idx * self.cfg.card_reveal_frames_per_line
        if frame_idx < start: return self.cfg.card_reveal_slide_px
        elapsed = frame_idx - start
        if elapsed >= self.cfg.card_reveal_frames_per_line: return 0
        return int(self.cfg.card_reveal_slide_px * (1 - elapsed / self.cfg.card_reveal_frames_per_line))

    @property
    def reveal_frame(self):
        return self._reveal_frame

    def advance_reveal(self):  # v8: only advance if not done
        if not self._reveal_done:
            self._reveal_frame += 1
    def reset_reveal(self): self._reveal_frame = 0; self._reveal_done = False  # v8: reset done flag for new clip

    # ── 11. Dynamic Color Grading ────────────────────────────────────
    def get_dynamic_preset(self, speaking, rms):
        if not self.cfg.dynamic_color_grading: return None
        if speaking:
            return self.cfg.dynamic_cg_speech_high_preset if rms > self.cfg.dynamic_cg_rms_threshold else self.cfg.dynamic_cg_speech_low_preset
        return self.cfg.dynamic_cg_silence_preset

    # ── 12. Depth-of-Field ───────────────────────────────────────────
    # (Integrated into _blur_fill via cfg.dof_* params)

    # ── 13. Ken Burns Effect ─────────────────────────────────────────
    def update_ken_burns(self, current_speaker_id):
        if not self.cfg.ken_burns_enabled: return 0.0, 0.0, 1.0
        if current_speaker_id != self._kb_last_spk:
            self._kb_last_spk = current_speaker_id
            self._kb_stable_cnt = 0; self._kb_offset_x = 0.0; self._kb_offset_y = 0.0; self._kb_zoom = 1.0
            return 0.0, 0.0, 1.0
        self._kb_stable_cnt += 1
        if self._kb_stable_cnt < self.cfg.ken_burns_min_stable_frames:
            return self._kb_offset_x, self._kb_offset_y, self._kb_zoom
        speed = self.cfg.ken_burns_drift_speed
        self._kb_offset_x += speed * 0.7; self._kb_offset_y -= speed * 0.3
        if self._kb_zoom < self.cfg.ken_burns_max_zoom: self._kb_zoom += speed * 0.5
        return self._kb_offset_x, self._kb_offset_y, self._kb_zoom


# ============================================================
# DOMINANT-FACE FILTER
# ============================================================
class DominantFaceFilter:
    def __init__(self, cfg): self.cfg = cfg
    def filter(self, faces, frame_h, frame_w):
        if not faces: return faces
        min_h = frame_h * self.cfg.min_face_size_ratio
        sized = {tid: f for tid,f in faces.items() if f.h >= min_h}
        if not sized: return faces
        sorted_by_size = sorted(sized.values(), key=lambda f: f.h, reverse=True)
        dominant_h = sorted_by_size[0].h
        threshold_h = dominant_h * self.cfg.size_dominance_ratio
        dominant_filtered = {tid: f for tid,f in sized.items() if f.h >= threshold_h}
        if not dominant_filtered: dominant_filtered = {sorted_by_size[0].track_id: sorted_by_size[0]}
        edge = self.cfg.face_edge_reject_ratio
        x_lo, x_hi = frame_w*edge, frame_w*(1-edge)
        y_lo, y_hi = frame_h*edge, frame_h*(1-edge)
        edge_filtered = {tid: f for tid,f in dominant_filtered.items()
                         if x_lo<=f.cx<=x_hi and y_lo<=f.cy<=y_hi}
        if not edge_filtered: edge_filtered = {sorted_by_size[0].track_id: sorted_by_size[0]}
        if len(edge_filtered) > self.cfg.max_live_faces:
            top = sorted(edge_filtered.values(), key=lambda f: f.h, reverse=True)
            edge_filtered = {f.track_id: f for f in top[:self.cfg.max_live_faces]}
        return edge_filtered


# ============================================================
# AUDIO VAD
# ============================================================
def get_audio_mask(video_path, fps, total_frames):
    temp = "/kaggle/working/temp_audio_vad.wav"
    mask = np.ones(total_frames, dtype=bool)
    try:
        subprocess.run(["ffmpeg","-y","-i",video_path,"-vn","-acodec","pcm_s16le",
                        "-ar","16000","-ac","1",temp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        waveform, sr = torchaudio.load(temp)
        if waveform.shape[0]>1: waveform = waveform.mean(0,keepdim=True)
        model,utils = torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True)
        (get_ts,*_) = utils; mask[:] = False
        for ts in get_ts(waveform,model,sampling_rate=sr):
            s = max(0,int(ts['start']/sr*fps)); e = min(total_frames,int(ts['end']/sr*fps))
            mask[s:e] = True
    except Exception as e: print(f"  Audio VAD failed ({e}), assuming always speaking.")
    finally:
        if os.path.exists(temp): os.remove(temp)
    return mask


# ============================================================
# AUDIO ENERGY TRACKER
# ============================================================
class AudioEnergyTracker:
    def __init__(self, cfg): self.cfg = cfg
    def compute(self, video_path, fps, total_frames):
        temp = "/kaggle/working/temp_rms_audio.wav"
        rms = np.zeros(total_frames, dtype=np.float32)
        try:
            subprocess.run(["ffmpeg","-y","-i",video_path,"-vn","-acodec","pcm_s16le",
                            "-ar","16000","-ac","1",temp],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            waveform, sr = torchaudio.load(temp)
            waveform = waveform.squeeze().numpy().astype(np.float32)
            spf = int(sr/fps)
            for i in range(total_frames):
                s = i*spf; chunk = waveform[s:min(s+spf,len(waveform))]
                if s>=len(waveform): break
                rms[i] = float(np.sqrt(np.mean(chunk**2)))
            k = np.ones(self.cfg.zoom_rms_window)/self.cfg.zoom_rms_window
            rms = np.convolve(rms,k,mode='same')
        except Exception as e: print(f"  Audio energy failed ({e})")
        finally:
            if os.path.exists(temp): os.remove(temp)
        return rms


# ============================================================
# FACE RE-ID
# ============================================================
class FaceReID:
    def __init__(self, cfg): self.cfg=cfg; self.gallery={}; self.frame_idx=0
    def _embed(self, crop):
        if crop is None or crop.size==0: return None
        try:
            img=cv2.resize(crop,(64,64)); img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
            feat=hog.compute(img).flatten().astype(np.float32)
            n=np.linalg.norm(feat); return feat/(n+1e-6)
        except Exception: return None
    def _cos(self,a,b): return float(np.dot(a,b))
    def _mean(self,gid):
        e=list(self.gallery[gid]["embeds"])
        if not e: return None
        m=np.mean(e,axis=0); return m/(np.linalg.norm(m)+1e-6)
    def update_gallery(self,tid,crop):
        e=self._embed(crop)
        if e is None: return
        if tid not in self.gallery:
            self.gallery[tid]={"embeds":deque(maxlen=self.cfg.reid_gallery_size),"last_seen":self.frame_idx}
        self.gallery[tid]["embeds"].append(e); self.gallery[tid]["last_seen"]=self.frame_idx
    def try_reidentify(self,new_tid,crop,active_ids):
        e=self._embed(crop)
        if e is None: return None
        best_id=None; best=self.cfg.reid_match_threshold
        for gid,data in self.gallery.items():
            if gid in active_ids: continue
            if self.frame_idx-data["last_seen"]>self.cfg.reid_max_lost_frames: continue
            ref=self._mean(gid)
            if ref is None: continue
            s=self._cos(e,ref)
            if s>best: best=s; best_id=gid
        return best_id
    def retire(self,tid):
        if tid in self.gallery: self.gallery[tid]["last_seen"]=self.frame_idx
    def purge_stale(self):
        stale=[g for g,d in self.gallery.items() if self.frame_idx-d["last_seen"]>self.cfg.reid_max_lost_frames]
        for g in stale: del self.gallery[g]
    def tick(self): self.frame_idx+=1


# ============================================================
# SCENE CUT DETECTOR
# ============================================================
class SceneCutDetector:
    def __init__(self, cfg): self.cfg=cfg; self.prev=None; self.cooldown=0
    def is_cut(self, frame):
        if not self.cfg.scene_cut_enabled: return False
        t=cv2.resize(frame,(160,90),interpolation=cv2.INTER_LINEAR)
        tg=cv2.cvtColor(t,cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut=False
        if self.prev is not None and self.cooldown==0:
            if np.mean(np.abs(tg-self.prev))>self.cfg.scene_cut_threshold:
                cut=True; self.cooldown=self.cfg.scene_cut_cooldown
        self.prev=tg
        if self.cooldown>0: self.cooldown-=1
        return cut


# ============================================================
# PANEL CROP VALIDATOR
# ============================================================
def _panel_crop_covers_face(crop, face, fh, cfg):
    fx1,fy1=face.cx-face.w/2,face.cy-face.h/2; fx2,fy2=face.cx+face.w/2,face.cy+face.h/2
    cx1,cy1,cx2,cy2=crop[0],crop[1],crop[0]+crop[2],crop[1]+crop[3]
    ix=max(0,min(fx2,cx2)-max(fx1,cx1)); iy=max(0,min(fy2,cy2)-max(fy1,cy1))
    iof=ix*iy/max(1.0,face.w*face.h)
    if iof<cfg.panel_min_face_coverage: return False
    cy_crop=crop[1]+crop[3]/2
    if abs(cy_crop-face.cy)/max(1.0,face.h)>cfg.panel_max_vert_drift: return False
    return True


# ============================================================
# TALKNET ACTIVE SPEAKER DETECTOR
# ============================================================
class TalkNetSpeakerDetector:
    MAX_BUFFER = 75

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self._ready = False
        self.face_buffers: Dict[int, deque] = {}
        self.score_history: Dict[int, deque] = {}
        self.mfcc: Optional[np.ndarray] = None
        self._load_model()

    def _load_model(self):
        if not self.cfg.talknet_enabled:
            print("  TalkNet ASD: DISABLED by config"); return
        model_path = os.path.join(TALKNET_REPO, "pretrain_TalkSet.model")
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 5_000_000:
            print("  ⚠ TalkNet weights not found or too small — falling back to lip-score"); return
        try:
            from talkNet import talkNet as TalkNetModel
            self.model = TalkNetModel()
            self.model.loadParameters(model_path)
            self.model.eval()
            self._ready = True
            print("  ✓ TalkNet-ASD loaded — Active Speaker Detection ENABLED")
        except Exception as e:
            print(f"  ⚠ TalkNet load failed ({e}) — falling back to lip-score")
            self.model = None; self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self.model is not None

    def precompute_audio(self, video_path: str, fps: float, total_frames: int):
        if not self.is_ready: return
        temp = "/kaggle/working/temp_talknet_audio.wav"
        try:
            subprocess.run(["ffmpeg","-y","-i",video_path,"-vn","-acodec","pcm_s16le","-ar","16000","-ac","1",temp],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            from scipy.io import wavfile
            import python_speech_features
            sr, audio = wavfile.read(temp)
            winlen = 0.025 * 25 / fps; winstep = 0.010 * 25 / fps
            self.mfcc = python_speech_features.mfcc(audio, sr, numcep=13, winlen=winlen, winstep=winstep)
            print(f"  ✓ TalkNet MFCC computed: {self.mfcc.shape[0]} frames")
        except Exception as e:
            print(f"  ⚠ TalkNet MFCC failed ({e})"); self.mfcc = None
        finally:
            if os.path.exists(temp): os.remove(temp)

    def buffer_face(self, track_id: int, frame: np.ndarray, bbox: tuple):
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2 = min(x2, frame.shape[1]); y2 = min(y2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (112, 112))
        face = ((face / 255.0 - 0.4161) / 0.1688).astype(np.float32)
        if track_id not in self.face_buffers:
            self.face_buffers[track_id] = deque(maxlen=self.MAX_BUFFER)
        self.face_buffers[track_id].append(face)

    def purge_lost(self, active_ids: set):
        lost = [tid for tid in self.face_buffers if tid not in active_ids]
        for tid in lost:
            del self.face_buffers[tid]; self.score_history.pop(tid, None)

    def score_faces(self, frame_idx: int, fps: float) -> Dict[int, float]:
        if not self.is_ready or self.mfcc is None: return {}
        scores: Dict[int, float] = {}
        mfcc_fps = 100.0; mfcc_per_vid_frame = mfcc_fps / fps
        with torch.no_grad():
            for tid, buffer in self.face_buffers.items():
                T_buf = len(buffer)
                if T_buf < self.cfg.talknet_min_frames: continue
                duration_scores = []
                for duration in self.cfg.talknet_durations:
                    vid_frames = int(duration * fps)
                    n_batches = max(1, math.ceil(T_buf / vid_frames))
                    batch_scores = []
                    for bi in range(n_batches):
                        v_start = max(0, T_buf - (n_batches - bi) * vid_frames)
                        v_end = min(T_buf, v_start + vid_frames)
                        if v_end - v_start < self.cfg.talknet_min_frames: continue
                        seg_frames = list(buffer)[v_start:v_end]
                        inputV = torch.FloatTensor(np.stack(seg_frames)).unsqueeze(0).cuda()
                        buf_start_frame = frame_idx - T_buf + 1
                        abs_v_start = buf_start_frame + v_start; abs_v_end = buf_start_frame + v_end
                        a_start = max(0, int(abs_v_start * mfcc_per_vid_frame))
                        a_end = min(len(self.mfcc), int(abs_v_end * mfcc_per_vid_frame))
                        audio_seg = self.mfcc[a_start:a_end]
                        if len(audio_seg) < 4: continue
                        audio_seg = audio_seg[:len(audio_seg) - len(audio_seg) % 4]
                        inputA = torch.FloatTensor(audio_seg).unsqueeze(0).cuda()
                        try:
                            embedA = self.model.model.forward_audio_frontend(inputA)
                            embedV = self.model.model.forward_visual_frontend(inputV)
                            embedA, embedV = self.model.model.forward_cross_attention(embedA, embedV)
                            out = self.model.model.forward_audio_visual_backend(embedA, embedV)
                            seg_score = self.model.lossAV.forward(out, labels=None)
                            batch_scores.extend(seg_score.tolist())
                        except Exception: continue
                    if batch_scores: duration_scores.append(batch_scores)
                if not duration_scores: continue
                max_len = max(len(s) for s in duration_scores)
                padded = [s + [s[-1]] * (max_len - len(s)) for s in duration_scores]
                avg_scores = np.mean(padded, axis=0)
                sw = min(self.cfg.talknet_smooth_window, len(avg_scores))
                smoothed = float(np.mean(avg_scores[-sw:]))
                scores[tid] = smoothed
                if tid not in self.score_history:
                    self.score_history[tid] = deque(maxlen=self.cfg.talknet_smooth_window * 3)
                self.score_history[tid].append(smoothed)
        return scores

    def get_smoothed_score(self, track_id: int) -> float:
        hist = self.score_history.get(track_id)
        if not hist: return 0.0
        sw = min(self.cfg.talknet_smooth_window, len(hist))
        return float(np.mean(list(hist)[-sw:]))


# ============================================================
# SPLIT SCREEN RENDERER
# ============================================================
class SplitScreenRenderer:
    def __init__(self,cfg):
        self.cfg=cfg; self.state="SINGLE"; self.progress=0.0
        self.both_cnt=0; self.single_cnt=0; self.confidence_cnt=0
        self.panel_ids=[None,None]; self.panel_crops=[None,None]
        self.panel_targets=[None,None]; self.panel_kf=[None,None]
        self.panel_freeze_cnt=[0,0]; self.panel_frozen_crop=[None,None]
        self.panel_reassign_cnt=[0,0]; self.panel_reassign_candidate=[None,None]
        self._cold_start_cnt=0
    def _make_kf(self,cx,cy):
        kf=KalmanFilter(dim_x=4,dim_z=2); kf.x=np.array([cx,cy,0.,0.])
        kf.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        kf.H=np.eye(2,4); kf.P=np.eye(4)*10
        kf.Q[0:2,0:2]*=self.cfg.kalman_noise; kf.Q[2:4,2:4]*=self.cfg.kalman_noise*0.1; kf.R*=0.01; return kf
    def _smoothstep(self,t): t=np.clip(t,0,1); return t*t*(3-2*t)
    def _crop_face(self,face,fw,fh,pw,ph,other=None):
        aspect=ph/pw; pad=self.cfg.split_padding_ratio
        cw=face.w*pad; ch=cw*aspect
        if ch>fh: ch=fh; cw=ch/aspect
        if cw>fw: cw=fw; ch=cw*aspect
        # Step 1: Center crop on face
        cx=face.cx-cw/2; cy=face.cy-ch*self.cfg.split_face_vertical_pos+face.h*0.1
        # Step 2: Push panels apart so they don't show the same area
        bias=self.cfg.split_overlap_bias
        if other and bias>0:
            if face.cx<other.cx: cx-=(other.cx-face.cx)*bias
            else: cx+=(face.cx-other.cx)*bias
        # Step 3: Shrink if panels still overlap
        if other and self.cfg.split_overlap_check:
            for _ in range(12):
                if not (cx<other.cx<cx+cw): break
                cw*=0.92; ch=cw*aspect; cx=face.cx-cw/2
                if bias>0:
                    if face.cx<other.cx: cx-=(other.cx-face.cx)*bias
                    else: cx+=(face.cx-other.cx)*bias
        # Step 4: Clamp to frame bounds
        cx=max(0,min(cx,fw-cw)); cy=max(0,min(cy,fh-ch))
        # ── FIX: Safe-zone centering ──────────────────────────────────
        # After bias + overlap adjustments + frame clamping, the face may
        # have drifted to the panel edge (especially in close-distance
        # face-to-face podcasts).  Ensure the face center stays within
        # the "safe zone" (30 %–70 % of panel width, 25 %–75 % height).
        # If it drifted out, shift the crop to re-center the face.
        SAFE_X_MIN, SAFE_X_MAX = 0.30, 0.70
        SAFE_Y_MIN, SAFE_Y_MAX = 0.25, 0.75
        face_rel_x = (face.cx - cx) / cw  # 0=left edge, 1=right edge
        face_rel_y = (face.cy - cy) / ch
        if face_rel_x < SAFE_X_MIN:
            cx -= (SAFE_X_MIN - face_rel_x) * cw
        elif face_rel_x > SAFE_X_MAX:
            cx += (face_rel_x - SAFE_X_MAX) * cw
        if face_rel_y < SAFE_Y_MIN:
            cy -= (SAFE_Y_MIN - face_rel_y) * ch
        elif face_rel_y > SAFE_Y_MAX:
            cy += (face_rel_y - SAFE_Y_MAX) * ch
        # Re-clamp after centering adjustment
        cx=max(0,min(cx,fw-cw)); cy=max(0,min(cy,fh-ch))
        crop=[cx,cy,cw,ch]
        if not _panel_crop_covers_face(crop,face,fh,self.cfg):
            fb_cw=min(face.w*pad,fw); fb_ch=min(fb_cw*aspect,fh); fb_cw=fb_ch/aspect
            crop=[max(0,min(face.cx-fb_cw/2,fw-fb_cw)),max(0,min(face.cy-fb_ch*self.cfg.split_face_vertical_pos,fh-fb_ch)),fb_cw,fb_ch]
        return crop
    def _update_panel(self,idx,face,fw,fh,pw,ph,other=None):
        if self.panel_kf[idx] is None: self.panel_kf[idx]=self._make_kf(face.cx,face.cy)
        self.panel_kf[idx].predict(); self.panel_kf[idx].update([face.cx,face.cy])
        kx,ky=self.panel_kf[idx].x[:2]
        sf=type('F',(),{'cx':kx,'cy':ky,'w':face.w,'h':face.h})()
        tgt=self._crop_face(sf,fw,fh,pw,ph,other)
        if self.panel_crops[idx] is None: self.panel_crops[idx]=list(tgt); return
        lr=self.cfg.tracking_lerp; dz=self.cfg.dead_zone_px
        if self.panel_targets[idx] is None: self.panel_targets[idx]=list(tgt)
        if abs(tgt[0]-self.panel_targets[idx][0])>dz or abs(tgt[1]-self.panel_targets[idx][1])>dz:
            self.panel_targets[idx]=list(tgt)
        c=self.panel_crops[idx]; t=self.panel_targets[idx]
        cand=[c[0]+(t[0]-c[0])*lr,c[1]+(t[1]-c[1])*lr,c[2]+(tgt[2]-c[2])*lr,c[3]+(tgt[3]-c[3])*lr]
        if _panel_crop_covers_face(cand,face,fh,self.cfg): self.panel_crops[idx]=cand
        else: self.panel_crops[idx]=self._crop_face(face,fw,fh,pw,ph,other)
    def _extract_panel(self,frame,crop,ow,oh):
        x1,y1=int(max(0,crop[0])),int(max(0,crop[1]))
        x2,y2=int(min(frame.shape[1],crop[0]+crop[2])),int(min(frame.shape[0],crop[1]+crop[3]))
        p=frame[y1:y2,x1:x2]
        if p.size==0: p=frame
        return cv2.resize(p,(ow,oh),interpolation=cv2.INTER_LANCZOS4)
    def should_split(self,faces,audio_active,camera):
        if not self.cfg.split_enabled or len(faces)<2: return False
        # FIX: Old logic required BOTH faces to have speaking scores > 0.001.
        # In normal conversation only one person speaks at a time — the listener
        # gets a NEGATIVE TalkNet score, so the AND condition never triggered.
        # New logic: split when 2+ stable faces are present and audio is active.
        if not audio_active: return False
        # Require faces tracked for at least a few frames to avoid flicker
        stable = sum(1 for f in faces.values() if f.frames_alive >= 5)
        return stable >= 2
    def notify_scene_cut(self):
        self._cold_start_cnt=self.cfg.split_cold_start_frames; self.confidence_cnt=0; self.both_cnt=0
    def _reset(self):
        self.panel_ids=[None,None]; self.panel_crops=[None,None]; self.panel_targets=[None,None]; self.panel_kf=[None,None]
        self.panel_freeze_cnt=[0,0]; self.panel_frozen_crop=[None,None]; self.panel_reassign_cnt=[0,0]; self.panel_reassign_candidate=[None,None]
        self.confidence_cnt=0; self.both_cnt=0; self._cold_start_cnt=self.cfg.split_cold_start_frames
    def update(self,frame,faces,single_crop,audio_active,camera):
        tw,th=self.cfg.target_width,self.cfg.target_height; fw,fh=frame.shape[1],frame.shape[0]; gap=self.cfg.split_gap_px
        want=self.should_split(faces,audio_active,camera)
        in_blackout=self._cold_start_cnt>0
        if in_blackout: self._cold_start_cnt-=1
        if want:
            self.both_cnt+=1; self.single_cnt=0
            if not in_blackout and len(faces)>=2: self.confidence_cnt+=1
            else: self.confidence_cnt=max(0,self.confidence_cnt-1)
        else:
            self.single_cnt+=1; self.both_cnt=0; self.confidence_cnt=max(0,self.confidence_cnt-2)
        can_split=(not in_blackout and self.both_cnt>=self.cfg.split_both_speaking_frames and self.confidence_cnt>=self.cfg.split_face_confidence_frames)
        if self.state=="SINGLE":
            if can_split and len(faces)>=2:
                self.state="OPENING"; self.progress=0.0; self._assign_panels(faces,fw,fh)
                self.panel_freeze_cnt=[0,0]; self.panel_frozen_crop=[None,None]
        elif self.state=="OPENING":
            self.progress+=1.0/self.cfg.split_transition_frames
            if self.progress>=1.0: self.progress=1.0; self.state="SPLIT"
        elif self.state=="SPLIT":
            if self.single_cnt>=self.cfg.split_exit_frames: self.state="CLOSING"; self.progress=1.0
            if self.cfg.split_panel_freeze_enabled:
                for idx in range(2):
                    if self.panel_freeze_cnt[idx]>self.cfg.split_panel_freeze_max_frames: self.state="CLOSING"; self.progress=1.0; break
        elif self.state=="CLOSING":
            self.progress-=1.0/self.cfg.split_transition_frames
            if self.progress<=0.0: self.progress=0.0; self.state="SINGLE"; self._reset()
        t=self._smoothstep(self.progress)
        if self.state=="SINGLE" or t==0.0:
            x1,y1=int(max(0,single_crop[0])),int(max(0,single_crop[1]))
            x2,y2=int(min(fw,single_crop[0]+single_crop[2])),int(min(fh,single_crop[1]+single_crop[3]))
            p=frame[y1:y2,x1:x2]
            if p.size==0: p=frame
            return cv2.resize(p,(tw,th),interpolation=cv2.INTER_LANCZOS4)
        self._refresh_panels(faces,fw,fh)
        return self._composite(frame,t,single_crop,fw,fh,tw,th,gap)
    def _assign_panels(self,faces,fw,fh):
        def _spk_score(f): return f.talknet_score if f.talknet_score != 0.0 else f.lip_score
        ranked=sorted(faces.values(),key=lambda f:_spk_score(f),reverse=True)
        self.panel_ids=[ranked[0].track_id,ranked[1].track_id if len(ranked)>1 else None]
        tw,th,gap=self.cfg.target_width,self.cfg.target_height,self.cfg.split_gap_px
        ph=(th-gap)//2 if self.cfg.split_mode=="top_bottom" else th; pw=tw if self.cfg.split_mode=="top_bottom" else (tw-gap)//2
        f0=faces.get(self.panel_ids[0]); f1=faces.get(self.panel_ids[1]) if self.panel_ids[1] is not None else None
        for idx,tid in enumerate(self.panel_ids):
            if tid is not None and tid in faces:
                other=f1 if idx==0 else f0
                self.panel_crops[idx]=self._crop_face(faces[tid],fw,fh,pw,ph,other)
                self.panel_targets[idx]=list(self.panel_crops[idx]); self.panel_kf[idx]=self._make_kf(faces[tid].cx,faces[tid].cy)
    def _refresh_panels(self,faces,fw,fh):
        tw,th,gap=self.cfg.target_width,self.cfg.target_height,self.cfg.split_gap_px
        ph=(th-gap)//2 if self.cfg.split_mode=="top_bottom" else th; pw=tw if self.cfg.split_mode=="top_bottom" else (tw-gap)//2
        def _spk_score(f): return f.talknet_score if f.talknet_score != 0.0 else f.lip_score
        for idx in range(2):
            tid=self.panel_ids[idx]
            if tid is not None and tid in faces:
                self.panel_freeze_cnt[idx]=0; self.panel_frozen_crop[idx]=None; self.panel_reassign_cnt[idx]=0; self.panel_reassign_candidate[idx]=None
                self._update_panel(idx,faces[tid],fw,fh,pw,ph,faces.get(self.panel_ids[1-idx])); continue
            self.panel_freeze_cnt[idx]+=1
            if self.cfg.split_panel_freeze_enabled and self.panel_freeze_cnt[idx]<=self.cfg.split_panel_freeze_max_frames:
                if self.panel_frozen_crop[idx] is None and self.panel_crops[idx]: self.panel_frozen_crop[idx]=list(self.panel_crops[idx])
                if self.panel_frozen_crop[idx]: self.panel_crops[idx]=list(self.panel_frozen_crop[idx])
                continue
            if faces:
                other_tid=self.panel_ids[1-idx]
                cands=[f for k,f in faces.items() if k!=other_tid and f.h>=fh*self.cfg.min_face_size_ratio]
                if cands:
                    # FIX: handle negative TalkNet scores — prefer speaker, fallback to size
                    best=max(cands,key=lambda f:_spk_score(f) if _spk_score(f)>0.0 else f.w*f.h)
                    if self.panel_reassign_candidate[idx]==best.track_id:
                        self.panel_reassign_cnt[idx]+=1
                        if self.panel_reassign_cnt[idx]>=self.cfg.split_panel_reassign_frames:
                            self.panel_ids[idx]=best.track_id; self.panel_kf[idx]=None
                            self.panel_crops[idx]=self._crop_face(best,fw,fh,pw,ph,faces.get(other_tid))
                            self.panel_targets[idx]=list(self.panel_crops[idx]); self.panel_freeze_cnt[idx]=0; self.panel_frozen_crop[idx]=None
                            self.panel_reassign_cnt[idx]=0; self.panel_reassign_candidate[idx]=None
                    else: self.panel_reassign_candidate[idx]=best.track_id; self.panel_reassign_cnt[idx]=1
    def _composite(self,frame,t,sc,fw,fh,tw,th,gap):
        if self.cfg.split_mode=="top_bottom":
            # v8: seamless split — gap=0 by default, bottom panel fills remaining pixels
            gap = self.cfg.split_gap_px
            if gap > 0:
                phs = (th - gap) // 2
            else:
                phs = th // 2  # v8: seamless, no gap
            pw=tw; canvas=np.zeros((th,tw,3),dtype=np.uint8)
            for idx in range(2):
                crop=self.panel_crops[idx]
                if crop is None:
                    # Defensive: use single_crop as fallback so panel is never black
                    if sc is not None:
                        sx1,sy1=int(max(0,sc[0])),int(max(0,sc[1]))
                        sx2,sy2=int(min(fw,sc[0]+sc[2])),int(min(fh,sc[1]+sc[3]))
                        p=frame[sy1:sy2,sx1:sx2]
                        if p.size>0: crop=[sc[0],sc[1],sc[2],sc[3]]
                    if crop is None: continue
                # v8: bottom panel uses th-phs to fill remaining pixels when gap=0
                panel_h = phs if idx == 0 else (th - phs if gap <= 0 else phs)
                pi=self._extract_panel(frame,crop,pw,panel_h)
                if gap > 0:
                    off=int((1-t)*phs); yd=-off if idx==0 else phs+gap+off
                else:
                    # v8: Seamless — no gap offset
                    off=int((1-t)*phs); yd=-off if idx==0 else phs+off
                sy1=max(0,-yd); sy2=min(panel_h,th-yd); dy1=max(0,yd); dy2=dy1+(sy2-sy1)
                if sy2>sy1 and dy2<=th: canvas[dy1:dy2,0:pw]=pi[sy1:sy2,:]
            if gap > 0:  # v8: only draw gap when gap > 0
                VisualFX.draw_split_gap(canvas, self.cfg, is_top_bottom=True)
        else:
            pws=(tw-gap)//2; ph=th; canvas=np.zeros((th,tw,3),dtype=np.uint8)
            for idx in range(2):
                crop=self.panel_crops[idx]
                if crop is None:
                    if sc is not None:
                        sx1,sy1=int(max(0,sc[0])),int(max(0,sc[1]))
                        sx2,sy2=int(min(fw,sc[0]+sc[2])),int(min(fh,sc[1]+sc[3]))
                        p=frame[sy1:sy2,sx1:sx2]
                        if p.size>0: crop=[sc[0],sc[1],sc[2],sc[3]]
                    if crop is None: continue
                pi=self._extract_panel(frame,crop,pws,ph)
                off=int((1-t)*pws); xd=-off if idx==0 else pws+gap+off
                sx1=max(0,-xd); sx2=min(pws,tw-xd); dx1=max(0,xd); dx2=dx1+(sx2-sx1)
                if sx2>sx1 and dx2<=tw: canvas[0:ph,dx1:dx2]=pi[:,sx1:sx2]
            VisualFX.draw_split_gap(canvas, self.cfg, is_top_bottom=False)
        if t<1.0:
            x1,y1=int(max(0,sc[0])),int(max(0,sc[1]))
            x2,y2=int(min(fw,sc[0]+sc[2])),int(min(fh,sc[1]+sc[3]))
            p=frame[y1:y2,x1:x2]
            if p.size==0: p=frame
            sf=cv2.resize(p,(tw,th),interpolation=cv2.INTER_LANCZOS4)
            canvas=cv2.addWeighted(sf,1-t,canvas,t,0)
        return canvas


# ============================================================
# TRACKED FACE
# ============================================================
@dataclass
class TrackedFace:
    track_id: int
    bbox: Tuple[float,float,float,float]
    cx: float; cy: float; w: float; h: float
    landmarks: Optional[np.ndarray] = None
    mouth_ratio_history: deque = field(default_factory=lambda: deque(maxlen=12))
    lip_score: float = 0.0
    talknet_score: float = 0.0
    frames_lost: int = 0
    frames_alive: int = 0
    kalman_filter: Optional[KalmanFilter] = None


# ============================================================
# IOU TRACKER
# ============================================================
class FaceTracker:
    def __init__(self): self.tracks={}; self.next_id=1  # start at 1 — tid=0 is truthy-bug prone
    def _iou(self,b1,b2):
        x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1]); x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
        inter=max(0,x2-x1)*max(0,y2-y1); union=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
        return inter/(union+1e-6)
    def _make_kf(self,cx,cy):
        kf=KalmanFilter(dim_x=4,dim_z=2); kf.x=np.array([cx,cy,0.,0.])
        kf.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        kf.H=np.eye(2,4); kf.P*=100; kf.Q[0:2,0:2]*=0.01; kf.R*=1; return kf
    def update(self,bboxes,lms,shape,reid=None,frame=None):
        if not self.tracks:
            for bb,lm in zip(bboxes,lms):
                x1,y1,x2,y2=bb; tid=self.next_id; self.next_id+=1
                self.tracks[tid]=TrackedFace(tid,bb,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,lm,kalman_filter=self._make_kf((x1+x2)/2,(y1+y2)/2))
            return self.tracks
        for t in self.tracks.values():
            if t.kalman_filter: t.kalman_filter.predict(); t.cx,t.cy=t.kalman_filter.x[:2]; t.bbox=(t.cx-t.w/2,t.cy-t.h/2,t.cx+t.w/2,t.cy+t.h/2)
        tids=list(self.tracks.keys()); iou_m=np.zeros((len(tids),len(bboxes)))
        for i,tid in enumerate(tids):
            for j,bb in enumerate(bboxes): iou_m[i,j]=self._iou(self.tracks[tid].bbox,bb)
        mt,mb=set(),set()
        for idx in np.argsort(iou_m.ravel())[::-1]:
            i,j=divmod(idx,len(bboxes))
            if i in mt or j in mb or iou_m[i,j]<0.2: continue
            tid=tids[i]; x1,y1,x2,y2=bboxes[j]
            self.tracks[tid].bbox=bboxes[j]; self.tracks[tid].cx=(x1+x2)/2; self.tracks[tid].cy=(y1+y2)/2
            self.tracks[tid].w=x2-x1; self.tracks[tid].h=y2-y1; self.tracks[tid].landmarks=lms[j]; self.tracks[tid].frames_lost=0
            if self.tracks[tid].kalman_filter: self.tracks[tid].kalman_filter.update([(x1+x2)/2,(y1+y2)/2])
            mt.add(i); mb.add(j)
        for j,(bb,lm) in enumerate(zip(bboxes,lms)):
            if j not in mb:
                x1,y1,x2,y2=bb; tid=self.next_id; self.next_id+=1
                if reid and frame is not None:
                    fx1,fy1=int(max(0,x1)),int(max(0,y1)); fx2,fy2=int(min(frame.shape[1],x2)),int(min(frame.shape[0],y2))
                    crop=frame[fy1:fy2,fx1:fx2]; old=reid.try_reidentify(tid,crop,set(self.tracks.keys()))
                    if old: tid=old
                self.tracks[tid]=TrackedFace(tid,bb,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,lm,kalman_filter=self._make_kf((x1+x2)/2,(y1+y2)/2))
        for tid in [t for t in self.tracks if self.tracks[t].frames_lost>20]:
            if reid: reid.retire(tid)
            del self.tracks[tid]
        for tid in self.tracks: self.tracks[tid].frames_lost+=1; self.tracks[tid].frames_alive+=1
        return self.tracks


# ============================================================
# BULLETPROOF CAMERA
# ============================================================
class BulletproofCamera:
    def __init__(self,cfg):
        self.cfg=cfg; self.aspect=cfg.target_height/cfg.target_width
        self.current_x=0.0; self.current_y=0.0; self.locked_w=0.0; self.locked_h=0.0
        self.target_x=0.0; self.target_y=0.0
        self.kf=KalmanFilter(dim_x=4,dim_z=2); self.kf.x=np.zeros(4)
        self.kf.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.kf.H=np.eye(2,4); self.kf.P*=1000
        self.kf.Q[0:2,0:2]*=cfg.kalman_noise; self.kf.Q[2:4,2:4]*=cfg.kalman_noise*0.1; self.kf.R*=0.01
        self.state="IDLE"; self.current_speaker_id=None; self.pending_id=None
        self.pending_cnt=0; self.pending_kf=None; self.initialized=False; self.face_ever_seen=False
        self.current_zoom=1.0; self.target_zoom=1.0; self.punch_zoom_factor=1.0
        self.sticky_cnt=0; self.current_crop_w=0.0; self.current_crop_h=0.0; self._dom_filter=None
        # v8: Smooth transition crop (Change 3 — camera transition glitch fix)
        self._transition_crop = None        # crop we're transitioning FROM
        self._transition_frames_left = 0    # countdown
        self._transition_total_frames = 8   # number of frames for transition (increased from 5)
        self._transition_target_crop = None # target crop we're transitioning TO
    def set_dominant_filter(self,f): self._dom_filter=f
    def _calc_safe_crop(self,cx,cy,fw,fh,fw_,fh_):
        cw=fw_*self.cfg.face_padding_ratio; ch=cw*self.aspect
        if ch>fh: ch=fh; cw=ch/self.aspect
        if cw>fw: cw=fw; ch=cw*self.aspect
        cx=cx-cw/2; cy=cy-ch*self.cfg.face_vertical_pos+fh_*0.1
        return max(0.0,min(cx,fw-cw)),max(0.0,min(cy,fh-ch)),cw,ch
    def _make_kf(self,cx,cy):
        kf=KalmanFilter(dim_x=4,dim_z=2); kf.x=np.array([cx,cy,0.,0.])
        kf.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        kf.H=np.eye(2,4); kf.P=np.eye(4)*10
        kf.Q[0:2,0:2]*=self.cfg.kalman_noise; kf.Q[2:4,2:4]*=self.cfg.kalman_noise*0.1; kf.R*=0.01; return kf
    def _snap(self,face,fw,fh):
        if self.pending_kf: self.kf=self.pending_kf; self.pending_kf=None
        else: self.kf=self._make_kf(face.cx,face.cy)
        nx,ny,nw,nh=self._calc_safe_crop(face.cx,face.cy,fw,fh,face.w,face.h)
        # v8: Start smooth transition from old crop to new crop (Change 3)
        # Instead of immediately jumping, save old crop and interpolate over
        # ~5 frames to avoid the garbled partial-crop flash.
        self._transition_crop = (self.current_x, self.current_y, self.locked_w, self.locked_h)
        self._transition_frames_left = self._transition_total_frames
        self.current_x=self.target_x=nx; self.current_y=self.target_y=ny
        self.locked_w=nw; self.locked_h=nh; self.current_speaker_id=face.track_id
        self.state="FOLLOWING"; self.pending_id=None; self.pending_cnt=0; self.sticky_cnt=0
    def _filter_faces(self,faces,fh,fw):
        if self._dom_filter: viable=self._dom_filter.filter(faces,fh,fw)
        else: viable={tid:f for tid,f in faces.items() if f.h>=fh*self.cfg.min_face_size_ratio}
        if self.current_speaker_id is not None:
            viable={tid:f for tid,f in viable.items() if f.frames_alive>=self.cfg.face_track_min_frames or tid==self.current_speaker_id}
        if not viable and faces: largest=max(faces.values(),key=lambda f:f.h); viable={largest.track_id:largest}
        return viable
    def _center_score(self,face):
        if not self.face_ever_seen or not self.current_speaker_id: return 0.0
        cam_cx=self.current_x+self.locked_w/2; cam_cy=self.current_y+self.locked_h/2
        diag=np.sqrt(self.locked_w**2+self.locked_h**2)
        if diag<1: return 0.0
        dist=np.sqrt((face.cx-cam_cx)**2+(face.cy-cam_cy)**2)
        return max(0.0,1.0-dist/diag)
    def _select(self,faces,speaking,fh,fw):
        viable=self._filter_faces(faces,fh,fw)
        if not viable: return None
        if len(viable)==1: return list(viable.values())[0]
        # FIX: Old scoring let face size (ss*50) dominate over speaking detection.
        # A non-speaker with a bigger frontal face could outscore the actual speaker.
        # New scoring: TalkNet-confirmed speakers get a MASSIVE bonus that face
        # size cannot overcome. Face size becomes a tiebreaker, not the main signal.
        has_talknet=any(f.talknet_score!=0.0 for f in viable.values())
        scored=[]
        for tid,f in viable.items():
            sc=0.0; spk=f.talknet_score if f.talknet_score!=0.0 else f.lip_score
            ss=(f.w*f.h)/(fh*fh+1e-6)
            if has_talknet and speaking:
                # TalkNet is scoring — let it dominate the decision
                if f.talknet_score > self.cfg.talknet_speaking_threshold:
                    # Confirmed speaker: huge bonus + proportional TalkNet score
                    sc += 500.0
                    sc += spk * 50.0
                elif f.talknet_score < 0:
                    # Confirmed non-speaker: strong penalty
                    sc -= 200.0
                    sc += spk * 20.0
                else:
                    # Not yet scored by TalkNet — use lip score as provisional
                    sc += spk * 150.0
                # Face size is only a tiebreaker when TalkNet is active
                sc += ss * 10.0
            elif speaking:
                # No TalkNet — lip score + moderate size (rebalanced)
                sc += spk * 200.0
                sc += ss * 20.0
            else:
                # Silence — fall back to face size
                sc += ss * 50.0
            sc += self._center_score(f) * self.cfg.face_center_bias * 20.0
            if tid==self.current_speaker_id:
                self.sticky_cnt+=1
                sc+=min(self.cfg.face_stickiness_frames,self.sticky_cnt)*self.cfg.face_stickiness_size_bonus*30.0
            scored.append((f,sc))
        scored.sort(key=lambda x:x[1],reverse=True); best=scored[0][0]
        if best.track_id!=self.current_speaker_id: self.sticky_cnt=0
        return best
    def update(self,faces,shape,speaking,rms=0.0):
        fh,fw=shape[:2]
        if not self.initialized:
            self.locked_w=min(fw,fh/self.aspect); self.locked_h=self.locked_w*self.aspect
            self.current_x=(fw-self.locked_w)/2; self.current_y=(fh-self.locked_h)/2
            self.target_x=self.current_x; self.target_y=self.current_y; self.initialized=True
        if not faces: return self.current_x,self.current_y,self.locked_w,self.locked_h
        active=self._select(faces,speaking,fh,fw)
        if active is None: return self.current_x,self.current_y,self.locked_w,self.locked_h
        if not self.face_ever_seen: self.face_ever_seen=True; self._snap(active,fw,fh); return self.current_x,self.current_y,self.locked_w,self.locked_h
        if active.track_id!=self.current_speaker_id:
            if self.pending_id!=active.track_id: self.pending_id=active.track_id; self.pending_cnt=1; self.pending_kf=self._make_kf(active.cx,active.cy)
            else:
                if self.pending_kf: self.pending_kf.predict(); self.pending_kf.update([active.cx,active.cy])
                self.pending_cnt+=1; needed=self.cfg.hysteresis_frames
                if not speaking: needed=max(needed,self.cfg.face_stickiness_frames)
                # FIX: When TalkNet strongly confirms a different speaker, switch fast
                # (bypass most of the hysteresis). Old code always required 15 frames
                # even when TalkNet was confident, causing the camera to stay stuck on
                # the non-speaker for 0.5 seconds.
                if speaking and active.talknet_score > 1.0:
                    cur_face = faces.get(self.current_speaker_id)
                    cur_tn = cur_face.talknet_score if cur_face else 0.0
                    # Strong TalkNet signal for new speaker + current speaker is silent
                    if cur_tn < 0 or (active.talknet_score - cur_tn) > 2.0:
                        needed = min(needed, 3)  # switch in just 3 frames (~0.1s)
                if self.pending_cnt>=needed: self._snap(active,fw,fh)
        else: self.pending_id=None; self.pending_cnt=0; self.pending_kf=None
        if self.state=="FOLLOWING" and self.current_speaker_id in faces:
            face=faces[self.current_speaker_id]; self.kf.predict(); self.kf.update([face.cx,face.cy])
            kx,ky=self.kf.x[0],self.kf.x[1]; dx,dy,_,_=self._calc_safe_crop(kx,ky,fw,fh,face.w,face.h)
            if abs(dx-self.target_x)>self.cfg.dead_zone_px or abs(dy-self.target_y)>self.cfg.dead_zone_px: self.target_x=dx; self.target_y=dy
            self.current_x+=(self.target_x-self.current_x)*self.cfg.tracking_lerp; self.current_y+=(self.target_y-self.current_y)*self.cfg.tracking_lerp
        self.current_x=max(0,min(self.current_x,fw-self.locked_w)); self.current_y=max(0,min(self.current_y,fh-self.locked_h))
        if self.cfg.zoom_enabled and rms>self.cfg.zoom_rms_threshold:
            t=np.clip((rms-self.cfg.zoom_rms_threshold)/(self.cfg.zoom_rms_peak-self.cfg.zoom_rms_threshold+1e-6),0,1)
            self.target_zoom=1.0+(self.cfg.zoom_max_factor-1.0)*t; lerp=self.cfg.zoom_in_lerp
        else: self.target_zoom=1.0; lerp=self.cfg.zoom_out_lerp
        self.current_zoom+=(self.target_zoom-self.current_zoom)*lerp
        if self.punch_zoom_factor>1.001: self.current_zoom*=self.punch_zoom_factor
        cw=self.locked_w/self.current_zoom; ch=self.locked_h/self.current_zoom
        cx=self.current_x+(self.locked_w-cw)/2; cy=self.current_y+(self.locked_h-ch)/2
        self.current_crop_w=cw; self.current_crop_h=ch
        # v8: Smooth transition interpolation (Change 3)
        # When transitioning from old crop to new crop, hold old crop for first
        # half then quickly ease to new crop. This avoids the garbled partial-crop
        # flash that appears when blending between two different speaker positions.
        if self._transition_crop is not None and self._transition_frames_left > 0:
            old_cx, old_cy, old_cw, old_ch = self._transition_crop
            total = max(1, self._transition_total_frames)
            elapsed = total - self._transition_frames_left
            hold_frames = total // 2  # hold old crop for first half
            if elapsed < hold_frames:
                # Hold phase: stay on old crop position (no garbled intermediate)
                cx, cy, cw, ch = old_cx, old_cy, old_cw, old_ch
            else:
                # Transition phase: ease from old to new over remaining frames
                t_blend = (elapsed - hold_frames) / max(1, total - hold_frames)
                # Smoothstep for ease-in-out
                t_smooth = t_blend * t_blend * (3.0 - 2.0 * t_blend)
                cx = old_cx + (cx - old_cx) * t_smooth
                cy = old_cy + (cy - old_cy) * t_smooth
                cw = old_cw + (cw - old_cw) * t_smooth
                ch = old_ch + (ch - old_ch) * t_smooth
            self._transition_frames_left -= 1
            if self._transition_frames_left <= 0:
                self._transition_crop = None
        return max(0,min(cx,fw-cw)),max(0,min(cy,fh-ch)),cw,ch


# ============================================================
# PORTRAIT SOURCE DETECTION
# ============================================================
def detect_source_portrait(video_path):
    cap=cv2.VideoCapture(video_path); w,h=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT); cap.release()
    return w>0 and h>0 and abs(h/w-16/9)<0.09


# ============================================================
# FONT MANAGER
# ============================================================
class FontManager:
    FONT_DIR="/kaggle/working/fonts"
    FONTS={
        "bold":{"url":"https://cdn.jsdelivr.net/gh/JulietaUla/Montserrat@master/fonts/ttf/Montserrat-ExtraBold.ttf","file":"Montserrat-ExtraBold.ttf","fallback":"https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf"},
        "regular":{"url":"https://cdn.jsdelivr.net/gh/JulietaUla/Montserrat@master/fonts/ttf/Montserrat-SemiBold.ttf","file":"Montserrat-SemiBold.ttf","fallback":"https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-SemiBold.ttf"},
        "devanagari":{"url":"","file":"NotoSansDevanagari-Bold.ttf","fallback":""},
    }
    _setup_done = False

    @classmethod
    def setup(cls):
        if cls._setup_done: return
        os.makedirs(cls.FONT_DIR, exist_ok=True)
        deva_sys="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"
        deva_dst=os.path.join(cls.FONT_DIR,"NotoSansDevanagari-Bold.ttf")
        if not os.path.exists(deva_dst) or os.path.getsize(deva_dst)<50000:
            if not os.path.exists(deva_sys):
                print("  Installing fonts-noto-core via apt...")
                subprocess.run(["apt-get","install","-y","-q","fonts-noto-core"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if os.path.exists(deva_sys): shutil.copy2(deva_sys,deva_dst); print(f"  ✓ NotoSansDevanagari-Bold.ttf ({os.path.getsize(deva_dst)//1024}KB)")
            else: print("  ⚠ Could not install NotoSansDevanagari")
        for name,info in cls.FONTS.items():
            if name=="devanagari": continue
            path=os.path.join(cls.FONT_DIR,info["file"])
            if not os.path.exists(path) or os.path.getsize(path)<50000:
                if os.path.exists(path): os.remove(path)
                print(f"  Downloading font: {name} ...")
                for url in [info["url"],info.get("fallback","")]:
                    if not url: continue
                    try:
                        import requests
                        r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=30); r.raise_for_status()
                        with open(path,"wb") as f: f.write(r.content)
                        if os.path.getsize(path)<50000: os.remove(path); raise ValueError("Too small")
                        print(f"  ✓ {info['file']} ({os.path.getsize(path)//1024}KB)"); break
                    except Exception as e: print(f"  ✗ {e}")
        cls._setup_done = True

    @classmethod
    def get(cls,name,size):
        path=os.path.join(cls.FONT_DIR,cls.FONTS.get(name,cls.FONTS["bold"])["file"])
        if os.path.exists(path) and os.path.getsize(path)>50000: return ImageFont.truetype(path,size)
        for sys_path in ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
            if os.path.exists(sys_path): return ImageFont.truetype(sys_path,size)
        return ImageFont.load_default()


# ============================================================
# EMOJI HELPER
# ============================================================
_EMOJI_PATTERN=re.compile("[\U0001F300-\U0001F9FF\U0001FA00-\U0001FA9F\U00002600-\U000027BF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0000200D]+",flags=re.UNICODE)
_EMOJI_CACHE: dict = {}

def _twemoji_key(s):
    parts=[]
    for ch in s:
        cp=ord(ch)
        if cp in(0xFE0F,0xFE0E): continue
        parts.append('-' if cp==0x200D else f'{cp:x}')
    return '-'.join(p for p in parts if p)

def _dl_emoji(s,size):
    key=(s,size)
    if key in _EMOJI_CACHE: return _EMOJI_CACHE[key]
    k=_twemoji_key(s); result=None
    if k:
        try:
            import requests
            r=requests.get(f"https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{k}.png",headers={"User-Agent":"Mozilla/5.0"},timeout=10); r.raise_for_status()
            result=Image.open(BytesIO(r.content)).convert("RGBA").resize((size,size),Image.LANCZOS)
        except Exception: pass
    _EMOJI_CACHE[key]=result; return result

def _split_emoji(text):
    res=[]; last=0
    for m in _EMOJI_PATTERN.finditer(text):
        if m.start()>last: res.append((text[last:m.start()],False))
        res.append((m.group(),True)); last=m.end()
    if last<len(text): res.append((text[last:],False))
    return res


# ============================================================
# SOCIAL CARD RENDERER  (Devanagari-aware + Animated Reveal + Glow/Bulge v8)
# ============================================================
class SocialCardRenderer:
    def __init__(self,cfg):
        self.cfg=cfg; FontManager.setup()
        self._overlay=None; self._deva_font_main=None; self._deva_font_sub=None
        # ── Per-line overlay storage for animated reveal ──
        self._line_overlays = []
        self._line_y_positions = []
        self._line_is_sub = []
        self._total_lines = 0
        self._reveal_complete_overlay = None
        self._build()

    def _build(self):
        cfg=self.cfg; ow=cfg.target_width; oh=cfg.target_height; ah=int(oh*cfg.card_padding_top)
        mw=ow-int(ow*0.08)

        # ── BUG FIX #1: Auto-shrink font when text overflows card area ──
        # Long mixed-script captions like "₹50K तक Salary कमाने वाले आज ही जान ले"
        # can wrap to many lines that exceed ah (card area height). Previously,
        # overflow lines were silently clipped by _paste_rgba(). Now we iteratively
        # reduce font size until all lines fit, with a minimum floor of 28px.
        MIN_MAIN_FS = 28
        fs = max(60, min(int(ah * cfg.card_text_size), int(ow * 0.13)))

        while fs >= MIN_MAIN_FS:
            sfs = max(20, int(fs * 0.55))
            bf = FontManager.get("bold", fs)
            sf = FontManager.get("regular", sfs)
            deva_main = FontManager.get("devanagari", fs)
            deva_sub  = FontManager.get("devanagari", sfs)

            mlines = self._wrap(cfg.card_text, bf, mw, fs, deva_main)
            slines = self._wrap(cfg.card_subtext, sf, mw, sfs, deva_sub) if cfg.card_subtext else []

            bb_dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            draw_dummy = ImageDraw.Draw(bb_dummy)
            bb = draw_dummy.textbbox((0, 0), "AygH", font=bf); lh = bb[3] - bb[1]
            sbb = draw_dummy.textbbox((0, 0), "AyjgH", font=sf); slh = sbb[3] - sbb[1]

            mg = int(lh * 0.35); mh = len(mlines) * lh + (len(mlines) - 1) * mg
            sg = int(slh * 0.25); gb = int(fs * 0.4)
            sh = len(slines) * slh + (len(slines) - 1) * sg if slines else 0
            total_h = mh + gb + sh

            if total_h <= ah or fs <= MIN_MAIN_FS:
                break  # Text fits, or we've hit minimum font size

            # Text doesn't fit — shrink font and retry
            fs -= 4

        # Store fonts for _draw_line / _wrap access
        self._deva_font_main = deva_main
        self._deva_font_sub  = deva_sub

        y_start = max(int(fs * 0.1), (ah - total_h) // 2)

        r,g,b = cfg.card_text_color

        # ── Build per-line overlays for animated reveal ──
        self._line_overlays = []
        self._line_y_positions = []
        self._line_is_sub = []
        self._total_lines = len(mlines) + len(slines)

        y = y_start
        for segs in mlines:
            # v8 FIX: 30% extra height for Devanagari matras (upper/lower
            # diacritical marks) that extend beyond Latin baseline metrics.
            # Without this, Hindi text like "कमाने" gets vertically clipped.
            line_h = int(lh * 1.3) + mg
            line_img = Image.new("RGBA", (ow, line_h), (0, 0, 0, 0))
            line_draw = ImageDraw.Draw(line_img)
            self._draw_line(line_draw, line_img, segs, 0, bf, fs, (r,g,b,255), ow, True, deva_main)
            self._line_overlays.append(np.array(line_img))
            self._line_y_positions.append(y)
            self._line_is_sub.append(False)
            y += lh + mg

        if slines:
            y += gb; sr,sg2,sb=cfg.card_subtext_color
            for segs in slines:
                sub_h = int(slh * 1.3) + sg  # v8 FIX: extra height for Devanagari matras
                sub_img = Image.new("RGBA", (ow, sub_h), (0, 0, 0, 0))
                sub_draw = ImageDraw.Draw(sub_img)
                self._draw_line(sub_draw, sub_img, segs, 0, sf, sfs, (sr,sg2,sb,255), ow, False, deva_sub)
                self._line_overlays.append(np.array(sub_img))
                self._line_y_positions.append(y)
                self._line_is_sub.append(True)
                y += slh + sg

        # ── Full static overlay (fallback) ──
        img=Image.new("RGBA",(ow,ah),(0,0,0,0)); draw=ImageDraw.Draw(img)
        y = y_start
        for segs in mlines: self._draw_line(draw,img,segs,y,bf,fs,(r,g,b,255),ow,True,deva_main); y+=lh+mg
        if slines:
            y+=gb; sr,sg2,sb=cfg.card_subtext_color
            for segs in slines: self._draw_line(draw,img,segs,y,sf,sfs,(sr,sg2,sb,255),ow,False,deva_sub); y+=slh+sg
        self._overlay=np.array(img)

    def _wrap(self,text,font,mw,fs,deva_font=None):
        """Wrap text into lines that fit within max width.

        BUG FIX #4: Added deva_font parameter so subtext uses the correct
        (smaller) Devanagari font for width measurement instead of always
        using the main (larger) deva font. Without this, Devanagari subtext
        wraps prematurely because it's measured as wider than it renders.
        """
        words=text.split(' '); lines=[]; cs=[]
        if deva_font is None:
            deva_font=self._deva_font_main or font
        for w in words:
            ws=_split_emoji(w); ts=cs+([(" ",False)] if cs else [])+ws
            if self._meas_mixed(ts,font,deva_font,fs)<=mw: cs=ts
            else:
                if cs: lines.append(cs)
                cs=ws
        if cs: lines.append(cs)
        return lines

    def _meas_mixed(self,segs,latin_font,deva_font,fs):
        w=0
        for s,is_emoji in segs:
            if not s: continue
            if is_emoji: w+=fs
            else:
                for chunk,is_deva in _segment_by_script(s):
                    fnt=deva_font if is_deva else latin_font
                    w+=fnt.getbbox(chunk)[2]-fnt.getbbox(chunk)[0]
        return w

    def _draw_line(self,draw,img,segs,y,font,fs,color,cw,outline,deva_font=None):
        """Draw a single text line with script-aware font selection.

        BUG FIX #2: Added deva_font parameter so subtext lines use the
        correct (smaller) Devanagari font instead of always using the main
        (larger) one. Previously, Hindi subtext rendered at the wrong size.
        """
        if deva_font is None:
            deva_font=self._deva_font_main or font
        expanded=[]
        for (s,is_emoji) in segs:
            if is_emoji or not s: expanded.append((s,is_emoji,font)); continue
            for (chunk,is_deva) in _segment_by_script(s):
                expanded.append((chunk,False,deva_font if is_deva else font))
        tw=sum(fs if ie else (fnt.getbbox(s)[2]-fnt.getbbox(s)[0]) for s,ie,fnt in expanded if s)
        x=max(0,(cw-tw)//2)
        for (s,is_emoji,fnt) in expanded:
            if not s: continue
            if is_emoji:
                em=_dl_emoji(s,fs)
                if em: ey=y+int(fs*0.85)-em.size[1]; img.paste(em,(int(x),max(y,ey)),em)
                else: draw.text((int(x),y),s,font=fnt,fill=color)
                x+=fs
            else:
                r,g,b,a=color
                if outline:
                    # v8: Glow effect (colored bleed from theme)
                    glow_enabled = getattr(self.cfg, 'caption_glow_enabled', True)
                    glow_color = getattr(self.cfg, 'caption_glow_color', (255, 40, 40))
                    if glow_enabled:
                        for r_dist in [4, 3, 2, 1]:
                            ga = (*glow_color, max(10, int(50 / r_dist)))
                            for dx, dy in [(-r_dist,0),(r_dist,0),(0,-r_dist),(0,r_dist),
                                           (-r_dist,-r_dist),(r_dist,-r_dist),(-r_dist,r_dist),(r_dist,r_dist)]:
                                draw.text((int(x)+dx, y+dy), s, font=fnt, fill=ga)
                    # v8: Bulge effect (white puff outline)
                    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-2),(0,2),(-2,0),(2,0)]:
                        draw.text((int(x)+dx, y+dy), s, font=fnt, fill=(255, 255, 255, 55))
                    # Standard dark outline
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        draw.text((int(x)+dx, y+dy), s, font=fnt, fill=(0, 0, 0, 120))
                draw.text((int(x), y), s, font=fnt, fill=(r,g,b,a))
                x+=fnt.getbbox(s)[2]-fnt.getbbox(s)[0]

    def _rounded_mask(self,w,h,r):
        mask=np.zeros((h,w),dtype=np.uint8); r=min(r,w//2,h//2)
        cv2.rectangle(mask,(r,0),(w-r,h),255,-1); cv2.rectangle(mask,(0,r),(w,h-r),255,-1)
        for cx,cy,sa,ea in[(r,r,180,90),(w-r,r,270,90),(r,h-r,90,90),(w-r,h-r,0,90)]:
            cv2.ellipse(mask,(cx,cy),(r,r),sa,0,ea,255,-1)
        return mask

    def _shadow(self,canvas,vx,vy,vw,vh,r):
        cfg=self.cfg; ox,oy=cfg.card_shadow_offset; blur=cfg.card_shadow_blur|1; alpha=cfg.card_shadow_opacity
        smask=self._rounded_mask(vw,vh,r).astype(np.float32)/255.0
        smask=cv2.GaussianBlur(smask,(blur,blur),0)
        sx1,sy1=vx+ox,vy+oy; sx2,sy2=sx1+vw,sy1+vh; ch,cw=canvas.shape[:2]
        ssx1,ssy1=max(0,-sx1),max(0,-sy1); ssx2,ssy2=vw-max(0,sx2-cw),vh-max(0,sy2-ch)
        dsx1,dsy1=max(0,sx1),max(0,sy1); dsx2,dsy2=dsx1+(ssx2-ssx1),dsy1+(ssy2-ssy1)
        if dsx2>dsx1 and dsy2>dsy1:
            region=canvas[dsy1:dsy2,dsx1:dsx2].astype(np.float32)
            s=smask[ssy1:ssy2,ssx1:ssx2,np.newaxis]
            canvas[dsy1:dsy2,dsx1:dsx2]=np.clip(region*(1-s*alpha),0,255).astype(np.uint8)
        return canvas

    def _paste_rgba(self,canvas,rgba,x,y):
        h,w=rgba.shape[:2]; ch,cw=canvas.shape[:2]
        x1,y1=max(0,x),max(0,y); x2,y2=min(cw,x+w),min(ch,y+h)
        if x2<=x1 or y2<=y1: return
        patch=rgba[y1-y:y2-y,x1-x:x2-x]
        a=patch[:,:,3:4].astype(np.float32)/255.0
        rgb=patch[:,:,:3][:,:,::-1].astype(np.float32)
        bg=canvas[y1:y2,x1:x2].astype(np.float32)
        canvas[y1:y2,x1:x2]=np.clip(bg*(1-a)+rgb*a,0,255).astype(np.uint8)

    def render(self,video_frame,frame_idx=0,vfx=None):
        cfg=self.cfg; ow=cfg.target_width; oh=cfg.target_height
        canvas=np.full((oh,ow,3),cfg.card_bg_color,dtype=np.uint8)
        pl=int(ow*cfg.card_padding_sides); pt=int(oh*cfg.card_padding_top); pb=int(oh*cfg.card_padding_bottom)
        vx,vy=pl,pt; vw=ow-pl*2; vh=oh-pt-pb; cr=int(vw*cfg.card_corner_radius)
        panel=cv2.resize(video_frame,(vw,vh),interpolation=cv2.INTER_LANCZOS4)
        if cfg.card_shadow: canvas=self._shadow(canvas,vx,vy,vw,vh,cr)
        mask=self._rounded_mask(vw,vh,cr); m3=mask[:,:,np.newaxis].astype(np.float32)/255.0
        roi=canvas[vy:vy+vh,vx:vx+vw].astype(np.float32)
        canvas[vy:vy+vh,vx:vx+vw]=(roi*(1-m3)+panel.astype(np.float32)*m3).astype(np.uint8)
        ah=max(8,int(oh*0.007)); ay=vy-ah-int(oh*0.012)
        if ay>0:
            r2=ah//2
            cv2.rectangle(canvas,(vx+r2,ay),(vx+vw-r2,ay+ah),cfg.card_accent_color,-1)
            cv2.circle(canvas,(vx+r2,ay+r2),r2,cfg.card_accent_color,-1)
            cv2.circle(canvas,(vx+vw-r2,ay+r2),r2,cfg.card_accent_color,-1)

        # ── Animated text reveal or static overlay ──
        use_reveal = (cfg.card_animated_reveal and vfx is not None and self._line_overlays)
        if use_reveal:
            reveal_frame = vfx.reveal_frame
            for line_idx in range(len(self._line_overlays)):
                alpha = vfx.get_reveal_alpha(line_idx, reveal_frame)
                if alpha <= 0.0: continue
                offset_y = vfx.get_reveal_offset(line_idx, reveal_frame)
                line_rgba = self._line_overlays[line_idx]
                y_pos = self._line_y_positions[line_idx] + offset_y
                if alpha < 1.0:
                    modulated = line_rgba.copy()
                    modulated[:, :, 3] = (modulated[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
                    self._paste_rgba(canvas, modulated, 0, y_pos)
                else:
                    self._paste_rgba(canvas, line_rgba, 0, y_pos)
        else:
            if self._overlay is not None: self._paste_rgba(canvas,self._overlay,0,0)
        return canvas


# ============================================================
# v8: LIVE GLOWING CAPTION RENDERER (karaoke-style word sync)
# ============================================================
# v8: Neon color palette for live caption highlight cycling
LIVE_CAPTION_NEON_PALETTE = [
    (0, 255, 255),    # Neon Cyan
    (255, 0, 255),    # Neon Magenta
    (0, 255, 128),    # Neon Green
    (255, 255, 0),    # Neon Yellow
    (255, 100, 0),    # Neon Orange
    (128, 0, 255),    # Neon Purple
    (0, 128, 255),    # Neon Blue
    (255, 0, 128),    # Neon Pink
    (0, 255, 200),    # Neon Mint
    (255, 50, 50),    # Neon Red
    (200, 255, 0),    # Neon Lime
    (0, 200, 255),    # Neon Sky Blue
    (255, 0, 200),    # Neon Hot Pink
    (100, 255, 100),  # Neon Spring Green
    (255, 200, 0),    # Neon Gold
    (180, 0, 255),    # Neon Violet
]

# v8: Attention elements for live captions
LIVE_CAPTION_ATTENTION_EMOJIS = ["🔥", "💡", "⚡", "✨", "💥", "🎯", "👀", "🚨",
                                   "⭐", "💎", "🔔", "🌟", "💫", "🎬", "📢", "🏆"]


class LiveCaptionRenderer:
    """Renders word-by-word glowing captions synced to Whisper timestamps.

    Shows 3-5 words at a time. The currently spoken word is highlighted
    with a bright glow; past words are dimmer; upcoming words are faint.
    Uses the same script-aware rendering pipeline (Devanagari + Latin).

    v8 improvements:
      - Bottom gradient background instead of solid black strip
      - Word wrapping to prevent overflow
      - Compatible with card enabled (constrained to video region)
      - Neon color cycling for highlight words
      - Attention elements (emoji pulses) alongside captions
    """

    def __init__(self, cfg: Config, words: List[Dict]):
        """
        Args:
            cfg: Config with live_caption_* fields.
            words: List of dicts with 'start', 'end', 'word' keys
                   (from Whisper word-level timestamps).
        """
        self.cfg = cfg
        self.words = words  # [{start, end, word}, ...]
        self.max_words = getattr(cfg, 'live_caption_max_words', 4)
        FontManager.setup()

        # Pre-compute word groups for O(1) lookup per frame
        # Each group: {start, end, words: [str], active_idx: int}
        self._groups: List[Dict] = []
        self._build_groups()
        # Track group index for neon cycling and attention elements
        self._last_group_start = -1.0
        self._attention_frame_counter = 0

    def _build_groups(self):
        """Build overlapping word groups (sliding window of max_words)."""
        if not self.words:
            return
        gap = getattr(self.cfg, 'live_caption_word_gap_sec', 0.05)
        n = len(self.words)
        i = 0
        while i < n:
            # Take up to max_words words, but break at natural pauses
            group_words = []
            group_start = self.words[i]['start']
            for j in range(i, min(i + self.max_words, n)):
                group_words.append(self.words[j])
            group_end = group_words[-1]['end'] + gap
            self._groups.append({
                'start': group_start,
                'end': group_end,
                'words': group_words,
                'active_idx': 0,  # will be resolved per-frame
            })
            # Advance by max_words (non-overlapping groups for clean display)
            i += self.max_words

    def _find_group_and_active(self, t: float):
        """Find which word group is active at time t and which word is current.

        Returns (group_dict, active_word_index) or (None, -1).
        """
        for g in self._groups:
            if g['start'] <= t <= g['end']:
                # Find the active word within this group
                for wi, w in enumerate(g['words']):
                    if w['start'] <= t <= w['end']:
                        return g, wi
                # Between words in the group — use the last spoken word
                last_spoken = -1
                for wi, w in enumerate(g['words']):
                    if t >= w['start']:
                        last_spoken = wi
                return g, max(0, last_spoken)
        return None, -1

    def _measure_word_width(self, w_text, latin_font, deva_font, fs):
        """Measure pixel width of a single word (script-aware)."""
        w_segs = _split_emoji(w_text)
        tw = 0
        for seg_text, is_emoji in w_segs:
            if not seg_text:
                continue
            if is_emoji:
                tw += fs
            else:
                for chunk, is_deva in _segment_by_script(seg_text):
                    fnt = deva_font if is_deva else latin_font
                    tw += fnt.getbbox(chunk)[2] - fnt.getbbox(chunk)[0]
        return tw

    def render(self, canvas: np.ndarray, frame_time: float,
               video_region_top: Optional[int] = None,
               video_region_bottom: Optional[int] = None,
               video_region_left: Optional[int] = None,
               video_region_right: Optional[int] = None) -> np.ndarray:
        """Render live caption overlay on the canvas.

        Args:
            canvas: BGR numpy array (output frame).
            frame_time: Current time in seconds.
            video_region_top: Y coordinate where video region starts (below card text).
                              When card is enabled, pass int(canvas_h * card_padding_top).
                              When no card, pass None (0 assumed).
            video_region_bottom: Y coordinate where video region ends.
                                 When card is enabled, pass canvas_h - int(canvas_h * card_padding_bottom).
                                 When no card, pass None (canvas height assumed).
            video_region_left: X coordinate where video region starts (card side padding).
                               When card is enabled, pass int(canvas_w * card_padding_sides).
                               When no card, pass None (0 assumed).
            video_region_right: X coordinate where video region ends.
                                When card is enabled, pass canvas_w - int(canvas_w * card_padding_sides).
                                When no card, pass None (canvas width assumed).
        Returns:
            Modified canvas with caption overlay.
        """
        group, active_idx = self._find_group_and_active(frame_time)
        if group is None:
            return canvas

        cfg = self.cfg
        oh, ow = canvas.shape[:2]
        # Video region boundaries (constrain caption to video part when card is on)
        vid_top = video_region_top if video_region_top is not None else 0
        vid_bottom = video_region_bottom if video_region_bottom is not None else oh
        vid_left = video_region_left if video_region_left is not None else 0
        vid_right = video_region_right if video_region_right is not None else ow
        eff_h = vid_bottom - vid_top  # effective height of video region
        words_list = group['words']

        # ── Font setup ──
        fs = max(24, int(oh * getattr(cfg, 'live_caption_font_size', 0.055)))
        latin_font = FontManager.get("bold", fs)
        deva_font = FontManager.get("devanagari", fs)

        # ── Measure word widths ──
        word_widths = []
        space_w = latin_font.getbbox(" ")[2] - latin_font.getbbox(" ")[0]
        for w_dict in words_list:
            word_widths.append(self._measure_word_width(w_dict['word'], latin_font, deva_font, fs))

        # Total width with spaces between words
        total_w = sum(word_widths) + space_w * (len(words_list) - 1)

        # ── Word wrapping (2b): split into lines if overflow ──
        max_line_w = int(ow * 0.90)  # 90% of frame width
        lines = []  # each line: list of (word_index, word_dict)
        current_line = []
        current_line_w = 0
        for wi, w_dict in enumerate(words_list):
            ww = word_widths[wi]
            needed = ww + (space_w if current_line else 0)
            if current_line and current_line_w + needed > max_line_w:
                # Overflow — start new line
                lines.append(current_line)
                current_line = [(wi, w_dict)]
                current_line_w = ww
            else:
                current_line.append((wi, w_dict))
                current_line_w += needed
        if current_line:
            lines.append(current_line)

        # ── Position (constrained to video region) ──
        position = getattr(cfg, 'live_caption_position', 'bottom_center')
        line_h = int(fs * 1.5)
        total_text_h = len(lines) * line_h

        if position == 'mid_center':
            text_y_base = vid_top + eff_h // 2
        else:  # bottom_center
            text_y_base = vid_bottom - int(fs * 0.8) - total_text_h

        # ── Bottom gradient background (2a) ──
        bg_opacity = getattr(cfg, 'live_caption_bg_opacity', 0.72)
        # Gradient covers bottom ~22% of video region, constrained to video part only
        # When card is on, gradient is limited horizontally to the video panel area
        gradient_h = int(eff_h * 0.22)
        gradient_y1 = max(vid_top, vid_bottom - gradient_h)
        if gradient_y1 < vid_bottom:
            roi = canvas[gradient_y1:vid_bottom, vid_left:vid_right].astype(np.float32)
            actual_grad_h = vid_bottom - gradient_y1
            # Create vertical gradient: transparent at top, semi-opaque black at bottom
            grad_alpha = np.linspace(0, bg_opacity, actual_grad_h, dtype=np.float32)
            grad_alpha_3d = grad_alpha[:, np.newaxis, np.newaxis]  # (actual_grad_h, 1, 1)
            bg_color = np.array([0, 0, 0], dtype=np.float32)
            blended = roi * (1 - grad_alpha_3d) + bg_color * 255 * grad_alpha_3d
            canvas[gradient_y1:vid_bottom, vid_left:vid_right] = np.clip(blended, 0, 255).astype(np.uint8)

        # ── Neon color cycling (2d) ──
        neon_cycle = getattr(cfg, 'live_caption_neon_cycle', True)
        group_idx = 0
        for gi, g in enumerate(self._groups):
            if g is group:
                group_idx = gi
                break
        if neon_cycle:
            highlight_color_rgb = LIVE_CAPTION_NEON_PALETTE[group_idx % len(LIVE_CAPTION_NEON_PALETTE)]
            glow_color_rgb = highlight_color_rgb  # glow matches highlight
        else:
            highlight_color_rgb = getattr(cfg, 'live_caption_highlight_color', (255, 255, 50))
            glow_color_rgb = getattr(cfg, 'live_caption_glow_color', (0, 220, 255))
        text_color_rgb = getattr(cfg, 'live_caption_text_color', (255, 255, 255))

        # ── Attention elements (6): emoji pulse on new group start ──
        attention_enabled = getattr(cfg, 'live_caption_attention_elements', True)
        attention_alpha = 0.0
        attention_emoji = None
        if attention_enabled and group['start'] != self._last_group_start:
            self._last_group_start = group['start']
            self._attention_frame_counter = 0
        if attention_enabled and active_idx == 0:
            self._attention_frame_counter += 1
            # Fade in over 3 frames, fade out over 5 frames (total 8 frames)
            fc = self._attention_frame_counter
            if fc <= 3:
                attention_alpha = fc / 3.0
            elif fc <= 8:
                attention_alpha = 1.0 - (fc - 3) / 5.0
            else:
                attention_alpha = 0.0
            attention_emoji = LIVE_CAPTION_ATTENTION_EMOJIS[group_idx % len(LIVE_CAPTION_ATTENTION_EMOJIS)]

        # ── Render each line ──
        for line_idx, line_words in enumerate(lines):
            # Calculate line width
            line_total_w = sum(word_widths[wi] for wi, _ in line_words) + space_w * (len(line_words) - 1)
            text_x_start = max(10, (ow - line_total_w) // 2)
            text_y = text_y_base + line_idx * line_h

            # Render attention emoji ABOVE the first line (centered)
            if line_idx == 0 and attention_alpha > 0 and attention_emoji is not None:
                em = _dl_emoji(attention_emoji, fs)
                if em:
                    em_x = text_x_start + line_total_w // 2 - fs // 2
                    em_y = text_y - fs - 4
                    self._overlay_emoji(canvas, em, max(vid_left, em_x), max(vid_top, em_y), attention_alpha)

            x_cursor = text_x_start
            for wi, w_dict in line_words:
                w_text = w_dict['word']

                # Determine color and glow intensity for this word
                if wi == active_idx:
                    # Currently spoken word — neon highlight + strong glow
                    color_rgb = highlight_color_rgb
                    glow_intensity = 1.0
                    outline_strength = 3
                elif wi < active_idx:
                    # Already spoken — white text, mild glow
                    color_rgb = text_color_rgb
                    glow_intensity = 0.4
                    outline_strength = 1
                else:
                    # Upcoming — dimmer text, no glow
                    color_rgb = tuple(c // 2 for c in text_color_rgb)
                    glow_intensity = 0.0
                    outline_strength = 1

                # Render word segments (script-aware)
                w_segs = _split_emoji(w_text)
                for seg_text, is_emoji in w_segs:
                    if not seg_text:
                        continue
                    if is_emoji:
                        em = _dl_emoji(seg_text, fs)
                        if em:
                            ey = text_y + int(fs * 0.1) - em.size[1]
                            self._overlay_emoji(canvas, em, int(x_cursor), max(text_y, ey), glow_intensity)
                        x_cursor += fs
                    else:
                        for chunk, is_deva in _segment_by_script(seg_text):
                            fnt = deva_font if is_deva else latin_font
                            self._draw_glow_text(
                                canvas, chunk, int(x_cursor), text_y, fnt,
                                color_rgb, glow_color_rgb, glow_intensity, outline_strength
                            )
                            chunk_w = fnt.getbbox(chunk)[2] - fnt.getbbox(chunk)[0]
                            x_cursor += chunk_w

                # Add space between words
                x_cursor += space_w

        return canvas

    def _draw_glow_text(self, canvas, text, x, y, font, color_rgb, glow_rgb,
                        glow_intensity, outline_strength):
        """Draw a single text segment with glow effect directly on BGR canvas.

        Uses PIL for rendering, then composites back.
        """
        h, w = canvas.shape[:2]
        # Create a small PIL image just for this text
        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0] + 20  # padding for glow
        th = bbox[3] - bbox[1] + 20
        if tw < 1 or th < 1:
            return
        ox, oy = 10 - bbox[0], 10 - bbox[1]  # offset to center in padded image

        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        r, g, b = color_rgb
        a = 255

        # Glow layers (outer to inner)
        if glow_intensity > 0:
            gr, gg, gb = glow_rgb
            for radius in [5, 4, 3, 2, 1]:
                ga = max(5, int(40 * glow_intensity / radius))
                glow_fill = (gr, gg, gb, ga)
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            draw.text((ox + dx, oy + dy), text, font=font, fill=glow_fill)

        # Dark outline for readability
        if outline_strength >= 2:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((ox + dx, oy + dy), text, font=font, fill=(0, 0, 0, 160))
        else:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((ox + dx, oy + dy), text, font=font, fill=(0, 0, 0, 100))

        # Main text
        draw.text((ox, oy), text, font=font, fill=(r, g, b, a))

        # Composite onto canvas
        rgba = np.array(img)
        ch, cw = canvas.shape[:2]
        # Position on canvas
        px, py = x - 10, y - 10  # account for padding
        x1, y1 = max(0, px), max(0, py)
        x2, y2 = min(cw, px + tw), min(ch, py + th)
        if x2 <= x1 or y2 <= y1:
            return
        sx1, sy1 = x1 - px, y1 - py
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        patch = rgba[sy1:sy2, sx1:sx2]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        rgb = patch[:, :, :3][:, :, ::-1].astype(np.float32)  # RGB→BGR
        bg = canvas[y1:y2, x1:x2].astype(np.float32)
        canvas[y1:y2, x1:x2] = np.clip(bg * (1 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)

    def _overlay_emoji(self, canvas, emoji_img, x, y, glow_intensity):
        """Overlay a PIL RGBA emoji onto BGR canvas."""
        h, w = canvas.shape[:2]
        ew, eh = emoji_img.size
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + ew), min(h, y + eh)
        if x2 <= x1 or y2 <= y1:
            return
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        patch = np.array(emoji_img)[sy1:sy2, sx1:sx2]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        if glow_intensity < 1.0:
            alpha *= glow_intensity
        rgb = patch[:, :, :3][:, :, ::-1].astype(np.float32)
        bg = canvas[y1:y2, x1:x2].astype(np.float32)
        canvas[y1:y2, x1:x2] = np.clip(bg * (1 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)


# ============================================================
# v8: AUTO B-ROLL INSERTER (Pexels Free API)
# ============================================================
class BRollInserter:
    """Downloads relevant stock footage from Pexels and overlays it
    as brief B-roll during frame processing.

    Uses the free Pexels API (https://www.pexels.com/api/).
    After Whisper transcription, extracts keywords from the transcript
    and searches Pexels for portrait videos. During frame processing,
    overlays B-roll footage with dissolve transitions when a keyword
    timestamp is reached.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = getattr(cfg, 'broll_enabled', False)
        self.api_key = getattr(cfg, 'broll_pexels_api_key', '_pexel_api_key')
        self.duration_sec = getattr(cfg, 'broll_duration_sec', 3.0)
        self.max_per_clip = getattr(cfg, 'broll_max_per_clip', 3)
        self.style = getattr(cfg, 'broll_style', 'dissolve')
        # Runtime state
        self._clips = []           # list of {keyword, start, end, frames, cap, fps}
        self._active_clip = None   # currently playing B-roll
        self._active_start_frame = -1
        self._active_duration_frames = 0
        self._broll_frames_cache = []  # pre-read frames for active clip
        self._keywords_extracted = False
        self._download_dir = "/kaggle/working/broll_clips"

    def extract_keywords(self, whisper_words):
        """Extract meaningful keywords from Whisper word-level timestamps.

        Filters out common stop words and short words, returns unique
        keywords with their average timestamps for B-roll trigger points.
        """
        if self._keywords_extracted or not whisper_words:
            return
        self._keywords_extracted = True
        if not self.enabled or not self.api_key:
            return

        # Simple stop word list
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between',
            'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but',
            'and', 'or', 'if', 'while', 'about', 'up', 'it', 'its',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
        }

        # Group words and extract keywords with timestamps
        keyword_times = {}  # keyword -> [list of timestamps]
        for w in whisper_words:
            word = w['word'].lower().strip('.,!?;:"\'-')
            if len(word) < 4 or word in stop_words:
                continue
            if word not in keyword_times:
                keyword_times[word] = []
            keyword_times[word].append((w['start'] + w['end']) / 2.0)

        # Pick top keywords by frequency (max 8)
        sorted_kw = sorted(keyword_times.items(), key=lambda x: len(x[1]), reverse=True)
        top_keywords = sorted_kw[:8]

        # Download B-roll clips for each keyword
        os.makedirs(self._download_dir, exist_ok=True)
        for keyword, times in top_keywords:
            clip_count = 0
            for t in times:
                if clip_count >= self.max_per_clip:
                    break
                clip_path = self._search_and_download(keyword, t)
                if clip_path:
                    self._clips.append({
                        'keyword': keyword,
                        'time': t,
                        'path': clip_path,
                    })
                    clip_count += 1

        if self._clips:
            print(f"  ✓ B-Roll: {len(self._clips)} clips queued for {len(top_keywords)} keywords")
        else:
            print(f"  ℹ B-Roll: no clips downloaded")

    def _search_and_download(self, keyword, timestamp):
        """Search Pexels for a portrait video and download it."""
        if not self.api_key:
            return None
        try:
            headers = {"Authorization": self.api_key}
            params = {
                "query": keyword,
                "per_page": 3,
                "orientation": "portrait",
                "size": "medium",
            }
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            videos = data.get("videos", [])
            if not videos:
                return None

            # Pick the first video with a suitable file
            for video in videos:
                for vf in video.get("video_files", []):
                    width = vf.get("width", 0)
                    height = vf.get("height", 0)
                    # Prefer portrait or square videos
                    if height >= width and width >= 360:
                        link = vf.get("link")
                        if link:
                            fname = f"broll_{keyword}_{int(timestamp)}.mp4"
                            fpath = os.path.join(self._download_dir, fname)
                            if os.path.exists(fpath):
                                return fpath
                            # Download
                            r = requests.get(link, timeout=60, stream=True)
                            if r.status_code == 200:
                                with open(fpath, "wb") as f:
                                    for chunk in r.iter_content(8192):
                                        f.write(chunk)
                                if os.path.getsize(fpath) > 10000:
                                    return fpath
                                else:
                                    os.remove(fpath)
            return None
        except Exception as e:
            print(f"  ⚠ B-Roll download failed for '{keyword}': {e}")
            return None

    def get_overlay_frame(self, canvas, frame_idx, fps):
        """Check if B-roll should be overlaid at this frame and return composited canvas.

        Returns:
            Modified canvas with B-roll overlay, or original canvas if no B-roll active.
        """
        if not self.enabled or not self._clips:
            return canvas

        frame_time = frame_idx / max(1, fps)
        oh, ow = canvas.shape[:2]

        # Check if any B-roll clip should trigger
        if self._active_clip is None:
            for clip in self._clips:
                if abs(frame_time - clip['time']) < 0.1:
                    # Trigger this clip
                    try:
                        cap = cv2.VideoCapture(clip['path'])
                        if not cap.isOpened():
                            continue
                        clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        total_frames = int(clip_fps * self.duration_sec)
                        # Pre-read frames
                        frames = []
                        for _ in range(total_frames):
                            ret, f = cap.read()
                            if not ret:
                                break
                            f = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                            frames.append(f)
                        cap.release()
                        if frames:
                            self._active_clip = frames
                            self._active_start_frame = frame_idx
                            self._active_duration_frames = len(frames)
                            break
                    except Exception:
                        continue

        # If active B-roll, composite it
        if self._active_clip is not None:
            elapsed = frame_idx - self._active_start_frame
            if elapsed < self._active_duration_frames:
                broll_frame = self._active_clip[elapsed]
                # Compute dissolve alpha
                dissolve_frames = min(8, self._active_duration_frames // 4)
                if elapsed < dissolve_frames:
                    alpha = elapsed / max(1, dissolve_frames)
                elif elapsed > self._active_duration_frames - dissolve_frames:
                    alpha = (self._active_duration_frames - elapsed) / max(1, dissolve_frames)
                else:
                    alpha = 1.0

                if self.style == "pip":
                    # Picture-in-picture: smaller overlay in corner
                    pip_w, pip_h = ow // 3, oh // 3
                    pip_frame = cv2.resize(broll_frame, (pip_w, pip_h))
                    px, py = ow - pip_w - 10, 10
                    roi = canvas[py:py+pip_h, px:px+pip_w].astype(np.float32)
                    pip_f = pip_frame.astype(np.float32)
                    blended = roi * (1 - alpha) + pip_f * alpha
                    canvas[py:py+pip_h, px:px+pip_w] = np.clip(blended, 0, 255).astype(np.uint8)
                elif self.style == "split":
                    # Split: left half B-roll, right half original
                    half = ow // 2
                    left = canvas[:, :half].astype(np.float32)
                    right_broll = broll_frame[:, :half].astype(np.float32)
                    blended = left * (1 - alpha) + right_broll * alpha
                    canvas[:, :half] = np.clip(blended, 0, 255).astype(np.uint8)
                else:
                    # dissolve (default): full-frame crossfade
                    canvas_f = canvas.astype(np.float32)
                    broll_f = broll_frame.astype(np.float32)
                    blended = canvas_f * (1 - alpha) + broll_f * alpha
                    canvas = np.clip(blended, 0, 255).astype(np.uint8)
                return canvas
            else:
                # B-roll finished
                self._active_clip = None
                self._active_start_frame = -1

        return canvas

    def cleanup(self):
        """Remove downloaded B-roll clips."""
        try:
            if os.path.isdir(self._download_dir):
                shutil.rmtree(self._download_dir)
        except Exception:
            pass


# ============================================================
# STORAGE UTILITY
# ============================================================
def free_gb(path="/kaggle/working"):
    st=shutil.disk_usage(path); return st.free/1024**3


# ============================================================
# SINGLE-VIDEO PROCESSOR  (v8 — TalkNet ASD + Visual FX + Bulge)
# ============================================================
class SingleVideoProcessor:
    _vcache: dict = {}

    def __init__(self, cfg: Config, yolo: YOLO, use_face_landmarks: bool,
                 talknet: TalkNetSpeakerDetector):
        self.cfg = cfg; self.yolo = yolo; self.use_face_landmarks = use_face_landmarks
        self.talknet = talknet; self.talknet_active = talknet.is_ready
        # ── v8: Live caption integration ──────────────────────────────
        # When live_caption_enabled, auto-disable split mode and social card
        # (live captions replace the static card — they're incompatible)
        self.live_caption_active = getattr(cfg, 'live_caption_enabled', False)
        if self.live_caption_active:
            cfg.split_enabled = False
            # v8 FIX: Live caption now coexists with card — don't force-disable it.
            # The caption is constrained to the video region above the card.
            print("  ℹ Live Caption ON — split mode disabled, card preserved")
        self.card_renderer  = SocialCardRenderer(cfg) if cfg.card_enabled else None
        self.is_portrait    = cfg.portrait_stabilize_only or detect_source_portrait(cfg.input_video)
        self.tracker        = FaceTracker()
        self.dom_filter     = DominantFaceFilter(cfg)
        self.camera         = BulletproofCamera(cfg)
        self.camera.set_dominant_filter(self.dom_filter)
        self.energy_tracker = AudioEnergyTracker(cfg)
        self.split_renderer = SplitScreenRenderer(cfg)
        self.reid           = FaceReID(cfg)
        self.cut_detect     = SceneCutDetector(cfg)
        self.color_grader   = ColorGrader(cfg.color_grading)
        self.vfx = VisualFX(cfg)
        self._current_cg_preset = cfg.color_grading.preset_name if cfg.color_grading else None
        self.switches = 0; self.last_spk = None
        # ── v8: B-Roll inserter (must init BEFORE _init_live_caption) ──
        self.broll_inserter = BRollInserter(cfg)
        # ── v8: Live caption renderer ────────────────────────────────
        self.live_caption_renderer: Optional[LiveCaptionRenderer] = None
        if self.live_caption_active:
            self._init_live_caption(cfg)

    def _init_live_caption(self, cfg: Config):
        """Run Whisper on the input video to get word-level timestamps,
        then create the LiveCaptionRenderer. If correct_whisper is on,
        show the transcript and allow manual editing.
        """
        print("  ⏳ Live Caption: running Whisper for word-level timestamps...")
        try:
            from faster_whisper import WhisperModel
            # Try GPU first, fallback to CPU
            try:
                model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                                     download_root="/kaggle/working/whisper_models")
            except Exception:
                model = WhisperModel("large-v3", device="cpu", compute_type="int8",
                                     download_root="/kaggle/working/whisper_models")
            # Extract audio for Whisper
            temp_audio = "/kaggle/working/temp_live_caption_audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", cfg.input_video, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", temp_audio],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            language = getattr(cfg, 'caption_language', 'hinglish')
            # v8: Improved Hindi/Hinglish transcription — use initial_prompt
            # to prime the model for code-mixed Hindi-English content, and
            # beam_size=5 for better accuracy on Hindi word boundaries.
            # Without initial_prompt, Whisper often misrecognizes Hindi words
            # or hallucinates English when the audio is Hindi-dominant.
            if language == "hinglish":
                whisper_lang = "hi"
                initial_prompt = "यह एक हिंदी-अंग्रेजी मिश्रित वीडियो है। नमस्ते, आज हम बात करेंगे"
            elif language == "english":
                whisper_lang = "en"
                initial_prompt = None
            else:
                whisper_lang = None
                initial_prompt = None
            segments_iter, info = model.transcribe(
                temp_audio, language=whisper_lang, word_timestamps=True,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
                initial_prompt=initial_prompt, beam_size=5,
                condition_on_previous_text=True)
            # Collect all words
            all_words = []
            for seg in segments_iter:
                if seg.words:
                    for w in seg.words:
                        all_words.append({
                            'start': w.start,
                            'end': w.end,
                            'word': w.word.strip(),
                        })
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            # Unload Whisper to free VRAM
            del model
            torch.cuda.empty_cache()
            try:
                import gc; gc.collect()
            except Exception:
                pass
            if not all_words:
                print("  ⚠ Live Caption: no words transcribed — disabling feature")
                self.live_caption_active = False
                return
            print(f"  ✓ Live Caption: {len(all_words)} words transcribed")
            # ── Manual correction mode ─────────────────────────────────
            if getattr(cfg, 'live_caption_correct_whisper', False):
                all_words = self._correct_whisper_interactive(all_words)
            self.live_caption_renderer = LiveCaptionRenderer(cfg, all_words)
            # v8: Extract B-roll keywords from Whisper transcript (Change 5)
            if self.broll_inserter.enabled:
                self.broll_inserter.extract_keywords(all_words)
        except Exception as e:
            print(f"  ⚠ Live Caption init failed: {e}")
            self.live_caption_active = False

    def _correct_whisper_interactive(self, words: List[Dict]) -> List[Dict]:
        """Print the Whisper transcript and allow manual editing via input().

        The user can edit individual words or press Enter to keep as-is.
        Returns the corrected word list.
        """
        print("\n" + "=" * 60)
        print("LIVE CAPTION — Whisper Correction Mode")
        print("=" * 60)
        print("  Below is the transcript with timestamps.")
        print("  To edit a word, type the replacement and press Enter.")
        print("  To keep as-is, just press Enter.")
        print("  Type 'DONE' to finish editing early.\n")
        # Show transcript in groups of ~8 words for readability
        corrected = list(words)  # copy
        i = 0
        while i < len(corrected):
            # Print a chunk of 8 words
            chunk_end = min(i + 8, len(corrected))
            chunk_str = " ".join(f"[{i+j}]{w['word']}" for j, w in enumerate(corrected[i:chunk_end]))
            t_start = corrected[i]['start']
            t_end = corrected[chunk_end - 1]['end']
            print(f"  {t_start:6.1f}s–{t_end:6.1f}s  {chunk_str}")
            # Allow editing each word in the chunk
            for j in range(i, chunk_end):
                current = corrected[j]['word']
                try:
                    replacement = input(f"    [{j}] '{current}' → ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n  (editing skipped)")
                    return corrected
                if replacement.upper() == 'DONE':
                    print("  ✓ Editing complete.")
                    return corrected
                if replacement and replacement != current:
                    corrected[j] = dict(corrected[j])  # copy
                    corrected[j]['word'] = replacement
                    print(f"    ✓ Changed to '{replacement}'")
            i = chunk_end
        print("\n  ✓ All words reviewed. Correction complete.")
        print("=" * 60 + "\n")
        return corrected

    def _detect(self, frame):
        results = self.yolo(frame, verbose=False, conf=self.cfg.yolo_confidence, imgsz=self.cfg.yolo_imgsz)[0]
        bboxes, lms = [], []; fh, fw = frame.shape[:2]; min_det_h = fh * self.cfg.min_face_size_ratio
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0]); x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            if y2-y1 < min_det_h: continue
            if self.use_face_landmarks:
                bboxes.append((x1,y1,x2,y2)); kpts = results.keypoints[i].xy[0].cpu().numpy() if results.keypoints else None; lms.append(kpts)
            else:
                if cls_id == 0: bh = y2-y1; bboxes.append((x1+(x2-x1)*0.15, y1, x2-(x2-x1)*0.15, y1+bh*0.35)); lms.append(None)
        return bboxes, lms

    def _update_lip_scores(self, faces):
        for f in faces.values():
            if f.landmarks is not None and len(f.landmarks)==5:
                le,re,lm,rm = f.landmarks[0],f.landmarks[1],f.landmarks[3],f.landmarks[4]
                ed = np.linalg.norm(le-re); mw = np.linalg.norm(lm-rm)
                f.mouth_ratio_history.append(mw/(ed+1e-6))
                if len(f.mouth_ratio_history)>=4: f.lip_score=np.std(list(f.mouth_ratio_history))
            else: f.lip_score = 0.0  # FIX: was random noise — gave non-speakers a false positive signal

    def _update_speaker_scores(self, faces, frame, frame_idx, fps):
        self._update_lip_scores(faces)
        if self.talknet_active:
            for tid, f in faces.items():
                x1,y1,x2,y2 = f.bbox; bw, bh = x2-x1, y2-y1; pad = 0.40
                cx, cy = (x1+x2)/2, (y1+y2)/2; sw = bw*(1+pad); sh = bh*(1+pad)
                self.talknet.buffer_face(tid, frame, (cx-sw/2, cy-sh/2, cx+sw/2, cy+sh/2))
            if frame_idx % self.cfg.talknet_infer_every == 0:
                tn_scores = self.talknet.score_faces(frame_idx, fps)
                if tn_scores and not hasattr(self, '_tn_first_score_logged'):
                    self._tn_first_score_logged = True
                    print(f"  ✓ TalkNet first scores at frame {frame_idx}: {dict((tid,f'{s:+.2f}') for tid,s in tn_scores.items())}")
                for tid, score in tn_scores.items():
                    if tid in faces: faces[tid].talknet_score = score
            self.talknet.purge_lost(set(faces.keys()))

    def _merge_audio(self):
        """Re-encode to H.264 + merge audio from original video.

        Two-pass approach:
          1. OpenCV VideoWriter writes frames as mp4v (MPEG-4 Part 2) —
             only codec that reliably works on Kaggle's ffmpeg build.
          2. ffmpeg re-encodes to H.264 (libx264 software encoder) with
             BT.709 color space metadata — ensures platform compatibility
             and correct color interpretation by players.

        Color space note: The mp4v→H.264 re-encode preserves whatever
        colors mp4v wrote. The BT.709 flags tell the player how to
        interpret them correctly (most modern players assume BT.709
        for HD/4K content by default, but explicit flags prevent
        mismatch with BT.601).
        """
        temp = self.cfg.output_video
        final = temp.replace(".mp4", "_final.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", temp,
            "-i", self.cfg.input_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-color_range", "tv",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            "-movflags", "+faststart",
            final
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  ⚠ H.264 re-encode failed: {result.stderr[-500:]}")
                if os.path.exists(final): os.remove(final)
                print(f"  ℹ Output remains as mp4v (MPEG-4 Part 2)")
                return
            if os.path.exists(final):
                # Verify codec with ffprobe
                try:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=codec_name,width,height",
                         "-of", "csv=p=0", final],
                        capture_output=True, text=True, timeout=30)
                    codec_info = probe.stdout.strip()
                except Exception:
                    codec_info = "unknown"
                # BUG FIX #6: Use os.replace() instead of os.remove()+os.rename()
                # The old code did os.remove(temp) then os.rename(final, temp).
                # If the process crashed between these two calls, the original
                # output was deleted but the final hadn't been renamed = data loss.
                # os.replace() is atomic on POSIX and handles the replacement safely.
                os.replace(final, temp)
                mb = os.path.getsize(temp) / 1024 / 1024
                print(f"  ✓ Re-encoded: H.264 ({codec_info}) | {mb:.1f}MB | Audio: AAC 192k | Color: BT.709")
            else:
                print("  ⚠ H.264 re-encode: output not found, keeping mp4v")
        except subprocess.TimeoutExpired:
            print("  ⚠ H.264 re-encode timed out (600s), keeping mp4v")
            if os.path.exists(final): os.remove(final)
        except Exception as e:
            print(f"  ⚠ H.264 re-encode error: {e}")
            if os.path.exists(final): os.remove(final)

    def _blur_fill(self, cropped, tw, th, frame, zoom_factor=1.0):
        ch, cw = cropped.shape[:2]
        if abs(cw/ch-tw/th)<0.02: return cv2.resize(cropped,(tw,th),interpolation=cv2.INTER_LANCZOS4)
        blur_str = self.cfg.dof_blur_strength if self.cfg.dof_enabled else 0.35
        if self.cfg.dof_enabled and zoom_factor > 1.02:
            blur_str = min(0.85, blur_str + self.cfg.dof_zoom_blur_boost * (zoom_factor - 1.0) / 0.18)
        ksize = self.cfg.dof_bg_kernel_size | 1
        bg = cv2.GaussianBlur(cv2.resize(frame,(tw,th),interpolation=cv2.INTER_LINEAR),(ksize,ksize),0)
        bg = (bg*blur_str).astype(np.uint8)
        sc = min(tw/cw,th/ch); fw2,fh2 = int(cw*sc),int(ch*sc)
        fg = cv2.resize(cropped,(fw2,fh2),interpolation=cv2.INTER_LANCZOS4)
        ox,oy = (tw-fw2)//2,(th-fh2)//2; res=bg.copy(); res[oy:oy+fh2,ox:ox+fw2]=fg; return res

    def _vignette(self, img, strength=0.3, radius=0.75):
        h,w = img.shape[:2]; key=(w,h,strength,radius)
        if key not in self._vcache:
            xs=np.linspace(-1,1,w,dtype=np.float32); ys=np.linspace(-1,1,h,dtype=np.float32)
            xg,yg=np.meshgrid(xs,ys); d=np.sqrt(xg**2+yg**2)/np.sqrt(2.0)
            m=np.clip((d-radius)/(1-radius+1e-6),0,1); m=m*m*(3-2*m)
            self._vcache[key]=np.stack([(1-strength*m).astype(np.float32)]*3,axis=-1)
        return (img.astype(np.float32)*self._vcache[key]).clip(0,255).astype(np.uint8)

    def _apply_dynamic_cg(self, preset_name):
        if preset_name == self._current_cg_preset: return
        try:
            cg = ColorGradingConfig(mode="preset", preset_name=preset_name, intensity=self.cfg.color_grading_intensity)
            new_grader = ColorGrader(cg)
            if new_grader._ready: self.color_grader = new_grader; self._current_cg_preset = preset_name
        except Exception: pass

    _bulge_cache: dict = {}

    def _apply_video_bulge(self, frame):
        """v8: Subtle barrel distortion to make video area bulge out."""
        if not self.cfg.video_bulge_enabled:
            return frame
        h, w = frame.shape[:2]
        key = (w, h, self.cfg.video_bulge_strength)
        if key not in self._bulge_cache:
            cx, cy = w / 2.0, h / 2.0
            k = self.cfg.video_bulge_strength
            x = np.arange(w, dtype=np.float32)
            y = np.arange(h, dtype=np.float32)
            xg, yg = np.meshgrid(x, y)
            dx = (xg - cx) / cx
            dy = (yg - cy) / cy
            r2 = dx * dx + dy * dy
            factor = 1.0 + k * r2
            map_x = (cx + dx * factor * cx).astype(np.float32)
            map_y = (cy + dy * factor * cy).astype(np.float32)
            self._bulge_cache[key] = (map_x, map_y)
        map_x, map_y = self._bulge_cache[key]
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def run(self):
        cap=cv2.VideoCapture(self.cfg.input_video)
        fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ow,oh=(fw,fh) if self.is_portrait else (self.cfg.target_width,self.cfg.target_height)
        mode = "TalkNet ASD+Audio" if self.talknet_active else ("Face Landmarks+Audio" if self.use_face_landmarks else "Body+Audio VAD")
        print(f"  Video: {fw}x{fh} @ {fps:.1f}fps, {total} frames -> {ow}x{oh}")
        print(f"  Mode: {mode} | imgsz={self.cfg.yolo_imgsz}")
        audio_mask = get_audio_mask(self.cfg.input_video, fps, total)
        rms_per_frame = self.energy_tracker.compute(self.cfg.input_video, fps, total)
        if self.talknet_active:
            self.talknet.precompute_audio(self.cfg.input_video, fps, total)
            self.talknet.face_buffers.clear(); self.talknet.score_history.clear()
        out=cv2.VideoWriter(self.cfg.output_video,cv2.VideoWriter_fourcc(*'mp4v'),fps,(ow,oh))
        if not out.isOpened():
            print(f"  ✗ VideoWriter failed to open! Trying CAP_FFMPEG fallback...")
            out=cv2.VideoWriter(self.cfg.output_video,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'mp4v'),fps,(ow,oh))
        if not out.isOpened():
            raise RuntimeError(f"VideoWriter could not open output: {self.cfg.output_video}")
        t0=time.time()
        for i in range(total):
            ret,frame=cap.read()
            if not ret: break
            is_cut=self.cut_detect.is_cut(frame)
            if is_cut:
                self.camera.face_ever_seen=False; self.split_renderer.notify_scene_cut()
                if hasattr(self,'vfx'): self.vfx.reset_scene()
                if self.talknet_active: self.talknet.face_buffers.clear(); self.talknet.score_history.clear()
            if self.is_portrait:
                of=cv2.resize(frame,(ow,oh),interpolation=cv2.INTER_LANCZOS4)
                of=self._vignette(of); of=self.color_grader.apply(of)
                if self.cfg.card_enabled and self.card_renderer is not None: of=self.card_renderer.render(of, frame_idx=i, vfx=self.vfx)
                # v8: Live glowing caption overlay (constrained to video region above card)
                if self.live_caption_active and self.live_caption_renderer is not None:
                    frame_time = i / fps if fps > 0 else 0.0
                    if self.cfg.card_enabled:
                        _vrt_p = int(oh * self.cfg.card_padding_top)
                        _vrb_p = oh - int(oh * self.cfg.card_padding_bottom)
                        _vrl_p = int(ow * self.cfg.card_padding_sides)
                        _vrr_p = ow - int(ow * self.cfg.card_padding_sides)
                    else:
                        _vrt_p = None; _vrb_p = None; _vrl_p = None; _vrr_p = None
                    of = self.live_caption_renderer.render(of, frame_time, video_region_top=_vrt_p, video_region_bottom=_vrb_p, video_region_left=_vrl_p, video_region_right=_vrr_p)
                out.write(of); continue
            bboxes,lms=self._detect(frame)
            faces=self.tracker.update(bboxes,lms,frame.shape,reid=self.reid if self.cfg.reid_enabled else None,frame=frame)
            if self.cfg.face_beautify_enabled and faces: frame = VisualFX.beautify_face(frame, faces, self.cfg)
            if self.cfg.reid_enabled:
                for tid,face in faces.items():
                    x1,y1=int(max(0,face.bbox[0])),int(max(0,face.bbox[1])); x2,y2=int(min(fw,face.bbox[2])),int(min(fh,face.bbox[3]))
                    self.reid.update_gallery(tid,frame[y1:y2,x1:x2])
                self.reid.purge_stale(); self.reid.tick()
            self._update_speaker_scores(faces, frame, i, fps)
            filtered_faces=self.dom_filter.filter(faces,fh,fw) if faces else faces
            speaking=audio_mask[i] if i<len(audio_mask) else True
            rms=float(rms_per_frame[i]) if i<len(rms_per_frame) else 0.0
            in_split = self.split_renderer.state in ("SPLIT","OPENING")
            effective_rms = 0.0 if in_split else rms
            crop=self.camera.update(faces,frame.shape,speaking,rms=effective_rms)
            if hasattr(self,'vfx') and self.cfg.ken_burns_enabled:
                kb_ox, kb_oy, kb_z = self.vfx.update_ken_burns(self.camera.current_speaker_id)
                if kb_z > 1.001 or abs(kb_ox) > 0.01:
                    cx, cy, cw, ch = crop; cx -= kb_ox*cw; cy -= kb_oy*ch
                    nw = cw/kb_z; nh = ch/kb_z; cx += (cw-nw)/2; cy += (ch-nh)/2
                    cx = max(0,min(cx,frame.shape[1]-nw)); cy = max(0,min(cy,frame.shape[0]-nh))
                    crop = (cx,cy,nw,nh)
            if self.camera.current_speaker_id!=self.last_spk:
                if self.last_spk is not None and self.camera.current_speaker_id is not None: self.switches+=1
                if hasattr(self,'vfx') and self.cfg.punch_zoom_enabled: self.camera.punch_zoom_factor = self.vfx.update_punch_zoom(True)
                self.last_spk=self.camera.current_speaker_id
            else:
                if hasattr(self,'vfx') and self.cfg.punch_zoom_enabled: self.camera.punch_zoom_factor = self.vfx.update_punch_zoom(False)
            of=self.split_renderer.update(frame,filtered_faces,crop,speaking,self.camera)
            of=self._blur_fill(of,ow,oh,frame,zoom_factor=self.camera.current_zoom)
            if hasattr(self,'vfx'): self.vfx.draw_speaker_glow(of, filtered_faces, self.camera.current_speaker_id, i)
            of=self._vignette(of)
            if hasattr(self,'vfx') and self.cfg.dynamic_color_grading:
                dyn_preset = self.vfx.get_dynamic_preset(speaking, rms)
                if dyn_preset and dyn_preset != self._current_cg_preset: self._apply_dynamic_cg(dyn_preset)
            of=self.color_grader.apply(of)
            of = self._apply_video_bulge(of)  # v8: video bulge effect
            if hasattr(self,'vfx'):
                of = self.vfx.apply_film_grain(of, i)
                self.vfx.advance_reveal()
            # ── BUG FIX #3: Render card FIRST, then apply frame-level effects ──
            # The card renderer creates a brand new canvas from card_bg_color,
            # which completely replaces the video frame. Any effects applied
            # before card render (waveform border, letterbox, watermark) were
            # painted on the old frame and lost when the card canvas overwrote it.
            # Now we render the card first, then apply frame-level overlays
            # (waveform border, letterbox, watermark) on top of the final canvas
            # so they're visible regardless of card_enabled.
            if self.cfg.card_enabled and self.card_renderer is not None: of=self.card_renderer.render(of, frame_idx=i, vfx=self.vfx)
            if hasattr(self,'vfx'):
                of = VisualFX.apply_waveform_border(of, rms, i, self.cfg)  # v8: waveform border
                of = self.vfx.apply_letterbox(of, self.cfg)
                # v8 FIX: Pass video region boundaries so watermark stays in video part
                if self.cfg.card_enabled:
                    _card_y_offset = int(oh * self.cfg.card_padding_top)
                    _card_y_end = oh - int(oh * self.cfg.card_padding_bottom)
                else:
                    _card_y_offset = None
                    _card_y_end = None
                of = self.vfx.apply_watermark(of, i, card_y_offset=_card_y_offset, card_y_end=_card_y_end)
            # v8: Live glowing caption overlay (after all other FX, constrained to video region)
            if self.live_caption_active and self.live_caption_renderer is not None:
                frame_time = i / fps if fps > 0 else 0.0
                if self.cfg.card_enabled:
                    _vrt = int(oh * self.cfg.card_padding_top)
                    _vrb = oh - int(oh * self.cfg.card_padding_bottom)
                    _vrl = int(ow * self.cfg.card_padding_sides)
                    _vrr = ow - int(ow * self.cfg.card_padding_sides)
                else:
                    _vrt = None; _vrb = None; _vrl = None; _vrr = None
                of = self.live_caption_renderer.render(of, frame_time, video_region_top=_vrt, video_region_bottom=_vrb, video_region_left=_vrl, video_region_right=_vrr)
            # v8: B-Roll overlay (Change 5)
            if self.broll_inserter.enabled:
                of = self.broll_inserter.get_overlay_frame(of, i, fps)
            out.write(of)
            if i%100==0:
                el=time.time()-t0; spd=(i+1)/el; eta=(total-i)/max(spd,0.01)/60
                tn_info = ""
                if self.talknet_active and filtered_faces:
                    scores = [f"{tid}:{f.talknet_score:+.2f}(lip:{f.lip_score:.3f})" for tid,f in filtered_faces.items()]
                    if scores: tn_info = f" TN:[{','.join(scores)}]"
                spk_id = self.camera.current_speaker_id
                spk_info = f" Spk:{spk_id}" if spk_id is not None else " Spk:None"
                print(f"    [{i:5d}/{total}] {spd:4.1f}fps ETA:{eta:.1f}m Faces:{len(faces)}->{len(filtered_faces)} Switches:{self.switches} Split:{self.split_renderer.state}{spk_info}{tn_info}")
        cap.release(); out.release(); self._merge_audio()
        elapsed=time.time()-t0; print(f"  Done in {elapsed:.0f}s  Switches:{self.switches}"); return elapsed


# ============================================================
# MODEL DOWNLOAD UTILITY
# ============================================================
YOLO_FACE_MODELS = {
    "yolov8n-face": {"url": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt", "size_mb": 6.1, "has_keypoints": True},
    "yolov11n-face": {"url": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt", "size_mb": 5.2, "has_keypoints": False},
}

def download_face_model(model_name: str, save_dir: str = "/kaggle/working") -> Optional[str]:
    if model_name not in YOLO_FACE_MODELS: print(f"  ⚠ Unknown model: {model_name}"); return None
    info = YOLO_FACE_MODELS[model_name]; path = os.path.join(save_dir, f"{model_name}.pt")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        actual_mb = os.path.getsize(path)/1024/1024; print(f"  ✓ {model_name}.pt already exists ({actual_mb:.1f}MB)"); return path
    print(f"  Downloading {model_name}.pt ...")
    try:
        import requests
        r = requests.get(info["url"], headers={"User-Agent":"Mozilla/5.0"}, timeout=120, stream=True); r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        actual_mb = os.path.getsize(path)/1024/1024
        if actual_mb < 1.0: os.remove(path); return None
        print(f"  ✓ {model_name}.pt ({actual_mb:.1f}MB)"); return path
    except Exception as e:
        if os.path.exists(path): os.remove(path)
        print(f"  ✗ Download failed: {e}"); return None


# ============================================================
# BATCH RUNNER
# ============================================================
class BatchReframer:
    def __init__(self, batch_cfg: BatchConfig, jobs: List[VideoJob]):
        self.batch_cfg = batch_cfg; self.jobs = jobs; self.yolo = None; self.use_face_landmarks = False; self.talknet = None; self._load_models()

    def _load_models(self):
        variant = self.batch_cfg.yolo_face_variant; model_loaded = False
        if variant in ("auto", "yolov8n-face"):
            path = download_face_model("yolov8n-face")
            if path:
                try: self.yolo = YOLO(path).to('cuda'); self.use_face_landmarks = True; model_loaded = True; print(f"  ✓ Loaded yolov8n-face (POSE, 5 keypoints)")
                except Exception as e: print(f"  ⚠ Failed to load yolov8n-face: {e}")
        if not model_loaded and variant in ("auto", "yolov11n-face"):
            path = download_face_model("yolov11n-face")
            if path:
                try: self.yolo = YOLO(path).to('cuda'); self.use_face_landmarks = False; model_loaded = True; print(f"  ✓ Loaded yolov11n-face (DETECT, no keypoints)")
                except Exception as e: print(f"  ⚠ Failed to load yolov11n-face: {e}")
        if not model_loaded:
            print("  ⚠ Falling back to YOLOv8n body detection"); self.yolo = YOLO("yolov8n.pt").to('cuda'); self.use_face_landmarks = False
        names = getattr(self.yolo, 'names', {}); print(f"  Model task: {getattr(self.yolo,'task','detect')}  |  Classes: {names}"); print(f"  imgsz: {self.batch_cfg.yolo_imgsz}")
        self.talknet = TalkNetSpeakerDetector(self.batch_cfg)
        if self.talknet.is_ready: print("  Speaker detection: TalkNet ASD (audio-visual sync)")
        else: fallback = "lip-keypoints" if self.use_face_landmarks else "face-size + VAD"; print(f"  Speaker detection: {fallback} (TalkNet unavailable)")

    def _check_storage(self) -> bool:
        gb = free_gb()
        if gb < self.batch_cfg.min_free_gb: print(f"  ⚠ Only {gb:.1f}GB free — skipping."); return False
        return True

    def _cleanup_temps(self):
        for f in ["/kaggle/working/temp_audio_vad.wav","/kaggle/working/temp_rms_audio.wav","/kaggle/working/temp_talknet_audio.wav"]:
            if os.path.exists(f): os.remove(f)

    def run(self):
        total_jobs = len(self.jobs); results = []; batch_start = time.time()
        tn_status = "TalkNet ASD" if self.talknet.is_ready else ("lip-keypoints" if self.use_face_landmarks else "face-size+VAD")
        print(f"\n{'='*60}")
        print(f"BATCH START — {total_jobs} video(s)")
        print(f"Free disk: {free_gb():.1f}GB  |  Speaker detection: {tn_status}")
        print(f"YOLO model: {'Face+Landmarks' if self.use_face_landmarks else 'Face/Body'}")
        print(f"{'='*60}\n")
        for idx, job in enumerate(self.jobs):
            print(f"[{idx+1}/{total_jobs}] {os.path.basename(job.input_video)}")
            print(f"  Caption: {job.card_text}")
            if not self._check_storage(): results.append((job, "SKIPPED_DISK", 0)); continue
            if not os.path.exists(job.input_video):
                print(f"  ✗ Input not found: {job.input_video}"); results.append((job, "MISSING_INPUT", 0)); continue
            os.makedirs(os.path.dirname(os.path.abspath(job.output_video)), exist_ok=True)
            cfg = _make_config(self.batch_cfg, job)
            elapsed = 0.0
            try:
                proc = SingleVideoProcessor(cfg, self.yolo, self.use_face_landmarks, self.talknet)
                elapsed = proc.run(); results.append((job, "OK", elapsed))
                print(f"  Output: {job.output_video}  ({free_gb():.1f}GB free)")
                if job.delete_input_after: os.remove(job.input_video); print(f"  Deleted input: {job.input_video}")
            except Exception as e:
                import traceback; print(f"  ✗ ERROR: {e}"); traceback.print_exc(); results.append((job, f"ERROR: {e}", elapsed))
            finally: self._cleanup_temps(); torch.cuda.empty_cache()
            print()
        total_elapsed = time.time()-batch_start
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE — {total_elapsed/60:.1f} min total")
        print(f"{'='*60}")
        ok = sum(1 for _,s,_ in results if s=="OK")
        print(f"  ✓ Succeeded : {ok}/{total_jobs}")
        print(f"  ✗ Failed    : {sum(1 for _,s,_ in results if s.startswith('ERROR'))}")
        print(f"  ⚠ Skipped  : {sum(1 for _,s,_ in results if 'SKIP' in s or 'MISS' in s)}")
        print(f"\nPer-video results:")
        for job,status,elapsed in results:
            name=os.path.basename(job.input_video); t=f"{elapsed:.0f}s" if elapsed else "—"
            print(f"  {status:20s} {t:6s}  {name}")
        print(f"{'='*60}\n")
        return results


# ============================================================
# v8 AUTOMATION LAYER
# ============================================================

# ──────────────────────────────────────────────────────────────
# ClipDownloader — yt-dlp wrapper for YouTube downloads
# ──────────────────────────────────────────────────────────────
class ClipDownloader:
    """Downloads audio-only and clip sections from YouTube URLs."""

    WORK_DIR = "/kaggle/working"

    def __init__(self, url: str):
        self.url = url
        self._title = None
        self._duration = None

    def get_metadata(self) -> Dict:
        """Fetch video title and duration via yt-dlp --print."""
        if self._title is not None:
            return {"title": self._title, "duration": self._duration}
        try:
            cmd = [
                "yt-dlp", "--print", "%(title)s|||%(duration)s",
                "--skip-download", self.url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("|||")
                self._title = parts[0] if len(parts) > 0 else "unknown"
                self._duration = float(parts[1]) if len(parts) > 1 and parts[1] != "NA" else 0.0
            else:
                self._title = "unknown"
                self._duration = 0.0
        except Exception as e:
            print(f"  ⚠ Metadata fetch failed: {e}")
            self._title = "unknown"
            self._duration = 0.0
        return {"title": self._title, "duration": self._duration}

    def download_audio(self, output_path: Optional[str] = None) -> Optional[str]:
        """Download audio-only for Whisper transcription (m4a format)."""
        if output_path is None:
            output_path = os.path.join(self.WORK_DIR, "whisper_audio")
        try:
            cmd = [
                "yt-dlp",
                "-f", "bestaudio[ext=m4a]/bestaudio",
                "--extract-audio",
                "--audio-format", "m4a",
                "-o", output_path + ".%(ext)s",
                self.url
            ]
            print(f"  ⬇ Downloading audio for transcription...")
            subprocess.run(cmd, check=True, timeout=600)
            # Find the actual file
            for ext in [".m4a", ".webm", ".opus", ".mp3"]:
                candidate = output_path + ext
                if os.path.exists(candidate):
                    print(f"  ✓ Audio downloaded: {candidate}")
                    return candidate
            print("  ⚠ Audio file not found after download")
            return None
        except Exception as e:
            print(f"  ⚠ Audio download failed: {e}")
            return None

    def download_section(self, start_sec: float, end_sec: float,
                         output_path: Optional[str] = None) -> Optional[str]:
        """Download a specific clip section using yt-dlp --download-sections."""
        if output_path is None:
            output_path = os.path.join(self.WORK_DIR, f"clip_{int(start_sec)}_{int(end_sec)}")
        try:
            section_spec = f"*{start_sec}-{end_sec}"
            cmd = [
                "yt-dlp",
                "--download-sections", section_spec,
                "--force-keyframes-at-cuts",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "-o", output_path + ".%(ext)s",
                self.url
            ]
            print(f"  ⬇ Downloading clip {start_sec:.0f}s–{end_sec:.0f}s...")
            subprocess.run(cmd, check=True, timeout=600)
            for ext in [".mp4", ".mkv", ".webm"]:
                candidate = output_path + ext
                if os.path.exists(candidate):
                    print(f"  ✓ Clip downloaded: {candidate} ({os.path.getsize(candidate)/1024/1024:.1f}MB)")
                    return candidate
            print("  ⚠ Clip file not found after download")
            return None
        except Exception as e:
            print(f"  ⚠ Clip download failed ({start_sec:.0f}s–{end_sec:.0f}s): {e}")
            return None


# ──────────────────────────────────────────────────────────────
# WhisperTranscriber — faster-whisper with CTranslate2 GPU
# ──────────────────────────────────────────────────────────────
class WhisperTranscriber:
    """Transcribes audio using faster-whisper (CTranslate2 GPU backend)."""

    def __init__(self, model_size: str = "large-v3", language: Optional[str] = None):
        self.model_size = model_size
        self.language = language
        self.model = None
        self._loaded = False

    def load(self):
        """Load the Whisper model onto GPU."""
        if self._loaded:
            return
        try:
            from faster_whisper import WhisperModel
            print(f"  ⏳ Loading Whisper {self.model_size} (CTranslate2 GPU)...")
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="float16",
                download_root="/kaggle/working/whisper_models"
            )
            self._loaded = True
            print(f"  ✓ Whisper {self.model_size} loaded on GPU")
        except Exception as e:
            print(f"  ⚠ Whisper GPU load failed, trying CPU: {e}")
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root="/kaggle/working/whisper_models"
                )
                self._loaded = True
                print(f"  ✓ Whisper {self.model_size} loaded on CPU (fallback)")
            except Exception as e2:
                print(f"  ✗ Whisper load failed entirely: {e2}")
                self.model = None
                self._loaded = False

    def transcribe(self, audio_path: str) -> List[Dict]:
        """Transcribe audio file, returning word-level timestamps.

        Returns list of dicts with keys: start, end, text, words
        where words is a list of {start, end, word, probability}.
        """
        if not self._loaded or self.model is None:
            print("  ⚠ Whisper not loaded — cannot transcribe")
            return []
        try:
            print(f"  ⏳ Transcribing {os.path.basename(audio_path)}...")
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=self.language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
                beam_size=5,
                condition_on_previous_text=True,
            )
            segments = []
            for seg in segments_iter:
                words = []
                if seg.words:
                    for w in seg.words:
                        words.append({
                            "start": w.start,
                            "end": w.end,
                            "word": w.word.strip(),
                            "probability": w.probability,
                        })
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "words": words,
                })
            detected_lang = info.language if info.language else "unknown"
            print(f"  ✓ Transcription done: {len(segments)} segments, language={detected_lang}")
            return segments
        except Exception as e:
            print(f"  ⚠ Transcription failed: {e}")
            return []

    def unload(self):
        """Unload the model and free GPU VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc = None
        try:
            import gc as _gc
            gc = _gc
        except ImportError:
            pass
        if gc:
            gc.collect()
        print("  ✓ Whisper model unloaded — VRAM freed")


# ──────────────────────────────────────────────────────────────
# ClipScorer — Two-tier scoring: rule-based + LLM
# ──────────────────────────────────────────────────────────────
class ClipScorer:
    """Scores clip windows using rule-based heuristics + optional LLM."""

    # Hindi/English hook words for Hindi/English code-mixed content
    HOOK_WORDS_HINDI = {
        "शॉकिंग", "जबरदस्त", "अद्भुत", "खतरनाक", "सच", "झूठ", "गरीब",
        "अमीर", "मुफ्त", "फ्री", "सीक्रेट", "ट्रिक", "चमत्कार", "कमाल",
        "बेस्ट", "वर्स्ट", "हकीकत", "सबसे", "पहली", "आखिरी", "बड़ी",
        "छोटी", "सिखो", "जानो", "करो", "मत", "कभी", "आज", "कल",
    }
    HOOK_WORDS_ENGLISH = {
        "shocking", "amazing", "secret", "trick", "hack", "free", "best",
        "worst", "never", "always", "truth", "lie", "must", "should",
        "dangerous", "insane", "crazy", "unbelievable", "incredible",
        "warning", "alert", "breaking", "exposed", "revealed", "proven",
        "money", "profit", "loss", "rich", "poor", "success", "fail",
        "why", "how", "what", "when", "where", "who", "which",
    }
    HOOK_WORDS = HOOK_WORDS_HINDI | HOOK_WORDS_ENGLISH

    def __init__(self, llm_config: Optional[Dict] = None):
        self.llm_config = llm_config or {}
        self._llm_available = bool(self.llm_config.get("api_key", ""))

    def _rule_score(self, text: str, start: float, end: float,
                    prev_text: str = "", next_text: str = "") -> float:
        """Score a clip window using rule-based heuristics (0-100)."""
        score = 0.0
        text_lower = text.lower()
        words = text_lower.split()

        # 1. Hook words (Hindi + English) — high impact
        hook_count = sum(1 for w in words if w in self.HOOK_WORDS or any(hw in w for hw in self.HOOK_WORDS))
        score += min(hook_count * 8.0, 40.0)

        # 2. Question patterns
        question_marks = text.count("?") + text.count("？")
        question_words = sum(1 for qw in ["क्यों", "कैसे", "क्या", "कहां", "कब", "कौन",
                                          "why", "how", "what", "when", "where", "who"]
                            if qw in text_lower)
        score += min((question_marks + question_words) * 6.0, 25.0)

        # 3. Exclamations — emotional intensity
        exclamations = text.count("!") + text.count("！")
        score += min(exclamations * 5.0, 20.0)

        # 4. Topic boundary detection — if text differs significantly from neighbors
        if prev_text:
            overlap = len(set(words) & set(prev_text.lower().split()))
            total = max(len(set(words) | set(prev_text.lower().split())), 1)
            boundary_score = 1.0 - (overlap / total)
            score += boundary_score * 8.0

        # 5. Duration preference — slightly favor 60s clips
        duration = end - start
        if 50 <= duration <= 70:
            score += 5.0
        elif 40 <= duration <= 90:
            score += 3.0

        # 6. Speaking density — more words per second = more engaging
        word_count = len(words)
        if duration > 0:
            density = word_count / duration
            if density > 2.5:
                score += 5.0
            elif density > 1.5:
                score += 3.0

        # 7. Numbers and data — tends to be hook content
        has_numbers = any(c.isdigit() for c in text)
        if has_numbers:
            score += 3.0

        # 8. Emoji-like indicators in text
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FA9F\U00002600-\U000027BF]', text))
        score += min(emoji_count * 2.0, 10.0)

        return min(score, 100.0)

    def _llm_score(self, text: str, start: float, end: float) -> float:
        """Score a clip window using LLM (0-100). Returns 0 if LLM unavailable."""
        if not self._llm_available:
            return 0.0
        api_key = self.llm_config.get("api_key", "")
        base_url = self.llm_config.get("base_url", "https://api.openai.com/v1")
        model = self.llm_config.get("model", "gpt-4o-mini")
        if not api_key:
            return 0.0

        prompt = (
            f"Rate this video clip transcript for YouTube Shorts/Reels viral potential (0-100).\n"
            f"Consider: hook strength, emotional impact, curiosity gap, shareability.\n"
            f"Clip: {start:.0f}s to {end:.0f}s\n"
            f"Transcript: {text}\n\n"
            f"Reply with ONLY a number between 0 and 100. No explanation."
        )
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.3,
                },
                timeout=15,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Extract number from response
                match = re.search(r'(\d+(?:\.\d+)?)', content)
                if match:
                    return float(min(100.0, max(0.0, float(match.group(1)))))
            return 0.0
        except Exception as e:
            print(f"  ⚠ LLM scoring failed: {e}")
            return 0.0

    def score_windows(self, windows: List[Dict]) -> List[Dict]:
        """Score all windows. Two-pass: rule-based on ALL, then LLM on top-50."""
        # Pass 1: Rule-based scoring on all windows
        for w in windows:
            w["rule_score"] = self._rule_score(
                w.get("text", ""),
                w.get("start", 0),
                w.get("end", 0),
                w.get("prev_text", ""),
                w.get("next_text", ""),
            )
            w["llm_score"] = 0.0
            w["combined_score"] = w["rule_score"]

        if not self._llm_available or not windows:
            return windows

        # Pass 2: LLM scoring on top-50 candidates by rule score
        sorted_windows = sorted(windows, key=lambda w: w["rule_score"], reverse=True)
        top_candidates = sorted_windows[:50]

        print(f"  ⏳ LLM scoring top {len(top_candidates)} candidates...")
        for i, w in enumerate(top_candidates):
            w["llm_score"] = self._llm_score(
                w.get("text", ""),
                w.get("start", 0),
                w.get("end", 0),
            )
            # Combined: rule*0.3 + llm*0.7
            w["combined_score"] = w["rule_score"] * 0.3 + w["llm_score"] * 0.7
            if (i + 1) % 10 == 0:
                print(f"    LLM scored {i+1}/{len(top_candidates)}")

        # Propagate combined score to all windows
        for w in windows:
            if "combined_score" not in w or w["llm_score"] == 0.0:
                w["combined_score"] = w["rule_score"] * 0.3 + w.get("llm_score", 0.0) * 0.7

        return windows


# ──────────────────────────────────────────────────────────────
# AutoClipGenerator — Full automation pipeline
# ──────────────────────────────────────────────────────────────
class AutoClipGenerator:
    """Full pipeline: download audio → transcribe → score → select → download clips."""

    WORK_DIR = "/kaggle/working"

    # Caption emoji pairs — one leading + one trailing for eye-catch
    EMOJI_PAIRS = [
        ("🔥", "😳🚨"), ("💰", "😱📈"), ("🤫", "🚨💡"), ("😱", "⚠️🔥"),
        ("💪", "💯⚡"), ("🎯", "🔥👀"), ("⚡", "💥😳"), ("🚀", "🎯💰"),
        ("🧠", "💡🤯"), ("👀", "😳🔥"), ("📈", "💰🚀"), ("🏆", "🔥💯"),
        ("💡", "🧠⚡"), ("⚠️", "🚨😱"), ("🎬", "🔥👀"), ("💯", "💪🔥"),
        ("❌", "😱⚠️"), ("🌟", "🔥💯"), ("📌", "👀💡"), ("🤯", "🔥😳"),
    ]

    # Hinglish caption templates (keyword + Hindi connector)
    HINGLISH_TEMPLATES = [
        "{keyword} की सच्चाई",
        "{keyword} का राज़",
        "{keyword} में ये छुपा है",
        "{keyword} से पहले जान लो",
        "{keyword} कैसे काम करता है",
        "{keyword} का सीक्रेट",
        "{keyword} ने कर दिया हैरान",
        "{keyword} कब तक चलेगा",
        "{keyword} का असली चेहरा",
        "{keyword} से बचो",
        "{keyword} की पूरी कहानी",
        "{keyword} में ये गलती मत करो",
        "{keyword} का जवाब",
        "{keyword} पर विश्वास मत करो",
        "{keyword} का Truth",
        "{keyword} का Real Reason",
        "{keyword} क्यों ज़रूरी है",
        "{keyword} का Best Tarika",
        "{keyword} ने बदल दी Game",
        "{keyword} का Shocking Truth",
    ]


    # v8: English caption templates
    ENGLISH_TEMPLATES = [
        "{keyword} TRUTH Revealed",
        "{keyword} SECRET Exposed",
        "{keyword} You MUST Know",
        "{keyword} HACK Changed Everything",
        "{keyword} MISTAKE to Avoid",
        "The REAL {keyword}",
        "{keyword} SHOCKING Facts",
        "Why {keyword} MATTERS",
        "{keyword} GONE Wrong",
        "{keyword} CHANGED Everything",
        "{keyword} They Don't Tell You",
        "{keyword} NOBODY Talks About",
        "{keyword} Will BLOW Your Mind",
        "Stop Ignoring {keyword}",
        "{keyword} The UGLY Truth",
        "{keyword} What They HIDE",
        "{keyword} RUINED Everything",
        "{keyword} Dark Side EXPOSED",
        "{keyword} You're Doing WRONG",
        "{keyword} CHANGED the Game",
    ]

    # English keywords that often appear in Hindi podcasts (keep UPPER for impact)
    KEYWORD_EXTRACT_PATTERNS = [
        r'\b(money|profit|loss|business|invest|rich|poor|success|fail|income|tax|save|earn)\b',
        r'\b(health|gym|doctor|hospital|medicine|diet|fitness|body|disease|cure)\b',
        r'\b(job|career|salary|company|boss|interview|promotion|fire|hire)\b',
        r'\b(love|relationship|marriage|divorce|cheat|trust|family)\b',
        r'\b(trick|hack|secret|scam|fraud|fake|real|truth|lie|exposed)\b',
        r'\b(danger|risk|safe|warning|alert|shocking|amazing|unbelievable)\b',
        r'\b(startup|AI|tech|app|phone|social media|YouTube|Instagram)\b',
        r'\b(crypto|stock|market|trading|loan|EMI|bank|credit)\b',
        r'\b(study|exam|college|student|education|school)\b',
        r'\b(law|court|police|crime|jail|justice|case)\b',
    ]

    def __init__(self, pipeline_config: Dict, batch_config: BatchConfig):
        self.config = pipeline_config
        self.batch_config = batch_config
        self.min_clips = pipeline_config.get("min_clips", 7)
        self.max_clips = pipeline_config.get("max_clips", 12)
        self.min_clip_sec = pipeline_config.get("min_clip_sec", 30)
        self.max_clip_sec = pipeline_config.get("max_clip_sec", 90)
        self.whisper_model = pipeline_config.get("whisper_model", "large-v3")
        self.language = pipeline_config.get("language", None)
        self.llm_config = pipeline_config.get("llm_config", {})
        self.source_url = pipeline_config.get("source_url", "")

    def _create_windows(self, segments: List[Dict], video_duration: float) -> List[Dict]:
        """Create sliding windows at 15-second steps with multiple durations."""
        step = 15  # seconds
        durations = [45, 60, 75, 90]
        windows = []

        # Build full text with timestamps for neighbor lookup
        all_text_by_time = {}
        for seg in segments:
            all_text_by_time[seg["start"]] = seg.get("text", "")

        for dur in durations:
            if dur < self.min_clip_sec or dur > self.max_clip_sec:
                continue
            start = 0.0
            while start + dur <= video_duration:
                end = start + dur
                # Gather text for this window from segments
                window_text_parts = []
                for seg in segments:
                    if seg["start"] < end and seg["end"] > start:
                        window_text_parts.append(seg.get("text", ""))
                window_text = " ".join(window_text_parts)

                # Get neighboring text for boundary detection
                prev_text_parts = []
                next_text_parts = []
                for seg in segments:
                    if seg["end"] <= start and seg["end"] > start - 30:
                        prev_text_parts.append(seg.get("text", ""))
                    if seg["start"] >= end and seg["start"] < end + 30:
                        next_text_parts.append(seg.get("text", ""))

                windows.append({
                    "start": start,
                    "end": end,
                    "duration": dur,
                    "text": window_text,
                    "prev_text": " ".join(prev_text_parts),
                    "next_text": " ".join(next_text_parts),
                })
                start += step

        return windows

    def _select_non_overlapping(self, scored_windows: List[Dict],
                                 min_clips: int, max_clips: int) -> List[Dict]:
        """Select top-N non-overlapping clips with progressive threshold lowering."""
        # Sort by combined score descending
        sorted_windows = sorted(scored_windows, key=lambda w: w["combined_score"], reverse=True)

        selected = []
        for threshold in [70, 50, 35, 25, 15, 5, 0]:
            candidates = [w for w in sorted_windows if w["combined_score"] >= threshold]
            for w in candidates:
                # Check overlap with already selected clips
                overlaps = False
                for s in selected:
                    if w["start"] < s["end"] and w["end"] > s["start"]:
                        overlaps = True
                        break
                if not overlaps:
                    selected.append(w)
                    if len(selected) >= max_clips:
                        break
            if len(selected) >= min_clips:
                break

        # Sort selected by start time for logical ordering
        selected.sort(key=lambda w: w["start"])
        return selected

    def _extract_keyword(self, text: str) -> str:
        """Extract the most impactful English keyword from transcript text."""
        text_lower = text.lower()
        # Try each keyword pattern in priority order
        for pattern in self.KEYWORD_EXTRACT_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        # Fallback: find any capitalized English word (3+ chars)
        eng_words = re.findall(r'\b([A-Za-z]{3,})\b', text)
        if eng_words:
            # Pick the most frequent or first meaningful one
            from collections import Counter
            freq = Counter(w.upper() for w in eng_words if len(w) >= 3)
            # Filter out common filler words
            fillers = {"THE", "AND", "BUT", "FOR", "NOT", "YOU", "ARE", "WAS",
                       "WERE", "THIS", "THAT", "WITH", "FROM", "HAVE", "HAS",
                       "THEY", "THEM", "BEEN", "WILL", "WOULD", "COULD", "SHOULD",
                       "THERE", "THEIR", "WHAT", "WHICH", "WHEN", "YOUR", "JUST"}
            for word, count in freq.most_common(10):
                if word not in fillers:
                    return word
        return "THIS"

    def _extract_hindi_phrase(self, text: str, max_chars: int = 25) -> str:
        """Extract a short Hindi phrase from the transcript text."""
        # Find Hindi (Devanagari) words in the text
        hindi_words = re.findall(r'[\u0900-\u097F]+', text)
        if not hindi_words:
            return ""
        # Try to build a meaningful short phrase (2-3 Hindi words)
        # Find contiguous Hindi segments
        hindi_segments = re.findall(r'[\u0900-\u097F]+(?:\s+[\u0900-\u097F]+)*', text)
        for seg in hindi_segments:
            words = seg.split()
            if 1 <= len(words) <= 4:
                phrase = seg.strip()
                if len(phrase) <= max_chars:
                    return phrase
        # Fallback: take first 2 Hindi words
        if len(hindi_words) >= 2:
            return f"{hindi_words[0]} {hindi_words[1]}"
        return hindi_words[0] if hindi_words else ""

    def _generate_caption(self, text: str, clip_idx: int) -> str:
        """v8: Auto-generate caption with language support (hinglish/english).

        Hinglish: "KEYWORD हिंदी_phrase 😳🚨" — English keyword UPPER + Hindi phrase + emoji pair.
        English: "KEYWORD TRUTH Revealed 🔥🚨" — English keyword UPPER + English phrase + emoji pair.
        """
        # v8: Determine max chars based on language
        lang = getattr(self.batch_config, 'caption_language', 'hinglish')
        MAX_CHARS = 50 if lang == "english" else 40  # v8: 50 for english, 40 for hinglish
        lead_emoji, trail_emoji = self.EMOJI_PAIRS[clip_idx % len(self.EMOJI_PAIRS)]

        # If LLM is available, try LLM caption first
        if self.llm_config.get("api_key", ""):
            llm_caption = self._llm_caption(text, clip_idx)
            if llm_caption and len(llm_caption) <= MAX_CHARS:
                return llm_caption
            elif llm_caption:
                # Truncate smartly at word boundary
                llm_caption = llm_caption[:MAX_CHARS - 3].rsplit(" ", 1)[0] + "..."
                return llm_caption

        # v8: Language-aware rule-based caption generation
        keyword = self._extract_keyword(text)
        if lang == "english":
            # v8: English templates — no Devanagari
            template = self.ENGLISH_TEMPLATES[clip_idx % len(self.ENGLISH_TEMPLATES)]
            caption_body = template.format(keyword=keyword)
        else:
            # Hinglish templates (default)
            template = self.HINGLISH_TEMPLATES[clip_idx % len(self.HINGLISH_TEMPLATES)]
            caption_body = template.format(keyword=keyword)

        # Build full caption with emojis
        caption = f"{lead_emoji} {caption_body} {trail_emoji}"

        # Hard enforce char limit — trim body if needed
        emoji_overhead = len(f"{lead_emoji} ") + len(f" {trail_emoji}")
        max_body = MAX_CHARS - emoji_overhead
        if len(caption_body) > max_body:
            if lang == "english":
                # v8: Shorter English fallbacks
                short_templates = [
                    f"{keyword} TRUTH",
                    f"{keyword} SECRET",
                    f"{keyword} HACK",
                    f"{keyword} ALERT",
                    f"{keyword} ⚡",
                ]
            else:
                short_templates = [
                    f"{keyword} का Truth",
                    f"{keyword} की सच्चाई",
                    f"{keyword} का राज़",
                    f"{keyword} से बचो",
                    f"{keyword} ⚡",
                ]
            for st in short_templates:
                test = f"{lead_emoji} {st} {trail_emoji}"
                if len(test) <= MAX_CHARS:
                    return test
            # Last resort: just keyword + emojis
            caption = f"{lead_emoji} {keyword} {trail_emoji}"

        return caption

    def _llm_caption(self, text: str, clip_idx: int) -> Optional[str]:
        """Generate a Hinglish caption using LLM. Returns None if unavailable."""
        api_key = self.llm_config.get("api_key", "")
        base_url = self.llm_config.get("base_url", "https://api.openai.com/v1")
        model = self.llm_config.get("model", "gpt-4o-mini")
        if not api_key:
            return None

        # Take first 300 chars of transcript to keep prompt small
        snippet = text.strip()[:300]
        lead_emoji, trail_emoji = self.EMOJI_PAIRS[clip_idx % len(self.EMOJI_PAIRS)]

        prompt = (
            f"Create a viral Hindi+English (Hinglish) Instagram Reel caption for this video clip.\n"
            f"RULES:\n"
            f"- MAXIMUM 45 characters total (including emojis)\n"
            f"- Format: English KEYWORD in UPPER CASE + Hindi phrase in Devanagari + 1-2 emojis\n"
            f"- Style examples: 'GYM Trainers की सच्चाई 😳🚨', 'MONEY का Real Secret 🤫💰', 'BUSINESS में ये गलती ❌😱'\n"
            f"- The English keyword should be the main topic (e.g. MONEY, HEALTH, JOB, LOVE, SUCCESS)\n"
            f"- The Hindi phrase should be short (2-3 words max)\n"
            f"- Start with emoji {lead_emoji}, end with {trail_emoji}\n"
            f"- Do NOT use full English sentences\n"
            f"- Reply with ONLY the caption, nothing else\n"
            f"\nTranscript: {snippet}"
        )
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 30,
                    "temperature": 0.7,
                },
                timeout=15,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Clean up: remove quotes if LLM wrapped them
                content = content.strip('"\'`')
                # Verify it's under 50 chars
                if len(content) <= 50 and content:
                    return content
                # Try to salvage by truncating
                if content:
                    return content[:47].rsplit(" ", 1)[0] + "..."
            return None
        except Exception as e:
            print(f"  ⚠ LLM caption failed: {e}")
            return None

    def _batch_llm_captions(self, clips: List[Dict]) -> Dict[int, str]:
        """Generate Hinglish captions for ALL clips in a single LLM call.

        Returns {clip_index: caption_string} — much faster than per-clip calls.
        Falls back to per-clip if batch fails.
        """
        api_key = self.llm_config.get("api_key", "")
        base_url = self.llm_config.get("base_url", "https://api.openai.com/v1")
        model = self.llm_config.get("model", "gpt-4o-mini")
        if not api_key:
            return {}

        # Build numbered clip list for the prompt
        clip_lines = []
        for i, clip in enumerate(clips):
            snippet = clip.get("text", "").strip()[:200]
            lead_emoji, trail_emoji = self.EMOJI_PAIRS[i % len(self.EMOJI_PAIRS)]
            clip_lines.append(f"{i+1}. [{lead_emoji}...{trail_emoji}] {snippet}")

        clips_text = "\n".join(clip_lines)

        # v8: Language-aware batch prompt
        lang = getattr(self.batch_config, 'caption_language', 'hinglish')
        max_chars = 50 if lang == "english" else 40  # v8: different limits

        if lang == "english":
            prompt = (
                f"Create viral ENGLISH Instagram Reel captions for these {len(clips)} video clips.\n"
                f"RULES for EACH caption:\n"
                f"- MAXIMUM {max_chars} characters (including emojis)\n"
                f"- Format: English KEYWORD in UPPER CASE + short English phrase + 1-2 emojis\n"
                f"- Style: 'MONEY Secret EXPOSED 🔥🚨', 'HEALTH Hack Nobody Tells You 💪😳'\n"
                f"- Use the emoji pair shown in brackets for each clip\n"
                f"- Do NOT use Hindi/Devanagari — English ONLY\n"
                f"- Reply with ONLY numbered captions, one per line:\n"
                f"1. caption here\n2. caption here\n...\n\n"
                f"Clips:\n{clips_text}"
            )
        else:
            prompt = (
                f"Create viral Hinglish (Hindi+English) Instagram Reel captions for these {len(clips)} video clips.\n"
                f"RULES for EACH caption:\n"
                f"- MAXIMUM {max_chars} characters (including emojis)\n"
                f"- Format: English KEYWORD in UPPER CASE + short Hindi phrase in Devanagari + 1-2 emojis\n"
                f"- Style: 'GYM Trainers की सच्चाई 😳🚨', 'MONEY का Real Secret 🤫💰', 'BUSINESS में ये गलती ❌😱'\n"
                f"- Use the emoji pair shown in brackets for each clip\n"
                f"- Do NOT use full English sentences — Hinglish only\n"
                f"- Reply with ONLY numbered captions, one per line:\n"
                f"1. caption here\n2. caption here\n...\n\n"
                f"Clips:\n{clips_text}"
            )
        try:
            print(f"  ⏳ Batch LLM caption generation for {len(clips)} clips...")
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                captions = {}
                for line in content.split("\n"):
                    line = line.strip().strip('"\'`')
                    if not line:
                        continue
                    # Parse "1. caption" or "1) caption" format
                    match = re.match(r'^(\d+)[.)]\s*(.+)', line)
                    if match:
                        idx = int(match.group(1)) - 1  # 0-based
                        caption = match.group(2).strip().strip('"\'`')
                        # v8: Use language-specific char limit
                        if len(caption) <= max_chars and caption:
                            captions[idx] = caption
                        elif caption:
                            captions[idx] = caption[:max_chars - 3].rsplit(" ", 1)[0] + "..."
                print(f"  ✓ Batch LLM captions: {len(captions)}/{len(clips)} generated")
                return captions
            return {}
        except Exception as e:
            print(f"  ⚠ Batch LLM captions failed: {e}")
            return {}

    def generate(self) -> Tuple[List[VideoJob], Dict]:
        """Full pipeline: download → transcribe → score → select → download clips.

        Returns (list of VideoJob, manifest dict for review).
        """
        print("\n" + "="*60)
        print("AUTOCLIP GENERATOR — Starting automation pipeline")
        print("="*60)

        # Step 1: Download audio for transcription
        downloader = ClipDownloader(self.source_url)
        metadata = downloader.get_metadata()
        video_title = metadata.get("title", "unknown")
        video_duration = metadata.get("duration", 0.0)
        print(f"  Video: {video_title}")
        print(f"  Duration: {video_duration:.0f}s ({video_duration/60:.1f} min)")

        if video_duration < self.min_clip_sec:
            print(f"  ⚠ Video too short ({video_duration:.0f}s < {self.min_clip_sec}s minimum)")
            return [], {}

        audio_path = downloader.download_audio()
        if not audio_path:
            print("  ✗ Audio download failed — cannot proceed")
            return [], {}

        # Step 2: Transcribe with Whisper
        transcriber = WhisperTranscriber(
            model_size=self.whisper_model,
            language=self.language,
        )
        transcriber.load()
        segments = transcriber.transcribe(audio_path)

        # Unload Whisper immediately to free VRAM
        transcriber.unload()

        if not segments:
            print("  ✗ No transcription segments — cannot proceed")
            return [], {}

        # Save transcript for review
        transcript_path = os.path.join(self.WORK_DIR, "full_transcript.json")
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Full transcript saved: {transcript_path}")
        except Exception as e:
            print(f"  ⚠ Could not save transcript: {e}")

        # Step 3: Create sliding windows
        windows = self._create_windows(segments, video_duration)
        print(f"  Created {len(windows)} sliding windows")

        # Step 4: Score all windows
        scorer = ClipScorer(llm_config=self.llm_config)
        scored_windows = scorer.score_windows(windows)

        # Step 5: Select top non-overlapping clips
        selected = self._select_non_overlapping(scored_windows, self.min_clips, self.max_clips)
        print(f"  Selected {len(selected)} clips from {len(scored_windows)} windows")

        if not selected:
            print("  ⚠ No clips selected — try lowering thresholds")
            return [], {}

        # Step 5b: Batch-generate Hinglish captions (1 LLM call for all clips)
        llm_captions = self._batch_llm_captions(selected)

        # Step 6: Download clip sections + assign captions
        jobs = []
        manifest_clips = []
        for i, clip in enumerate(selected):
            clip_filename = f"autoclip_{i+1:02d}_{int(clip['start'])}s_{int(clip['end'])}s"
            clip_path = downloader.download_section(
                clip["start"], clip["end"],
                output_path=os.path.join(self.WORK_DIR, clip_filename)
            )
            if not clip_path:
                print(f"  ⚠ Clip {i+1} download failed — skipping")
                continue

            # Use LLM caption if available, else rule-based Hinglish
            if i in llm_captions:
                caption = llm_captions[i]
            else:
                caption = self._generate_caption(clip.get("text", ""), i)
            output_path = os.path.join(self.WORK_DIR, f"output_{clip_filename}.mp4")

            job = VideoJob(
                input_video=clip_path,
                output_video=output_path,
                card_text=caption,
                color_grading_preset=self.batch_config.color_grading_preset,
                color_grading_intensity=self.batch_config.color_grading_intensity,
                delete_input_after=True,
            )
            jobs.append(job)

            manifest_clips.append({
                "clip_index": i + 1,
                "start": clip["start"],
                "end": clip["end"],
                "duration": clip["duration"],
                "rule_score": clip.get("rule_score", 0),
                "llm_score": clip.get("llm_score", 0),
                "combined_score": clip.get("combined_score", 0),
                "caption": caption,
                "text_preview": clip.get("text", "")[:200],
                "input_video": clip_path,
                "output_video": output_path,
            })

        # Step 7: Save clips manifest for review
        manifest = {
            "source_url": self.source_url,
            "video_title": video_title,
            "video_duration": video_duration,
            "whisper_model": self.whisper_model,
            "total_windows": len(scored_windows),
            "selected_clips": len(manifest_clips),
            "clips": manifest_clips,
        }
        manifest_path = os.path.join(self.WORK_DIR, "clips_manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Clips manifest saved: {manifest_path}")
        except Exception as e:
            print(f"  ⚠ Could not save manifest: {e}")

        # Clean up audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"  ✓ Cleaned up audio file")
            except Exception:
                pass

        print(f"\n  ✓ AutoClipGenerator complete: {len(jobs)} clips ready for processing")
        return jobs, manifest


# ──────────────────────────────────────────────────────────────
# QCGen — Thumbnail contact sheet generator
# ──────────────────────────────────────────────────────────────
class QCGen:
    """Generates FFmpeg-style thumbnail contact sheets using OpenCV."""

    @staticmethod
    def generate_grid(video_paths: List[str],
                      output_path: str = "/kaggle/working/QC_PREVIEW.png",
                      cols: int = 3, rows: int = 3,
                      thumb_w: int = 360, thumb_h: int = 640):
        """Create a 3x3 grid of evenly-spaced frames per video, stacked vertically.

        For each video:
          - Extract cols*rows frames at evenly spaced timestamps
          - Arrange in a grid with timestamp labels and filename
          - Stack all grids vertically into a single QC_PREVIEW.png
        """
        if not video_paths:
            print("  ⚠ No videos for QC grid")
            return

        grids = []
        for vp in video_paths:
            if not os.path.exists(vp):
                print(f"  ⚠ QC: missing {os.path.basename(vp)}")
                continue
            try:
                cap = cv2.VideoCapture(vp)
                if not cap.isOpened():
                    print(f"  ⚠ QC: cannot open {os.path.basename(vp)}")
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                duration = total_frames / fps if fps > 0 else 0
                n_thumbs = cols * rows
                # Evenly spaced frame indices
                if total_frames <= n_thumbs:
                    indices = list(range(total_frames))
                else:
                    indices = [int(i * total_frames / n_thumbs) for i in range(n_thumbs)]

                thumbs = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        resized = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_LANCZOS4)
                        # Add timestamp overlay
                        ts = idx / fps if fps > 0 else 0
                        ts_text = f"{int(ts//60):02d}:{int(ts%60):02d}"
                        font_scale = 0.5
                        thickness = 1
                        cv2.putText(resized, ts_text, (8, thumb_h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (255, 255, 255), thickness, cv2.LINE_AA)
                        # Background for timestamp
                        (tw, th_text), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(resized, (4, thumb_h - 10 - th_text - 4),
                                      (12 + tw, thumb_h - 6), (0, 0, 0), -1)
                        cv2.putText(resized, ts_text, (8, thumb_h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (255, 255, 255), thickness, cv2.LINE_AA)
                        thumbs.append(resized)
                    else:
                        thumbs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
                cap.release()

                if not thumbs:
                    continue

                # Pad to fill grid if fewer frames than grid slots
                while len(thumbs) < n_thumbs:
                    thumbs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

                # Build grid
                grid_rows = []
                for r in range(rows):
                    row_thumbs = thumbs[r * cols: (r + 1) * cols]
                    row_img = np.hstack(row_thumbs)
                    grid_rows.append(row_img)
                grid = np.vstack(grid_rows)

                # Add filename label at top
                label_h = 40
                label = np.full((label_h, grid.shape[1], 3), 30, dtype=np.uint8)
                fname = os.path.basename(vp)
                cv2.putText(label, fname, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)
                grid_with_label = np.vstack([label, grid])
                grids.append(grid_with_label)

            except Exception as e:
                print(f"  ⚠ QC grid failed for {os.path.basename(vp)}: {e}")

        if not grids:
            print("  ⚠ No QC grids generated")
            return

        # Stack all grids vertically
        final = np.vstack(grids)

        # Add header
        header_h = 50
        header = np.full((header_h, final.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(header, f"QC PREVIEW — {len(video_paths)} video(s)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 200, 255), 2, cv2.LINE_AA)
        final_with_header = np.vstack([header, final])

        try:
            cv2.imwrite(output_path, final_with_header)
            print(f"  ✓ QC preview saved: {output_path} ({final_with_header.shape[1]}x{final_with_header.shape[0]})")
        except Exception as e:
            print(f"  ⚠ Failed to save QC preview: {e}")


# ──────────────────────────────────────────────────────────────
# ManifestLoader — JSON manifest reader
# ──────────────────────────────────────────────────────────────
class ManifestLoader:
    """Reads JSON manifest file and produces BatchConfig + List[VideoJob]."""

    @staticmethod
    def load(manifest_path: str) -> Tuple[BatchConfig, List[VideoJob]]:
        """Load manifest JSON and return (BatchConfig, List[VideoJob]).

        Manifest format:
        {
            "batch": { ... BatchConfig overrides ... },
            "clips": [
                {"url": "...", "start": 120, "end": 210, "caption": "..."},
                {"input_video": "...", "caption": "..."}
            ]
        }
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Build BatchConfig from manifest overrides
        batch_kwargs = {}
        if "batch" in manifest:
            for key, value in manifest["batch"].items():
                if hasattr(BatchConfig, key):
                    batch_kwargs[key] = value
        batch_cfg = BatchConfig(**batch_kwargs)

        # Build VideoJob list
        jobs = []
        for i, clip_def in enumerate(manifest.get("clips", [])):
            input_video = clip_def.get("input_video", "")
            output_video = clip_def.get("output_video", "")

            # If URL provided, download the section first
            url = clip_def.get("url", "")
            start = clip_def.get("start", 0)
            end = clip_def.get("end", 0)

            if url and not input_video:
                downloader = ClipDownloader(url)
                if end > start:
                    clip_filename = f"manifest_clip_{i+1:02d}_{int(start)}s_{int(end)}s"
                    input_video_path = downloader.download_section(
                        start, end,
                        output_path=os.path.join("/kaggle/working", clip_filename)
                    )
                    if input_video_path:
                        input_video = input_video_path
                    else:
                        print(f"  ⚠ Manifest clip {i+1}: download failed, skipping")
                        continue
                else:
                    print(f"  ⚠ Manifest clip {i+1}: invalid start/end, skipping")
                    continue

            if not input_video or not os.path.exists(input_video):
                print(f"  ⚠ Manifest clip {i+1}: input not found, skipping")
                continue

            if not output_video:
                output_video = os.path.join(
                    "/kaggle/working",
                    f"output_manifest_{i+1:02d}.mp4"
                )

            caption = clip_def.get("caption", f"Clip {i+1}")
            job = VideoJob(
                input_video=input_video,
                output_video=output_video,
                card_text=caption,
                card_subtext=clip_def.get("subtext", ""),
                color_grading_preset=clip_def.get("color_grading_preset"),
                color_grading_intensity=clip_def.get("color_grading_intensity", 0.85),
                delete_input_after=clip_def.get("delete_input_after", bool(url)),
            )
            jobs.append(job)

        print(f"  ✓ Manifest loaded: {len(jobs)} clips from {manifest_path}")
        return batch_cfg, jobs


# ──────────────────────────────────────────────────────────────
# PipelineOrchestrator — Three-phase execution
# ──────────────────────────────────────────────────────────────
class PipelineOrchestrator:
    """Three-phase execution: Whisper → GPU Reframe → QC Grid.

    Phase 1: Whisper transcription + LLM scoring + clip download (GPU for Whisper)
    Phase 2: Unload Whisper, then GPU processing of all clips via BatchReframer
    Phase 3: QC thumbnail grid generation (CPU/disk)
    """

    def __init__(self, pipeline_config: Dict, batch_config: BatchConfig):
        self.pipeline_config = pipeline_config
        self.batch_config = batch_config
        self.source_url = pipeline_config.get("source_url", "")

    def run(self):
        """Execute the three-phase pipeline."""
        total_start = time.time()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Whisper transcription + LLM scoring + clip download
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "█"*60)
        print("█  PHASE 1: Transcription + Clip Selection + Download")
        print("█"*60)

        if not self.source_url or "PASTE_YOUR_URL" in self.source_url:
            print("  ✗ No source URL provided — set PIPELINE['source_url']")
            print("  ℹ To use manual mode, comment out PipelineOrchestrator and use ManifestLoader")
            return

        generator = AutoClipGenerator(self.pipeline_config, self.batch_config)
        jobs, manifest = generator.generate()

        if not jobs:
            print("  ✗ No clips generated — pipeline cannot continue")
            return

        print(f"\n  Phase 1 complete: {len(jobs)} clips downloaded and scored")

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: GPU processing of all clips via BatchReframer
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "█"*60)
        print("█  PHASE 2: GPU Reframing + Visual FX")
        print("█"*60)
        print(f"  Processing {len(jobs)} clips with YOLO + TalkNet + Visual FX...")

        # Pre-download verification: all clips must exist before GPU processing
        valid_jobs = []
        for job in jobs:
            if os.path.exists(job.input_video):
                valid_jobs.append(job)
            else:
                print(f"  ⚠ Missing input: {job.input_video} — skipping")
        jobs = valid_jobs

        if not jobs:
            print("  ✗ No valid clips to process")
            return

        # Ensure Whisper VRAM is freed before loading YOLO + TalkNet
        torch.cuda.empty_cache()
        try:
            import gc
            gc.collect()
        except ImportError:
            pass

        runner = BatchReframer(self.batch_config, jobs)
        results = runner.run()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: QC thumbnail grid generation (CPU/disk)
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "█"*60)
        print("█  PHASE 3: QC Thumbnail Grid")
        print("█"*60)

        output_videos = [j.output_video for j, status, _ in results if status == "OK"]
        if output_videos:
            QCGen.generate_grid(output_videos)
        else:
            print("  ⚠ No successful outputs for QC grid")

        # Final summary
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE — {total_elapsed/60:.1f} min total")
        print(f"  Clips processed: {sum(1 for _,s,_ in results if s=='OK')}/{len(jobs)}")
        print(f"  QC preview: /kaggle/working/QC_PREVIEW.png")
        print(f"  Manifest: /kaggle/working/clips_manifest.json")
        print(f"{'='*60}\n")


# ============================================================
# ENTRY POINTS — Modal-callable API + legacy CLI
# ============================================================
#
# This module exposes two callable entry points for the Modal GPU
# container (see modal_app.py):
#
#   1) run_automation(pipeline_config, batch_overrides, secrets) -> dict
#      — Full YouTube URL → auto-select clips → reframe pipeline.
#
#   2) run_manifest(manifest_dict, secrets) -> dict
#      — Manual mode: caller supplies explicit clip definitions.
#
# Both functions:
#   • Replace the Kaggle-specific temp paths with /tmp paths.
#   • Accept secrets (Groq/Pexels API keys) from Modal Secret.
#   • Return a JSON-serializable dict of job results, suitable for
#     streaming back to FastAPI → Supabase.
# ============================================================


# Default automation pipeline config (used when caller omits overrides)
_DEFAULT_PIPELINE = {
    "source_url": "",
    "min_clips": 5,
    "max_clips": 10,
    "min_clip_sec": 30,
    "max_clip_sec": 45,
    "whisper_model": "large-v3",
    "language": "en",
    "caption_language": "hinglish",
    "llm_config": {
        "api_key": "",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3-32b",
    },
}

# Default BatchConfig overrides (used when caller omits overrides)
_DEFAULT_BATCH = {
    "card_theme": "classic_white",
    "caption_language": "hinglish",
    "target_width": 1080,
    "target_height": 1920,
    "min_free_gb": 2.0,
    "yolo_imgsz": 1280,
    "yolo_face_variant": "auto",
    "talknet_enabled": True,
    "talknet_infer_every": 5,
    "talknet_min_frames": 25,
    "talknet_smooth_window": 5,
    "talknet_speaking_threshold": 0.0,
    "talknet_durations": (1, 2, 3),
    "color_grading_preset": "vibrant",
    "color_grading_intensity": 0.85,
    "yolo_confidence": 0.50,
    "size_dominance_ratio": 0.65,
    "max_live_faces": 2,
    "face_edge_reject_ratio": 0.08,
    "face_track_min_frames": 8,
    "split_cold_start_frames": 40,
    "split_face_confidence_frames": 25,
    "split_both_speaking_frames": 10,
    "split_exit_frames": 25,
    "hysteresis_frames": 15,
    "face_stickiness_frames": 10,
    "face_stickiness_size_bonus": 0.35,
    "split_gap_style": "gradient",
    "split_gap_accent_line": False,
    "punch_zoom_enabled": True,
    "speaker_glow_enabled": True,
    "film_grain_enabled": True,
    "split_panel_rounded_corners": True,
    "watermark_enabled": True,
    "watermark_path": "@clipskari",
    "watermark_opacity": 0.4,
    "watermark_position": "top_left",
    "face_beautify_enabled": True,
    "border_glow_enabled": True,
    "letterbox_enabled": False,
    "card_animated_reveal": True,
    "dynamic_color_grading": True,
    "dof_enabled": True,
    "ken_burns_enabled": True,
    "live_caption_enabled": False,
}


def _patch_kaggle_paths():
    """Redirect /kaggle/working paths to /tmp (Modal ephemeral storage)."""
    global TALKNET_REPO
    # TalkNet repo lives in /opt/talknet (mounted by Modal image)
    TALKNET_REPO = "/opt/talknet"
    if not os.path.isdir(TALKNET_REPO):
        TALKNET_REPO = os.environ.get("TALKNET_REPO", "/tmp/TalkNet-ASD")
    sys.path.insert(0, TALKNET_REPO)


def run_automation(
    pipeline_config: Optional[Dict] = None,
    batch_overrides: Optional[Dict] = None,
    secrets: Optional[Dict] = None,
) -> Dict:
    """Run full YouTube URL → auto-selected reframed clips pipeline.

    Args:
        pipeline_config: Dict with keys from _DEFAULT_PIPELINE. At minimum,
            source_url must be set.
        batch_overrides: Dict of BatchConfig field name → value overrides.
        secrets: Dict with optional keys:
            groq_api_key: Groq API key for LLM scoring + caption generation.
            pexels_api_key: Pexels API key for B-roll.

    Returns:
        Dict with keys:
            status: "ok" | "error"
            clips: List of {"output_path", "caption", "score", "duration_sec"}
            qc_grid_path: Optional[str]
            manifest: Dict (the manifest.json content)
            error: Optional[str]
            elapsed_sec: float
    """
    import time as _time
    t0 = _time.time()
    _patch_kaggle_paths()

    secrets = secrets or {}
    pipeline = {**_DEFAULT_PIPELINE, **(pipeline_config or {})}
    pipeline.setdefault("llm_config", {})
    pipeline["llm_config"] = {
        "api_key": secrets.get("groq_api_key", ""),
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3-32b",
    }

    try:
        batch_kwargs = {**_DEFAULT_BATCH, **(batch_overrides or {})}
        batch = BatchConfig(**batch_kwargs)

        orchestrator = PipelineOrchestrator(pipeline, batch)
        orchestrator.run()

        # Read back the manifest produced by AutoClipGenerator
        manifest_path = "/tmp/clips_manifest.json"
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        qc_path = "/tmp/QC_PREVIEW.png"
        clips_out = []
        for clip in manifest.get("clips", []):
            out_path = clip.get("output_video", "")
            if out_path and os.path.exists(out_path):
                clips_out.append({
                    "output_path": out_path,
                    "caption": clip.get("caption", ""),
                    "start_sec": clip.get("start", 0),
                    "end_sec": clip.get("end", 0),
                    "duration_sec": clip.get("end", 0) - clip.get("start", 0),
                    "score": clip.get("score", 0),
                    "size_bytes": os.path.getsize(out_path),
                })

        return {
            "status": "ok",
            "clips": clips_out,
            "qc_grid_path": qc_path if os.path.exists(qc_path) else None,
            "manifest": manifest,
            "elapsed_sec": _time.time() - t0,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": f"{e}\n{traceback.format_exc()}",
            "elapsed_sec": _time.time() - t0,
        }


def run_manifest(
    manifest: Dict,
    secrets: Optional[Dict] = None,
) -> Dict:
    """Run manual mode: caller supplies explicit clip definitions.

    Args:
        manifest: Dict matching the manifest.json schema, e.g.:
            {
              "batch": { ... BatchConfig overrides ... },
              "clips": [
                {"url": "...", "start": 120, "end": 210, "caption": "..."},
                {"input_video": "...", "caption": "..."}
              ]
            }
        secrets: Optional dict with groq_api_key, pexels_api_key.

    Returns:
        Dict with keys:
            status: "ok" | "error"
            clips: List of {"output_path", "caption", "size_bytes"}
            qc_grid_path: Optional[str]
            error: Optional[str]
            elapsed_sec: float
    """
    import time as _time
    t0 = _time.time()
    _patch_kaggle_paths()

    secrets = secrets or {}

    try:
        # Write manifest to /tmp and load via ManifestLoader
        manifest_path = "/tmp/modal_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        batch_cfg, jobs = ManifestLoader.load(manifest_path)
        runner = BatchReframer(batch_cfg, jobs)
        results = runner.run()

        qc_path = "/tmp/QC_PREVIEW.png"
        success_outputs = [j.output_video for j, s, _ in results if s == "OK"]
        if success_outputs:
            QCGen.generate_grid(success_outputs, output_path=qc_path)

        clips_out = []
        for job, status, elapsed in results:
            out_path = job.output_video
            clips_out.append({
                "output_path": out_path,
                "caption": job.card_text,
                "status": status,
                "size_bytes": os.path.getsize(out_path) if os.path.exists(out_path) else 0,
                "elapsed_sec": elapsed,
            })

        return {
            "status": "ok" if any(c["status"] == "OK" for c in clips_out) else "error",
            "clips": clips_out,
            "qc_grid_path": qc_path if os.path.exists(qc_path) else None,
            "elapsed_sec": _time.time() - t0,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": f"{e}\n{traceback.format_exc()}",
            "elapsed_sec": _time.time() - t0,
        }


def cli_main():
    """Legacy CLI entrypoint — used for local testing without Modal."""
    _patch_kaggle_paths()

    # Default demo config — replace source_url to test
    pipeline = {
        **_DEFAULT_PIPELINE,
        "source_url": os.environ.get("TEST_YT_URL", ""),
        "llm_config": {
            "api_key": os.environ.get("GROQ_API_KEY", ""),
            "base_url": "https://api.groq.com/openai/v1",
            "model": "qwen/qwen3-32b",
        },
    }
    result = run_automation(pipeline)
    print("\n" + "=" * 60)
    print("RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli_main()
