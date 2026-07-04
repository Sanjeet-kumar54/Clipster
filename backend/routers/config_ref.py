"""
Configuration reference router — exposes theme catalog, batch config
defaults, and FX options so the frontend can render a picker.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/config", tags=["config"])


# Mirror of CARD_THEMES in reframer.py — kept in sync manually.
# If reframer.py adds a new theme, add it here too.
THEMES = [
    {
        "id": "classic_white",
        "name": "Classic White",
        "description": "Crisp white card with red accent — corporate clean.",
        "swatch": {"bg": "#FFFFFF", "text": "#0A0A0A", "accent": "#FF2828"},
    },
    {
        "id": "neon_void",
        "name": "Neon Void",
        "description": "Deep purple-black with violet glow — futuristic.",
        "swatch": {"bg": "#0D0D1A", "text": "#F0EEFF", "accent": "#B14FFF"},
    },
    {
        "id": "brat_summer",
        "name": "Brat Summer",
        "description": "Lime green with black text — viral 2024 aesthetic.",
        "swatch": {"bg": "#CAFF3A", "text": "#0D0D0D", "accent": "#0D0D0D"},
    },
    {
        "id": "glazed_donut",
        "name": "Glazed Donut",
        "description": "Warm cream with terracotta accent — soft & friendly.",
        "swatch": {"bg": "#FDEEE7", "text": "#3B1A10", "accent": "#E8845A"},
    },
    {
        "id": "digital_dusk",
        "name": "Digital Dusk",
        "description": "Midnight blue with electric cyan — tech podcast.",
        "swatch": {"bg": "#060F2E", "text": "#E0F2FF", "accent": "#00D4FF"},
    },
    {
        "id": "pink_pill",
        "name": "Pink Pill",
        "description": "Hot pink with white text — bold creator vibe.",
        "swatch": {"bg": "#FF2D78", "text": "#FFFFFF", "accent": "#FFFFFF"},
    },
    {
        "id": "matcha_latte",
        "name": "Matcha Latte",
        "description": "Soft green with olive accent — calm & organic.",
        "swatch": {"bg": "#EFF5E9", "text": "#1A2E10", "accent": "#4A7C30"},
    },
    {
        "id": "midnight_chrome",
        "name": "Midnight Chrome",
        "description": "Charcoal black with silver glow — premium dark.",
        "swatch": {"bg": "#111111", "text": "#F5F5F5", "accent": "#C8C8C8"},
    },
    {
        "id": "burnt_orange",
        "name": "Burnt Orange",
        "description": "Deep black with molten orange — podcast hot.",
        "swatch": {"bg": "#1A0300", "text": "#FFEDCC", "accent": "#FF6B00"},
    },
]


COLOR_GRADING_PRESETS = [
    {"id": "off", "name": "Off"},
    {"id": "cinematic", "name": "Cinematic"},
    {"id": "moody", "name": "Moody"},
    {"id": "vibrant", "name": "Vibrant"},
    {"id": "bleach_bypass", "name": "Bleach Bypass"},
    {"id": "golden_hour", "name": "Golden Hour"},
    {"id": "teal_orange", "name": "Teal & Orange"},
    {"id": "matte", "name": "Matte"},
]


WATERMARK_POSITIONS = [
    {"id": "top_left", "name": "Top Left"},
    {"id": "top_right", "name": "Top Right"},
    {"id": "bottom_left", "name": "Bottom Left"},
    {"id": "bottom_right", "name": "Bottom Right"},
]


VISUAL_FX = [
    {"id": "punch_zoom", "name": "Punch Zoom on Speaker Switch", "default": True},
    {"id": "speaker_glow", "name": "Speaker Glow Ring", "default": True},
    {"id": "film_grain", "name": "Film Grain Overlay", "default": True},
    {"id": "border_waveform", "name": "Audio-Reactive Waveform Border", "default": True},
    {"id": "split_panel_rounded_corners", "name": "Rounded Panel Corners", "default": True},
    {"id": "watermark", "name": "Watermark / Branding", "default": True},
    {"id": "face_beautify", "name": "Face Beautification", "default": True},
    {"id": "border_glow", "name": "Border Glow", "default": True},
    {"id": "letterbox", "name": "Cinematic Letterbox", "default": False},
    {"id": "card_animated_reveal", "name": "Animated Text Reveal", "default": True},
    {"id": "dynamic_color_grading", "name": "Dynamic Color Grading", "default": True},
    {"id": "dof", "name": "Depth-of-Field Blur", "default": True},
    {"id": "ken_burns", "name": "Ken Burns Drift", "default": True},
    {"id": "live_caption", "name": "Live Word-by-Word Caption", "default": False},
    {"id": "video_bulge", "name": "Video Bulge Effect", "default": True},
]


@router.get("/themes")
async def get_themes():
    """Card color theme catalog."""
    return {"themes": THEMES}


@router.get("/color-grading")
async def get_color_grading():
    """Color grading preset catalog."""
    return {"presets": COLOR_GRADING_PRESETS}


@router.get("/watermark-positions")
async def get_watermark_positions():
    return {"positions": WATERMARK_POSITIONS}


@router.get("/visual-fx")
async def get_visual_fx():
    """Toggleable visual effects catalog."""
    return {"effects": VISUAL_FX}


@router.get("/all")
async def get_all_config():
    """All config catalogs in one call (used on job submission page)."""
    return {
        "themes": THEMES,
        "color_grading_presets": COLOR_GRADING_PRESETS,
        "watermark_positions": WATERMARK_POSITIONS,
        "visual_fx": VISUAL_FX,
        "caption_languages": [
            {"id": "hinglish", "name": "Hinglish (Hindi+English mix)"},
            {"id": "english", "name": "English"},
        ],
        "output_resolutions": [
            {"id": "1080x1920", "name": "1080×1920 (Full HD portrait)", "default": True},
            {"id": "2160x3840", "name": "2160×3840 (4K portrait)", "default": False},
            {"id": "720x1280", "name": "720×1280 (HD portrait)", "default": False},
        ],
    }
