"""Product identity shared by the application, packaging scripts and installers."""

from __future__ import annotations

APP_NAME = "AI Upscaler Studio"
APP_TAGLINE = "Powered by NVIDIA"
APP_VERSION = "1.0.0"
APP_CODENAME = "Imaginoromancophilia"
APP_SLUG = "ai-upscaler-studio"
APP_AUTHOR = "Sanjay Kumar S"
APP_HOMEPAGE = "https://github.com/Ayan-Crafts/Nvidia-AI-Video-Upscaler"
NVIDIA_DRIVER_URL = "https://www.nvidia.com/en-us/drivers/"

WINDOW_TITLE = f"{APP_NAME} — {APP_TAGLINE}"
USER_AGENT = f"AI-Upscaler-Studio/{APP_VERSION} (model-manager)"


def display_version() -> str:
    return f"V{APP_VERSION} — {APP_CODENAME}"
