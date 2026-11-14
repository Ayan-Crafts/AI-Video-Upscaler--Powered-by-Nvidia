"""Platform-aware locations for resources and per-user data.

Read-only resources (bundled FFmpeg, licences) live next to the code: the
repository root when running from source, or the PyInstaller resource
directory (``sys._MEIPASS``) in packaged builds. The AppImage mount point and
installation directories are read-only, so every writable location below is a
per-user directory:

========== ============================================== ==============================================
Purpose    Windows                                        Linux (XDG)
========== ============================================== ==============================================
models     %LOCALAPPDATA%\\AI Upscaler Studio\\models       $XDG_CACHE_HOME/ai-upscaler-studio/models
logs       %LOCALAPPDATA%\\AI Upscaler Studio\\logs         $XDG_STATE_HOME/ai-upscaler-studio/logs
config     %APPDATA%\\AI Upscaler Studio                   $XDG_CONFIG_HOME/ai-upscaler-studio
data       %LOCALAPPDATA%\\AI Upscaler Studio               $XDG_DATA_HOME/ai-upscaler-studio
temp       %TEMP%\\ai-upscaler-studio                       $TMPDIR/ai-upscaler-studio
========== ============================================== ==============================================
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from app.version import APP_NAME, APP_SLUG

ENV_MODEL_DIR = "AI_UPSCALER_STUDIO_MODEL_DIR"
ENV_FFMPEG_DIR = "AI_UPSCALER_STUDIO_FFMPEG_DIR"
LEGACY_APP_DIR_NAME = "NVIDIA AI Video Studio"


def is_windows() -> bool:
    return os.name == "nt"


def is_frozen() -> bool:
    """True inside a PyInstaller bundle (installer, portable ZIP, AppImage or .deb)."""
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Directory containing read-only application resources."""
    if is_frozen():
        bundle_dir = getattr(sys, "_MEIPASS", None)
        return Path(bundle_dir) if bundle_dir else Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_ffmpeg_dir() -> Path:
    return resource_root() / "ffmpeg"


def icon_path() -> Path:
    """Application icon (bundled under ``icons/``; packaging assets when run from source)."""
    bundled = resource_root() / "icons" / "ai-upscaler-studio.png"
    if bundled.is_file():
        return bundled
    return resource_root() / "packaging" / "common" / "icons" / "ai-upscaler-studio.png"


def licenses_dir() -> Path:
    return resource_root() / "licenses"


def executable_name(name: str) -> str:
    return f"{name}.exe" if is_windows() else name


def _env_path(variable: str, fallback: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value) if value else fallback


def _windows_local_appdata() -> Path:
    return _env_path("LOCALAPPDATA", Path.home() / "AppData" / "Local")


def _windows_roaming_appdata() -> Path:
    return _env_path("APPDATA", Path.home() / "AppData" / "Roaming")


def user_data_dir() -> Path:
    if is_windows():
        return _windows_local_appdata() / APP_NAME
    return _env_path("XDG_DATA_HOME", Path.home() / ".local" / "share") / APP_SLUG


def user_cache_dir() -> Path:
    if is_windows():
        return _windows_local_appdata() / APP_NAME / "cache"
    return _env_path("XDG_CACHE_HOME", Path.home() / ".cache") / APP_SLUG


def user_config_dir() -> Path:
    if is_windows():
        return _windows_roaming_appdata() / APP_NAME
    return _env_path("XDG_CONFIG_HOME", Path.home() / ".config") / APP_SLUG


def model_dir() -> Path:
    """Model checkpoint cache. ``AI_UPSCALER_STUDIO_MODEL_DIR`` overrides it."""
    override = os.environ.get(ENV_MODEL_DIR)
    if override:
        return Path(override)
    if is_windows():
        return _windows_local_appdata() / APP_NAME / "models"
    return user_cache_dir() / "models"


def log_dir() -> Path:
    if is_windows():
        return _windows_local_appdata() / APP_NAME / "logs"
    return _env_path("XDG_STATE_HOME", Path.home() / ".local" / "state") / APP_SLUG / "logs"


def temp_dir() -> Path:
    return Path(tempfile.gettempdir()) / APP_SLUG


def default_output_dir() -> Path:
    videos = Path.home() / "Videos"
    return videos if videos.is_dir() else Path.home()


def legacy_model_dirs() -> list[Path]:
    """Model caches used by pre-1.0 developer builds, reused to avoid re-downloads."""
    if is_windows():
        return [_windows_local_appdata() / LEGACY_APP_DIR_NAME / "models"]
    return [
        _env_path("XDG_DATA_HOME", Path.home() / ".local" / "share")
        / LEGACY_APP_DIR_NAME
        / "models"
    ]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
