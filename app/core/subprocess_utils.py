"""Cross-platform subprocess options for external tools such as FFmpeg.

The packaged Windows build is a GUI (no-console) executable. Without
``CREATE_NO_WINDOW`` every FFmpeg/nvidia-smi call would flash a console window,
and the frozen process has no valid standard input to inherit. The Windows-only
flag is never passed on Linux.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def no_window_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def run_tool(
    args: list[str], timeout: float = 10.0
) -> subprocess.CompletedProcess[str]:
    """Run a short-lived tool, capturing text output, without a console window."""
    return subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        **no_window_kwargs(),
    )
