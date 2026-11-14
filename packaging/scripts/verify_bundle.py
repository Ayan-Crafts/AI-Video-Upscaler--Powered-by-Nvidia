"""Inspect a built application directory and run its self-test in isolation.

Static checks: launcher present, bundled FFmpeg/FFprobe (executable on Linux),
licences, Qt platform plugin, CUDA libraries, and no development-only files.

Runtime check: launches the frozen app with ``--self-test`` using a stripped
environment (minimal system PATH, no PYTHONPATH/PYTHONHOME, isolated HOME on
Linux) so a developer's Python or FFmpeg installation cannot satisfy a
missing packaged dependency.

Usage: python verify_bundle.py <app-dir> [--launcher PATH] [--e2e] [--require-gpu] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_cuda_runtime import bundle_report

IS_WINDOWS = sys.platform.startswith("win")
FORBIDDEN = ("pytest", "pytestqt", "ruff", "PyInstaller", "torchvision", "tkinter")


def static_checks(app_dir: Path) -> dict[str, object]:
    internal = app_dir / "_internal"
    exe = app_dir / ("AI Upscaler Studio.exe" if IS_WINDOWS else "ai-upscaler-studio")
    suffix = ".exe" if IS_WINDOWS else ""
    ffmpeg = internal / "ffmpeg" / f"ffmpeg{suffix}"
    ffprobe = internal / "ffmpeg" / f"ffprobe{suffix}"
    # PySide6 wheels keep plugins in PySide6/plugins (Windows) or PySide6/Qt/plugins (Linux).
    plugin_name = "qwindows.dll" if IS_WINDOWS else "libqxcb.so"
    plugin_candidates = [internal / "PySide6" / sub / "platforms" / plugin_name
                         for sub in ("plugins", "Qt/plugins")]
    platform_plugin = next((c for c in plugin_candidates if c.exists()), plugin_candidates[0])
    issues: list[str] = []
    for path in (exe, ffmpeg, ffprobe, platform_plugin, internal / "licenses"):
        if not path.exists():
            issues.append(f"missing: {path.relative_to(app_dir)}")
    if not IS_WINDOWS:
        for path in (exe, ffmpeg, ffprobe):
            if path.exists() and not os.access(path, os.X_OK):
                issues.append(f"not executable: {path.relative_to(app_dir)}")
    for name in FORBIDDEN:
        if (internal / name).exists():
            issues.append(f"development package bundled: {name}")
    dev_files = [p for p in internal.rglob("*") if p.suffix.lower() in {".lib", ".h", ".hpp", ".cuh"}]
    if dev_files:
        issues.append(f"{len(dev_files)} header/import-library files bundled, e.g. {dev_files[0].name}")
    cuda = bundle_report(app_dir)
    issues += [f"missing CUDA library: {stem}" for stem in cuda["missing_cuda_libraries"]]
    # Count symlinks (PyInstaller creates them on Linux) once, not as their targets.
    files = [p for p in app_dir.rglob("*") if p.is_file() and not p.is_symlink()]
    return {
        "launcher": str(exe),
        "platform_plugin": str(platform_plugin.relative_to(app_dir)),
        "file_count": len(files),
        "symlink_count": sum(1 for p in app_dir.rglob("*") if p.is_symlink()),
        "size_bytes": sum(p.lstat().st_size for p in files),
        "issues": issues,
        "ok": not issues,
    }


def isolated_env(home: Path) -> dict[str, str]:
    if IS_WINDOWS:
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        keep = ("SystemRoot", "SystemDrive", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                "LOCALAPPDATA", "APPDATA", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERNAME",
                "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "ProgramData",
                "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles",
                "AI_UPSCALER_STUDIO_MODEL_DIR")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        env["PATH"] = os.pathsep.join([
            rf"{system_root}\System32", system_root, rf"{system_root}\System32\Wbem",
        ])
    else:
        keep = ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "QT_QPA_PLATFORM", "LANG",
                "AI_UPSCALER_STUDIO_MODEL_DIR", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                "APPIMAGE_EXTRACT_AND_RUN")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        env.update(PATH="/usr/bin:/bin", HOME=str(home), LANG=env.get("LANG", "C.UTF-8"))
    return env


def run_self_test(launcher: list[str], *, e2e: bool, require_gpu: bool, timeout: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="aus verify ") as tmp:
        report_path = Path(tmp) / "self-test report.json"
        args = [*launcher, "--self-test", "--report", str(report_path)]
        if e2e:
            args.append("--e2e")
        if require_gpu:
            args.append("--require-gpu")
        result = subprocess.run(args, env=isolated_env(Path(tmp)), capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                stdin=subprocess.DEVNULL, check=False)
        outcome: dict[str, object] = {"command": args, "exit_code": result.returncode,
                                      "stderr_tail": result.stderr[-4000:]}
        if report_path.is_file():
            outcome["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        outcome["ok"] = result.returncode == 0 and bool(outcome.get("report", {}).get("passed"))
        return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("app_dir", type=Path)
    parser.add_argument("--launcher", help="Run this instead of the bundle executable (e.g. an AppImage).")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {}
    if not args.skip_static:
        result["static"] = static_checks(args.app_dir)
    default_launcher = args.app_dir / ("AI Upscaler Studio.exe" if IS_WINDOWS else "ai-upscaler-studio")
    launcher = [args.launcher or str(default_launcher)]
    result["self_test"] = run_self_test(launcher, e2e=args.e2e, require_gpu=args.require_gpu,
                                        timeout=args.timeout)
    result["ok"] = all(section["ok"] for section in result.values() if isinstance(section, dict))
    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    summary = {
        "static_issues": result.get("static", {}).get("issues"),
        "size_bytes": result.get("static", {}).get("size_bytes"),
        "self_test_exit": result["self_test"]["exit_code"],
        "self_test_failures": result["self_test"].get("report", {}).get("failures"),
        "ok": result["ok"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if not result["ok"]:
        print(result["self_test"]["stderr_tail"], file=sys.stderr)
        sys.exit("Bundle verification FAILED")


if __name__ == "__main__":
    main()
