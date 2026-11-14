# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec shared by the Windows and Ubuntu builds of AI Upscaler Studio.
#
# One-directory ("onedir") mode is used on purpose: the CUDA PyTorch runtime is
# several GB, and a onefile executable would re-extract it to a temporary
# directory on every launch. The onedir tree is wrapped by Inno Setup (Windows),
# a portable ZIP, an AppImage and a .deb.
#
# Inputs (environment variables set by packaging/windows/build.ps1 and
# packaging/linux/build.sh):
#   AUS_REPO_ROOT  repository root
#   AUS_STAGE_DIR  directory containing ffmpeg/ and licenses/ to bundle

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

REPO = Path(os.environ["AUS_REPO_ROOT"]).resolve()
STAGE = Path(os.environ["AUS_STAGE_DIR"]).resolve()
IS_WINDOWS = sys.platform.startswith("win")
ICONS = REPO / "packaging" / "common" / "icons"

sys.path.insert(0, str(REPO))
from app.version import APP_NAME, APP_VERSION  # noqa: E402

EXE_NAME = APP_NAME if IS_WINDOWS else "ai-upscaler-studio"

hiddenimports = collect_submodules("app") + ["pynvml"]

# Never bundle development/test tooling or unused heavy packages. torchvision
# and triton are not used by the application (no torch.compile, no
# torchvision ops), so they are excluded to keep the bundle smaller.
excludes = [
    "tkinter", "_tkinter", "pytest", "pytestqt", "ruff", "coverage", "IPython", "matplotlib",
    "torchvision", "triton", "torch.utils.tensorboard", "tensorboard", "PyInstaller",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick", "PySide6.QtQml",
    "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf",
]

datas = [
    (str(STAGE / "ffmpeg"), "ffmpeg"),
    (str(STAGE / "licenses"), "licenses"),
    (str(ICONS / "ai-upscaler-studio.png"), "icons"),
]

a = Analysis(
    [str(REPO / "packaging" / "pyinstaller" / "launcher.py")],
    pathex=[str(REPO)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Development artefacts shipped inside the torch wheel are not needed at
# runtime: C++ headers, static/import libraries and CMake files.
_DROP_SUFFIXES = (".h", ".hpp", ".cuh", ".lib", ".a", ".cmake", ".pyi")
_DROP_PARTS = ("torch/include/", "torch/share/cmake/", "torch/test/")
if IS_WINDOWS:
    # torch/bin only holds protoc.exe on Windows. On Linux it holds
    # torch_shm_manager, which `import torch` requires, so it must be kept.
    _DROP_PARTS += ("torch/bin/",)


def _keep(entry) -> bool:
    dest = entry[0].replace("\\", "/")
    if dest.lower().endswith(_DROP_SUFFIXES):
        return False
    return not any(part in dest for part in _DROP_PARTS)


a.datas = [entry for entry in a.datas if _keep(entry)]
a.binaries = [entry for entry in a.binaries if _keep(entry)]

pyz = PYZ(a.pure)

version_file = None
if IS_WINDOWS:
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
    )

    parts = tuple(int(p) for p in APP_VERSION.split(".")) + (0,)
    version_file = VSVersionInfo(
        ffi=FixedFileInfo(filevers=parts, prodvers=parts),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", "Sanjay Kumar S"),
                StringStruct("FileDescription", f"{APP_NAME} — Powered by NVIDIA"),
                StringStruct("FileVersion", APP_VERSION),
                StringStruct("InternalName", "ai-upscaler-studio"),
                StringStruct("OriginalFilename", f"{EXE_NAME}.exe"),
                StringStruct("ProductName", APP_NAME),
                StringStruct("ProductVersion", APP_VERSION),
                StringStruct("LegalCopyright", "GPL-3.0-or-later"),
            ])]),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts CUDA/Qt DLLs; never enable it.
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICONS / "ai-upscaler-studio.ico") if IS_WINDOWS else None,
    version=version_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AI-Upscaler-Studio",
)
