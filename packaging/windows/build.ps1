<#
.SYNOPSIS
  Build AI Upscaler Studio for Windows x64: PyInstaller app directory,
  portable ZIP and Inno Setup installer, with verification.

.DESCRIPTION
  Reproducible steps (all versions pinned in packaging/requirements and
  packaging/common):
    1. Create a clean Python 3.12 virtual environment with uv (build\venv-win).
    2. Install the pinned runtime (CUDA 12.6 PyTorch) + build/test tools.
    3. Refuse to continue if PyTorch is a CPU-only build.
    4. Run ruff, pytest and compileall (unless -SkipTests).
    5. Download FFmpeg 9.0.2 (pinned SHA-256), verify encoders and licence.
    6. Stage licences, run PyInstaller (onedir).
    7. Verify the bundle and run its --self-test in a stripped environment.
    8. Create the portable ZIP and the Inno Setup installer.
    9. Write SHA256SUMS.txt and BUILD_MANIFEST-windows-x64.json.

  Outputs go to dist\release (ignored by Git).

.PARAMETER E2E
  Also run the frozen app's end-to-end self-test (downloads models, needs a
  CUDA-capable NVIDIA GPU).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -E2E
#>
param(
    [switch]$SkipTests,
    [switch]$SkipInstaller,
    [switch]$E2E,
    [string]$InnoSetupCompiler = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "==> $Label" -ForegroundColor Cyan
    # Native tools write progress to stderr; judge success by exit code only.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

# Pinned versions.
$Versions = @{}
Get-Content "packaging\common\versions.env" | Where-Object { $_ -match '^\s*([A-Z_]+)=(.*)$' } |
    ForEach-Object { $Versions[$Matches[1]] = $Matches[2].Trim() }
$AppVersion = $Versions["APP_VERSION"]
$PythonVersion = $Versions["PYTHON_VERSION"]

$BuildDir = Join-Path $Repo "build\windows"
$Venv = Join-Path $Repo "build\venv-win"
$Python = Join-Path $Venv "Scripts\python.exe"
$Stage = Join-Path $BuildDir "stage"
$OutDir = Join-Path $Repo "dist\release"
$AppDirName = "AI-Upscaler-Studio"
$PyiDist = Join-Path $BuildDir "pyinstaller-dist"
$AppDir = Join-Path $PyiDist $AppDirName
$Base = "AI-Upscaler-Studio-v$AppVersion-Windows-x64"
New-Item -ItemType Directory -Force $BuildDir, $OutDir | Out-Null

# 1-2. Clean, pinned build environment.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required (https://docs.astral.sh/uv/). Install it, then re-run."
}
Invoke-Checked "Install Python $PythonVersion" { uv python install $PythonVersion }
if (-not (Test-Path $Python)) {
    Invoke-Checked "Create build venv" { uv venv --python $PythonVersion --seed $Venv }
}
Invoke-Checked "Install pinned dependencies" {
    & $Python -m pip install --disable-pip-version-check -q `
        -r packaging\requirements\runtime.txt -r packaging\requirements\build.txt `
        -r packaging\requirements\test.txt
}

# 3. Never package a CPU-only PyTorch.
Invoke-Checked "Verify CUDA PyTorch build" {
    & $Python packaging\scripts\verify_cuda_runtime.py --expected-cuda 12.6 `
        --json (Join-Path $BuildDir "cuda-environment.json")
}

# 4. Tests.
if (-not $SkipTests) {
    Invoke-Checked "ruff" { & $Python -m ruff check . }
    New-Item -ItemType Directory -Force (Join-Path $BuildDir "pytest-tmp") | Out-Null
    Invoke-Checked "pytest" {
        & $Python -m pytest -q -p no:cacheprovider --basetemp (Join-Path $BuildDir "pytest-tmp")
    }
    Invoke-Checked "compileall" { & $Python -m compileall -q app tests scripts packaging\scripts }
}

# 5. FFmpeg.
$FfmpegExtract = Join-Path $BuildDir "ffmpeg-extract"
Invoke-Checked "Download and verify FFmpeg" {
    & $Python packaging\scripts\fetch_binary.py ffmpeg_windows --extract $FfmpegExtract
}
$FfmpegRoot = Get-ChildItem $FfmpegExtract -Directory | Select-Object -First 1
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force (Join-Path $Stage "ffmpeg") | Out-Null
Copy-Item (Join-Path $FfmpegRoot.FullName "bin\ffmpeg.exe") (Join-Path $Stage "ffmpeg")
Copy-Item (Join-Path $FfmpegRoot.FullName "bin\ffprobe.exe") (Join-Path $Stage "ffmpeg")
$FfmpegReport = Join-Path $BuildDir "ffmpeg-report.json"
Invoke-Checked "Verify FFmpeg encoders and licence" {
    & $Python packaging\scripts\verify_ffmpeg.py (Join-Path $Stage "ffmpeg") --json $FfmpegReport
}

# 6. Licences + PyInstaller.
Invoke-Checked "Stage licences" {
    & $Python packaging\scripts\stage_licenses.py (Join-Path $Stage "licenses") `
        --ffmpeg-dir $FfmpegRoot.FullName --ffmpeg-report $FfmpegReport
}
$env:AUS_REPO_ROOT = $Repo
$env:AUS_STAGE_DIR = $Stage
Invoke-Checked "PyInstaller" {
    & $Python -m PyInstaller --noconfirm --clean --log-level WARN `
        --distpath $PyiDist --workpath (Join-Path $BuildDir "pyinstaller-work") `
        packaging\pyinstaller\ai_upscaler_studio.spec
}

# 7. Verification (stripped PATH: no developer Python/FFmpeg).
$VerifyArgs = @("packaging\scripts\verify_bundle.py", $AppDir, "--json", (Join-Path $BuildDir "verify-bundle.json"))
if ($E2E) { $VerifyArgs += @("--e2e", "--require-gpu") }
# Build-time verification uses its own model cache, never the user's.
$env:AI_UPSCALER_STUDIO_MODEL_DIR = Join-Path $BuildDir "verify-models"
try {
    Invoke-Checked "Verify bundle and run self-test" { & $Python @VerifyArgs }
} finally {
    Remove-Item Env:\AI_UPSCALER_STUDIO_MODEL_DIR -ErrorAction SilentlyContinue
}

# 8. Portable ZIP and installer.
Get-ChildItem $OutDir -Filter "$Base*" -ErrorAction SilentlyContinue | Remove-Item -Force
Invoke-Checked "Portable ZIP" {
    & $Python packaging\scripts\make_zip.py $AppDir (Join-Path $OutDir "$Base-Portable.zip") --root "AI Upscaler Studio"
}

$InnoVersion = "not built"
if (-not $SkipInstaller) {
    $Candidates = @(
        $InnoSetupCompiler,
        (Get-Command ISCC.exe -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }),
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
    if (-not $Candidates) { throw "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isdl.php or pass -InnoSetupCompiler." }
    $Iscc = @($Candidates)[0]
    # ISCC.exe carries no version resource; read the installed version from the uninstall key.
    $InnoVersion = "6 (version unknown)"
    foreach ($Hive in "HKCU", "HKLM") {
        $Key = "${Hive}:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"
        $Value = (Get-ItemProperty $Key -ErrorAction SilentlyContinue).DisplayVersion
        if (-not $Value) {
            $Key = "${Hive}:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"
            $Value = (Get-ItemProperty $Key -ErrorAction SilentlyContinue).DisplayVersion
        }
        if ($Value) { $InnoVersion = $Value; break }
    }
    Invoke-Checked "Inno Setup ($InnoVersion)" {
        & $Iscc /Qp "/DAppVersion=$AppVersion" "/DSourceDir=$AppDir" "/DOutputDir=$OutDir" packaging\windows\installer.iss
    }
}

# 9. Checksums and manifest.
$PyiVersion = (& $Python -c "import PyInstaller; print(PyInstaller.__version__)").Trim()
Invoke-Checked "Build manifest" {
    & $Python packaging\scripts\write_manifest.py --platform windows-x64 `
        --environment "Windows $([Environment]::OSVersion.Version) / uv CPython $PythonVersion" `
        --artifacts $OutDir --ffmpeg-report $FfmpegReport `
        --tool "PyInstaller=$PyiVersion" --tool "Inno Setup=$InnoVersion" `
        --output (Join-Path $OutDir "BUILD_MANIFEST-windows-x64.json")
}
$Manifests = @(Get-ChildItem $OutDir -Filter "BUILD_MANIFEST-*.json" | ForEach-Object { $_.FullName })
Invoke-Checked "Merge manifests" {
    & $Python packaging\scripts\write_manifest.py --merge @Manifests --output (Join-Path $OutDir "BUILD_MANIFEST.json")
}
Invoke-Checked "SHA256SUMS" { & $Python packaging\scripts\generate_checksums.py $OutDir }

Write-Host "`nWindows build complete. Artifacts in $OutDir" -ForegroundColor Green
Get-ChildItem $OutDir | Format-Table Name, Length -AutoSize
