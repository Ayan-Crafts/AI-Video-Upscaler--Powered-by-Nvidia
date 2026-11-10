$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "app/ui/main_window.py")) {
    throw "Run this script from an extracted overlay inside the existing repository root."
}

python scripts/patch_ai_ui.py
if ($LASTEXITCODE -ne 0) { throw "UI integration patch failed." }

python -m compileall -q app tests scripts
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

Write-Host "AI pipeline overlay applied. Next run: pip install -e `".[ai,dev,nvidia]`""
