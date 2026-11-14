# Superseded by the reproducible V1 packaging pipeline. Kept as a thin wrapper
# so existing instructions keep working. See docs/BUILD_AND_DISTRIBUTION.md.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "..\packaging\windows\build.ps1") @args
exit $LASTEXITCODE
