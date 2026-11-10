$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip
python -m pip install -e ".[dev,nvidia]"
python -m pip install pyinstaller
# Development-only package. Include a pinned PyTorch CUDA runtime, model adapters,
# license-reviewed FFmpeg binaries and verified model manifests before a full AI release.
pyinstaller --noconfirm --clean --windowed --name "NVIDIA-AI-Video-Studio" `
  --collect-all PySide6 --collect-submodules app --paths . app\main.py
Write-Host "Development package created under dist\. This is not yet a full AI installer."
