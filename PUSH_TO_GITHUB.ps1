$ErrorActionPreference = "Stop"
$origin = git remote get-url origin 2>$null
if (-not $origin) {
    git remote add origin "https://github.com/Sanjay1712KSK/Nvidia-AI-Upscaler-Studio.git"
}
git push -u origin main
