# scripts/docker_push.ps1
# Build and push training image to Docker Hub (or GHCR)
# Usage: .\scripts\docker_push.ps1 -User yourdockerhubuser
#
# Prerequisites:
#   docker login
#   (optional) set $env:DOCKER_USER before running

param(
    [string]$User = $env:DOCKER_USER,
    [string]$Tag  = "latest"
)

if (-not $User) {
    Write-Host "ERROR: provide -User <dockerhub_username> or set DOCKER_USER env var" -ForegroundColor Red
    exit 1
}

$ErrorActionPreference = "Stop"
$ROOT  = Split-Path -Parent $PSScriptRoot
$IMAGE = "$User/spot-rl-train:$Tag"

Write-Host "=== Spot RL — Build & Push Training Image ===" -ForegroundColor Cyan
Write-Host "Image : $IMAGE"
Write-Host "Root  : $ROOT"
Write-Host ""

Set-Location $ROOT

# Build
Write-Host "--- Building ---" -ForegroundColor Yellow
docker build -f Dockerfile.train -t $IMAGE .
if (-not $?) { Write-Host "Build failed" -ForegroundColor Red; exit 1 }

# Push
Write-Host ""
Write-Host "--- Pushing ---" -ForegroundColor Yellow
docker push $IMAGE
if (-not $?) { Write-Host "Push failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Done! Image pushed: $IMAGE" -ForegroundColor Green
Write-Host ""
Write-Host "=== RunPod setup ===" -ForegroundColor Cyan
Write-Host "1. New Pod -> Custom Image -> Docker Image: $IMAGE"
Write-Host "2. Container Disk: 20GB, Volume Disk: 50GB at /workspace"
Write-Host "3. Expose ports: 8888 (Jupyter)"
Write-Host "4. Environment variables (optional, for MLflow):"
Write-Host "   MLFLOW_USERNAME=..."
Write-Host "   MLFLOW_PASSWORD=..."
Write-Host "   AWS_ACCESS_KEY_ID=..."
Write-Host "   AWS-SECRET_ACCESS_KEY=..."
Write-Host ""
Write-Host "5. After pod starts, open Terminal and run:"
Write-Host "   cd /workspace/spot-rl"
Write-Host "   # Copy data (if not already on volume):"
Write-Host "   # Upload spot_rl_kaggle.zip -> unzip, then copy data/processed/"
Write-Host "   jupyter lab --ip=0.0.0.0 --no-browser --allow-root &"
Write-Host "   # or: python experiments/train.py"
