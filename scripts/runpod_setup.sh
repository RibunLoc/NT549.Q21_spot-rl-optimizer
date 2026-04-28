#!/bin/bash
# RunPod setup script — run once after pod is created.
# Usage: bash scripts/runpod_setup.sh
#
# Prerequisites on RunPod:
#   - RunPod Pytorch template (CUDA 12.4 + torch pre-installed)
#   - Upload spot_rl_kaggle.zip to /workspace/
#
# After setup, run training with:
#   python experiments/train_runpod.py --episodes 5000

set -e

WORKDIR=${WORKDIR:-/workspace/spot-rl}
ZIP_PATH=${ZIP_PATH:-/workspace/spot_rl_kaggle.zip}

echo "=== RunPod Setup for Spot RL Training ==="

# 1. Verify GPU
echo "--- GPU check ---"
nvidia-smi | head -20

# 2. Create workdir and unzip
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ -f "$ZIP_PATH" ]; then
    echo "--- Unzipping code ---"
    unzip -oq "$ZIP_PATH" -d .
    ls -la
else
    echo "ERROR: $ZIP_PATH not found. Upload spot_rl_kaggle.zip to /workspace/ first."
    exit 1
fi

# 3. Install minimal deps (torch pre-installed in RunPod Pytorch image)
echo "--- Installing deps ---"
pip install --no-cache-dir -q -r requirements-train.txt

# 4. Verify env loads
echo "--- Smoke test ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from envs.spot_orchestrator_env import SpotOrchestratorEnv
from envs.instance_catalog import STATE_DIM
env = SpotOrchestratorEnv(data_path='data/processed/multipool_stable.csv', seed=42)
obs, _ = env.reset()
assert obs.shape[0] == STATE_DIM, f'STATE_DIM mismatch: {obs.shape[0]} vs {STATE_DIM}'
print(f'Env OK: STATE_DIM={STATE_DIM}, obs={obs.shape}')
"

echo ""
echo "=== Setup done. Open JupyterLab and run the notebook: ==="
echo "  notebooks/runpod_train.ipynb"
echo ""
echo "=== MLflow tracking (optional): export before opening Jupyter: ==="
echo "  export MLFLOW_USERNAME=..."
echo "  export MLFLOW_PASSWORD=..."
echo "  export AWS_ACCESS_KEY_ID=..."
echo "  export AWS_SECRET_ACCESS_KEY=..."
