#!/bin/bash
# RunPod container entrypoint
# Runs automatically when pod starts (set as "On-start Command" in RunPod UI)
# or call manually: bash /workspace/spot-rl/scripts/runpod_entrypoint.sh
#
# What it does:
#   1. Copy data/processed CSVs from volume (if present) into workdir
#   2. Smoke-test the env
#   3. Start JupyterLab on port 8888

set -e

WORKDIR=/workspace/spot-rl
DATA_SRC=${DATA_SRC:-/workspace/data/processed}   # volume-mounted data
DATA_DST=$WORKDIR/data/processed

echo "=== Spot RL RunPod Entrypoint ==="
cd $WORKDIR

# 1. Copy processed CSVs from volume if available
if [ -d "$DATA_SRC" ] && [ "$(ls -A $DATA_SRC/*.csv 2>/dev/null)" ]; then
    echo "--- Copying data from $DATA_SRC ---"
    mkdir -p $DATA_DST
    cp $DATA_SRC/*.csv $DATA_DST/
    echo "  Copied: $(ls $DATA_DST/*.csv | wc -l) CSV files"
else
    echo "WARNING: No CSVs at $DATA_SRC"
    echo "  Upload spot_rl_kaggle.zip and run:"
    echo "  unzip spot_rl_kaggle.zip -d /workspace/spot-rl"
fi

# 2. Smoke test
echo ""
echo "--- Smoke test ---"
python -c "
import torch
print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import sys; sys.path.insert(0, '.')
from envs.spot_orchestrator_env import SpotOrchestratorEnv
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS
print(f'STATE_DIM={STATE_DIM}  N_ACTIONS={N_ACTIONS}')
print('Env import OK')
" || echo "WARNING: smoke test failed — check source code"

# 3. Start JupyterLab
echo ""
echo "--- Starting JupyterLab on port 8888 ---"
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir=$WORKDIR \
    2>&1 | tee /workspace/jupyter.log &

echo "JupyterLab started (PID $!)"
echo "Open: http://<pod-ip>:8888"
echo ""
echo "Notebooks:"
echo "  notebooks/kaggle_train.ipynb  — same as Kaggle version"
echo "  notebooks/runpod_train.ipynb  — RunPod-specific paths"
echo ""

# Keep container alive
wait
