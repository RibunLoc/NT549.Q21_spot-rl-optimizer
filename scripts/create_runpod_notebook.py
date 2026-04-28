"""Create notebooks/runpod_train.ipynb from kaggle_train.ipynb by replacing
Kaggle-specific cells with RunPod/JupyterLab generic equivalents.

Also fixes v14 bug `range(168)` → `range(env.max_steps)` in BOTH notebooks.
"""
import json
import shutil
from pathlib import Path

NB_KAGGLE = Path('notebooks/kaggle_train.ipynb')
NB_RUNPOD = Path('notebooks/runpod_train.ipynb')


# RunPod/JupyterLab cells (generic, no /kaggle paths)

CELL2_RUNPOD = '''# Step 2: Setup project path (RunPod / JupyterLab / local)
# Prerequisites:
#   - spot_rl_kaggle.zip already unzipped into current directory
#   - Or this notebook lives inside the unzipped dir (notebooks/runpod_train.ipynb)
import os, sys

# Detect project root: either CWD or parent dir (if notebook sits in notebooks/)
DST = os.getcwd()
if os.path.basename(DST) == 'notebooks':
    DST = os.path.dirname(DST)

assert os.path.exists(os.path.join(DST, 'envs')), \\
    f'envs/ not found in {DST}. Unzip spot_rl_kaggle.zip first.'

# Ensure package __init__.py files exist
for pkg in ['envs', 'agents', 'utils']:
    init = os.path.join(DST, pkg, '__init__.py')
    if not os.path.exists(init):
        open(init, 'w').close()

# Change CWD to project root so `data/processed/*.csv` relative paths work
os.chdir(DST)
sys.path.insert(0, DST)
print(f'Project root: {DST}')
print(f'CWD set to:   {os.getcwd()}')
'''

CELL6_RUNPOD = '''# Step 5b: MLflow setup — credentials from env vars
#
# On RunPod, set these before launching Jupyter:
#   export MLFLOW_USERNAME=...
#   export MLFLOW_PASSWORD=...
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#
# Or set them inline below (NOT recommended — do not commit secrets)
import os, mlflow

mlflow_user = os.environ.get('MLFLOW_USERNAME')
mlflow_pass = os.environ.get('MLFLOW_PASSWORD')

MLFLOW_URI = None
if mlflow_user and mlflow_pass:
    try:
        MLFLOW_URI = f'https://{mlflow_user}:{mlflow_pass}@mlflow.holoc.id.vn'
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment('spot-rl-generalist')
        mlflow.enable_system_metrics_logging()
        mlflow.tracking.MlflowClient().search_experiments()
        print('MLflow connected: mlflow.holoc.id.vn')
        print('System metrics logging: ENABLED (CPU/RAM/GPU)')
        print(f'S3 artifact store ready: {bool(os.environ.get("AWS_ACCESS_KEY_ID"))}')
    except Exception as e:
        print(f'MLflow connect failed: {e}')
        MLFLOW_URI = None
else:
    print('MLflow disabled (no MLFLOW_USERNAME/PASSWORD env vars).')
    print('Training will proceed without experiment tracking.')
'''

CELL7_RUNPOD = '''# Step 5c: (OPTIONAL) Resume from checkpoint
# Set RESUME_CHECKPOINT to a .pth file to continue training
# Example: RESUME_CHECKPOINT = 'results/v13/models/best_mixed.pth'
RESUME_CHECKPOINT = None
SAVE_DIR = os.path.join(DST, 'results', 'v14', 'models')
os.makedirs(SAVE_DIR, exist_ok=True)
print(f'SAVE_DIR: {SAVE_DIR}')

if RESUME_CHECKPOINT:
    agent.load(RESUME_CHECKPOINT)
    print(f'Resumed: {RESUME_CHECKPOINT}')
    print(f'  epsilon={agent.epsilon:.4f}  steps={agent.steps_done:,}')
else:
    print('Fresh training')
'''


def fix_v14_range_bug(nb):
    """Replace hard-coded `range(168)` → `range(env.max_steps)` in training cell."""
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'for step in range(168):' in src:
            cell['source'] = src.replace(
                'for step in range(168):',
                'for step in range(env.max_steps):'
            )
            cell['outputs'] = []
            cell['execution_count'] = None
            return True
    return False


def main():
    # 1. Fix v14 bug in Kaggle notebook
    nb_k = json.load(NB_KAGGLE.open(encoding='utf-8'))
    if fix_v14_range_bug(nb_k):
        json.dump(nb_k, NB_KAGGLE.open('w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print(f'Fixed range(168) in {NB_KAGGLE}')

    # 2. Create RunPod notebook = copy Kaggle + replace Kaggle-specific cells
    shutil.copy(NB_KAGGLE, NB_RUNPOD)
    nb_r = json.load(NB_RUNPOD.open(encoding='utf-8'))

    nb_r['cells'][2]['source'] = CELL2_RUNPOD
    nb_r['cells'][2]['outputs'] = []
    nb_r['cells'][2]['execution_count'] = None

    nb_r['cells'][6]['source'] = CELL6_RUNPOD
    nb_r['cells'][6]['outputs'] = []
    nb_r['cells'][6]['execution_count'] = None

    nb_r['cells'][7]['source'] = CELL7_RUNPOD
    nb_r['cells'][7]['outputs'] = []
    nb_r['cells'][7]['execution_count'] = None

    # Update title markdown
    nb_r['cells'][0]['source'] = (
        '# Spot RL Training -- RunPod / JupyterLab\n'
        'Train DQN generalist agent for AWS Spot Instance optimization.\n\n'
        'This notebook runs on RunPod, Paperspace, local JupyterLab, or any generic env.\n'
        'For Kaggle, use `kaggle_train.ipynb` instead.'
    )

    json.dump(nb_r, NB_RUNPOD.open('w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'Created {NB_RUNPOD}')


if __name__ == '__main__':
    main()
