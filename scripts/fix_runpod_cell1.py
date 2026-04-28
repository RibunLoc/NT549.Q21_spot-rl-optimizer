"""Harden cell 1 of runpod_train.ipynb: install deps explicitly for current kernel
and verify pandas/torch are importable BEFORE moving to cell 3.
"""
import json

NB = 'notebooks/runpod_train.ipynb'

CELL1 = '''# Step 1: Install dependencies (RunPod / JupyterLab)
# Use sys.executable to install into the SAME kernel the notebook is using.
# --ignore-installed skips distutils-installed conflicts (blinker etc. in RunPod image).
# No version pins: let pip pick versions compatible with the pre-installed torch/cuda stack.
import sys, subprocess

DEPS = ['gymnasium', 'pandas', 'mlflow', 'tqdm', 'python-dotenv']

print(f'Using Python: {sys.executable}')
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    '--ignore-installed',      # skip distutils conflicts (blinker)
    '--no-deps',                # don't touch torch/numpy/matplotlib already in image
    *DEPS,
])
# Separately ensure scientific stack is present (install only if missing)
for pkg, imp in [('numpy', 'numpy'), ('matplotlib', 'matplotlib'), ('pandas', 'pandas')]:
    try:
        __import__(imp)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

# Torch check (should be pre-installed on RunPod Pytorch image)
try:
    import torch
    print(f'torch available: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except ImportError:
    print('torch missing — install manually via RunPod image template')
    raise

# Verify critical imports
import pandas, numpy, gymnasium, torch
print(f'pandas {pandas.__version__}, numpy {numpy.__version__}, '
      f'gymnasium {gymnasium.__version__}, torch {torch.__version__}')
print('Install OK — safe to proceed to cell 2.')
'''

nb = json.load(open(NB, encoding='utf-8'))
nb['cells'][1]['source'] = CELL1
nb['cells'][1]['outputs'] = []
nb['cells'][1]['execution_count'] = None
json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Hardened cell 1 of {NB}')
