# pack_kaggle.ps1 — Tao file zip de upload len Kaggle
# Usage: .\pack_kaggle.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$OUT  = "$ROOT\spot_rl_kaggle.zip"

Write-Host "=== Spot RL - Pack for Kaggle ===" -ForegroundColor Cyan
Write-Host "Root  : $ROOT"
Write-Host "Output: $OUT"
Write-Host ""

# Kiem tra pkl truoc khi chay
$pklCount = (Get-ChildItem "$ROOT\data\processed" -Filter "multipool_*.csv" -ErrorAction SilentlyContinue).Count
if ($pklCount -lt 4) {
    Write-Host "[!] Only $pklCount multipool_*.csv files found — need 4. Run preprocess first:" -ForegroundColor Red
    Write-Host "    python data/scripts/preprocess.py --per-scenario --instance-types m5.large,c5.xlarge,r5.large,m5.xlarge,c5.2xlarge" -ForegroundColor Yellow
    exit 1
}

# Delegate toan bo viec pack sang Python (cross-platform, no TMP dir needed)
python -c @"
import zipfile, os, sys

ROOT = r'$ROOT'.replace('\\', '/')
OUT  = r'$OUT'.replace('\\', '/')

DIRS = ['agents', 'envs', 'utils', 'experiments']

print('Packing...')
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Source code
    for d in DIRS:
        for root, dirs, files in os.walk(os.path.join(ROOT, d)):
            dirs[:] = [x for x in dirs if x != '__pycache__']
            for f in files:
                if f.endswith('.pyc'): continue
                abs_path = os.path.join(root, f)
                arcname = os.path.relpath(abs_path, ROOT).replace(os.sep, '/')
                zf.write(abs_path, arcname)
        print(f'  [+] {d}/')

    # Data processed — per-scenario CSV. If these files are present, Kaggle can
    # skip DVC entirely and train offline from the uploaded dataset.
    processed = os.path.join(ROOT, 'data', 'processed')
    for f in sorted(os.listdir(processed)):
        if f.startswith('multipool_') and f.endswith('.csv'):
            abs_path = os.path.join(processed, f)
            arcname  = f'data/processed/{f}'
            zf.write(abs_path, arcname)
            print(f'  [+] {arcname}  ({os.path.getsize(abs_path)/1024/1024:.1f} MB)')

    # DVC metadata — needed only when processed CSV files are not bundled or need refresh.
    dvc_items = ['.dvc/config', '.dvcignore', 'data/processed.dvc']
    for item in dvc_items:
        abs_path = os.path.join(ROOT, item)
        if os.path.exists(abs_path):
            zf.write(abs_path, item.replace(os.sep, '/'))
            print(f'  [+] {item}')

    # Notebook
    zf.write(os.path.join(ROOT, 'notebooks', 'kaggle_train.ipynb'), 'notebooks/kaggle_train.ipynb')
    print('  [+] notebooks/kaggle_train.ipynb')

    # Requirements
    zf.write(os.path.join(ROOT, 'requirements.txt'), 'requirements.txt')
    print('  [+] requirements.txt')

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f'\nDone! spot_rl_kaggle.zip ({size_mb:.1f} MB)')
print('\ndata/processed contents:')
with zipfile.ZipFile(OUT) as zf:
    for n in sorted(zf.namelist()):
        if 'processed' in n:
            print(f'  {n}')
"@

Write-Host ""
Write-Host "Upload spot_rl_kaggle.zip len Kaggle -> Datasets -> New Dataset" -ForegroundColor Yellow
