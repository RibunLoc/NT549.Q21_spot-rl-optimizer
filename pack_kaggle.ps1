# pack_kaggle.ps1
# Tao file zip de upload len Kaggle va chay training
# Usage: .\pack_kaggle.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$OUT  = "$ROOT\spot_rl_kaggle.zip"

Write-Host "=== Spot RL - Pack for Kaggle ===" -ForegroundColor Cyan
Write-Host "Root  : $ROOT"
Write-Host "Output: $OUT"
Write-Host ""

# Xoa zip cu neu co
if (Test-Path $OUT) {
    Remove-Item $OUT
    Write-Host "Removed old zip." -ForegroundColor Yellow
}

# Thu muc tam de chuan bi noi dung
$TMP = "$env:TEMP\spot_rl_pack"
if (Test-Path $TMP) { Remove-Item $TMP -Recurse -Force }
New-Item -ItemType Directory -Path $TMP | Out-Null

# --- 1. Source code ---
$DIRS = @("agents", "envs", "utils", "experiments")
foreach ($d in $DIRS) {
    $src = "$ROOT\$d"
    $dst = "$TMP\$d"
    Copy-Item $src $dst -Recurse
    Get-ChildItem $dst -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
    Write-Host "  [+] $d\" -ForegroundColor Green
}

# --- 2. Data processed (chi .csv) ---
$DATA_DST = "$TMP\data\processed"
New-Item -ItemType Directory -Path $DATA_DST | Out-Null
$csvFiles = Get-ChildItem "$ROOT\data\processed" -Filter "multipool_*.csv"
foreach ($f in $csvFiles) {
    Copy-Item $f.FullName "$DATA_DST\$($f.Name)"
    $sizeMB = [math]::Round($f.Length / 1MB, 1)
    Write-Host "  [+] data/processed/$($f.Name) ($sizeMB MB)" -ForegroundColor Green
}

# --- 3. Notebook ---
$NB_DST = "$TMP\notebooks"
New-Item -ItemType Directory -Path $NB_DST | Out-Null
Copy-Item "$ROOT\notebooks\kaggle_train.ipynb" "$NB_DST\kaggle_train.ipynb"
Write-Host "  [+] notebooks/kaggle_train.ipynb" -ForegroundColor Green

# --- 4. Requirements ---
Copy-Item "$ROOT\requirements.txt" "$TMP\requirements.txt"
Write-Host "  [+] requirements.txt" -ForegroundColor Green

# --- 5. Tao zip bang Python (forward slash, tuong thich Linux/Kaggle) ---
Write-Host ""
Write-Host "Compressing..." -ForegroundColor Cyan

$pyScript = @"
import zipfile, os, sys

tmp = sys.argv[1]
out = sys.argv[2]

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(tmp):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            abs_path = os.path.join(root, file)
            # arcname dung forward slash, bo prefix thu muc tam
            arcname = os.path.relpath(abs_path, tmp).replace(os.sep, '/')
            zf.write(abs_path, arcname)

print(f'Zip created: {out}')
"@

python -c $pyScript $TMP $OUT
Remove-Item $TMP -Recurse -Force

$zipSizeMB = [math]::Round((Get-Item $OUT).Length / 1MB, 1)
Write-Host ""
Write-Host "Done! -> $OUT ($zipSizeMB MB)" -ForegroundColor Cyan
Write-Host ""

# --- 6. Kiem tra noi dung zip ---
Write-Host "Contents:" -ForegroundColor White
$zipEntries = python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); [print(' ',n) for n in z.namelist()]" $OUT

Write-Host ""
Write-Host "Upload spot_rl_kaggle.zip len Kaggle -> Datasets -> New Dataset" -ForegroundColor Yellow
Write-Host "Path tren Kaggle: /kaggle/input/spot-rl-code/spot_rl_kaggle.zip" -ForegroundColor Yellow
