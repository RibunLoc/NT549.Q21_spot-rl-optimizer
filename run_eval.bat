@echo off
cd /d D:\Document\Seminar_MLOps\spot-rl-optimiztion
venv\Scripts\python.exe experiments\evaluate_kaggle.py --model results\kaggle\best_mixed_v41.pth --episodes 30 --output-dir results\kaggle\eval_v41 > results\kaggle\eval_v41_log.txt 2>&1
echo EXIT:%ERRORLEVEL% >> results\kaggle\eval_v41_log.txt
