"""Smoke test step_minutes config (v42). Run via Windows venv to avoid segfault."""
import subprocess, os, sys

os.chdir(r"D:\Document\Seminar_MLOps\spot-rl-optimiztion")

script = r"""
import sys, os
sys.path.insert(0, r'D:\Document\Seminar_MLOps\spot-rl-optimiztion')
import numpy as np
from envs.spot_orchestrator_env import SpotOrchestratorEnv

CSV = r'data/processed/multipool_stable.csv'

# ---- legacy v41 path: step_minutes=15 default ----
env15 = SpotOrchestratorEnv(data_path=CSV, max_steps=672, workload_config={'base_arrival_rate': 15.0, 'peak_multiplier': 2.0, 'avg_job_duration': 3}, seed=42)
print(f'[v41 15min] step_minutes={env15.step_minutes} sph={env15.steps_per_hour} step_hours={env15.step_hours}')
obs, info = env15.reset()
total_cost = 0.0
for _ in range(20):
    obs, r, term, trunc, info = env15.step(0)  # HOLD
    total_cost += info['step_cost']
    if term or trunc: break
print(f'[v41 15min] 20-step HOLD cost=${total_cost:.2f}')

# ---- v42 path: step_minutes=5 ----
env5 = SpotOrchestratorEnv(data_path=CSV, max_steps=2016, workload_config={'base_arrival_rate_per_hour': 60.0, 'peak_multiplier': 2.0, 'avg_job_duration_hours': 0.75}, seed=42, step_minutes=5)
print(f'[v42 5min] step_minutes={env5.step_minutes} sph={env5.steps_per_hour} step_hours={env5.step_hours:.4f}')
print(f'[v42 5min] arrival_rate={env5.workload_gen.base_arrival_rate:.3f} jobs/step  job_dur={env5.workload_gen.avg_job_duration} steps')
obs, info = env5.reset()
total_cost = 0.0
for _ in range(60):  # 3x more steps to cover same wallclock as 20×15min
    obs, r, term, trunc, info = env5.step(0)
    total_cost += info['step_cost']
    if term or trunc: break
print(f'[v42 5min] 60-step HOLD cost=${total_cost:.2f}  (should be similar to 15min cost)')

# verify time features re-bucket correctly
env5.reset()
for k in range(env5.steps_per_hour):
    obs, _, _, _, _ = env5.step(0)
# After steps_per_hour steps, hour feature should advance by 1
print(f'[v42 5min] after {env5.steps_per_hour} steps, current_step={env5.current_step}, hour bucket={(env5.current_step // env5.steps_per_hour) % 24}')

# verify reward coeff scaling
print()
print(f'[v41] fleet_size={env15.fleet_size_penalty_coeff:.4f} idle_fleet={env15.idle_fleet_penalty_coeff:.4f} sla_bonus={env15.sla_bonus_coeff:.4f}')
print(f'[v42] fleet_size={env5.fleet_size_penalty_coeff:.4f} idle_fleet={env5.idle_fleet_penalty_coeff:.4f} sla_bonus={env5.sla_bonus_coeff:.4f}')
print(f'[v42] sla_window cap will be {max(10, int(round(2.5 * env5.steps_per_hour)))} (vs {max(10, int(round(2.5 * env15.steps_per_hour)))} for v41)')

# full episode SLA comparison: run 1 random-action episode each, verify reward magnitudes comparable
np.random.seed(0)
def run_ep(env):
    obs, info = env.reset()
    total_r = 0; total_c = 0
    while True:
        a = np.random.randint(0, 121)
        obs, r, term, trunc, info = env.step(a)
        total_r += r; total_c += info['step_cost']
        if term or trunc: break
    return total_r, total_c, info.get('cost', 0), info.get('sla_compliance', 0)

env15b = SpotOrchestratorEnv(data_path=CSV, max_steps=672, workload_config={'base_arrival_rate': 1.25, 'peak_multiplier': 1.4, 'avg_job_duration': 12}, seed=99)
env5b = SpotOrchestratorEnv(data_path=CSV, max_steps=2016, workload_config={'base_arrival_rate_per_hour': 5.0, 'peak_multiplier': 1.4, 'avg_job_duration_hours': 3.0}, seed=99, step_minutes=5)
r15, sc15, c15, s15 = run_ep(env15b)
r5, sc5, c5, s5 = run_ep(env5b)
print(f'[v41 random-act ep] reward={r15:.1f} sum_step_cost={sc15:.1f} cost={c15:.1f} SLA={s15:.1%}')
print(f'[v42 random-act ep] reward={r5:.1f} sum_step_cost={sc5:.1f} cost={c5:.1f} SLA={s5:.1%}')
print('  -> reward magnitudes should be similar order (random-act so noisy, but not 3x off)')

print('OK')
"""
r = subprocess.run([r"venv\Scripts\python.exe", "-c", script], capture_output=True, text=True, timeout=120)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(r.returncode)
