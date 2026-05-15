"""Patch notebooks/runpod_train.ipynb cell 5 with v42=v41 (correct, full v41 reward set)."""
import json

NB_PATH = 'notebooks/runpod_train.ipynb'

NEW_CELL_5 = '''# Step 5: Setup env + agent (v42 = v41 with 5-min step resolution)
#
# Philosophy: v42 reuses ALL v41 hyperparameters (reward, workload, agent).
# The only change is step_minutes: 15 -> 5. Items scaled to keep SAME wallclock
# semantics across step resolution:
#   - MAX_STEPS:        672 @ 15min  -> 2016 @ 5min  (1 week wallclock)
#   - cooldown_steps:   4 @ 15min    -> 12 @ 5min    (60min wallclock)
#   - workload uses *_per_hour / *_hours keys; env auto-scales per steps_per_hour
# Agent hyperparams: epsilon_decay & target_update_freq count GRADIENT steps.
# 3x env steps/episode -> 3x grad steps/episode -> scale both by 3.
from envs.mixed_scenario_env import MixedScenarioEnv
from envs.spot_orchestrator_env import SpotOrchestratorEnv
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS
from agents.dqn_agent import DQNAgent
import inspect, os, torch

_params = list(inspect.signature(DQNAgent.__init__).parameters.keys())
print(f'DQNAgent params: {_params}')

if 'device0' not in dir():
    device0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB')
    torch.backends.cudnn.benchmark = True
else:
    vram_gb = 0
    print('CUDA unavailable -- training will run on CPU')
print(f'Training device: {device0}')

_data_dir = os.path.join(DST, 'data', 'processed')
SCENARIOS = {
    'stable':        os.path.join(_data_dir, 'multipool_stable.csv'),
    'volatile':      os.path.join(_data_dir, 'multipool_volatile.csv'),
    'spike':         os.path.join(_data_dir, 'multipool_spike.csv'),
    'az_divergence': os.path.join(_data_dir, 'multipool_az_divergence.csv'),
}
for name, path in SCENARIOS.items():
    size = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
    print(f'  {name}: {size} KB' if size else f'  {name}: MISSING -- {path}')

STEP_MINUTES = 5
STEPS_PER_HOUR = 60 // STEP_MINUTES   # 12
MAX_STEPS = 2016                       # 1 week @ 5 min (= v41 672 @ 15min)

# v41 workload (jobs/hour wallclock, NOT jobs/step):
#   v41: base_arrival_rate=10 jobs/15min step  = 40 jobs/hour
#   v41: avg_job_duration=3 steps @ 15min      = 45 min  = 0.75 hour
WORKLOAD_CFG = {
    'base_arrival_rate_per_hour': 40.0,
    'peak_multiplier':            1.4,
    'avg_job_duration_hours':     0.75,
    'spike_probability':          0.02,
    'spike_multiplier':           2.0,
}

# v41 reward (full set, copied from main branch). Only cooldown_steps rescaled.
REWARD_CFG = {
    'sla_penalty':              5.5,
    'interrupt_penalty':        0.1,
    'convert_to_od_penalty':    0.4,
    'convert_to_spot_penalty':  0.1,
    'rebalance_penalty':        0.8,
    'concentration_penalty':    0.05,
    'concentration_threshold':  0.8,
    'overprovision_penalty':    0.02,
    'fleet_size_penalty':       0.0025,
    'idle_fleet_penalty':       0.015,
    'empty_fleet_penalty':      1.0,
    'cooldown_steps':           12,    # v41: 4 @ 15min = 60min wallclock
    'provision_spot_bonus':     0.9,
    'sla_bonus':                1.0,
    'sla_bonus_threshold':      0.95,
}

SCENARIO_WEIGHTS = {'stable': 1.0, 'volatile': 1.0, 'spike': 1.0, 'az_divergence': 1.0}

env = MixedScenarioEnv(
    scenario_paths=SCENARIOS,
    max_steps=MAX_STEPS,
    workload_config=WORKLOAD_CFG,
    reward_config=REWARD_CFG,
    weights=SCENARIO_WEIGHTS,
    seed=42,
    step_minutes=STEP_MINUTES,
)
env._loss_alpha = 0.0

_lr_key = 'learning_rate' if 'learning_rate' in _params else 'lr'
_tu_key = 'target_update_freq' if 'target_update_freq' in _params else 'target_update'

# Agent = v41, scale epsilon_decay & target_update_freq x3 (gradient steps).
agent_cfg = {
    'state_dim':      STATE_DIM,
    'action_dim':     N_ACTIONS,
    'hidden_dim':     512,
    'network_type':   'factored',
    _lr_key:          2e-5,
    'gamma':          0.99,
    'batch_size':     512,
    'buffer_size':    500_000,
    'epsilon_start':  1.0,
    'epsilon_end':    0.08,
    'epsilon_decay':  1_350_000,   # v41: 450_000 * 3
    _tu_key:          24_000,      # v41: 8_000 * 3
    'use_per':        True,
    'per_alpha':      0.5,
    'per_beta_start': 0.5,
}
agent = DQNAgent(**agent_cfg, device=device0)

if 'RESUME_CHECKPOINT' not in dir():
    RESUME_CHECKPOINT = None
if 'START_EPISODE' not in dir():
    START_EPISODE = 0
if 'RESUME_LR' not in dir():
    RESUME_LR = None
if RESUME_CHECKPOINT:
    if not os.path.exists(RESUME_CHECKPOINT):
        raise FileNotFoundError(f'RESUME_CHECKPOINT not found: {RESUME_CHECKPOINT}')
    agent.load(RESUME_CHECKPOINT)
    if RESUME_LR is not None:
        for group in agent.optimizer.param_groups:
            group['lr'] = RESUME_LR
        print(f'Override resume LR: {RESUME_LR}')
    print(f'Resumed agent from: {RESUME_CHECKPOINT}')
    print(f'Loaded epsilon={agent.epsilon:.3f}, steps_done={agent.steps_done:,}, start_episode={START_EPISODE}')

N_GRADIENT_STEPS = 1
WARMUP_STEPS = agent_cfg['batch_size'] * 8

if 'SAVE_DIR' not in dir() or 'v42' not in SAVE_DIR:
    SAVE_DIR = os.path.join(DST, 'results', 'runpod_v42_step5min', 'models')
os.makedirs(SAVE_DIR, exist_ok=True)

print(f'\\nEnv: obs={env.observation_space.shape}, act={env.action_space.n}')
print(f'Agent: {agent_cfg["network_type"]} | params={sum(p.numel() for p in agent.q_network.parameters()):,}')
print('v42 = v41 with 5-min step resolution:')
print(f'  step_minutes  = {STEP_MINUTES} ({STEPS_PER_HOUR} steps/hour)')
print(f'  max_steps     = {MAX_STEPS} ({MAX_STEPS / STEPS_PER_HOUR:.0f}h = {MAX_STEPS / STEPS_PER_HOUR / 24:.2f} days)')
print(f'  arrival_rate  = {WORKLOAD_CFG["base_arrival_rate_per_hour"]} jobs/hour (= v41)')
print(f'  job_duration  = {WORKLOAD_CFG["avg_job_duration_hours"]} hour = 45 min (= v41)')
print(f'  cooldown      = {REWARD_CFG["cooldown_steps"]} steps = 60 min wallclock (= v41)')
print(f'  reward        = full v41 set (sla_penalty={REWARD_CFG["sla_penalty"]}, bonus={REWARD_CFG["sla_bonus"]}, ...)')
print(f'  lr            = {agent_cfg[_lr_key]}')
print(f'  batch_size    = {agent_cfg["batch_size"]}')
print(f'  buffer_size   = {agent_cfg["buffer_size"]:,}')
print(f'  epsilon_decay = {agent_cfg["epsilon_decay"]:,} grad steps (v41 450k * 3)')
print(f'  target_update = {agent_cfg[_tu_key]:,} grad steps (v41 8k * 3)')
print(f'  warmup        = {WARMUP_STEPS:,} transitions')
print(f'  SAVE_DIR      = {SAVE_DIR}')
'''

# Cell 7 hyperparameters that should match v41 main
CELL_7_UPDATES = {
    'NUM_EPISODES  = 1500': 'NUM_EPISODES  = 6000',
    'NUM_EPISODES  = 2500': 'NUM_EPISODES  = 6000',
    'MIN_EPISODES  = 700':  'MIN_EPISODES  = 3000',
    'MIN_EPISODES  = 1200': 'MIN_EPISODES  = 3000',
    'CHECKPOINT_EVERY = 200': 'CHECKPOINT_EVERY = 500',
    'EVAL_START    = 300':   'EVAL_START    = 500',
    "run_name = 'v42-runpod-rtx2000ada-step5min-smoke1d'":
        "run_name = 'v42-runpod-rtx2000ada-step5min-v41cfg'",
    "run_name = 'v42-runpod-rtx2000ada-step5min-v41aligned'":
        "run_name = 'v42-runpod-rtx2000ada-step5min-v41cfg'",
    "version  = 'v42-smoke1d'":        "version  = 'v42-step5min-v41cfg'",
    "version  = 'v42-v41aligned'":     "version  = 'v42-step5min-v41cfg'",
    "'features':                'step5min_288steps_1day_smoke',":
        "'features':                'step5min_2016steps_1week_v41cfg',",
    "'features':                'step5min_2016steps_1week_v41aligned',":
        "'features':                'step5min_2016steps_1week_v41cfg',",
}

# Restore REWARD_CFG[...] direct lookups (we now override every key v41 used).
RESTORE_KEYS = ['sla_penalty', 'empty_fleet_penalty', 'provision_spot_bonus',
                'idle_fleet_penalty', 'sla_bonus', 'sla_bonus_threshold',
                'fleet_size_penalty']
# Drop log_params lines that reference keys NOT in our REWARD_CFG anymore
DROP_LINES_CONTAINING = [
    "'initial_spot_instances':  REWARD_CFG",
    "'min_total_instances':     REWARD_CFG",
    "'queue_penalty':           REWARD_CFG",
]

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- Cell 5 ---
target = nb['cells'][5]
assert target['cell_type'] == 'code'
assert 'Step 5: Setup env + agent' in ''.join(target.get('source', []))
target['source'] = NEW_CELL_5.splitlines(keepends=True)
target['outputs'] = []
target['execution_count'] = None

# --- Cell 7 ---
cell7_src = ''.join(nb['cells'][7].get('source', []))
for old, new in CELL_7_UPDATES.items():
    cell7_src = cell7_src.replace(old, new)
# restore direct REWARD_CFG[k] lookups
for k in RESTORE_KEYS:
    cell7_src = cell7_src.replace(
        f"REWARD_CFG.get('{k}', 'v41-default')",
        f"REWARD_CFG['{k}']"
    )
# drop log_params lines for dropped keys
new_lines = []
for ln in cell7_src.splitlines(keepends=True):
    if any(s in ln for s in DROP_LINES_CONTAINING):
        continue
    new_lines.append(ln)
cell7_src = ''.join(new_lines)

nb['cells'][7]['source'] = cell7_src.splitlines(keepends=True)
nb['cells'][7]['outputs'] = []
nb['cells'][7]['execution_count'] = None

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')

print(f'Patched {NB_PATH}: cell 5 + cell 7 aligned with v41 main config.')
