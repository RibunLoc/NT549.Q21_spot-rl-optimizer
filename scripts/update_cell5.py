"""Update cell 5 for v14: 15-min steps, STATE_DIM=90, scaled hyperparams."""
import json

NB = 'notebooks/kaggle_train.ipynb'

cell5_src = '''# Step 5: Setup env + agent (v14)
from envs.mixed_scenario_env import MixedScenarioEnv
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS
from agents.dqn_agent import DQNAgent
import inspect

_params = list(inspect.signature(DQNAgent.__init__).parameters.keys())
print(f'DQNAgent params: {_params}')

SCENARIOS = {
    'stable':        'data/processed/multipool_stable.csv',
    'volatile':      'data/processed/multipool_volatile.csv',
    'spike':         'data/processed/multipool_spike.csv',
    'az_divergence': 'data/processed/multipool_az_divergence.csv',
}

# v14: 1 step = 15 min, rates scaled ÷4 from hourly equivalent
WORKLOAD_CFG = {
    'base_arrival_rate': 3.75,   # 15/h → 3.75/step
    'peak_multiplier':   2.0,
    'avg_job_duration':  12,     # 3h → 12 steps
}

# v14 reward config — same magnitudes as v13 (per-event penalties, scale-invariant)
REWARD_CFG = {
    'sla_penalty':              2.0,
    'interrupt_penalty':        0.1,
    'convert_to_od_penalty':    0.4,
    'convert_to_spot_penalty':  0.3,
    'rebalance_penalty':        0.8,
    'concentration_penalty':    0.05,
    'concentration_threshold':  0.8,
    'cooldown_steps':           2,    # 30 min (= boot time buffer)
}

env = MixedScenarioEnv(
    scenario_paths=SCENARIOS,
    max_steps=672,                   # 1 week @ 15-min steps
    workload_config=WORKLOAD_CFG,
    reward_config=REWARD_CFG,
    seed=42,
)

_lr_key = 'learning_rate' if 'learning_rate' in _params else 'lr'
_tu_key = 'target_update_freq' if 'target_update_freq' in _params else 'target_update'

# v14 agent: buffer + epsilon_decay scaled for 4× longer episodes
agent_cfg = {
    'state_dim':    STATE_DIM,        # 90
    'action_dim':   N_ACTIONS,        # 121
    'hidden_dim':   512,
    'network_type': 'factored',
    _lr_key:        3e-4,
    'gamma':        0.99,
    'batch_size':   256,
    'buffer_size':  1_000_000,        # was 500K
    'epsilon_start': 1.0,
    'epsilon_end':   0.05,
    'epsilon_decay': 1_200_000,       # was 300K
    _tu_key:        2000,             # was 500 (scale with longer episodes)
}
agent = DQNAgent(**agent_cfg, device=device0)
print(f'Env: obs={env.observation_space.shape}, act={env.action_space.n}')
print(f'Agent: {agent_cfg["network_type"]} | params={sum(p.numel() for p in agent.q_network.parameters()):,}')
print(f'Episode: {env.max_steps} steps (1 week @ 15-min)')
print(f'Reward cfg: cooldown={REWARD_CFG["cooldown_steps"]} (30 min), '
      f'sla={REWARD_CFG["sla_penalty"]}, rebalance={REWARD_CFG["rebalance_penalty"]}')
'''

nb = json.load(open(NB, encoding='utf-8'))
nb['cells'][5]['source'] = cell5_src
nb['cells'][5]['outputs'] = []
nb['cells'][5]['execution_count'] = None
json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Updated cell 5 in {NB}: v14 (15-min step, STATE_DIM=90)')
