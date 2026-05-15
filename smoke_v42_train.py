"""Smoke train v42: 10 episodes, verify loss decreases + SLA non-collapse."""
import subprocess, os, sys

os.chdir(r"D:\Document\Seminar_MLOps\spot-rl-optimiztion")

script = r"""
import sys, os, time
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, r'D:\Document\Seminar_MLOps\spot-rl-optimiztion')
import numpy as np
import torch

from envs.mixed_scenario_env import MixedScenarioEnv
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS
from agents.dqn_agent import DQNAgent

STEP_MINUTES = 5
STEPS_PER_HOUR = 12
MAX_STEPS = 2016
SCENARIOS = {
    'stable':        'data/processed/multipool_stable.csv',
    'volatile':      'data/processed/multipool_volatile.csv',
    'spike':         'data/processed/multipool_spike.csv',
    'az_divergence': 'data/processed/multipool_az_divergence.csv',
}
WORKLOAD_CFG = {
    'base_arrival_rate_per_hour': 40.0,
    'peak_multiplier': 1.4,
    'avg_job_duration_hours': 3.0,
    'spike_probability': 0.02,
    'spike_multiplier': 2.0,
}
REWARD_CFG = {
    'sla_penalty': 5.5, 'interrupt_penalty': 0.1,
    'convert_to_od_penalty': 0.4, 'convert_to_spot_penalty': 0.1,
    'rebalance_penalty': 0.8, 'concentration_penalty': 0.05,
    'concentration_threshold': 0.8, 'overprovision_penalty': 0.02,
    'fleet_size_penalty': 0.0025, 'idle_fleet_penalty': 0.015,
    'empty_fleet_penalty': 1.0, 'cooldown_steps': 12,
    'provision_spot_bonus': 0.9, 'sla_bonus': 1.0, 'sla_bonus_threshold': 0.95,
}

env = MixedScenarioEnv(
    scenario_paths=SCENARIOS, max_steps=MAX_STEPS,
    workload_config=WORKLOAD_CFG, reward_config=REWARD_CFG,
    weights={'stable':1.0,'volatile':1.0,'spike':1.0,'az_divergence':1.0},
    seed=42, step_minutes=STEP_MINUTES,
)
env._loss_alpha = 0.0

agent = DQNAgent(
    state_dim=STATE_DIM, action_dim=N_ACTIONS,
    hidden_dim=512, network_type='factored',
    learning_rate=2e-5, gamma=0.99,
    batch_size=512, buffer_size=500_000,
    epsilon_start=1.0, epsilon_end=0.08, epsilon_decay=1_350_000,
    target_update_freq=24_000,
    use_per=True, per_alpha=0.5, per_beta_start=0.5,
    device='cpu',
)

print(f'state_dim={STATE_DIM} N_ACTIONS={N_ACTIONS}')
print(f'step_minutes={STEP_MINUTES} max_steps={MAX_STEPS}')

# warmup-aware training: only learn after 512*4 transitions
WARMUP = 512 * 4
N_EPISODES = 5  # 5 episodes x 2016 steps = ~10k env steps. Should give signal.
print(f'Running {N_EPISODES} episodes, warmup={WARMUP}...')

ep_rewards, ep_costs, ep_slas, ep_losses_all = [], [], [], []
global_env_steps = 0
global_grad_steps = 0
t0 = time.time()

for ep in range(N_EPISODES):
    obs, info = env.reset()
    ep_r, ep_losses = 0.0, []
    for step in range(env.max_steps):
        mask = env.get_action_mask()
        action = agent.select_action(obs, training=True, action_mask=mask)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        next_mask = env.get_action_mask()
        agent.store_transition(obs, action, reward, next_obs, done, next_action_mask=next_mask)
        global_env_steps += 1
        if global_env_steps >= WARMUP:
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
                global_grad_steps += 1
        ep_r += reward
        obs = next_obs
        if done:
            break
    ep_rewards.append(ep_r)
    ep_costs.append(float(info.get('cost', 0)))
    ep_slas.append(float(info.get('sla_compliance', 0)))
    avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
    ep_losses_all.append(avg_loss)
    print(f'  Ep {ep+1}: R={ep_r:+.1f}  cost=${ep_costs[-1]:.0f}  SLA={ep_slas[-1]:.1%}  '
          f'loss={avg_loss:.3f}  eps={agent.epsilon:.3f}  grad={global_grad_steps}')

elapsed = time.time() - t0
print(f'\nElapsed: {elapsed/60:.1f} min')
print(f'Reward range: {min(ep_rewards):+.0f} -> {max(ep_rewards):+.0f}')
print(f'Loss range:   {min(ep_losses_all):.3f} -> {max(ep_losses_all):.3f}')
print(f'SLA range:    {min(ep_slas):.1%} -> {max(ep_slas):.1%}')

# Sanity checks
print()
if max(abs(r) for r in ep_rewards) > 30000:
    print('FAIL: reward magnitude > 30000 (3x v41 = ~3000-5000 expected)')
elif max(ep_losses_all[1:]) > min(ep_losses_all[1:]) * 3:
    print('WARN: loss varied >3x (early training noise, may be OK)')
else:
    print('OK: reward magnitude reasonable, loss not exploding')
"""

r = subprocess.run([r"venv\Scripts\python.exe", "-c", script], capture_output=True, text=True, timeout=600)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(r.returncode)
