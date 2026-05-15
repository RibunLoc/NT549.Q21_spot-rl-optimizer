import subprocess, os

os.chdir(r"D:\Document\Seminar_MLOps\spot-rl-optimiztion")

script = r"""
import os, sys
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, r'D:\Document\Seminar_MLOps\spot-rl-optimiztion')

import json
import numpy as np
import torch

# ---- load model ----
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS
from agents.dqn_agent import DQNAgent

model_path = r'results\kaggle\best_mixed_v41.pth'
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
network_type = ckpt.get('network_type', 'factored')
hidden_dim   = ckpt.get('hidden_dim', 512)
state_dim    = ckpt.get('state_dim', STATE_DIM)
action_dim   = ckpt.get('action_dim', N_ACTIONS)

dqn = DQNAgent(
    state_dim=state_dim, action_dim=action_dim,
    hidden_dim=hidden_dim, network_type=network_type,
    epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
    device='cpu',
)
dqn.load(model_path)
print(f"Model loaded: {network_type} state={state_dim} action={action_dim} hidden={hidden_dim}")

# ---- scenarios ----
from envs.spot_orchestrator_env import SpotOrchestratorEnv
from agents.baselines import OnDemandAgent, SpotAgent, ThresholdAgent, CheapestAZAgent, CheapestTypeAgent, RandomAgent

SCENARIOS = {
    "stable":        "data/processed/multipool_stable.csv",
    "volatile":      "data/processed/multipool_volatile.csv",
    "spike":         "data/processed/multipool_spike.csv",
    "az_divergence": "data/processed/multipool_az_divergence.csv",
}
WORKLOAD_CFG = {"base_arrival_rate": 15.0, "peak_multiplier": 2.0, "avg_job_duration": 3}
N_EPISODES = 30

def make_env(scenario, seed=0):
    return SpotOrchestratorEnv(
        data_path=SCENARIOS[scenario],
        max_steps=168,
        workload_config=WORKLOAD_CFG,
        seed=seed,
    )

def run_episode(env, agent, use_mask=False):
    obs, info = env.reset()
    total_reward = total_cost = 0.0
    ac = np.zeros(N_ACTIONS, dtype=int)
    while True:
        if hasattr(agent, 'replay_buffer'):
            mask   = env.get_action_mask() if use_mask else None
            action = agent.select_action(obs, training=False, action_mask=mask)
        else:
            action = agent.select_action(obs, info)
        ac[action] += 1
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        total_cost   += info.get('step_cost', 0.0)
        if term or trunc:
            break
    return {
        'reward': float(total_reward),
        'cost':   float(info.get('cost', total_cost)),
        'sla':    float(info.get('sla_compliance', 0.0)),
        'failed': int(info.get('failed_jobs', 0)),
        'ac':     ac,
    }

def evaluate(agent, scenario, use_mask=False):
    rewards, costs, slas = [], [], []
    ac_total = np.zeros(N_ACTIONS, dtype=int)
    for ep in range(N_EPISODES):
        env = make_env(scenario, seed=ep * 7 + 42)
        r = run_episode(env, agent, use_mask)
        rewards.append(r['reward']); costs.append(r['cost']); slas.append(r['sla'])
        ac_total += r['ac']
    return {
        'avg_reward': float(np.mean(rewards)), 'std_reward': float(np.std(rewards)),
        'avg_cost':   float(np.mean(costs)),   'std_cost':   float(np.std(costs)),
        'avg_sla':    float(np.mean(slas)),
        'costs': costs, 'rewards': rewards, 'slas': slas,
        'ac': ac_total,
    }

AGENTS = {
    'DQN':          (dqn,                              True),
    'OnDemand':     (OnDemandAgent(target_capacity=8),  False),
    'Spot':         (SpotAgent(target_capacity=8),      False),
    'Threshold':    (ThresholdAgent(),                  False),
    'CheapestAZ':   (CheapestAZAgent(),                False),
    'CheapestType': (CheapestTypeAgent(),               False),
    'Random':       (RandomAgent(seed=0),               False),
}

all_results = {}
for scn in SCENARIOS:
    print(f"\n{'='*55}\nScenario: {scn}\n{'='*55}")
    scn_res = {}
    for name, (agent, use_mask) in AGENTS.items():
        if hasattr(agent, 'reset'):
            agent.reset()
        res = evaluate(agent, scn, use_mask)
        scn_res[name] = res
        print(f"  {name:<14} cost=${res['avg_cost']:>7.1f}+/-{res['std_cost']:.0f}  SLA={res['avg_sla']*100:.1f}%  reward={res['avg_reward']:.1f}")
    all_results[scn] = scn_res

    od  = scn_res['OnDemand']['avg_cost']
    dqn_c = scn_res['DQN']['avg_cost']
    print(f"\n  >> DQN saves {(od-dqn_c)/od*100:.1f}% vs OnDemand (${od:.1f} -> ${dqn_c:.1f})")

# ---- print summary table ----
print("\n" + "="*80)
print(f"{'Agent':<14}  {'stable cost':>11} {'SLA':>6}  {'volatile cost':>13} {'SLA':>6}  {'spike cost':>11} {'SLA':>6}  {'az_div cost':>11} {'SLA':>6}")
print("-"*80)
for name in AGENTS:
    row = f"{name:<14}"
    for scn in SCENARIOS:
        r = all_results[scn][name]
        row += f"  ${r['avg_cost']:>7.1f}  {r['avg_sla']*100:>5.1f}%"
    print(row)

# ---- savings summary ----
print("\n--- DQN Savings vs OnDemand ---")
total_dqn_cost = 0; total_od_cost = 0
for scn in SCENARIOS:
    od  = all_results[scn]['OnDemand']['avg_cost']
    dqn_c = all_results[scn]['DQN']['avg_cost']
    pct = (od - dqn_c) / od * 100 if od > 0 else 0
    print(f"  {scn:<14}: saves {pct:+.1f}%  (${od:.1f} -> ${dqn_c:.1f})")
    total_dqn_cost += dqn_c; total_od_cost += od
overall = (total_od_cost - total_dqn_cost) / total_od_cost * 100 if total_od_cost > 0 else 0
print(f"  {'OVERALL':<14}: saves {overall:+.1f}%")

# ---- save JSON ----
import pathlib
out_dir = pathlib.Path(r'results\kaggle\eval_v41')
out_dir.mkdir(parents=True, exist_ok=True)

summary = {}
for scn in SCENARIOS:
    summary[scn] = {}
    for name, res in all_results[scn].items():
        od = all_results[scn]['OnDemand']['avg_cost']
        summary[scn][name] = {
            'avg_reward': res['avg_reward'], 'std_reward': res['std_reward'],
            'avg_cost':   res['avg_cost'],   'std_cost':   res['std_cost'],
            'avg_sla':    res['avg_sla'],
            'savings_vs_od_pct': (od - res['avg_cost']) / od * 100 if od > 0 else 0,
        }

with open(out_dir / 'eval_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {out_dir / 'eval_summary.json'}")

# ---- plot ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scenarios = list(SCENARIOS.keys())
agents    = list(AGENTS.keys())
colors    = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#795548','#00BCD4']

fig, axes = plt.subplots(2, 4, figsize=(22, 9))
fig.suptitle('DQN v41 Generalist vs Baselines — 4 Scenarios', fontsize=14, fontweight='bold')

for col, scn in enumerate(scenarios):
    costs = [all_results[scn][a]['avg_cost'] for a in agents]
    slas  = [all_results[scn][a]['avg_sla']  for a in agents]
    x = np.arange(len(agents))

    ax = axes[0, col]
    bars = ax.bar(x, costs, color=colors[:len(agents)])
    ax.set_title(f'[{scn}] Cost ($)', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(agents, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Avg Cost / Episode ($)')
    for bar, v in zip(bars, costs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'${v:.0f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1, col]
    bars2 = ax2.bar(x, [s*100 for s in slas], color=colors[:len(agents)])
    ax2.set_title(f'[{scn}] SLA (%)', fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(agents, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('SLA Compliance (%)')
    ax2.set_ylim(0, 110)
    ax2.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    for bar, v in zip(bars2, slas):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plot_path = out_dir / 'comparison_all_scenarios.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {plot_path}")

# savings heatmap
agents_no_od = [a for a in agents if a != 'OnDemand']
matrix = np.zeros((len(agents_no_od), 4))
for j, scn in enumerate(scenarios):
    od = all_results[scn]['OnDemand']['avg_cost']
    for i, ag in enumerate(agents_no_od):
        c = all_results[scn][ag]['avg_cost']
        matrix[i, j] = (od - c) / od * 100 if od > 0 else 0

fig2, ax3 = plt.subplots(figsize=(10, 5))
im = ax3.imshow(matrix, cmap='RdYlGn', vmin=-20, vmax=60)
ax3.set_xticks(range(4)); ax3.set_xticklabels(scenarios, fontsize=10)
ax3.set_yticks(range(len(agents_no_od))); ax3.set_yticklabels(agents_no_od, fontsize=10)
ax3.set_title('Cost Savings vs OnDemand (%) — v41', fontsize=12)
plt.colorbar(im, ax=ax3, label='Savings %')
for i in range(len(agents_no_od)):
    for j in range(4):
        ax3.text(j, i, f'{matrix[i,j]:+.1f}%', ha='center', va='center', fontsize=10,
                 color='black' if abs(matrix[i,j]) < 40 else 'white')
plt.tight_layout()
hmap_path = out_dir / 'savings_heatmap.png'
plt.savefig(hmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {hmap_path}")

# action distribution (DQN combined across all scenarios)
from envs.action_schema import N_POOL_OPS, N_POOLS, HOLD_ACTION, decode_action, OPERATION_NAMES
ac_combined = np.zeros(N_ACTIONS, dtype=int)
for scn in SCENARIOS:
    ac_combined += all_results[scn]['DQN']['ac']

op_counts = np.zeros(N_POOL_OPS + 1, dtype=int)
for idx, cnt in enumerate(ac_combined):
    if idx == HOLD_ACTION:
        op_counts[N_POOL_OPS] += cnt
    else:
        op, _, _ = decode_action(idx)
        if 0 <= op < N_POOL_OPS:
            op_counts[op] += cnt

labels = list(OPERATION_NAMES.values()) if isinstance(OPERATION_NAMES, dict) else list(OPERATION_NAMES)
total = op_counts.sum()
pcts  = op_counts / total * 100 if total > 0 else op_counts

fig3, ax4 = plt.subplots(figsize=(12, 4))
bars3 = ax4.bar(range(len(labels)), pcts,
                color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
ax4.set_xticks(range(len(labels)))
ax4.set_xticklabels(labels, rotation=30, ha='right')
ax4.set_ylabel('Action frequency (%)')
ax4.set_title('DQN v41 Action Distribution (all scenarios combined)')
for bar, v in zip(bars3, pcts):
    if v > 0.5:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
adist_path = out_dir / 'dqn_action_dist.png'
plt.savefig(adist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Action dist saved: {adist_path}")

print("\nDONE")
"""

result = subprocess.run(
    [r"venv\Scripts\python.exe", "-c", script],
    capture_output=True, text=True,
    cwd=r"D:\Document\Seminar_MLOps\spot-rl-optimiztion",
    timeout=600,
)

# save log
os.makedirs(r"results\kaggle\eval_v41", exist_ok=True)
with open(r"results\kaggle\eval_v41\run_log.txt", "w", encoding="utf-8") as f:
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== EXIT: {result.returncode} ===\n")

print("Exit code:", result.returncode)
print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
