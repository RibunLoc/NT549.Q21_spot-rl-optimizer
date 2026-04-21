"""
Evaluate generalist DQN (best_mixed.pth) vs 6 baselines on 4 scenarios.

Usage:
    python experiments/evaluate_kaggle.py
    python experiments/evaluate_kaggle.py --model results/kaggle/best_mixed.pth --episodes 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from envs.spot_orchestrator_env import SpotOrchestratorEnv
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS, decode_action, OPERATION_NAMES
from agents.dqn_agent import DQNAgent
from agents.baselines import (
    OnDemandAgent, SpotAgent, ThresholdAgent,
    CheapestAZAgent, CheapestTypeAgent, RandomAgent,
)


SCENARIOS = {
    "stable":        "data/processed/multipool_stable.csv",
    "volatile":      "data/processed/multipool_volatile.csv",
    "spike":         "data/processed/multipool_spike.csv",
    "az_divergence": "data/processed/multipool_az_divergence.csv",
}

WORKLOAD_CFG = {
    "base_arrival_rate": 15.0,
    "peak_multiplier":   2.0,
    "avg_job_duration":  3,
}


def make_env(scenario: str, seed: int = 0) -> SpotOrchestratorEnv:
    return SpotOrchestratorEnv(
        data_path=SCENARIOS[scenario],
        max_steps=168,
        workload_config=WORKLOAD_CFG,
        seed=seed,
    )


def load_dqn(model_path: str, device: str = "cpu") -> DQNAgent:
    import torch
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    network_type = ckpt.get("network_type")
    hidden_dim   = ckpt.get("hidden_dim")

    if network_type is None:
        sd = ckpt["q_network_state_dict"]
        if "trunk.0.weight" in sd:
            network_type = "factored"
            hidden_dim   = hidden_dim or sd["trunk.0.weight"].shape[0]
        elif "feature.0.weight" in sd:
            network_type = "dueling"
            hidden_dim   = hidden_dim or sd["feature.0.weight"].shape[0]
        else:
            raise RuntimeError(f"Unknown checkpoint format. Keys: {list(sd.keys())[:5]}")

    state_dim  = ckpt.get("state_dim",  STATE_DIM)
    action_dim = ckpt.get("action_dim", N_ACTIONS)

    agent = DQNAgent(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=hidden_dim, network_type=network_type,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
        device=device,
    )
    agent.load(model_path)
    print(f"Loaded {network_type} net | state={state_dim} action={action_dim} hidden={hidden_dim}")
    return agent


def run_episode(env, agent, use_mask: bool = False) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    total_cost   = 0.0
    action_counts = np.zeros(N_ACTIONS, dtype=int)

    while True:
        if hasattr(agent, "replay_buffer"):  # DQNAgent
            mask   = env.get_action_mask() if use_mask else None
            action = agent.select_action(obs, training=False, action_mask=mask)
        else:
            action = agent.select_action(obs, info)

        action_counts[action] += 1
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        total_cost   += info.get("step_cost", 0.0)

        if term or trunc:
            break

    return {
        "total_reward":   float(total_reward),
        "total_cost":     float(info.get("cost", total_cost)),
        "sla_rate":       float(info.get("sla_compliance", 0.0)),
        "failed_jobs":    int(info.get("failed_jobs", 0)),
        "completed_jobs": int(info.get("completed_jobs", 0)),
        "action_counts":  action_counts,
    }


def evaluate(agent, scenario: str, n_episodes: int, use_mask: bool = False) -> dict:
    results = []
    ac_total = np.zeros(N_ACTIONS, dtype=int)

    for ep in range(n_episodes):
        env = make_env(scenario, seed=ep * 7 + 42)
        r   = run_episode(env, agent, use_mask)
        results.append(r)
        ac_total += r["action_counts"]

    rewards = [r["total_reward"] for r in results]
    costs   = [r["total_cost"]   for r in results]
    slas    = [r["sla_rate"]     for r in results]

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_cost":   float(np.mean(costs)),
        "std_cost":   float(np.std(costs)),
        "avg_sla":    float(np.mean(slas)),
        "episode_costs":   costs,
        "episode_rewards": rewards,
        "action_counts":   ac_total,
    }


def print_table(all_results: dict):
    """Print markdown-style summary table."""
    scenarios = list(all_results.keys())
    agents    = list(all_results[scenarios[0]].keys())

    header = f"{'Agent':<16}" + "".join(
        f" {'Cost '+s:>14} {'SLA '+s:>8}" for s in scenarios
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for a in agents:
        row = f"{a:<16}"
        for s in scenarios:
            m = all_results[s][a]
            row += f"  ${m['avg_cost']:>8.1f}±{m['std_cost']:.0f}   {m['avg_sla']*100:>5.1f}%"
        print(row)

    # Savings row
    print("-" * len(header))
    od_costs = {s: all_results[s]["OnDemand"]["avg_cost"] for s in scenarios}
    dqn_row  = f"{'DQN savings':16}"
    for s in scenarios:
        dqn_cost = all_results[s]["DQN"]["avg_cost"]
        pct = (od_costs[s] - dqn_cost) / od_costs[s] * 100 if od_costs[s] > 0 else 0
        dqn_row += f"  {'vs OD: ':>9}{pct:+.1f}%    {'':>5}"
    print(dqn_row)
    print("=" * len(header))


def plot_comparison(all_results: dict, out_dir: Path):
    """4-scenario × 2-metric grid: cost + SLA bars."""
    scenarios = list(all_results.keys())
    agents    = list(all_results[scenarios[0]].keys())
    colors    = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#795548", "#00BCD4"]

    fig, axes = plt.subplots(2, len(scenarios), figsize=(5 * len(scenarios), 9))
    fig.suptitle("DQN Generalist vs Baselines — 4 Scenarios", fontsize=14, fontweight="bold")

    for col, scn in enumerate(scenarios):
        costs = [all_results[scn][a]["avg_cost"] for a in agents]
        slas  = [all_results[scn][a]["avg_sla"]  for a in agents]
        x     = np.arange(len(agents))

        # Cost
        ax = axes[0, col]
        bars = ax.bar(x, costs, color=colors[:len(agents)])
        ax.set_title(f"[{scn}] Cost ($)", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(agents, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Avg Cost / Episode ($)")
        for bar, v in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"${v:.0f}", ha="center", va="bottom", fontsize=8)

        # SLA
        ax2 = axes[1, col]
        bars2 = ax2.bar(x, [s * 100 for s in slas], color=colors[:len(agents)])
        ax2.set_title(f"[{scn}] SLA (%)", fontsize=11)
        ax2.set_xticks(x); ax2.set_xticklabels(agents, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("SLA Compliance (%)")
        ax2.set_ylim(0, 105)
        ax2.axhline(95, color="red", linestyle="--", linewidth=1, alpha=0.7)
        for bar, v in zip(bars2, slas):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = out_dir / "comparison_all_scenarios.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_action_dist(action_counts: np.ndarray, out_dir: Path):
    """DQN operation usage breakdown."""
    from envs.action_schema import N_POOL_OPS, N_POOLS, HOLD_ACTION

    op_counts = np.zeros(N_POOL_OPS + 1, dtype=int)
    for idx, cnt in enumerate(action_counts):
        if idx == HOLD_ACTION:
            op_counts[N_POOL_OPS] += cnt
        else:
            op, _, _ = decode_action(idx)
            if 0 <= op < N_POOL_OPS:
                op_counts[op] += cnt

    labels = list(OPERATION_NAMES) if not isinstance(OPERATION_NAMES, dict) else list(OPERATION_NAMES.values())
    total  = op_counts.sum()
    pcts   = op_counts / total * 100 if total > 0 else op_counts

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(labels)), pcts,
                  color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Action frequency (%)")
    ax.set_title("DQN Action Distribution (all scenarios)")
    for bar, v in zip(bars, pcts):
        if v > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = out_dir / "dqn_action_dist.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_savings_heatmap(all_results: dict, out_dir: Path):
    """Cost savings % vs OnDemand — DQN vs all baselines × all scenarios."""
    scenarios = list(all_results.keys())
    agents    = [a for a in all_results[scenarios[0]].keys() if a != "OnDemand"]

    matrix = np.zeros((len(agents), len(scenarios)))
    for j, scn in enumerate(scenarios):
        od = all_results[scn]["OnDemand"]["avg_cost"]
        for i, ag in enumerate(agents):
            c = all_results[scn][ag]["avg_cost"]
            matrix[i, j] = (od - c) / od * 100 if od > 0 else 0

    fig, ax = plt.subplots(figsize=(len(scenarios) * 2 + 2, len(agents) * 0.8 + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-20, vmax=60)
    ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yticks(range(len(agents)));    ax.set_yticklabels(agents, fontsize=10)
    ax.set_title("Cost Savings vs OnDemand (%)\n(green = cheaper than OnDemand)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Savings %")
    for i in range(len(agents)):
        for j in range(len(scenarios)):
            ax.text(j, i, f"{matrix[i,j]:+.1f}%", ha="center", va="center", fontsize=9,
                    color="black" if abs(matrix[i,j]) < 40 else "white")
    plt.tight_layout()
    path = out_dir / "savings_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="results/kaggle/best_mixed.pth")
    parser.add_argument("--episodes",   type=int, default=30)
    parser.add_argument("--output-dir", default="results/kaggle/eval")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Episodes per scenario: {args.episodes}")

    dqn = load_dqn(args.model, device=args.device)

    AGENTS = {
        "DQN":          (dqn,                             True),
        "OnDemand":     (OnDemandAgent(target_capacity=8), False),
        "Spot":         (SpotAgent(target_capacity=8),     False),
        "Threshold":    (ThresholdAgent(),                 False),
        "CheapestAZ":   (CheapestAZAgent(),               False),
        "CheapestType": (CheapestTypeAgent(),              False),
        "Random":       (RandomAgent(seed=0),              False),
    }

    all_results = {}
    ac_combined = np.zeros(N_ACTIONS, dtype=int)

    for scn in SCENARIOS:
        print(f"\n{'='*60}\nScenario: {scn}\n{'='*60}")
        scn_res = {}

        for name, (agent, use_mask) in AGENTS.items():
            if hasattr(agent, "reset"):
                agent.reset()
            print(f"  [{name}] evaluating {args.episodes} episodes...")
            res = evaluate(agent, scn, args.episodes, use_mask)
            scn_res[name] = res
            print(f"    cost=${res['avg_cost']:.1f}±{res['std_cost']:.0f}  "
                  f"SLA={res['avg_sla']*100:.1f}%  "
                  f"reward={res['avg_reward']:.1f}")

        all_results[scn] = scn_res
        ac_combined += scn_res["DQN"]["action_counts"]

        od  = scn_res["OnDemand"]["avg_cost"]
        dqn_c = scn_res["DQN"]["avg_cost"]
        print(f"  >> DQN saves {(od-dqn_c)/od*100:.1f}% vs OnDemand  (${od:.1f} -> ${dqn_c:.1f})")

    print_table(all_results)
    plot_comparison(all_results, out_dir)
    plot_action_dist(ac_combined, out_dir)
    plot_savings_heatmap(all_results, out_dir)

    # JSON summary
    summary = {}
    for scn, agents in all_results.items():
        summary[scn] = {}
        for name, res in agents.items():
            summary[scn][name] = {
                k: v for k, v in res.items()
                if not isinstance(v, (np.ndarray, list))
            }
        od = summary[scn]["OnDemand"]["avg_cost"]
        summary[scn]["DQN"]["savings_vs_ondemand_pct"] = (
            (od - summary[scn]["DQN"]["avg_cost"]) / od * 100 if od > 0 else 0
        )

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
