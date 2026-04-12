"""
Evaluate DQN agent + baselines across all scenarios and log to MLflow.

Usage:
    python experiments/evaluate_all.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from envs.spot_env import SpotInstanceEnv
from agents.dqn_agent import DQNAgent
from agents.baselines import (
    AlwaysOnDemandAgent, AlwaysSpotAgent, ThresholdBasedAgent, RandomAgent,
)
from utils.mlflow_logger import MLflowLogger
from utils.metrics import compute_evaluation_metrics
from utils.visualization import plot_action_distribution, plot_cost_comparison


# ── Scenarios ────────────────────────────────────────────────
SCENARIOS = {
    "stable": {
        "config": "experiments/configs/stable_price.yaml",
        "model_dir": "results/dqn_stable_v5",
    },
    "volatile": {
        "config": "experiments/configs/volatile_price.yaml",
        "model_dir": "results/dqn_volatile_v5",
    },
    "spike": {
        "config": "experiments/configs/workload_spike.yaml",
        "model_dir": "results/dqn_spike_v5",
    },
}

EVAL_EPISODES = 100
EVAL_SEEDS = 5


# ── Helpers ──────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(config: dict) -> SpotInstanceEnv:
    ec = config["env"]
    return SpotInstanceEnv(
        data_path=ec["data_path"],
        max_steps=ec.get("max_steps", 1000),
        sla_threshold=ec.get("sla_threshold", 0.95),
        spot_capacity=ec.get("spot_capacity", 10),
        ondemand_capacity=ec.get("ondemand_capacity", 5),
        workload_config=ec.get("workload", {}),
        cost_config=ec.get("cost", {}),
    )


def find_best_model(model_dir: str) -> Path:
    """Find best_model.pth in the latest run subdirectory."""
    base = Path(model_dir)
    if not base.exists():
        return None
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not subdirs:
        return None
    best = subdirs[-1] / "models" / "best_model.pth"
    return best if best.exists() else None


def make_dqn_agent(env, config, model_path):
    ac = config["agent"]
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=ac.get("learning_rate", 1e-4),
        gamma=ac.get("gamma", 0.99),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        batch_size=ac.get("batch_size", 64),
        buffer_size=ac.get("replay_buffer_size", 100000),
        target_update_freq=ac.get("target_update_freq", 1000),
    )
    agent.load(str(model_path))
    return agent


def safe_action(agent, obs, info):
    try:
        return agent.select_action(obs, training=False)
    except TypeError:
        return agent.select_action(obs, info)


def run_eval(env, agent, episodes, seeds, max_steps):
    """Run evaluation episodes and return per-episode rows + action counts."""
    rows = []
    action_counts = np.zeros(env.action_space.n, dtype=int)
    base_seeds = list(range(seeds)) if seeds > 0 else [0]

    for ep in range(1, episodes + 1):
        seed = base_seeds[(ep - 1) % len(base_seeds)] + ep
        obs, info = env.reset(seed=seed)
        if hasattr(agent, "reset"):
            agent.reset()

        total_reward = 0.0
        baseline_cost = 0.0

        for _ in range(max_steps):
            action = safe_action(agent, obs, info)
            action_counts[action] += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            total_cap = info.get("spot_instances", 0) + info.get("ondemand_instances", 0)
            baseline_cost += total_cap * env.cost_calc.ondemand_price

            obs = next_obs
            if terminated or truncated:
                break

        rows.append({
            "episode": ep,
            "reward": total_reward,
            "cost": info.get("cost", 0.0),
            "sla_compliance": info.get("sla_compliance", 0.0),
            "failed_jobs": info.get("failed_jobs", 0),
            "spot_instances": info.get("spot_instances", 0),
            "ondemand_instances": info.get("ondemand_instances", 0),
            "baseline_cost": baseline_cost,
        })

    return pd.DataFrame(rows), action_counts


def to_serializable(d: dict) -> dict:
    return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in d.items()}


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seeds", type=int, default=EVAL_SEEDS)
    args = parser.parse_args()

    reports_dir = Path("results/reports")
    plots_dir = Path("results/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mlf = MLflowLogger(experiment_name="spot-rl-evaluation")

    all_results = {}  # scenario → list of agent results

    for scenario_name, scenario_info in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating scenario: {scenario_name.upper()}")
        print(f"{'='*60}")

        config = load_config(scenario_info["config"])
        env = build_env(config)
        max_steps = config["training"].get("max_steps_per_episode", env.max_steps)
        ondemand_price = config["env"].get("cost", {}).get("ondemand_price", 0.096)

        # Build agents
        agents = {}

        # DQN
        model_path = find_best_model(scenario_info["model_dir"])
        if model_path:
            agents["DQN"] = make_dqn_agent(env, config, model_path)
            print(f"  DQN model: {model_path}")
        else:
            print(f"  WARNING: No DQN model found for {scenario_name}")

        # Baselines
        agents["OnDemand-Only"] = AlwaysOnDemandAgent(target_capacity=5)
        agents["Spot-Only"] = AlwaysSpotAgent(target_capacity=5)
        agents["Threshold-40%"] = ThresholdBasedAgent(
            threshold_ratio=0.4, ondemand_price=ondemand_price, target_capacity=5,
        )
        agents["Random"] = RandomAgent(num_actions=env.action_space.n, seed=42)

        scenario_results = []

        for agent_name, agent in agents.items():
            print(f"\n  Running {agent_name}...")
            df, action_counts = run_eval(
                env, agent, args.episodes, args.seeds, max_steps,
            )

            avg_baseline_cost = df["baseline_cost"].mean()
            baseline_for_metrics = max(avg_baseline_cost, df["cost"].mean(), 1.0)

            summary = compute_evaluation_metrics(
                episode_costs=df["cost"].tolist(),
                episode_sla=df["sla_compliance"].tolist(),
                baseline_cost=baseline_for_metrics,
            )
            summary.update({
                "agent_name": agent_name,
                "scenario": scenario_name,
                "avg_reward": float(df["reward"].mean()),
                "std_reward": float(df["reward"].std()),
                "avg_baseline_cost": float(avg_baseline_cost),
                "avg_spot": float(df["spot_instances"].mean()),
                "avg_ondemand": float(df["ondemand_instances"].mean()),
            })
            summary = to_serializable(summary)
            scenario_results.append(summary)

            # Print summary
            print(f"    Reward:  {summary['avg_reward']:>10.1f}")
            print(f"    Cost:    ${summary['avg_cost']:>9.1f}")
            print(f"    SLA:     {summary['avg_sla_compliance']:>9.1%}")
            print(f"    Savings: {summary['cost_savings_pct']:>9.1f}%")

            # Log to MLflow
            mlf.start_run(
                run_name=f"eval_{scenario_name}_{agent_name}",
                tags={
                    "scenario": scenario_name,
                    "agent": agent_name,
                    "eval_type": "evaluation",
                    "episodes": str(args.episodes),
                },
            )
            mlf.log_metrics({
                "avg_reward": summary["avg_reward"],
                "std_reward": summary["std_reward"],
                "avg_cost": summary["avg_cost"],
                "avg_sla": summary["avg_sla_compliance"],
                "sla_violation_rate": summary["sla_violation_rate_pct"],
                "cost_savings_pct": summary["cost_savings_pct"],
                "avg_spot_instances": summary["avg_spot"],
                "avg_ondemand_instances": summary["avg_ondemand"],
            })

            # Save per-agent detail CSV
            detail_path = reports_dir / f"eval_{scenario_name}_{agent_name}.csv"
            df.to_csv(detail_path, index=False)
            mlf.log_artifact(str(detail_path))

            # Action distribution plot
            action_plot_path = plots_dir / f"actions_{scenario_name}_{agent_name}.png"
            plot_action_distribution(
                action_counts=action_counts,
                action_names=getattr(env, "ACTION_NAMES", None),
                title=f"Actions — {agent_name} on {scenario_name}",
                save_path=str(action_plot_path),
            )
            mlf.log_artifact(str(action_plot_path))

            # Log DQN model artifact
            if agent_name == "DQN" and model_path:
                mlf.log_model(str(model_path), artifact_path="best_model")

            mlf.end_run()

        all_results[scenario_name] = scenario_results

        # ── Per-scenario comparison plot ─────────────────────
        comparison_results = [
            {"agent_name": r["agent_name"],
             "avg_cost": r["avg_cost"],
             "avg_sla": r["avg_sla_compliance"]}
            for r in scenario_results
        ]
        plot_cost_comparison(
            results=comparison_results,
            save_path=str(plots_dir / f"comparison_{scenario_name}.png"),
        )

    # ── Cross-scenario summary ───────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY — ALL SCENARIOS")
    print(f"{'='*60}")

    header = f"{'Scenario':<10} {'Agent':<16} {'Reward':>10} {'Cost':>10} {'SLA':>8} {'Savings':>8}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for scenario_name, results in all_results.items():
        for r in results:
            print(
                f"{scenario_name:<10} {r['agent_name']:<16} "
                f"{r['avg_reward']:>10.1f} "
                f"${r['avg_cost']:>9.1f} "
                f"{r['avg_sla_compliance']:>7.1%} "
                f"{r['cost_savings_pct']:>7.1f}%"
            )
            summary_rows.append(r)

    # Save full summary
    summary_path = reports_dir / "eval_all_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSaved full summary to {summary_path}")

    # Save as CSV too
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = reports_dir / "eval_all_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")

    print(f"\nView results: mlflow ui --host 0.0.0.0 --port 5000")
    print(f"Experiment: spot-rl-evaluation")


if __name__ == "__main__":
    main()
