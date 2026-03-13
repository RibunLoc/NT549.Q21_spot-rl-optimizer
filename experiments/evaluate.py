"""
Evaluation script for DQN agent.

Usage:
    python experiments/evaluate.py --config experiments/configs/stable_price.yaml \
        --model results/.../best_model.pth --episodes 100 --seeds 5
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from envs.spot_env import SpotInstanceEnv
from agents.dqn_agent import DQNAgent
from utils.metrics import compute_evaluation_metrics
from utils.visualization import plot_action_distribution, plot_cost_comparison


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_env(config: dict) -> SpotInstanceEnv:
    env_config = config['env']
    return SpotInstanceEnv(
        data_path=env_config['data_path'],
        max_steps=env_config.get('max_steps', 1000),
        sla_threshold=env_config.get('sla_threshold', 0.95),
        spot_capacity=env_config.get('spot_capacity', 10),
        ondemand_capacity=env_config.get('ondemand_capacity', 5),
        workload_config=env_config.get('workload', {}),
        cost_config=env_config.get('cost', {}),
    )


def select_action(agent, observation, info):
    try:
        return agent.select_action(observation, training=False)
    except TypeError:
        return agent.select_action(observation, info)


def run_episodes(env, agent, episodes: int, seeds: int, max_steps: int):
    episode_rows = []
    action_counts = np.zeros(env.action_space.n, dtype=int)

    base_seeds = list(range(seeds)) if seeds > 0 else [0]

    for ep in range(1, episodes + 1):
        seed = base_seeds[(ep - 1) % len(base_seeds)] + ep
        obs, info = env.reset(seed=seed)
        if hasattr(agent, 'reset'):
            agent.reset()

        total_reward = 0.0
        baseline_cost = 0.0

        for _ in range(max_steps):
            action = select_action(agent, obs, info)
            action_counts[action] += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            total_capacity = info.get('spot_instances', 0) + info.get('ondemand_instances', 0)
            baseline_cost += total_capacity * env.cost_calc.ondemand_price

            obs = next_obs
            if terminated or truncated:
                break

        episode_rows.append({
            'episode': ep,
            'reward': total_reward,
            'cost': info.get('cost', 0.0),
            'sla_compliance': info.get('sla_compliance', 0.0),
            'failed_jobs': info.get('failed_jobs', 0),
            'baseline_cost': baseline_cost,
        })

    return episode_rows, action_counts


def to_serializable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate DQN agent')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds to cycle (default: 5)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')

    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=config['agent'].get('learning_rate', 1e-4),
        gamma=config['agent'].get('gamma', 0.99),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        batch_size=config['agent'].get('batch_size', 64),
        buffer_size=config['agent'].get('replay_buffer_size', 100000),
        target_update_freq=config['agent'].get('target_update_freq', 1000),
    )
    agent.load(str(model_path))

    max_steps = config['training'].get('max_steps_per_episode', env.max_steps)

    episode_rows, action_counts = run_episodes(
        env, agent, args.episodes, args.seeds, max_steps
    )

    df = pd.DataFrame(episode_rows)
    avg_baseline_cost = df['baseline_cost'].mean() if not df.empty else 0.0

    baseline_for_metrics = avg_baseline_cost if avg_baseline_cost > 0 else df['cost'].mean() if not df.empty else 1.0
    summary = compute_evaluation_metrics(
        episode_costs=df['cost'].tolist(),
        episode_sla=df['sla_compliance'].tolist(),
        baseline_cost=baseline_for_metrics,
    )
    summary.update({
        'agent_name': 'DQN',
        'avg_reward': float(df['reward'].mean()) if not df.empty else 0.0,
        'avg_baseline_cost': float(avg_baseline_cost),
    })
    summary = to_serializable(summary)

    scenario_name = Path(args.config).stem
    output_root = Path(args.output_dir)
    reports_dir = output_root / 'reports'
    plots_dir = output_root / 'plots'
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / f'eval_{scenario_name}.json'
    detail_path = reports_dir / f'eval_{scenario_name}.csv'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    df.to_csv(detail_path, index=False)

    plot_action_distribution(
        action_counts=action_counts,
        action_names=env.ACTION_NAMES,
        title=f"Action Distribution - {scenario_name}",
        save_path=str(plots_dir / f"action_distribution_{scenario_name}.png"),
    )

    plot_cost_comparison(
        results=[{
            'agent_name': 'DQN',
            'avg_cost': summary['avg_cost'],
            'avg_sla': summary['avg_sla_compliance'],
        }],
        save_path=str(plots_dir / f"cost_sla_{scenario_name}.png"),
    )

    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved evaluation details to {detail_path}")


if __name__ == '__main__':
    main()
