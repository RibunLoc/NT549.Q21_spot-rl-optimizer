"""
Compare DQN against baseline agents across scenarios.

Usage:
    python experiments/compare_baselines.py \
        --scenarios stable,volatile,spike \
        --dqn-model results/.../best_model.pth
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
from agents.baselines import (
    AlwaysOnDemandAgent,
    AlwaysSpotAgent,
    ThresholdBasedAgent,
    RandomAgent,
)
from utils.metrics import compute_evaluation_metrics
from utils.visualization import plot_cost_comparison


SCENARIO_MAP = {
    'stable': 'experiments/configs/stable_price.yaml',
    'volatile': 'experiments/configs/volatile_price.yaml',
    'spike': 'experiments/configs/workload_spike.yaml',
}


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
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
            'baseline_cost': baseline_cost,
        })

    return episode_rows, action_counts


def evaluate_agent(env, agent, episodes: int, seeds: int, max_steps: int, baseline_cost: float, agent_name: str):
    rows, action_counts = run_episodes(env, agent, episodes, seeds, max_steps)
    df = pd.DataFrame(rows)

    baseline_for_metrics = baseline_cost if baseline_cost > 0 else df['cost'].mean() if not df.empty else 1.0
    summary = compute_evaluation_metrics(
        episode_costs=df['cost'].tolist(),
        episode_sla=df['sla_compliance'].tolist(),
        baseline_cost=baseline_for_metrics,
    )
    summary.update({
        'agent_name': agent_name,
        'avg_reward': float(df['reward'].mean()) if not df.empty else 0.0,
    })

    return summary, df, action_counts


def to_serializable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def resolve_scenarios(scenarios_arg: str):
    scenarios = [s.strip() for s in scenarios_arg.split(',') if s.strip()]
    resolved = []
    for s in scenarios:
        if s.endswith('.yaml') or '/' in s or '\\' in s:
            resolved.append((Path(s).stem, s))
        else:
            resolved.append((s, SCENARIO_MAP.get(s, s)))
    return resolved


def main():
    parser = argparse.ArgumentParser(description='Compare DQN and baselines')
    parser.add_argument('--scenarios', type=str, default='stable,volatile,spike',
                        help='Comma-separated scenarios (stable,volatile,spike) or paths')
    parser.add_argument('--dqn-model', type=str, default=None,
                        help='Path to DQN model (.pth). If omitted, DQN is skipped.')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes (default: 50)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds to cycle (default: 5)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')

    args = parser.parse_args()

    output_root = Path(args.output_dir)
    reports_dir = output_root / 'reports'
    plots_dir = output_root / 'plots'
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for scenario_name, config_path in resolve_scenarios(args.scenarios):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = load_config(str(config_path))
        env = build_env(config)
        max_steps = config['training'].get('max_steps_per_episode', env.max_steps)

        # Baseline: Always On-Demand (used as cost baseline)
        target_capacity = env.spot_capacity + env.ondemand_capacity
        ondemand_agent = AlwaysOnDemandAgent(target_capacity=target_capacity)
        ondemand_summary, ondemand_df, _ = evaluate_agent(
            env, ondemand_agent, args.episodes, args.seeds, max_steps, baseline_cost=0.0,
            agent_name='Always On-Demand'
        )
        baseline_cost = ondemand_summary['avg_cost']

        ondemand_summary = to_serializable(ondemand_summary)
        results = [ondemand_summary]
        all_rows.append({'scenario': scenario_name, **ondemand_summary})

        # Always Spot
        spot_agent = AlwaysSpotAgent(target_capacity=env.spot_capacity)
        spot_summary, _, _ = evaluate_agent(
            env, spot_agent, args.episodes, args.seeds, max_steps, baseline_cost,
            agent_name='Always Spot'
        )
        spot_summary = to_serializable(spot_summary)
        results.append(spot_summary)
        all_rows.append({'scenario': scenario_name, **spot_summary})

        # Threshold-based
        threshold_agent = ThresholdBasedAgent(
            threshold_ratio=0.4,
            ondemand_price=env.cost_calc.ondemand_price,
            target_capacity=target_capacity,
        )
        threshold_summary, _, _ = evaluate_agent(
            env, threshold_agent, args.episodes, args.seeds, max_steps, baseline_cost,
            agent_name='Threshold'
        )
        threshold_summary = to_serializable(threshold_summary)
        results.append(threshold_summary)
        all_rows.append({'scenario': scenario_name, **threshold_summary})

        # Random
        random_agent = RandomAgent(num_actions=env.action_space.n)
        random_summary, _, _ = evaluate_agent(
            env, random_agent, args.episodes, args.seeds, max_steps, baseline_cost,
            agent_name='Random'
        )
        random_summary = to_serializable(random_summary)
        results.append(random_summary)
        all_rows.append({'scenario': scenario_name, **random_summary})

        # DQN (optional)
        if args.dqn_model:
            model_path = Path(args.dqn_model)
            if not model_path.exists():
                raise FileNotFoundError(f"DQN model not found: {model_path}")

            dqn_agent = DQNAgent(
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
            dqn_agent.load(str(model_path))

            dqn_summary, _, _ = evaluate_agent(
                env, dqn_agent, args.episodes, args.seeds, max_steps, baseline_cost,
                agent_name='DQN'
            )
            dqn_summary = to_serializable(dqn_summary)
            results.append(dqn_summary)
            all_rows.append({'scenario': scenario_name, **dqn_summary})

        # Plot comparison for this scenario
        plot_cost_comparison(
            results=[{
                'agent_name': r['agent_name'],
                'avg_cost': r['avg_cost'],
                'avg_sla': r['avg_sla_compliance'],
            } for r in results],
            save_path=str(plots_dir / f'cost_sla_comparison_{scenario_name}.png'),
        )

    # Save combined comparison table
    comparison_df = pd.DataFrame(all_rows)
    comparison_path = reports_dir / 'baseline_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)

    summary_path = reports_dir / 'baseline_comparison.json'
    with open(summary_path, 'w') as f:
        json.dump(all_rows, f, indent=2)

    print(f"Saved baseline comparison to {comparison_path}")


if __name__ == '__main__':
    main()
