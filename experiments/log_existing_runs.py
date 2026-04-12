"""
Retroactively log existing training runs to MLflow.

Usage:
    python experiments/log_existing_runs.py
"""

import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import yaml
import pickle
import numpy as np
from utils.mlflow_logger import MLflowLogger


def parse_train_log(log_path: Path) -> list:
    """Parse train.log to extract per-episode metrics."""
    episodes = []
    pattern = re.compile(
        r'Episode (\d+)/\d+ \| '
        r'Reward: ([-\d.]+) \| '
        r'Avg Reward: ([-\d.]+) \| '
        r'Loss: ([\d.]+) \| '
        r'Epsilon: ([\d.]+) \| '
        r'Cost: \$([\d.]+) \| '
        r'SLA: ([\d.]+)%'
    )
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append({
                    'episode': int(m.group(1)),
                    'reward': float(m.group(2)),
                    'avg_reward': float(m.group(3)),
                    'loss': float(m.group(4)),
                    'epsilon': float(m.group(5)),
                    'cost': float(m.group(6)),
                    'sla_compliance': float(m.group(7)) / 100.0,
                })
    return episodes


def log_run(mlf: MLflowLogger, run_dir: Path, run_name: str, scenario: str):
    """Log a single training run to MLflow."""
    print(f"\n{'='*60}")
    print(f"Logging: {run_name} ({scenario})")
    print(f"Dir: {run_dir}")

    # Load config
    config_path = run_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        print(f"  Warning: no config.yaml found")

    # Parse log
    log_path = run_dir / 'logs' / 'train.log'
    if not log_path.exists():
        print(f"  Error: no train.log found, skipping")
        return
    episodes = parse_train_log(log_path)
    print(f"  Found {len(episodes)} logged episodes")

    if not episodes:
        print("  No episodes to log, skipping")
        return

    # Start MLflow run
    mlf.start_run(run_name=run_name, tags={
        "scenario": scenario,
        "agent_type": config.get('agent', {}).get('type', 'DQN'),
        "retroactive": "true",
        "source_dir": str(run_dir),
    })

    # Log config
    if config:
        mlf.log_config(config)

    # Log episode metrics
    for ep in episodes:
        step = ep['episode']
        mlf.log_metrics({
            "episode_reward": ep['reward'],
            "avg_reward": ep['avg_reward'],
            "loss": ep['loss'],
            "epsilon": ep['epsilon'],
            "cost": ep['cost'],
            "sla_compliance": ep['sla_compliance'],
        }, step=step)

    # Log final summary (last 100 episodes)
    last_episodes = episodes[-10:]  # last 10 logged points (each = 10 episodes)
    mlf.log_metrics({
        "final/avg_reward": np.mean([e['avg_reward'] for e in last_episodes]),
        "final/avg_cost": np.mean([e['cost'] for e in last_episodes]),
        "final/avg_sla": np.mean([e['sla_compliance'] for e in last_episodes]),
        "final/total_episodes": float(episodes[-1]['episode']),
    })

    # Log model artifacts
    model_dir = run_dir / 'models'
    best_model = model_dir / 'best_model.pth'
    if best_model.exists():
        mlf.log_model(str(best_model), artifact_path="best_model")
        print(f"  Logged best_model.pth")

    # Log config artifact
    if config_path.exists():
        mlf.log_artifact(str(config_path))

    # Log metrics.pkl if exists
    metrics_path = run_dir / 'metrics.pkl'
    if metrics_path.exists():
        mlf.log_artifact(str(metrics_path))

    mlf.end_run()
    print(f"  Done! ({len(episodes)} episodes logged)")


def main():
    results_dir = Path('results')

    # Define runs to log
    runs = [
        {
            'pattern': 'dqn_stable_v5',
            'scenario': 'stable',
        },
        {
            'pattern': 'dqn_volatile_v5',
            'scenario': 'volatile',
        },
        {
            'pattern': 'dqn_spike_v5',
            'scenario': 'spike',
        },
    ]

    mlf = MLflowLogger(experiment_name="spot-rl-optimization")

    for run_info in runs:
        run_dir_parent = results_dir / run_info['pattern']
        if not run_dir_parent.exists():
            print(f"Skipping {run_info['pattern']}: directory not found")
            continue

        # Find latest timestamp subdirectory
        subdirs = sorted([d for d in run_dir_parent.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"Skipping {run_info['pattern']}: no run subdirectories")
            continue

        latest_dir = subdirs[-1]
        log_run(mlf, latest_dir, run_info['pattern'], run_info['scenario'])

    print(f"\n{'='*60}")
    print("All runs logged to MLflow!")
    print("View dashboard: mlflow ui --port 5000")
    print("Open: http://localhost:5000")


if __name__ == '__main__':
    main()
