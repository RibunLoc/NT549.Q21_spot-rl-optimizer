"""
Training script for DQN agent.

Supports both single-pool (SpotInstanceEnv) and multi-pool (SpotOrchestratorEnv).

Usage:
    # Single-pool (legacy)
    python train.py --config configs/dqn_default.yaml --experiment-name dqn_stable

    # Multi-pool
    python train.py --config configs/multi_pool_stable.yaml --experiment-name mp_stable
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import logging
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.dqn_agent import DQNAgent
from utils.logger import setup_logger, TensorBoardLogger
from utils.metrics import MetricsTracker
from utils.mlflow_logger import MLflowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_env(env_config: dict):
    """Create environment based on config type."""
    env_type = env_config.get('type', 'single')

    if env_type == 'multi_pool':
        from envs.spot_orchestrator_env import SpotOrchestratorEnv
        env = SpotOrchestratorEnv(
            data_path=env_config['data_path'],
            max_steps=env_config.get('max_steps', 168),
            sla_threshold=env_config.get('sla_threshold', 0.95),
            workload_config=env_config.get('workload', {}),
            reward_config=env_config.get('reward', {}),
        )
    else:
        from envs.spot_env import SpotInstanceEnv
        env = SpotInstanceEnv(
            data_path=env_config['data_path'],
            max_steps=env_config.get('max_steps', 1000),
            sla_threshold=env_config.get('sla_threshold', 0.95),
            spot_capacity=env_config.get('spot_capacity', 20),
            ondemand_capacity=env_config.get('ondemand_capacity', 10),
            workload_config=env_config.get('workload', {}),
            cost_config=env_config.get('cost', {}),
        )

    return env


def evaluate(env, agent, num_episodes: int = 10) -> dict:
    """Run evaluation episodes and return average metrics."""
    eval_rewards = []
    eval_costs = []
    eval_sla = []

    for _ in range(num_episodes):
        state, info = env.reset()
        ep_reward = 0.0

        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            state = next_state
            if terminated or truncated:
                break

        eval_rewards.append(ep_reward)
        eval_costs.append(info.get('cost', 0))
        eval_sla.append(info.get('sla_compliance', 0))

    return {
        'avg_reward': float(np.mean(eval_rewards)),
        'avg_cost': float(np.mean(eval_costs)),
        'avg_sla': float(np.mean(eval_sla)),
    }


def train(config: dict, experiment_name: str,
          resume_path: str = None, resume_episode: int = 0):
    """
    Train DQN agent.

    Args:
        config: Configuration dictionary
        experiment_name: Name for this experiment
        resume_path: Path to checkpoint to resume from
        resume_episode: Episode number to resume from
    """
    # Global seed for reproducibility
    seed = config.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path('results') / experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_dir = exp_dir / 'models'
    log_dir = exp_dir / 'logs'
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    export_dir = Path('results') / 'models'
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save config to experiment dir for reproducibility
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Setup logging
    setup_logger(log_dir / 'train.log')
    tb_logger = TensorBoardLogger(log_dir)

    # MLflow tracking
    mlf = MLflowLogger(experiment_name="spot-rl-optimization")
    scenario_tag = "unknown"
    data_path = config.get('env', {}).get('data_path', '')
    for tag in ['stable', 'volatile', 'spike', 'az_divergence']:
        if tag in data_path:
            scenario_tag = tag
            break

    env_type = config.get('env', {}).get('type', 'single')
    mlf.start_run(run_name=experiment_name, tags={
        "scenario": scenario_tag,
        "env_type": env_type,
        "agent_type": config.get('agent', {}).get('type', 'DQN'),
        "resumed": str(resume_path is not None),
    })
    mlf.log_config(config)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Config: {config}")

    # Create environment
    env_config = config['env']
    env = create_env(env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    logger.info(f"Environment created: type={env_type}, state_dim={state_dim}, action_dim={action_dim}")

    # Create agent
    agent_config = config['agent']
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=agent_config.get('learning_rate', 1e-4),
        gamma=agent_config.get('gamma', 0.99),
        epsilon_start=agent_config.get('epsilon_start', 1.0),
        epsilon_end=agent_config.get('epsilon_end', 0.01),
        epsilon_decay=agent_config.get('epsilon_decay', 100000),
        batch_size=agent_config.get('batch_size', 64),
        buffer_size=agent_config.get('replay_buffer_size', 100000),
        target_update_freq=agent_config.get('target_update_freq', 1000),
    )

    logger.info("DQN agent created")

    # Resume from checkpoint if specified
    if resume_path is not None:
        agent.load(resume_path)
        logger.info(f"Resumed from checkpoint: {resume_path} (episode {resume_episode})")

    # Training parameters
    train_config = config['training']
    num_episodes = train_config.get('num_episodes', 5000)
    max_steps_per_episode = train_config.get('max_steps_per_episode', 1000)
    log_interval = train_config.get('log_interval', 10)
    save_interval = train_config.get('save_interval', 100)
    eval_interval = train_config.get('eval_interval', 0)
    eval_episodes = train_config.get('eval_episodes', 10)

    # Metrics tracker
    metrics = MetricsTracker()

    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes...")

    best_reward = -np.inf
    episode_rewards = []

    start_episode = resume_episode + 1
    for episode in range(start_episode, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss = []

        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, terminated or truncated)

            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        # Episode finished
        episode_rewards.append(episode_reward)
        metrics.add_episode(episode_reward, info)

        # Logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_loss = np.mean(episode_loss) if episode_loss else 0.0
            agent_metrics = agent.get_metrics()

            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent_metrics['epsilon']:.4f} | "
                f"Cost: ${info.get('cost', 0):.2f} | "
                f"SLA: {info.get('sla_compliance', 0):.2%}"
            )

            # TensorBoard logging
            tb_logger.log_scalar('train/episode_reward', episode_reward, episode)
            tb_logger.log_scalar('train/avg_reward', avg_reward, episode)
            tb_logger.log_scalar('train/loss', avg_loss, episode)
            tb_logger.log_scalar('train/epsilon', agent_metrics['epsilon'], episode)
            tb_logger.log_scalar('train/cost', info.get('cost', 0), episode)
            tb_logger.log_scalar('train/sla_compliance', info.get('sla_compliance', 0), episode)

            # Multi-pool specific metrics
            if env_type == 'multi_pool':
                tb_logger.log_scalar('train/total_instances', info.get('total_instances', 0), episode)
                tb_logger.log_scalar('train/total_vcpu', info.get('total_vcpu', 0), episode)

            # MLflow logging
            mlf.log_metrics({
                "episode_reward": episode_reward,
                "avg_reward": avg_reward,
                "loss": avg_loss,
                "epsilon": agent_metrics['epsilon'],
                "cost": info.get('cost', 0),
                "sla_compliance": info.get('sla_compliance', 0),
                "spot_instances": info.get('spot_instances', 0),
                "ondemand_instances": info.get('ondemand_instances', 0),
            }, step=episode)

        # Evaluation
        if eval_interval > 0 and episode % eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=eval_episodes)
            logger.info(
                f"  [Eval] Avg Reward: {eval_metrics['avg_reward']:.2f} | "
                f"Avg Cost: ${eval_metrics['avg_cost']:.2f} | "
                f"Avg SLA: {eval_metrics['avg_sla']:.2%}"
            )
            tb_logger.log_scalar('eval/avg_reward', eval_metrics['avg_reward'], episode)
            tb_logger.log_scalar('eval/avg_cost', eval_metrics['avg_cost'], episode)
            tb_logger.log_scalar('eval/avg_sla', eval_metrics['avg_sla'], episode)
            mlf.log_metrics({
                "eval/avg_reward": eval_metrics['avg_reward'],
                "eval/avg_cost": eval_metrics['avg_cost'],
                "eval/avg_sla": eval_metrics['avg_sla'],
            }, step=episode)

        # Save best model (check every episode)
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = model_dir / 'best_model.pth'
            agent.save(best_model_path)
            shutil.copy2(best_model_path, export_dir / f"{experiment_name}_best.pth")
            if episode % log_interval == 0:
                logger.info(f"New best model (reward: {best_reward:.2f})")

        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = model_dir / f'checkpoint_ep{episode}.pth'
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = model_dir / 'final_model.pth'
    agent.save(final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")
    shutil.copy2(final_model_path, export_dir / f"{experiment_name}_final.pth")

    # Save metrics
    metrics.save(exp_dir / 'metrics.pkl')

    # Log final summary & artifacts to MLflow
    summary = metrics.get_summary(window=100)
    mlf.log_metrics({
        "final/avg_reward": summary['avg_reward'],
        "final/avg_cost": summary['avg_cost'],
        "final/avg_sla": summary['avg_sla_compliance'],
        "final/avg_spot_usage": summary['avg_spot_usage'],
        "final/total_episodes": float(summary['num_episodes']),
    })
    mlf.log_model(str(model_dir / 'best_model.pth'), artifact_path="best_model")
    mlf.log_model(str(final_model_path), artifact_path="final_model")
    mlf.log_artifact(str(exp_dir / 'config.yaml'))
    mlf.log_artifact(str(exp_dir / 'metrics.pkl'))
    mlf.end_run()

    tb_logger.close()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--experiment-name', type=str, default='dqn_experiment',
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pth file to resume training')
    parser.add_argument('--resume-episode', type=int, default=0,
                        help='Episode number to resume from (e.g. 750)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    train(config, args.experiment_name,
          resume_path=args.resume, resume_episode=args.resume_episode)


if __name__ == '__main__':
    main()


# Example usage:
# Single-pool:
#   python train.py --config configs/dqn_default.yaml --experiment-name dqn_stable_price
# Multi-pool:
#   python train.py --config configs/multi_pool_stable.yaml --experiment-name mp_stable
