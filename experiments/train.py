"""
Training script for DQN agent.

Usage:
    python train.py --config configs/dqn_default.yaml
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

from envs.spot_env import SpotInstanceEnv
from agents.dqn_agent import DQNAgent
from utils.logger import setup_logger, TensorBoardLogger
from utils.metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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

    # Setup logging
    setup_logger(log_dir / 'train.log')
    tb_logger = TensorBoardLogger(log_dir)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Config: {config}")

    # Create environment
    env_config = config['env']
    env = SpotInstanceEnv(
        data_path=env_config['data_path'],
        max_steps=env_config.get('max_steps', 1000),
        sla_threshold=env_config.get('sla_threshold', 0.95),
        spot_capacity=env_config.get('spot_capacity', 10),
        ondemand_capacity=env_config.get('ondemand_capacity', 5),
        workload_config=env_config.get('workload', {}),
        cost_config=env_config.get('cost', {}),
    )

    logger.info(f"Environment created: state_dim={env.observation_space.shape[0]}, "
                f"action_dim={env.action_space.n}")

    # Create agent
    agent_config = config['agent']
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
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

        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = model_dir / f'checkpoint_ep{episode}.pth'
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = model_dir / 'best_model.pth'
                agent.save(best_model_path)
                logger.info(f"Saved best model (reward: {best_reward:.2f})")
                shutil.copy2(best_model_path, export_dir / f"{experiment_name}_best.pth")

    # Save final model
    final_model_path = model_dir / 'final_model.pth'
    agent.save(final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")
    shutil.copy2(final_model_path, export_dir / f"{experiment_name}_final.pth")

    # Save metrics
    metrics.save(exp_dir / 'metrics.pkl')

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
# python train.py --config configs/dqn_default.yaml --experiment-name dqn_stable_price
