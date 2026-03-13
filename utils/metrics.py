"""
Metrics tracking and evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import pickle
from pathlib import Path


class MetricsTracker:
    """
    Track and compute metrics during training and evaluation.
    """

    def __init__(self):
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_sla_compliance = []
        self.episode_spot_usage = []
        self.episode_info = []

    def add_episode(self, reward: float, info: Dict[str, Any]):
        """
        Add episode metrics.

        Args:
            reward: Episode reward
            info: Episode info dictionary
        """
        self.episode_rewards.append(reward)
        self.episode_costs.append(info.get('cost', 0.0))
        self.episode_sla_compliance.append(info.get('sla_compliance', 0.0))

        # Calculate spot usage ratio
        spot = info.get('spot_instances', 0)
        ondemand = info.get('ondemand_instances', 0)
        total = spot + ondemand
        spot_ratio = spot / total if total > 0 else 0.0
        self.episode_spot_usage.append(spot_ratio)

        self.episode_info.append(info)

    def get_summary(self, window: int = 100) -> Dict[str, float]:
        """
        Get summary statistics for recent episodes.

        Args:
            window: Number of recent episodes to consider

        Returns:
            Summary statistics dictionary
        """
        recent_rewards = self.episode_rewards[-window:]
        recent_costs = self.episode_costs[-window:]
        recent_sla = self.episode_sla_compliance[-window:]
        recent_spot_usage = self.episode_spot_usage[-window:]

        return {
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'avg_cost': np.mean(recent_costs) if recent_costs else 0.0,
            'avg_sla_compliance': np.mean(recent_sla) if recent_sla else 0.0,
            'avg_spot_usage': np.mean(recent_spot_usage) if recent_spot_usage else 0.0,
            'num_episodes': len(self.episode_rewards),
        }

    def compute_cost_savings(self, baseline_cost: float) -> float:
        """
        Compute cost savings compared to baseline.

        Args:
            baseline_cost: Baseline cost (e.g., all on-demand)

        Returns:
            Percentage cost savings
        """
        avg_cost = np.mean(self.episode_costs)
        savings = (baseline_cost - avg_cost) / baseline_cost * 100
        return savings

    def save(self, path: Path):
        """Save metrics to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        metrics_dict = {
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'episode_sla_compliance': self.episode_sla_compliance,
            'episode_spot_usage': self.episode_spot_usage,
            'episode_info': self.episode_info,
        }

        with open(path, 'wb') as f:
            pickle.dump(metrics_dict, f)

    def load(self, path: Path):
        """Load metrics from file."""
        with open(path, 'rb') as f:
            metrics_dict = pickle.load(f)

        self.episode_rewards = metrics_dict['episode_rewards']
        self.episode_costs = metrics_dict['episode_costs']
        self.episode_sla_compliance = metrics_dict['episode_sla_compliance']
        self.episode_spot_usage = metrics_dict['episode_spot_usage']
        self.episode_info = metrics_dict['episode_info']

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        return pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'cost': self.episode_costs,
            'sla_compliance': self.episode_sla_compliance,
            'spot_usage': self.episode_spot_usage,
        })


def compute_evaluation_metrics(
    episode_costs: List[float],
    episode_sla: List[float],
    baseline_cost: float,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        episode_costs: List of episode costs
        episode_sla: List of SLA compliance rates
        baseline_cost: Baseline cost (all on-demand)

    Returns:
        Metrics dictionary
    """
    avg_cost = np.mean(episode_costs)
    cost_savings = (baseline_cost - avg_cost) / baseline_cost * 100
    avg_sla = np.mean(episode_sla)
    sla_violations = sum(1 for sla in episode_sla if sla < 0.95) / len(episode_sla) * 100

    return {
        'avg_cost': avg_cost,
        'cost_savings_pct': cost_savings,
        'avg_sla_compliance': avg_sla,
        'sla_violation_rate_pct': sla_violations,
        'num_episodes': len(episode_costs),
    }
