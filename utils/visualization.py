"""
Visualization utilities for training analysis and evaluation.

Generates plots for:
- Training learning curves (reward, cost, SLA over episodes)
- Action distribution
- Cost comparison bar chart (DQN vs baselines)
- Episode timeline (price, instances, jobs over steps)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import List, Dict, Optional
import json


def set_style():
    """Set consistent matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def plot_training_curves(
    episode_rewards: List[float],
    episode_costs: List[float],
    episode_sla: List[float],
    window: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plot training learning curves with smoothing.

    Args:
        episode_rewards: List of episode rewards
        episode_costs: List of episode costs
        episode_sla: List of SLA compliance values
        window: Smoothing window size
        save_path: Path to save figure (None = show)
    """
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    episodes = np.arange(1, len(episode_rewards) + 1)

    def smooth(values, w):
        if len(values) < w:
            return values
        return pd.Series(values).rolling(window=w, min_periods=1).mean().values

    # Reward
    axes[0].plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    axes[0].plot(episodes, smooth(episode_rewards, window), color='blue',
                 linewidth=2, label=f'MA({window})')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Progress')
    axes[0].legend()

    # Cost
    axes[1].plot(episodes, episode_costs, alpha=0.3, color='red', label='Raw')
    axes[1].plot(episodes, smooth(episode_costs, window), color='red',
                 linewidth=2, label=f'MA({window})')
    axes[1].set_ylabel('Episode Cost ($)')
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
    axes[1].legend()

    # SLA
    axes[2].plot(episodes, episode_sla, alpha=0.3, color='green', label='Raw')
    axes[2].plot(episodes, smooth(episode_sla, window), color='green',
                 linewidth=2, label=f'MA({window})')
    axes[2].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='SLA Threshold (95%)')
    axes[2].set_ylabel('SLA Compliance')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylim(0, 1.05)
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_action_distribution(
    action_counts: np.ndarray,
    action_names: List[str] = None,
    title: str = "Action Distribution",
    save_path: Optional[str] = None,
):
    """
    Plot action distribution as a bar chart.

    Args:
        action_counts: Array of action counts
        action_names: List of action names
        title: Plot title
        save_path: Path to save figure
    """
    set_style()

    if action_names is None:
        action_names = [
            "Req Spot", "Req OnDemand", "Term Spot",
            "Term OnDemand", "Migrate→OD", "Migrate→Spot", "Do Nothing",
        ]

    total = action_counts.sum()
    percentages = action_counts / total * 100 if total > 0 else action_counts

    colors = ['#2196F3', '#FF9800', '#F44336', '#E91E63', '#9C27B0', '#00BCD4', '#607D8B']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(action_names, percentages, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, pct, count in zip(bars, percentages, action_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{pct:.1f}%\n({int(count)})',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_ylim(0, max(percentages) * 1.2)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved action distribution to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cost_comparison(
    results: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Plot cost comparison bar chart: DQN vs baselines.

    Args:
        results: List of evaluation result dicts (from evaluate.py)
        save_path: Path to save figure
    """
    set_style()

    names = [r['agent_name'] for r in results]
    costs = [r['avg_cost'] for r in results]
    sla_vals = [r['avg_sla'] for r in results]

    # Color: DQN = green, baselines = gray/blue
    colors = []
    for name in names:
        if 'DQN' in name:
            colors.append('#4CAF50')
        elif 'On-Demand' in name:
            colors.append('#F44336')
        elif 'Spot' in name and 'Threshold' not in name:
            colors.append('#2196F3')
        elif 'Random' in name:
            colors.append('#9E9E9E')
        else:
            colors.append('#FF9800')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Cost comparison
    bars1 = ax1.bar(names, costs, color=colors, edgecolor='white', linewidth=1.5)
    for bar, cost in zip(bars1, costs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'${cost:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Episode Cost ($)')
    ax1.set_title('Cost Comparison')
    ax1.tick_params(axis='x', rotation=30)

    # SLA comparison
    bars2 = ax2.bar(names, sla_vals, color=colors, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='SLA Threshold')
    for bar, sla in zip(bars2, sla_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{sla:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average SLA Compliance')
    ax2.set_title('SLA Compliance Comparison')
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cost comparison to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_episode_timeline(
    history_df: pd.DataFrame,
    title: str = "Episode Timeline",
    save_path: Optional[str] = None,
):
    """
    Plot an episode timeline: price, instances, jobs, reward.

    Args:
        history_df: DataFrame from env.get_episode_history()
        title: Plot title
        save_path: Path to save figure
    """
    set_style()

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    steps = history_df['step']

    # Spot price
    axes[0].plot(steps, history_df['spot_price'], color='orange', linewidth=1.5)
    axes[0].set_ylabel('Spot Price ($/hr)')
    axes[0].set_title(title)

    # Instance counts
    axes[1].fill_between(steps, 0, history_df['spot_instances'],
                         alpha=0.6, color='#2196F3', label='Spot')
    axes[1].fill_between(steps, history_df['spot_instances'],
                         history_df['spot_instances'] + history_df['ondemand_instances'],
                         alpha=0.6, color='#FF9800', label='On-Demand')
    axes[1].set_ylabel('Instances')
    axes[1].legend(loc='upper right')

    # Jobs
    axes[2].plot(steps, history_df['pending_jobs'], color='red', label='Pending', linewidth=1.5)
    axes[2].plot(steps, history_df['completed_jobs'], color='green', label='Completed (cumul.)', linewidth=1.5)
    axes[2].set_ylabel('Jobs')
    axes[2].legend(loc='upper left')

    # Reward
    axes[3].plot(steps, history_df['reward'], color='purple', alpha=0.5, linewidth=1)
    axes[3].plot(steps, history_df['reward'].rolling(window=20, min_periods=1).mean(),
                 color='purple', linewidth=2, label='MA(20)')
    axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Step Reward')
    axes[3].set_xlabel('Step')
    axes[3].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved episode timeline to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_spot_price_data(
    df: pd.DataFrame,
    title: str = "Spot Price History",
    save_path: Optional[str] = None,
):
    """
    Plot raw spot price data from CSV.

    Args:
        df: DataFrame with columns [timestamp, spot_price, instance_type, availability_zone]
        title: Plot title
        save_path: Path to save figure
    """
    set_style()

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    instance_types = df['instance_type'].unique()
    fig, ax = plt.subplots(figsize=(14, 6))

    for it in instance_types:
        subset = df[df['instance_type'] == it]
        # Group by AZ
        for az in subset['availability_zone'].unique()[:1]:  # Just first AZ per type
            az_data = subset[subset['availability_zone'] == az].sort_values('timestamp')
            ax.plot(az_data['timestamp'], az_data['spot_price'],
                    label=f'{it} ({az})', alpha=0.8, linewidth=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Spot Price ($/hr)')
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spot price plot to {save_path}")
    else:
        plt.show()
    plt.close()
