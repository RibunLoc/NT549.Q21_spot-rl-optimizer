"""
RL agents for Spot Instance optimization.
"""

from agents.dqn_agent import DQNAgent
from agents.baselines import (
    OnDemandAgent,
    SpotAgent,
    ThresholdAgent,
    CheapestAZAgent,
    CheapestTypeAgent,
    RandomAgent,
    BASELINES,
)

__all__ = [
    "DQNAgent",
    "OnDemandAgent",
    "SpotAgent",
    "ThresholdAgent",
    "CheapestAZAgent",
    "CheapestTypeAgent",
    "RandomAgent",
    "BASELINES",
]
