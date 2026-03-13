"""
Baseline strategies for comparison.

Implements simple heuristic-based policies:
- Always use on-demand (safe but expensive)
- Always use spot (cheap but risky)
- Threshold-based (request spot if price < threshold)
- Random (sanity check)
"""

import numpy as np
from typing import Dict, Any


class BaselineAgent:
    """Base class for baseline agents."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """
        Select action given observation.

        Args:
            observation: Environment observation
            info: Additional info from environment

        Returns:
            Action (int)
        """
        raise NotImplementedError

    def reset(self):
        """Reset agent state (if needed)."""
        pass


class AlwaysOnDemandAgent(BaselineAgent):
    """
    Always use on-demand instances.

    Strategy: Request on-demand when needed, never use spot.
    - Pros: No interruptions, guaranteed availability
    - Cons: Most expensive option
    """

    ACTION_REQUEST_ONDEMAND = 1
    ACTION_DO_NOTHING = 5

    def __init__(self, target_capacity: int = 5):
        """
        Initialize agent.

        Args:
            target_capacity: Target number of instances to maintain
        """
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Select action: request on-demand if below capacity."""
        # TODO: Parse observation to get current capacity
        # For now, simple logic: request on-demand if capacity low

        if info and 'ondemand_instances' in info:
            current_ondemand = info['ondemand_instances']
            if current_ondemand < self.target_capacity:
                return self.ACTION_REQUEST_ONDEMAND

        return self.ACTION_DO_NOTHING


class AlwaysSpotAgent(BaselineAgent):
    """
    Always use spot instances (ignore price).

    Strategy: Request spot when needed, never on-demand.
    - Pros: Cheapest option (on average)
    - Cons: High risk of interruptions, may violate SLA
    """

    ACTION_REQUEST_SPOT = 0
    ACTION_DO_NOTHING = 5

    def __init__(self, target_capacity: int = 5):
        """
        Initialize agent.

        Args:
            target_capacity: Target number of spot instances
        """
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Select action: request spot if below capacity."""
        if info and 'spot_instances' in info:
            current_spot = info['spot_instances']
            if current_spot < self.target_capacity:
                return self.ACTION_REQUEST_SPOT

        return self.ACTION_DO_NOTHING


class ThresholdBasedAgent(BaselineAgent):
    """
    Threshold-based policy: use spot if price below threshold.

    Strategy:
    - If spot price < threshold * on-demand price: request spot
    - Otherwise: request on-demand
    - Threshold typically 0.3-0.5 (30-50% of on-demand)
    """

    ACTION_REQUEST_SPOT = 0
    ACTION_REQUEST_ONDEMAND = 1
    ACTION_DO_NOTHING = 5

    def __init__(
        self,
        threshold_ratio: float = 0.4,
        ondemand_price: float = 0.096,
        target_capacity: int = 5,
    ):
        """
        Initialize agent.

        Args:
            threshold_ratio: Threshold as fraction of on-demand price (0-1)
            ondemand_price: On-demand price ($/hour)
            target_capacity: Target capacity
        """
        self.threshold_ratio = threshold_ratio
        self.ondemand_price = ondemand_price
        self.target_capacity = target_capacity
        self.price_threshold = threshold_ratio * ondemand_price

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Select action based on current spot price."""
        # TODO: Parse observation to extract spot price
        # Observation format: [spot_price, ..., num_spot, num_ondemand, ...]

        # Assume observation[0] is current spot price
        current_spot_price = observation[0]

        if info:
            total_instances = info.get('spot_instances', 0) + info.get('ondemand_instances', 0)
            if total_instances >= self.target_capacity:
                return self.ACTION_DO_NOTHING

        # Decide based on price
        if current_spot_price < self.price_threshold:
            return self.ACTION_REQUEST_SPOT
        else:
            return self.ACTION_REQUEST_ONDEMAND


class RandomAgent(BaselineAgent):
    """
    Random agent: select random action.

    Used as sanity check - any reasonable policy should beat this.
    """

    def __init__(self, num_actions: int = 6, seed: int = None):
        """
        Initialize agent.

        Args:
            num_actions: Number of available actions
            seed: Random seed
        """
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Select random action."""
        return self.rng.integers(0, self.num_actions)


# TODO: Add more sophisticated baselines
# - Time-of-day aware: use spot during low-price hours
# - Workload-aware: use on-demand during high-priority jobs
# - Interruption-aware: avoid spot when interruption prob high
