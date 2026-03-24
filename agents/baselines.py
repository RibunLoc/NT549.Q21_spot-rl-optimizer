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
    ACTION_DO_NOTHING = 6

    def __init__(self, target_capacity: int = 5):
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Request on-demand until capacity đủ cho pending + running jobs."""
        if info:
            current_ondemand = info.get('ondemand_instances', 0)
            pending = info.get('pending_jobs', 0)
            # Scale theo demand, không chỉ theo target cố định
            needed = max(self.target_capacity, pending)
            if current_ondemand < needed:
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
    ACTION_DO_NOTHING = 6

    def __init__(self, target_capacity: int = 5):
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Request spot until capacity đủ cho pending + running jobs."""
        if info:
            current_spot = info.get('spot_instances', 0)
            pending = info.get('pending_jobs', 0)
            needed = max(self.target_capacity, pending)
            if current_spot < needed:
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
    ACTION_DO_NOTHING = 6

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
        """Select action based on current spot price (from info, not normalized obs)."""
        if not info:
            return self.ACTION_DO_NOTHING

        current_spot_price = info.get('spot_price', self.ondemand_price)
        total_instances = info.get('spot_instances', 0) + info.get('ondemand_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending)

        if total_instances >= needed:
            return self.ACTION_DO_NOTHING

        # Chọn spot nếu giá dưới ngưỡng, ngược lại dùng on-demand
        if current_spot_price < self.price_threshold:
            return self.ACTION_REQUEST_SPOT
        else:
            return self.ACTION_REQUEST_ONDEMAND


class RandomAgent(BaselineAgent):
    """
    Random agent: select random action.

    Used as sanity check - any reasonable policy should beat this.
    """

    def __init__(self, num_actions: int = 7, seed: int = None):
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
