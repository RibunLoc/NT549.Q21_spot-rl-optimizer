"""
Baseline strategies for multi-pool (5 types × 3 AZs) environment.

6 baselines:
1. OnDemand — always request on-demand (safe, expensive)
2. Spot — always request spot (cheap, risky)
3. Threshold — spot if price < threshold, else on-demand
4. CheapestAZ — pick cheapest AZ for spot
5. CheapestType — pick cheapest instance type for spot
6. Random — uniform random (sanity check)
"""

import numpy as np
from typing import Dict, Any

from envs.instance_catalog import (
    N_TYPES, N_AZS, N_ACTIONS, INSTANCE_TYPES,
    OP_REQUEST_SPOT, OP_REQUEST_ONDEMAND, OP_DO_NOTHING,
    encode_action,
)


class BaselineAgent:
    """Base class for multi-pool baseline agents."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        raise NotImplementedError

    def reset(self):
        pass


class OnDemandAgent(BaselineAgent):
    """
    Always request on-demand instances.

    Strategy: round-robin across types and AZs to spread load.
    """

    def __init__(self, target_capacity: int = 10):
        self.target_capacity = target_capacity
        self._rr_idx = 0

    def reset(self):
        self._rr_idx = 0

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        if info:
            total = info.get('total_instances', 0)
            pending = info.get('pending_jobs', 0)
            needed = max(self.target_capacity, pending // 2)
            if total < needed:
                # Round-robin type/AZ
                t_idx = self._rr_idx % N_TYPES
                az_idx = (self._rr_idx // N_TYPES) % N_AZS
                self._rr_idx += 1
                return encode_action(OP_REQUEST_ONDEMAND, t_idx, az_idx)
        return encode_action(OP_DO_NOTHING, 0, 0)


class SpotAgent(BaselineAgent):
    """
    Always request spot instances.

    Strategy: round-robin across types and AZs.
    """

    def __init__(self, target_capacity: int = 10):
        self.target_capacity = target_capacity
        self._rr_idx = 0

    def reset(self):
        self._rr_idx = 0

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        if info:
            total = info.get('total_instances', 0)
            pending = info.get('pending_jobs', 0)
            needed = max(self.target_capacity, pending // 2)
            if total < needed:
                t_idx = self._rr_idx % N_TYPES
                az_idx = (self._rr_idx // N_TYPES) % N_AZS
                self._rr_idx += 1
                return encode_action(OP_REQUEST_SPOT, t_idx, az_idx)
        return encode_action(OP_DO_NOTHING, 0, 0)


class ThresholdAgent(BaselineAgent):
    """
    Threshold-based: use spot if price ratio < threshold, else on-demand.

    Uses observation features to estimate price level.
    """

    def __init__(self, threshold_ratio: float = 0.5, target_capacity: int = 10):
        self.threshold_ratio = threshold_ratio
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        if info is None:
            return encode_action(OP_DO_NOTHING, 0, 0)

        total = info.get('total_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending // 2)

        if total >= needed:
            return encode_action(OP_DO_NOTHING, 0, 0)

        # Top-1 cheapest combo: observation[0] = price_ratio / 2.0
        # observation[3] = az_id normalized
        top1_price_ratio = observation[0] * 2.0  # un-normalize
        top1_az = int(round(observation[3] * (N_AZS - 1)))

        # Infer type from observation[15] (best_price_rank)
        best_type = int(round(observation[15] * (N_TYPES - 1)))

        if top1_price_ratio < self.threshold_ratio:
            return encode_action(OP_REQUEST_SPOT, best_type, top1_az)
        else:
            return encode_action(OP_REQUEST_ONDEMAND, best_type, top1_az)


class CheapestAZAgent(BaselineAgent):
    """
    Always request spot in the cheapest AZ (for cheapest type).

    Uses top-1 cheapest combo from observation.
    """

    def __init__(self, target_capacity: int = 10):
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        if info is None:
            return encode_action(OP_DO_NOTHING, 0, 0)

        total = info.get('total_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending // 2)

        if total >= needed:
            return encode_action(OP_DO_NOTHING, 0, 0)

        # Cheapest AZ from observation[13]
        cheapest_az = int(round(observation[13] * (N_AZS - 1)))
        best_type = int(round(observation[15] * (N_TYPES - 1)))

        return encode_action(OP_REQUEST_SPOT, best_type, cheapest_az)


class CheapestTypeAgent(BaselineAgent):
    """
    Always request spot with cheapest type, spread across AZs.
    """

    def __init__(self, target_capacity: int = 10):
        self.target_capacity = target_capacity
        self._az_rr = 0

    def reset(self):
        self._az_rr = 0

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        if info is None:
            return encode_action(OP_DO_NOTHING, 0, 0)

        total = info.get('total_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending // 2)

        if total >= needed:
            return encode_action(OP_DO_NOTHING, 0, 0)

        best_type = int(round(observation[15] * (N_TYPES - 1)))
        az_idx = self._az_rr % N_AZS
        self._az_rr += 1

        return encode_action(OP_REQUEST_SPOT, best_type, az_idx)


class RandomAgent(BaselineAgent):
    """Uniform random action (sanity check)."""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


# Registry for easy access
BASELINES = {
    "ondemand": OnDemandAgent,
    "spot": SpotAgent,
    "threshold": ThresholdAgent,
    "cheapest_az": CheapestAZAgent,
    "cheapest_type": CheapestTypeAgent,
    "random": RandomAgent,
}
