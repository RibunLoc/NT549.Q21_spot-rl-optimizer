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

from envs.instance_catalog import N_TYPES, N_AZS, INSTANCE_TYPES
from envs.action_schema import (
    Operation, N_ACTIONS, encode_action,
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
                return encode_action(Operation.PROVISION_ONDEMAND, t_idx, az_idx)
        return encode_action(Operation.HOLD, 0, 0)


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
                return encode_action(Operation.PROVISION_SPOT, t_idx, az_idx)
        return encode_action(Operation.HOLD, 0, 0)


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
            return encode_action(Operation.HOLD, 0, 0)

        total = info.get('total_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending // 2)

        if total >= needed:
            return encode_action(Operation.HOLD, 0, 0)

        # Top-1 cheapest combo: observation[0] = price_ratio / 2.0
        # observation[3] = az_id normalized
        top1_price_ratio = observation[0] * 2.0  # un-normalize
        top1_az = int(round(observation[3] * (N_AZS - 1)))

        # Infer type from observation[15] (best_price_rank)
        best_type = int(round(observation[15] * (N_TYPES - 1)))

        if top1_price_ratio < self.threshold_ratio:
            return encode_action(Operation.PROVISION_SPOT, best_type, top1_az)
        else:
            return encode_action(Operation.PROVISION_ONDEMAND, best_type, top1_az)


class CheapestAZAgent(BaselineAgent):
    """
    Always request spot in the cheapest AZ (for cheapest type).

    Uses top-1 cheapest combo from observation.
    """

    def __init__(self, target_capacity: int = 10):
        self.target_capacity = target_capacity

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        if info is None:
            return encode_action(Operation.HOLD, 0, 0)

        total = info.get('total_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending // 2)

        if total >= needed:
            return encode_action(Operation.HOLD, 0, 0)

        # Cheapest AZ from observation[13]
        cheapest_az = int(round(observation[13] * (N_AZS - 1)))
        best_type = int(round(observation[15] * (N_TYPES - 1)))

        return encode_action(Operation.PROVISION_SPOT, best_type, cheapest_az)


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
            return encode_action(Operation.HOLD, 0, 0)

        total = info.get('total_instances', 0)
        pending = info.get('pending_jobs', 0)
        needed = max(self.target_capacity, pending // 2)

        if total >= needed:
            return encode_action(Operation.HOLD, 0, 0)

        best_type = int(round(observation[15] * (N_TYPES - 1)))
        az_idx = self._az_rr % N_AZS
        self._az_rr += 1

        return encode_action(Operation.PROVISION_SPOT, best_type, az_idx)


class AutoscalerAgent(BaselineAgent):
    """
    Autoscaler phản ứng theo target vCPU — giống cách ASG / Karpenter làm trong thực tế.
    Đây là baseline RL cần thắng; các baseline cũ ở trên không scale-in và không đạt SLA.

    Mỗi step:
      1. need_out = ceil((running + pending) × headroom). Thiếu → thêm instance:
         - use_spot: spot ở pool rẻ nhất theo $/vCPU, với SPS ≥ min_sps và risk ≤ max_risk,
           bỏ qua pool vừa launch fail trong `fail_backoff` step. Không có pool hợp lệ → OD.
         - không use_spot: OD rẻ nhất theo $/vCPU.
      2. need_in = ceil((running + pending) × scale_in_headroom) (> headroom để tránh dao động).
         Thừa (bỏ 1 instance vẫn ≥ need_in) → release: OD trước, rồi spot đắt nhất theo $/vCPU.
      3. use_spot và còn OD → CONVERT_TO_SPOT nếu pool đó có spot ổn định.
      4. HOLD.

    Cần info["pools"] và info["action_mask"] (evaluate_kaggle.py truyền vào).
    """

    def __init__(
        self,
        use_spot: bool = True,
        headroom: float = 1.25,
        scale_in_headroom: float = 1.6,
        min_sps: float = 0.3,
        max_risk: float = 0.35,
        fail_backoff: int = 4,
    ):
        self.use_spot = use_spot
        self.headroom = headroom
        self.scale_in_headroom = scale_in_headroom
        self.min_sps = min_sps
        self.max_risk = max_risk
        self.fail_backoff = fail_backoff
        self._backoff: Dict[int, int] = {}
        self._last_spot_pool: int = -1

    def reset(self):
        self._backoff = {}
        self._last_spot_pool = -1

    @staticmethod
    def _ok(mask, op, pool) -> bool:
        return mask is None or bool(mask[encode_action(op, pool["type_idx"], pool["az_idx"])])

    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        hold = encode_action(Operation.HOLD, 0, 0)
        if not info or "pools" not in info:
            return hold
        mask = info.get("action_mask")
        pools = info["pools"]

        # Launch spot bước trước bị AWS từ chối → tạm né pool đó
        if info.get("spot_launch_failed") and self._last_spot_pool >= 0:
            self._backoff[self._last_spot_pool] = self.fail_backoff
        self._backoff = {k: v - 1 for k, v in self._backoff.items() if v > 1}
        self._last_spot_pool = -1

        demand = info.get("running_jobs", 0) + info.get("pending_jobs", 0)
        cur_vcpu = info.get("total_vcpu", 0)
        need_out = int(np.ceil(demand * self.headroom))
        need_in = int(np.ceil(demand * self.scale_in_headroom))

        # 1. Scale out
        if cur_vcpu < need_out or (demand > 0 and cur_vcpu == 0):
            if self.use_spot:
                cands = [
                    (i, p) for i, p in enumerate(pools)
                    if self._ok(mask, Operation.PROVISION_SPOT, p)
                    and p["sps"] >= self.min_sps and p["risk"] <= self.max_risk
                    and i not in self._backoff
                ]
                if cands:
                    i, p = min(cands, key=lambda ip: ip[1]["price"] / ip[1]["vcpus"])
                    self._last_spot_pool = i
                    return encode_action(Operation.PROVISION_SPOT, p["type_idx"], p["az_idx"])
            cands = [p for p in pools if self._ok(mask, Operation.PROVISION_ONDEMAND, p)]
            if cands:
                p = min(cands, key=lambda q: q["od_price"] / q["vcpus"])
                return encode_action(Operation.PROVISION_ONDEMAND, p["type_idx"], p["az_idx"])
            return hold

        # 2. Scale in (OD trước, rồi spot đắt nhất / vCPU)
        removable = []
        for p in pools:
            if cur_vcpu - p["vcpus"] < need_in:
                continue
            if p["od"] > 0 and self._ok(mask, Operation.RELEASE_ONDEMAND, p):
                removable.append((0, -p["od_price"] / p["vcpus"], Operation.RELEASE_ONDEMAND, p))
            if p["spot"] > 0 and self._ok(mask, Operation.RELEASE_SPOT, p):
                removable.append((1, -p["price"] / p["vcpus"], Operation.RELEASE_SPOT, p))
        if removable:
            _, _, op, p = min(removable, key=lambda r: (r[0], r[1]))
            return encode_action(op, p["type_idx"], p["az_idx"])

        # 3. Đổi OD → spot khi spot ổn định
        if self.use_spot:
            cands = [
                (i, p) for i, p in enumerate(pools)
                if p["od"] > 0 and self._ok(mask, Operation.CONVERT_TO_SPOT, p)
                and p["sps"] >= self.min_sps and p["risk"] <= self.max_risk
                and i not in self._backoff
            ]
            if cands:
                i, p = min(cands, key=lambda ip: ip[1]["price"] / ip[1]["vcpus"])
                self._last_spot_pool = i
                return encode_action(Operation.CONVERT_TO_SPOT, p["type_idx"], p["az_idx"])

        return hold


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
    "autoscaler": AutoscalerAgent,
}
