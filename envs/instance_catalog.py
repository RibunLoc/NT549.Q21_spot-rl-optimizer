"""
Instance type catalog and AZ definitions for multi-pool spot optimization.

Contains ONLY static infrastructure data:
- Instance types (specs, prices)
- Availability zones
- Environment constraints (MAX_INSTANCES, MAX_VCPU, ...)
- State dimensions

Action-related constants (Operation enum, encode_action, decode_action, N_ACTIONS)
live in envs.action_schema. Import from there.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InstanceType:
    """AWS EC2 instance type specification."""
    name: str
    vcpus: int
    memory_gb: float
    ondemand_price: float  # $/hour in ap-southeast-1
    # Interruption model: P = base_rate + (1 - capacity_depth) * 0.35 + ...
    # base_rate: AWS natural churn even at stable price (from Spot Instance Advisor)
    # alpha:     price-sensitivity coefficient (legacy, weak signal in v14)
    base_interrupt_rate: float = 0.02
    interrupt_alpha: float = 0.5
    # Long-term average spot price as fraction of OD (from Spot Advisor "savings")
    # Used as STATIC feature: agent knows baseline savings expectation per type
    long_term_spot_ratio: float = 0.35  # ≈ spot avg is 35% of OD → 65% savings


# 5 instance types spanning compute, memory, and general-purpose
# alpha calibrated by instance family capacity depth (AWS Spot Advisor data):
#   small general (m5.large):  shallower pool → more sensitive → alpha=0.6
#   compute (c5.xlarge):       medium pool                     → alpha=0.5
#   memory (r5.large):         medium pool, higher base churn  → alpha=0.5, base=0.03
#   large general (m5.xlarge): deeper pool                     → alpha=0.4
#   large compute (c5.2xlarge):deepest pool, most stable       → alpha=0.3
INSTANCE_TYPES: List[InstanceType] = [
    InstanceType("m5.large",   2,  8.0, 0.096, 0.02, 0.6, 0.36),
    InstanceType("c5.xlarge",  4,  8.0, 0.170, 0.02, 0.5, 0.36),
    InstanceType("r5.large",   2, 16.0, 0.126, 0.03, 0.5, 0.35),
    InstanceType("m5.xlarge",  4, 16.0, 0.192, 0.02, 0.4, 0.35),
    InstanceType("c5.2xlarge", 8, 16.0, 0.340, 0.02, 0.3, 0.35),
]

AVAILABILITY_ZONES: List[str] = [
    "ap-southeast-1a",
    "ap-southeast-1b",
    "ap-southeast-1c",
]

# ─── Dimensions ──────────────────────────────────────────────────────────────
N_TYPES = len(INSTANCE_TYPES)    # 5
N_AZS = len(AVAILABILITY_ZONES)  # 3

# State dimensions (documented per feature group)
N_TOP_K = 3                 # top-3 cheapest combos
# v14: 10 features = 4 original + 2 static + 2 dynamic + 2 event flags
#   price_ratio, interrupt_prob, vcpu/$, az_id            [dynamic]
#   base_rate, baseline_savings                            [static]
#   sps_score, price_cv_24h                                [dynamic]
#   rebalance_flag, termination_flag                       [event]
N_TOP_K_FEATURES = 10
N_MULTI_AZ = 3              # spread, cheapest_az, concentration
N_MULTI_TYPE = 3            # price_rank, interrupt_rank, vcpu_ratio
N_INFRA = 3                 # spot, ondemand, capacity
N_WORKLOAD = 6              # pending, running, forecast, queue_wait, avg_cpu_demand, avg_ram_demand
N_TIME = 3                  # hour, day, progress
N_CURRENT = 5               # avg_price, avg_interrupt, spot_ratio, cost_rate, sla_health
N_EXTRA = 9                 # budget, idle_spot, trend, interrupt_streak,
                            # pool_price, price_trend, cheaper_avail, od_gap, sla_risk
N_POOL_CPU = N_TYPES * N_AZS   # 15: per-pool CPU util
N_POOL_RAM = N_TYPES * N_AZS   # 15: per-pool RAM util

STATE_DIM = (
    N_TOP_K * N_TOP_K_FEATURES   # 30
    + N_MULTI_AZ                  # 3
    + N_MULTI_TYPE                # 3
    + N_INFRA                     # 3
    + N_WORKLOAD                  # 6
    + N_TIME                      # 3
    + N_CURRENT                   # 5
    + N_EXTRA                     # 9
    + N_POOL_CPU                  # 15
    + N_POOL_RAM                  # 15
)  # = 92

# ─── Environment constraints ─────────────────────────────────────────────────
MAX_INSTANCES = 20                                                 # across all pools
MAX_PER_AZ = 10                                                    # per single AZ
MAX_VCPU = MAX_INSTANCES * max(t.vcpus for t in INSTANCE_TYPES)    # 160
MAX_JOBS = 100                                                     # queue cap
MAX_WAIT = 10                                                      # max queue wait (steps)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_instance_type(idx: int) -> InstanceType:
    """Get InstanceType by index."""
    return INSTANCE_TYPES[idx]


def get_od_price(type_idx: int) -> float:
    """Get on-demand price for instance type."""
    return INSTANCE_TYPES[type_idx].ondemand_price


def get_max_od_price() -> float:
    """Get maximum on-demand price across all types."""
    return max(t.ondemand_price for t in INSTANCE_TYPES)


# ─── Deprecated action-related re-exports ────────────────────────────────────
# Old code may still do `from envs.instance_catalog import N_ACTIONS, encode_action, ...`.
# Re-export from action_schema for backward compatibility. NEW CODE: import from
# envs.action_schema directly.
def _reexport_from_action_schema():
    """Lazy re-export to avoid circular import at module load."""
    from envs import action_schema as _a
    return _a


def __getattr__(name: str):
    # PEP 562: lazy attribute access for backward-compat action constants.
    _compat = {
        "N_ACTIONS", "N_POOL_ACTIONS", "N_OPS", "HOLD_ACTION", "DO_NOTHING_ACTION",
        "OP_REQUEST_SPOT", "OP_REQUEST_ONDEMAND",
        "OP_TERMINATE_SPOT", "OP_TERMINATE_ONDEMAND",
        "OP_MIGRATE_TO_ONDEMAND", "OP_MIGRATE_TO_SPOT",
        "OP_MIGRATE_SPOT_TO_SPOT", "OP_RESERVE_CAPACITY", "OP_DO_NOTHING",
        "OP_NAMES",
        "encode_action", "decode_action",
    }
    if name in _compat:
        schema = _reexport_from_action_schema()
        if name == "OP_NAMES":
            return schema.OPERATION_NAMES
        if name == "N_OPS":
            return schema.N_POOL_OPS
        return getattr(schema, name)
    raise AttributeError(f"module 'envs.instance_catalog' has no attribute {name!r}")
