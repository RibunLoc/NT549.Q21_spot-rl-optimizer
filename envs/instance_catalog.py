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


# 5 instance types spanning compute, memory, and general-purpose
INSTANCE_TYPES: List[InstanceType] = [
    InstanceType("m5.large",    2,   8.0, 0.096),
    InstanceType("c5.xlarge",   4,   8.0, 0.170),
    InstanceType("r5.large",    2,  16.0, 0.126),
    InstanceType("m5.xlarge",   4,  16.0, 0.192),
    InstanceType("c5.2xlarge",  8,  16.0, 0.340),
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
N_TOP_K_FEATURES = 4        # price, interrupt, vcpu/$, az_id
N_MULTI_AZ = 3              # spread, cheapest_az, concentration
N_MULTI_TYPE = 3            # price_rank, interrupt_rank, vcpu_ratio
N_INFRA = 3                 # spot, ondemand, capacity
N_WORKLOAD = 4              # pending, running, forecast, queue_wait
N_TIME = 3                  # hour, day, progress
N_CURRENT = 5               # avg_price, avg_interrupt, spot_ratio, cost_rate, sla_health
N_EXTRA = 9                 # budget, idle_spot, trend, interrupt_streak,
                            # pool_price, price_trend, cheaper_avail, od_gap, sla_risk
N_POOL_CPU = N_TYPES * N_AZS   # 15: per-pool CPU util
N_POOL_RAM = N_TYPES * N_AZS   # 15: per-pool RAM util

STATE_DIM = (
    N_TOP_K * N_TOP_K_FEATURES   # 12
    + N_MULTI_AZ                  # 3
    + N_MULTI_TYPE                # 3
    + N_INFRA                     # 3
    + N_WORKLOAD                  # 4
    + N_TIME                      # 3
    + N_CURRENT                   # 5
    + N_EXTRA                     # 9
    + N_POOL_CPU                  # 15
    + N_POOL_RAM                  # 15
)  # = 72

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
