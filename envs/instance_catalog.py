"""
Instance type catalog and AZ definitions for multi-pool spot optimization.

Defines the instance types, availability zones, and constants used by
SpotOrchestratorEnv for multi-type × multi-AZ cost-aware orchestration.
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

# Dimensions
N_TYPES = len(INSTANCE_TYPES)       # 5
N_AZS = len(AVAILABILITY_ZONES)    # 3
N_OPS = 7                           # 7 operations
N_ACTIONS = N_OPS * N_TYPES * N_AZS  # 105

# State dimensions
N_TOP_K = 3                         # top-3 cheapest combos
N_TOP_K_FEATURES = 4                # price, interrupt, vcpu/$, az_id
N_MULTI_AZ = 3                      # spread, cheapest_az, concentration
N_MULTI_TYPE = 3                    # price_rank, interrupt_rank, vcpu_ratio
N_INFRA = 3                         # spot, ondemand, capacity
N_WORKLOAD = 4                      # pending, running, forecast, queue_wait
N_TIME = 3                          # hour, day, progress
N_CURRENT = 5                       # avg_price, avg_interrupt, spot_ratio, cost_rate, sla_health

STATE_DIM = (
    N_TOP_K * N_TOP_K_FEATURES      # 12
    + N_MULTI_AZ                     # 3
    + N_MULTI_TYPE                   # 3
    + N_INFRA                        # 3
    + N_WORKLOAD                     # 4
    + N_TIME                         # 3
    + N_CURRENT                      # 5
)  # = 33

# Instance constraints
MAX_INSTANCES = 20                   # total across all types & AZs
MAX_PER_AZ = 10                     # max instances in a single AZ
MAX_VCPU = MAX_INSTANCES * max(t.vcpus for t in INSTANCE_TYPES)  # 160
MAX_JOBS = 100                      # queue cap
MAX_WAIT = 10                       # max queue wait (steps)

# Operation indices
OP_REQUEST_SPOT = 0
OP_REQUEST_ONDEMAND = 1
OP_TERMINATE_SPOT = 2
OP_TERMINATE_ONDEMAND = 3
OP_MIGRATE_TO_ONDEMAND = 4
OP_MIGRATE_TO_SPOT = 5
OP_DO_NOTHING = 6

OP_NAMES = [
    "REQUEST_SPOT", "REQUEST_ONDEMAND", "TERMINATE_SPOT",
    "TERMINATE_ONDEMAND", "MIGRATE_TO_ONDEMAND", "MIGRATE_TO_SPOT",
    "DO_NOTHING",
]


def decode_action(action: int):
    """Decode flat action index → (operation, type_idx, az_idx)."""
    op = action // (N_TYPES * N_AZS)
    remainder = action % (N_TYPES * N_AZS)
    type_idx = remainder // N_AZS
    az_idx = remainder % N_AZS
    return op, type_idx, az_idx


def encode_action(op: int, type_idx: int, az_idx: int) -> int:
    """Encode (operation, type_idx, az_idx) → flat action index."""
    return op * (N_TYPES * N_AZS) + type_idx * N_AZS + az_idx


def get_instance_type(idx: int) -> InstanceType:
    """Get InstanceType by index."""
    return INSTANCE_TYPES[idx]


def get_od_price(type_idx: int) -> float:
    """Get on-demand price for instance type."""
    return INSTANCE_TYPES[type_idx].ondemand_price


def get_max_od_price() -> float:
    """Get maximum on-demand price across all types."""
    return max(t.ondemand_price for t in INSTANCE_TYPES)
