"""
Action schema for Spot Instance Orchestrator — v2 (factored, AWS-aligned).

Design principles:
1. AWS EC2 API-aligned naming (PROVISION = run-instances, RELEASE = terminate-instances)
2. Factored representation: Action = (Operation, Pool), Pool = (Type, AZ)
3. Operations grouped by lifecycle phase (PROVISION / RELEASE / CONVERT / REBALANCE / RESERVE / HOLD)
4. Extensible: thêm Operation = thêm 1 IntEnum value, không đổi state/network dim

Taxonomy (9 ops):

    PROVISION phase — tạo capacity mới
        PROVISION_SPOT       — run-instances --market-type spot
        PROVISION_ONDEMAND   — run-instances (default billing)

    RELEASE phase — giải phóng capacity
        RELEASE_SPOT         — terminate-instances (spot)
        RELEASE_ONDEMAND     — terminate-instances (on-demand)

    CONVERT phase — đổi billing model, giữ workload ở cùng pool
        CONVERT_TO_ONDEMAND  — graceful fallback khi spot price/interrupt risk cao
        CONVERT_TO_SPOT      — opportunistic savings khi spot market ổn

    REBALANCE phase — tối ưu phân bố
        REBALANCE_SPOT       — chuyển spot từ pool đắt nhất → pool được chỉ định (rẻ hơn)

    RESERVE phase — pre-allocate capacity before interrupt (NEW in v2)
        RESERVE_CAPACITY     — provision OD phòng bị khi P(interrupt) cao

    Null action
        HOLD                 — no-op (đợi thêm 1 step)

Action encoding:
    Pool-targeted ops (8 ops × N_POOLS slots): index = op * N_POOLS + pool_idx
    HOLD (special):                            index = N_POOL_ACTIONS

    Với N_TYPES=5, N_AZS=3 → N_POOLS=15
    Total: 8 * 15 + 1 = 121 discrete actions (same như v1 để dataset cũ vẫn load được)
"""

from enum import IntEnum
from typing import Tuple, List

from envs.instance_catalog import (
    INSTANCE_TYPES, AVAILABILITY_ZONES, N_TYPES, N_AZS,
)


# ─── Operation enum ──────────────────────────────────────────────────────────
class Operation(IntEnum):
    """Lifecycle operations for EC2 instance orchestration."""
    # PROVISION phase
    PROVISION_SPOT       = 0
    PROVISION_ONDEMAND   = 1
    # RELEASE phase
    RELEASE_SPOT         = 2
    RELEASE_ONDEMAND     = 3
    # CONVERT phase (same pool, change billing)
    CONVERT_TO_ONDEMAND  = 4   # spot → on-demand
    CONVERT_TO_SPOT      = 5   # on-demand → spot
    # REBALANCE phase
    REBALANCE_SPOT       = 6   # move spot from expensive pool → target pool
    # RESERVE phase (new in v2)
    RESERVE_CAPACITY     = 7   # preemptive OD when interrupt risk high
    # Null
    HOLD                 = 8


# Operation groups (for logging, metrics, action masks)
PROVISION_OPS = {Operation.PROVISION_SPOT, Operation.PROVISION_ONDEMAND, Operation.RESERVE_CAPACITY}
RELEASE_OPS   = {Operation.RELEASE_SPOT, Operation.RELEASE_ONDEMAND}
CONVERT_OPS   = {Operation.CONVERT_TO_ONDEMAND, Operation.CONVERT_TO_SPOT}
REBALANCE_OPS = {Operation.REBALANCE_SPOT}


# Human-readable names (ordered by enum value)
OPERATION_NAMES: List[str] = [op.name for op in Operation]


# ─── Dimensions ──────────────────────────────────────────────────────────────
N_POOLS = N_TYPES * N_AZS                  # 15
N_POOL_OPS = 8                             # all Operations EXCEPT HOLD are pool-targeted
N_POOL_ACTIONS = N_POOL_OPS * N_POOLS      # 120
N_ACTIONS = N_POOL_ACTIONS + 1             # 121 (HOLD added as final action)
HOLD_ACTION = N_POOL_ACTIONS               # flat index of HOLD


# ─── Encoding / Decoding ─────────────────────────────────────────────────────
def encode_action(op: Operation, type_idx: int = 0, az_idx: int = 0) -> int:
    """Encode (operation, pool) → flat action index.

    HOLD ignores type_idx/az_idx.

    Raises:
        ValueError: if pool indices out of range for non-HOLD ops.
    """
    if op == Operation.HOLD:
        return HOLD_ACTION
    if not (0 <= type_idx < N_TYPES):
        raise ValueError(f"type_idx {type_idx} out of range [0, {N_TYPES})")
    if not (0 <= az_idx < N_AZS):
        raise ValueError(f"az_idx {az_idx} out of range [0, {N_AZS})")
    return int(op) * N_POOLS + type_idx * N_AZS + az_idx


def decode_action(action: int) -> Tuple[Operation, int, int]:
    """Decode flat action index → (operation, type_idx, az_idx).

    For HOLD, returns (Operation.HOLD, 0, 0) — pool indices are ignored.
    """
    if action == HOLD_ACTION:
        return Operation.HOLD, 0, 0
    if not (0 <= action < N_POOL_ACTIONS):
        raise ValueError(f"action {action} out of range [0, {N_ACTIONS})")
    op_idx = action // N_POOLS
    pool_idx = action % N_POOLS
    type_idx = pool_idx // N_AZS
    az_idx = pool_idx % N_AZS
    return Operation(op_idx), type_idx, az_idx


def pool_to_idx(type_idx: int, az_idx: int) -> int:
    """Pool flat index (0..N_POOLS-1)."""
    return type_idx * N_AZS + az_idx


def idx_to_pool(pool_idx: int) -> Tuple[int, int]:
    """Pool flat index → (type_idx, az_idx)."""
    return pool_idx // N_AZS, pool_idx % N_AZS


def action_label(action: int) -> str:
    """Human-readable label for logging/debug."""
    op, t, az = decode_action(action)
    if op == Operation.HOLD:
        return "HOLD"
    type_name = INSTANCE_TYPES[t].name
    az_name = AVAILABILITY_ZONES[az]
    return f"{op.name}({type_name}@{az_name})"


# ─── Backward-compat aliases (DEPRECATED, for migration only) ────────────────
# Old code uses OP_REQUEST_SPOT etc. Keep aliases so we don't break controllers
# that were exported before the rename. New code must import Operation.*
OP_REQUEST_SPOT        = int(Operation.PROVISION_SPOT)
OP_REQUEST_ONDEMAND    = int(Operation.PROVISION_ONDEMAND)
OP_TERMINATE_SPOT      = int(Operation.RELEASE_SPOT)
OP_TERMINATE_ONDEMAND  = int(Operation.RELEASE_ONDEMAND)
OP_MIGRATE_TO_ONDEMAND = int(Operation.CONVERT_TO_ONDEMAND)
OP_MIGRATE_TO_SPOT     = int(Operation.CONVERT_TO_SPOT)
OP_MIGRATE_SPOT_TO_SPOT = int(Operation.REBALANCE_SPOT)
OP_RESERVE_CAPACITY    = int(Operation.RESERVE_CAPACITY)
OP_DO_NOTHING          = int(Operation.HOLD)
DO_NOTHING_ACTION      = HOLD_ACTION
