"""
Test agent inference: đưa 1 state vector → agent trả action.

Usage:
    python scripts/test_agent_inference.py                           # preset scenarios
    python scripts/test_agent_inference.py --model <path.pth>        # chọn model khác
    python scripts/test_agent_inference.py --interactive             # nhập state bằng tay
    python scripts/test_agent_inference.py --from-env                # dùng state thật từ env

Agent expects state = np.ndarray shape (72,), all values normalized [0, 1].
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from envs.instance_catalog import STATE_DIM, INSTANCE_TYPES, AVAILABILITY_ZONES
from envs.action_schema import N_ACTIONS, action_label, decode_action
from agents.dqn_agent import DQNAgent


def load_agent(model_path: str, device: str = 'cpu') -> DQNAgent:
    """Khởi tạo agent với config khớp training và load checkpoint."""
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        hidden_dim=512,
        network_type='factored',
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=500_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=300_000,
        target_update_freq=500,
        device=device,
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # greedy (tắt exploration khi test)
    return agent


def query_agent(agent: DQNAgent, state: np.ndarray, top_k: int = 5, mask: np.ndarray = None):
    """Feed state → agent trả top-k actions theo Q-value."""
    assert state.shape == (STATE_DIM,), f"state phải shape ({STATE_DIM},), got {state.shape}"

    s = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.q_network(s).cpu().numpy().squeeze(0)  # shape (121,)

    if mask is not None:
        q_values_masked = q_values.copy()
        q_values_masked[~mask.astype(bool)] = -1e9
    else:
        q_values_masked = q_values

    top_indices = np.argsort(q_values_masked)[::-1][:top_k]

    print(f"\nTop-{top_k} actions theo Q-value:")
    print(f"{'Rank':<5} {'Action':<6} {'Q-value':<10} {'Label'}")
    print("-" * 80)
    for rank, a in enumerate(top_indices, 1):
        marker = ' *' if rank == 1 else '  '
        print(f"{rank:<5}{marker}{a:<4} {q_values[a]:<+10.3f} {action_label(int(a))}")

    chosen = int(top_indices[0])
    op, t, az = decode_action(chosen)
    print(f"\n→ Agent chọn action={chosen}")
    print(f"  Operation: {op.name}")
    if op.name != 'HOLD':
        print(f"  Instance type: {INSTANCE_TYPES[t].name} ({INSTANCE_TYPES[t].vcpus} vCPU, ${INSTANCE_TYPES[t].ondemand_price}/h OD)")
        print(f"  AZ: {AVAILABILITY_ZONES[az]}")
    return chosen, q_values


# ─── Preset scenarios (state vectors giả lập) ─────────────────────────────────

def make_state_empty_cluster_with_jobs() -> np.ndarray:
    """Cluster rỗng, có job đang chờ → agent nên PROVISION."""
    s = np.zeros(STATE_DIM, dtype=np.float32)
    # Pool stats (index 12..56): giá spot ~50% OD, interrupt ~5% → market tốt
    for pool_idx in range(15):
        base = 12 + pool_idx * 3  # 3 features/pool (spot_price_ratio, interrupt, vcpu/$)
        s[base + 0] = 0.5   # spot price ratio
        s[base + 1] = 0.05  # interrupt rate
        s[base + 2] = 0.8   # vcpu per dollar (cao = tốt)
    # Top-3 cheapest (index 0..11): pool rẻ nhất là m5.large az-1a
    s[0] = 0.4   # price_ratio top-1
    s[1] = 0.03  # interrupt top-1
    s[2] = 0.9   # vcpu/$ top-1
    s[3] = 0.0   # az_id top-1 (az-1a)
    # Infra (index 57..59): 0 spot, 0 OD, 0 vCPU
    # Workload (index 60..63): 10 pending jobs
    s[60] = 0.3  # pending/MAX_JOBS
    s[61] = 0.0  # running
    s[62] = 0.3  # forecast_1h
    s[63] = 0.0  # queue_wait
    # Time (index 64..66)
    s[64] = 0.5  # hour
    s[65] = 0.5  # day
    s[66] = 0.1  # progress
    # Current (index 67..71)
    s[67] = 0.5  # avg_spot_price
    s[68] = 0.05 # avg_interrupt
    s[69] = 0.0  # spot_ratio
    s[70] = 0.0  # cost_rate
    s[71] = 0.5  # sla_health
    return s


def make_state_high_interrupt_spot_heavy() -> np.ndarray:
    """Đang có nhiều spot, interrupt rate tăng cao → agent nên CONVERT_TO_OD hoặc RESERVE."""
    s = np.zeros(STATE_DIM, dtype=np.float32)
    for pool_idx in range(15):
        base = 12 + pool_idx * 3
        s[base + 0] = 0.7
        s[base + 1] = 0.35   # interrupt 35% -- rất cao
        s[base + 2] = 0.5
    s[0] = 0.6; s[1] = 0.3; s[2] = 0.6; s[3] = 0.0
    # Infra: 8 spot, 0 OD
    s[57] = 0.4   # num_spot
    s[58] = 0.0   # num_od
    s[59] = 0.3   # total_vcpu
    # Workload
    s[60] = 0.1   # pending
    s[61] = 0.4   # running (jobs đang chạy trên spot → nguy!)
    s[62] = 0.1   # forecast
    s[63] = 0.0
    s[64] = 0.5; s[65] = 0.3; s[66] = 0.5
    s[67] = 0.7; s[68] = 0.35  # avg_interrupt cao
    s[69] = 1.0   # spot_ratio = 100%
    s[70] = 0.6
    s[71] = 0.9
    return s


def make_state_stable_market_all_od() -> np.ndarray:
    """Market ổn định (interrupt thấp), agent đang dùng toàn OD → nên CONVERT_TO_SPOT."""
    s = np.zeros(STATE_DIM, dtype=np.float32)
    for pool_idx in range(15):
        base = 12 + pool_idx * 3
        s[base + 0] = 0.35  # spot rẻ
        s[base + 1] = 0.02  # interrupt rất thấp
        s[base + 2] = 0.9
    s[0] = 0.3; s[1] = 0.01; s[2] = 1.0; s[3] = 0.0
    s[57] = 0.0   # 0 spot
    s[58] = 0.5   # 10 OD
    s[59] = 0.4
    s[60] = 0.0; s[61] = 0.5; s[62] = 0.1; s[63] = 0.0
    s[64] = 0.5; s[65] = 0.5; s[66] = 0.5
    s[67] = 0.35; s[68] = 0.02
    s[69] = 0.0   # spot_ratio = 0%
    s[70] = 0.8   # cost_rate cao (đang dùng OD đắt)
    s[71] = 1.0   # SLA tốt
    return s


def make_state_workload_spike() -> np.ndarray:
    """Workload đột biến: nhiều pending + forecast cao → agent nên PROVISION nhanh."""
    s = np.zeros(STATE_DIM, dtype=np.float32)
    for pool_idx in range(15):
        base = 12 + pool_idx * 3
        s[base + 0] = 0.5
        s[base + 1] = 0.08
        s[base + 2] = 0.7
    s[0] = 0.45; s[1] = 0.05; s[2] = 0.8; s[3] = 0.0
    s[57] = 0.15  # 3 spot (ít)
    s[58] = 0.05  # 1 OD
    s[59] = 0.15
    s[60] = 0.8   # pending rất nhiều (24 jobs)
    s[61] = 0.15
    s[62] = 0.9   # forecast_1h rất cao → sắp spike tiếp
    s[63] = 0.6   # queue_wait cao
    s[64] = 0.5; s[65] = 0.3; s[66] = 0.3
    s[67] = 0.5; s[68] = 0.08
    s[69] = 0.75
    s[70] = 0.3
    s[71] = 0.4   # SLA đang xấu
    return s


PRESETS = {
    'empty_with_jobs':      ('Cluster rỗng + nhiều job chờ → kỳ vọng PROVISION_SPOT',
                             make_state_empty_cluster_with_jobs),
    'high_interrupt':       ('Market interrupt cao + toàn spot → kỳ vọng CONVERT_TO_ONDEMAND',
                             make_state_high_interrupt_spot_heavy),
    'stable_market_all_od': ('Market ổn định + toàn OD → kỳ vọng CONVERT_TO_SPOT',
                             make_state_stable_market_all_od),
    'workload_spike':       ('Workload spike + capacity ít → kỳ vọng PROVISION mạnh',
                             make_state_workload_spike),
}


def run_presets(agent: DQNAgent):
    """Chạy tất cả preset scenarios, xem agent phản ứng sao."""
    for name, (desc, state_fn) in PRESETS.items():
        print("\n" + "=" * 80)
        print(f"SCENARIO: {name}")
        print(f"  {desc}")
        print("=" * 80)
        state = state_fn()
        query_agent(agent, state, top_k=5)


def run_interactive(agent: DQNAgent):
    """Cho user nhập state vector qua stdin."""
    print(f"\nNhập {STATE_DIM} giá trị [0,1] ngăn cách dấu cách (hoặc 'random' để random):")
    line = input("> ").strip()
    if line.lower() == 'random':
        state = np.random.rand(STATE_DIM).astype(np.float32)
        print(f"  Dùng random state: {state[:10].round(3)}...")
    else:
        values = [float(x) for x in line.split()]
        if len(values) != STATE_DIM:
            raise ValueError(f"Cần {STATE_DIM} giá trị, nhập {len(values)}")
        state = np.array(values, dtype=np.float32)
    query_agent(agent, state, top_k=10)


def run_from_env(agent: DQNAgent, scenario: str = 'stable', seed: int = 0):
    """Lấy state thật từ SpotOrchestratorEnv cho agent quyết định."""
    from envs.mixed_scenario_env import MixedScenarioEnv

    env = MixedScenarioEnv(scenario_paths={
        'stable':        'data/processed/multipool_stable.csv',
        'volatile':      'data/processed/multipool_volatile.csv',
        'spike':         'data/processed/multipool_spike.csv',
        'az_divergence': 'data/processed/multipool_az_divergence.csv',
    }, max_steps=168, seed=seed)
    obs, info = env.reset()
    print(f"\nScenario: {info['scenario']}")
    print(f"Initial state shape: {obs.shape}")
    print(f"State first 10 values: {obs[:10].round(3)}")

    mask = env.get_action_mask()
    query_agent(agent, obs, top_k=10, mask=mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='results/kaggle/best_mixed_v1.pth')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--interactive', action='store_true', help='Nhập state bằng tay')
    ap.add_argument('--from-env', action='store_true', help='Lấy state thật từ env')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f"Loading agent: {args.model}")
    agent = load_agent(args.model, device=args.device)
    print(f"  epsilon (set to 0.0 cho greedy): {agent.epsilon}")
    print(f"  state_dim={STATE_DIM}  action_dim={N_ACTIONS}")

    if args.interactive:
        run_interactive(agent)
    elif args.from_env:
        run_from_env(agent, seed=args.seed)
    else:
        run_presets(agent)


if __name__ == '__main__':
    main()
