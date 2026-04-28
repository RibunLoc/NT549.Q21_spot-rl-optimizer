"""
Local evaluation script — test model sát thực trước khi deploy.

Metrics:
  - Cost vs OnDemand baseline (savings %)
  - SLA compliance
  - Action chatter rate (flip cùng pool trong 3 steps liên tiếp)
  - Migration overhead (số lần convert/rebalance)
  - Stress scenarios: interrupt burst, price spike, workload burst

Usage:
    python scripts/local_eval.py --model results/kaggle/best_mixed_v1.pth
    python scripts/local_eval.py --model results/kaggle/best_mixed_v1.pth --episodes 50
    python scripts/local_eval.py --model results/kaggle/best_mixed_v1.pth --stress-only
"""

import argparse
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.spot_orchestrator_env import SpotOrchestratorEnv
from envs.instance_catalog import STATE_DIM, INSTANCE_TYPES, N_TYPES, N_AZS
from envs.action_schema import (
    N_ACTIONS, HOLD_ACTION, N_POOLS,
    decode_action, encode_action, Operation, OPERATION_NAMES,
)
from agents.dqn_agent import DQNAgent
from agents.baselines import OnDemandAgent, SpotAgent, ThresholdAgent


# ─── Data paths (.pkl) — absolute, CWD-independent ────────────────────────────

SCENARIO_PATHS = {
    "stable":        str(PROJECT_ROOT / "data/processed/multipool_stable.csv"),
    "volatile":      str(PROJECT_ROOT / "data/processed/multipool_volatile.csv"),
    "spike":         str(PROJECT_ROOT / "data/processed/multipool_spike.csv"),
    "az_divergence": str(PROJECT_ROOT / "data/processed/multipool_az_divergence.csv"),
}

WORKLOAD_CFG = {
    "base_arrival_rate": 15.0,
    "peak_multiplier":   2.0,
    "avg_job_duration":  3,
}


# ─── Agent loader ─────────────────────────────────────────────────────────────

def load_agent(model_path: str, device: str = "cpu") -> DQNAgent:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    network_type = ckpt.get("network_type")
    hidden_dim   = ckpt.get("hidden_dim")
    if network_type is None:
        sd = ckpt["q_network_state_dict"]
        if "trunk.0.weight" in sd:
            network_type = "factored"
            hidden_dim   = hidden_dim or sd["trunk.0.weight"].shape[0]
        elif "feature.0.weight" in sd:
            network_type = "dueling"
            hidden_dim   = hidden_dim or sd["feature.0.weight"].shape[0]
        else:
            raise RuntimeError(f"Unknown checkpoint. Keys: {list(sd.keys())[:5]}")
    state_dim  = ckpt.get("state_dim",  STATE_DIM)
    action_dim = ckpt.get("action_dim", N_ACTIONS)
    agent = DQNAgent(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=hidden_dim, network_type=network_type,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
        device=device,
    )
    agent.load(model_path)
    return agent


# ─── Chatter detector ─────────────────────────────────────────────────────────

class ChatterTracker:
    """
    Theo dõi action chatter: agent flip action trên cùng pool trong window steps.

    Chatter = pool nhận action ở step T và step T+k (k <= window) mà không có
    lý do thay đổi market (giá thay đổi <5%).
    """
    def __init__(self, window: int = 3):
        self.window = window
        self.pool_action_history: dict = defaultdict(list)  # pool_key → [(step, op)]
        self.chatter_events = 0
        self.total_actions = 0

    def record(self, step: int, action: int):
        if action == HOLD_ACTION:
            return
        op, t, az = decode_action(action)
        pool_key = (t, az)
        history = self.pool_action_history[pool_key]
        history.append((step, op))
        self.total_actions += 1
        # Chatter: cùng pool có action cả ở step trước và step này trong window
        recent = [s for s, _ in history if step - s <= self.window and s != step]
        if recent:
            self.chatter_events += 1
        # Giữ window sạch
        self.pool_action_history[pool_key] = [
            (s, o) for s, o in history if step - s <= self.window
        ]

    def chatter_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.chatter_events / self.total_actions


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(env: SpotOrchestratorEnv, agent, use_mask: bool = True) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    step_costs = []
    actions_taken = []
    op_counts = defaultdict(int)
    chatter = ChatterTracker(window=3)

    step = 0
    while True:
        if hasattr(agent, "replay_buffer"):  # DQNAgent
            mask   = env.get_action_mask() if use_mask else None
            action = agent.select_action(obs, training=False, action_mask=mask)
        else:
            action = agent.select_action(obs, info)

        chatter.record(step, action)
        actions_taken.append(action)

        if action != HOLD_ACTION:
            op, _, _ = decode_action(action)
            op_counts[op.name] += 1

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        step_costs.append(info.get("step_cost", 0.0))
        step += 1

        if term or trunc:
            break

    total_cost = float(info.get("cost", sum(step_costs)))
    migrations = (
        op_counts.get("CONVERT_TO_ONDEMAND", 0)
        + op_counts.get("CONVERT_TO_SPOT", 0)
        + op_counts.get("REBALANCE_SPOT", 0)
    )

    return {
        "total_reward":   total_reward,
        "total_cost":     total_cost,
        "sla_rate":       float(info.get("sla_compliance", 0.0)),
        "failed_jobs":    int(info.get("failed_jobs", 0)),
        "completed_jobs": int(info.get("completed_jobs", 0)),
        "chatter_rate":   chatter.chatter_rate(),
        "migrations":     migrations,
        "hold_pct":       actions_taken.count(HOLD_ACTION) / max(len(actions_taken), 1),
        "op_counts":      dict(op_counts),
        "step_costs":     step_costs,
    }


def evaluate_scenario(agent, scenario: str, n_episodes: int,
                       use_mask: bool = True, data_path: str = None) -> dict:
    path = data_path or SCENARIO_PATHS[scenario]
    results = []
    for ep in range(n_episodes):
        env = SpotOrchestratorEnv(
            data_path=path,
            max_steps=168,
            workload_config=WORKLOAD_CFG,
            seed=ep * 7 + 42,
        )
        results.append(run_episode(env, agent, use_mask))

    costs    = [r["total_cost"]   for r in results]
    slas     = [r["sla_rate"]     for r in results]
    chatters = [r["chatter_rate"] for r in results]
    migs     = [r["migrations"]   for r in results]
    holds    = [r["hold_pct"]     for r in results]

    return {
        "avg_cost":         float(np.mean(costs)),
        "std_cost":         float(np.std(costs)),
        "avg_sla":          float(np.mean(slas)),
        "avg_chatter_rate": float(np.mean(chatters)),
        "avg_migrations":   float(np.mean(migs)),
        "avg_hold_pct":     float(np.mean(holds)),
        "episode_costs":    costs,
        "episode_slas":     slas,
    }


# ─── Stress scenarios ─────────────────────────────────────────────────────────

def run_stress_tests(agent, n_episodes: int = 20) -> dict:
    """
    3 stress scenarios inject điều kiện bất thường vào environment:
      1. interrupt_burst  — P(interrupt) tăng cao đột ngột ở tất cả pools
      2. price_spike      — giá spot tăng 3× OD sau 50 steps
      3. workload_burst   — arrival rate × 5 ở giữa episode
    """
    print("\n=== STRESS TESTS ===")

    stress_results = {}

    # Stress 1: Interrupt burst — dùng volatile data + reward config nhạy cảm
    print("  [stress-1] interrupt_burst ...")
    results = []
    for ep in range(n_episodes):
        env = SpotOrchestratorEnv(
            data_path=SCENARIO_PATHS["volatile"],
            max_steps=168,
            workload_config=WORKLOAD_CFG,
            reward_config={"interrupt_penalty": 2.0},  # nhạy hơn với interrupt
            seed=ep * 13 + 7,
        )
        results.append(run_episode(env, agent))
    stress_results["interrupt_burst"] = {
        "avg_cost": float(np.mean([r["total_cost"] for r in results])),
        "avg_sla":  float(np.mean([r["sla_rate"]   for r in results])),
        "avg_chatter": float(np.mean([r["chatter_rate"] for r in results])),
        "avg_migrations": float(np.mean([r["migrations"] for r in results])),
    }

    # Stress 2: AZ divergence — giá và interrupt mỗi AZ lệch nhau mạnh
    print("  [stress-2] az_divergence ...")
    results = []
    for ep in range(n_episodes):
        env = SpotOrchestratorEnv(
            data_path=SCENARIO_PATHS["az_divergence"],
            max_steps=168,
            workload_config=WORKLOAD_CFG,
            seed=ep * 17 + 3,
        )
        results.append(run_episode(env, agent))
    stress_results["az_divergence"] = {
        "avg_cost": float(np.mean([r["total_cost"] for r in results])),
        "avg_sla":  float(np.mean([r["sla_rate"]   for r in results])),
        "avg_chatter": float(np.mean([r["chatter_rate"] for r in results])),
        "avg_migrations": float(np.mean([r["migrations"] for r in results])),
    }

    # Stress 3: Workload burst — arrival rate cao x3
    print("  [stress-3] workload_burst ...")
    results = []
    for ep in range(n_episodes):
        env = SpotOrchestratorEnv(
            data_path=SCENARIO_PATHS["spike"],
            max_steps=168,
            workload_config={
                "base_arrival_rate": 30.0,  # 2× default
                "peak_multiplier":   4.0,
                "avg_job_duration":  3,
            },
            seed=ep * 11 + 5,
        )
        results.append(run_episode(env, agent))
    stress_results["workload_burst"] = {
        "avg_cost": float(np.mean([r["total_cost"] for r in results])),
        "avg_sla":  float(np.mean([r["sla_rate"]   for r in results])),
        "avg_chatter": float(np.mean([r["chatter_rate"] for r in results])),
        "avg_migrations": float(np.mean([r["migrations"] for r in results])),
    }

    return stress_results


# ─── Report printer ───────────────────────────────────────────────────────────

def print_scenario_report(all_results: dict, od_costs: dict):
    PASS_SLA    = 0.95
    PASS_CHATTER = 0.10  # < 10% chatter rate = acceptable
    PASS_SAVINGS = 30.0  # > 30% cheaper than OD

    print("\n" + "=" * 90)
    print(f"{'Scenario':<16} {'Cost':>8} {'±':>6} {'SLA':>7} {'Savings':>9} "
          f"{'Chatter':>9} {'Migrations':>11} {'HOLD%':>7} {'Status'}")
    print("-" * 90)

    all_pass = True
    for scn, res in all_results.items():
        od = od_costs.get(scn, 0)
        savings_pct = (od - res["avg_cost"]) / od * 100 if od > 0 else 0
        sla_ok      = res["avg_sla"] >= PASS_SLA
        chatter_ok  = res["avg_chatter_rate"] <= PASS_CHATTER
        savings_ok  = savings_pct >= PASS_SAVINGS
        status = "PASS" if (sla_ok and chatter_ok and savings_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        flags = []
        if not sla_ok:      flags.append(f"SLA={res['avg_sla']:.1%}<{PASS_SLA:.0%}")
        if not chatter_ok:  flags.append(f"chatter={res['avg_chatter_rate']:.1%}>{PASS_CHATTER:.0%}")
        if not savings_ok:  flags.append(f"savings={savings_pct:.1f}%<{PASS_SAVINGS:.0f}%")
        flag_str = " | ".join(flags) if flags else ""

        print(f"{scn:<16} ${res['avg_cost']:>7.1f} ±{res['std_cost']:>5.0f} "
              f"{res['avg_sla']:>7.1%} {savings_pct:>+8.1f}% "
              f"{res['avg_chatter_rate']:>9.1%} {res['avg_migrations']:>11.1f} "
              f"{res['avg_hold_pct']:>7.1%}  [{status}] {flag_str}")

    print("=" * 90)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return all_pass


def print_stress_report(stress_results: dict):
    print("\n--- Stress Test Results ---")
    print(f"{'Scenario':<20} {'Cost':>8} {'SLA':>7} {'Chatter':>9} {'Migrations':>11}")
    print("-" * 60)
    for scn, res in stress_results.items():
        sla_flag = "" if res["avg_sla"] >= 0.90 else " !"
        print(f"{scn:<20} ${res['avg_cost']:>7.1f} {res['avg_sla']:>7.1%}{sla_flag} "
              f"{res['avg_chatter']:>9.1%} {res['avg_migrations']:>11.1f}")
    print("-" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="results/kaggle/best_mixed_v1.pth")
    ap.add_argument("--episodes",    type=int, default=30)
    ap.add_argument("--device",      default="cpu")
    ap.add_argument("--stress-only", action="store_true")
    ap.add_argument("--output",      default="results/kaggle/local_eval.json")
    args = ap.parse_args()

    print(f"Model : {args.model}")
    print(f"Episodes per scenario: {args.episodes}")

    agent = load_agent(args.model, device=args.device)
    print(f"Loaded agent | state_dim={agent.state_dim} action_dim={agent.action_dim}")

    # Baselines để tính OD reference cost
    od_agent   = OnDemandAgent(target_capacity=8)
    spot_agent = SpotAgent(target_capacity=8)

    all_results = {}
    od_costs    = {}

    if not args.stress_only:
        for scn in SCENARIO_PATHS:
            print(f"\n--- Scenario: {scn} ---")

            # DQN
            print(f"  [DQN] {args.episodes} episodes...")
            dqn_res = evaluate_scenario(agent, scn, args.episodes, use_mask=True)
            print(f"    cost=${dqn_res['avg_cost']:.1f}  SLA={dqn_res['avg_sla']:.1%}  "
                  f"chatter={dqn_res['avg_chatter_rate']:.1%}  "
                  f"migrations={dqn_res['avg_migrations']:.1f}  "
                  f"hold={dqn_res['avg_hold_pct']:.1%}")

            # OnDemand baseline
            print(f"  [OnDemand] {args.episodes} episodes...")
            od_res = evaluate_scenario(od_agent, scn, args.episodes, use_mask=False)
            od_costs[scn] = od_res["avg_cost"]
            print(f"    cost=${od_res['avg_cost']:.1f}  SLA={od_res['avg_sla']:.1%}")

            # Spot baseline
            print(f"  [Spot] {args.episodes} episodes...")
            spot_res = evaluate_scenario(spot_agent, scn, args.episodes, use_mask=False)
            print(f"    cost=${spot_res['avg_cost']:.1f}  SLA={spot_res['avg_sla']:.1%}")

            all_results[scn] = {
                "DQN":      dqn_res,
                "OnDemand": od_res,
                "Spot":     spot_res,
            }

        print_scenario_report(
            {s: r["DQN"] for s, r in all_results.items()},
            od_costs
        )

    # Stress tests
    stress_results = run_stress_tests(agent, n_episodes=min(args.episodes, 20))
    print_stress_report(stress_results)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for scn, agents_res in all_results.items():
        summary[scn] = {}
        for name, res in agents_res.items():
            summary[scn][name] = {
                k: v for k, v in res.items()
                if not isinstance(v, list)
            }
        od = summary[scn].get("OnDemand", {}).get("avg_cost", 0)
        dqn_cost = summary[scn]["DQN"]["avg_cost"]
        summary[scn]["DQN"]["savings_pct"] = (
            (od - dqn_cost) / od * 100 if od > 0 else 0
        )
    summary["stress"] = stress_results
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
