"""
Evaluate DQN generalist vs baselines trên 4 scenario, dùng CHUNG config với lúc train.

Env (workload, reward, cooldown, interrupt model, obs version, max_steps) lấy từ --config,
nên eval khớp đúng điều kiện train. Trước đây script tạo env không có reward_config
(cooldown 2 thay vì 4) và workload mặc định khác notebook.

Baseline chính để so sánh:
  - Autoscaler-Spot: autoscaler target-vCPU, spot-first, fallback OD (giống ASG/Karpenter)
  - Autoscaler-OD:   cùng logic nhưng chỉ dùng OD → mốc "tiết kiệm so với OD" công bằng
Các baseline cũ (OnDemand, Spot, ...) giữ lại để so với số liệu cũ (--no-legacy-baselines để bỏ).

Usage:
    python experiments/evaluate_kaggle.py --config experiments/configs/generalist/v41.yaml \
        --model results/kaggle/best_mixed_v41.pth --output-dir results/eval/v41
    # chỉ chạy baseline (không cần model):
    python experiments/evaluate_kaggle.py --config experiments/configs/generalist/s3_interrupt_realistic.yaml \
        --output-dir results/eval/s3_baselines
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.action_schema import N_ACTIONS, decode_action, OPERATION_NAMES  # noqa: E402
from agents.dqn_agent import DQNAgent  # noqa: E402
from agents.baselines import (  # noqa: E402
    AutoscalerAgent, OnDemandAgent, SpotAgent, ThresholdAgent,
    CheapestAZAgent, CheapestTypeAgent, RandomAgent,
)
from experiments.common import load_config, make_eval_env, scenario_paths, config_name  # noqa: E402

DEFAULT_CONFIG = "experiments/configs/generalist/v41.yaml"
REF_AGENT = "Autoscaler-OD"


class OnnxQAgent:
    """Chạy file .onnx đã export (đúng artifact controller Go đang dùng) như một DQN greedy."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.state_dim = int(inp.shape[1])

    def select_action(self, state, training: bool = False, action_mask=None) -> int:
        q = self.sess.run(None, {self.input_name: np.asarray(state, dtype=np.float32)[None, :]})[0][0]
        if action_mask is not None:
            q = np.where(action_mask, q, -np.inf)
        return int(np.argmax(q))


def load_dqn(model_path: str, device: str = "cpu"):
    if model_path.endswith(".onnx"):
        agent = OnnxQAgent(model_path)
        print(f"Loaded ONNX model | state={agent.state_dim}")
        return agent
    import torch
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt["q_network_state_dict"]
    network_type = ckpt.get("network_type")
    hidden_dim = ckpt.get("hidden_dim")
    if network_type is None:
        if "trunk.0.weight" in sd:
            network_type = "factored"
            hidden_dim = hidden_dim or sd["trunk.0.weight"].shape[0]
        elif "feature.0.weight" in sd:
            network_type = "dueling"
            hidden_dim = hidden_dim or sd["feature.0.weight"].shape[0]
        else:
            raise RuntimeError(f"Unknown checkpoint format. Keys: {list(sd.keys())[:5]}")

    state_dim = ckpt.get("state_dim")
    if state_dim is None:
        from envs.instance_catalog import STATE_DIM
        state_dim = STATE_DIM
    action_dim = ckpt.get("action_dim", N_ACTIONS)

    agent = DQNAgent(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=hidden_dim or 512, network_type=network_type,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1,
        use_per=False, buffer_size=1, device=device,
    )
    agent.q_network.load_state_dict(sd)
    agent.q_network.eval()
    print(f"Loaded {network_type} net | state={state_dim} action={action_dim} hidden={hidden_dim} "
          f"| train steps={ckpt.get('steps_done')} lr_schedule={ckpt.get('lr_schedule', 'step (legacy)')}")
    return agent


def run_episode(env, agent, is_dqn: bool) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    interruptions = 0
    action_counts = np.zeros(N_ACTIONS, dtype=int)

    while True:
        mask = env.get_action_mask()
        if is_dqn:
            action = agent.select_action(obs, training=False, action_mask=mask)
        else:
            action = agent.select_action(obs, dict(info, action_mask=mask))
        action_counts[action] += 1
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        interruptions += int(info.get("step_interruptions", 0))
        if term or trunc:
            break

    return {
        "total_reward": float(total_reward),
        "total_cost": float(info.get("cost", 0.0)),
        "sla_rate": float(info.get("sla_compliance", 0.0)),
        "failed_jobs": int(info.get("failed_jobs", 0)),
        "completed_jobs": int(info.get("completed_jobs", 0)),
        "interruptions": interruptions,
        "action_counts": action_counts,
    }


def evaluate(agent, cfg: dict, scenario: str, n_episodes: int, is_dqn: bool) -> dict:
    e = cfg.get("eval", {})
    seed_base, stride = int(e.get("seed_base", 42)), int(e.get("seed_stride", 7))
    results = []
    ac_total = np.zeros(N_ACTIONS, dtype=int)
    for ep in range(n_episodes):
        if hasattr(agent, "reset"):
            agent.reset()
        env = make_eval_env(cfg, scenario, seed=seed_base + ep * stride)
        r = run_episode(env, agent, is_dqn)
        results.append(r)
        ac_total += r["action_counts"]

    rewards = [r["total_reward"] for r in results]
    costs = [r["total_cost"] for r in results]
    slas = [r["sla_rate"] for r in results]
    hold_pct = 0.0
    if ac_total.sum() > 0:
        hold_pct = float(ac_total[-1] / ac_total.sum() * 100)
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "avg_sla": float(np.mean(slas)),
        "min_sla": float(np.min(slas)),
        "avg_failed_jobs": float(np.mean([r["failed_jobs"] for r in results])),
        "avg_interruptions": float(np.mean([r["interruptions"] for r in results])),
        "hold_pct": hold_pct,
        "episode_costs": costs,
        "action_counts": ac_total,
    }


def _savings(ref_cost: float, cost: float) -> float:
    return (ref_cost - cost) / ref_cost * 100 if ref_cost > 0 else 0.0


def print_table(all_results: dict, ref: str):
    scenarios = list(all_results.keys())
    agents = list(all_results[scenarios[0]].keys())
    header = f"{'Agent':<16}" + "".join(f" {'Cost ' + s[:8]:>14} {'SLA':>6} {'vs ' + ref[-2:]:>7}" for s in scenarios)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for a in agents:
        row = f"{a:<16}"
        for s in scenarios:
            m = all_results[s][a]
            sv = _savings(all_results[s][ref]["avg_cost"], m["avg_cost"]) if ref in all_results[s] else 0.0
            row += f"  ${m['avg_cost']:>8.1f}±{m['std_cost']:<3.0f} {m['avg_sla']*100:>5.1f}% {sv:>+6.1f}%"
        print(row)
    print("=" * len(header))
    print(f"'vs {ref[-2:]}' = % tiết kiệm so với {ref}. Chỉ so sánh agent có SLA ≥ 95%.")


def plot_comparison(all_results: dict, out_dir: Path, title: str):
    scenarios = list(all_results.keys())
    agents = list(all_results[scenarios[0]].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(agents), 2)))

    fig, axes = plt.subplots(2, len(scenarios), figsize=(5 * len(scenarios), 9), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for col, scn in enumerate(scenarios):
        costs = [all_results[scn][a]["avg_cost"] for a in agents]
        slas = [all_results[scn][a]["avg_sla"] for a in agents]
        x = np.arange(len(agents))

        ax = axes[0, col]
        bars = ax.bar(x, costs, color=colors[:len(agents)])
        ax.set_title(f"[{scn}] Cost ($)", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Avg Cost / Episode ($)")
        for bar, v in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${v:.0f}",
                    ha="center", va="bottom", fontsize=7)

        ax2 = axes[1, col]
        bars2 = ax2.bar(x, [s * 100 for s in slas], color=colors[:len(agents)])
        ax2.set_title(f"[{scn}] SLA (%)", fontsize=11)
        ax2.set_xticks(x); ax2.set_xticklabels(agents, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("SLA Compliance (%)")
        ax2.set_ylim(0, 105)
        ax2.axhline(95, color="red", linestyle="--", linewidth=1, alpha=0.7)
        for bar, v in zip(bars2, slas):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v*100:.1f}%",
                     ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = out_dir / "comparison_all_scenarios.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_action_dist(action_counts: np.ndarray, out_dir: Path):
    from envs.action_schema import N_POOL_OPS, HOLD_ACTION
    op_counts = np.zeros(N_POOL_OPS + 1, dtype=int)
    for idx, cnt in enumerate(action_counts):
        if idx == HOLD_ACTION:
            op_counts[N_POOL_OPS] += cnt
        else:
            op, _, _ = decode_action(idx)
            op_counts[op] += cnt
    labels = list(OPERATION_NAMES)
    total = op_counts.sum()
    pcts = op_counts / total * 100 if total > 0 else op_counts

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(labels)), pcts, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Action frequency (%)")
    ax.set_title("DQN Action Distribution (all scenarios)")
    for bar, v in zip(bars, pcts):
        if v > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = out_dir / "dqn_action_dist.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_savings_heatmap(all_results: dict, out_dir: Path, ref: str):
    scenarios = list(all_results.keys())
    agents = [a for a in all_results[scenarios[0]].keys() if a != ref]
    matrix = np.zeros((len(agents), len(scenarios)))
    for j, scn in enumerate(scenarios):
        ref_cost = all_results[scn][ref]["avg_cost"]
        for i, ag in enumerate(agents):
            matrix[i, j] = _savings(ref_cost, all_results[scn][ag]["avg_cost"])

    fig, ax = plt.subplots(figsize=(len(scenarios) * 2 + 2, len(agents) * 0.8 + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-20, vmax=60)
    ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yticks(range(len(agents))); ax.set_yticklabels(agents, fontsize=10)
    ax.set_title(f"Cost Savings vs {ref} (%)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Savings %")
    for i in range(len(agents)):
        for j in range(len(scenarios)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", fontsize=9,
                    color="black" if abs(matrix[i, j]) < 40 else "white")
    plt.tight_layout()
    path = out_dir / "savings_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Config dùng lúc train (env/workload/reward/sim lấy từ đây)")
    parser.add_argument("--model", default=None,
                        help="Checkpoint DQN (.pth) hoặc model đã export (.onnx); bỏ trống = chỉ chạy baseline")
    parser.add_argument("--episodes", type=int, default=None, help="Mặc định = eval.episodes trong config")
    parser.add_argument("--scenarios", default=None, help="vd: stable,spike (mặc định: tất cả)")
    parser.add_argument("--output-dir", default="results/kaggle/eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-legacy-baselines", action="store_true",
                        help="Bỏ các baseline cũ (OnDemand, Spot, Threshold, ...)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_eps = args.episodes or int(cfg.get("eval", {}).get("episodes", 30))
    scenarios = list(scenario_paths(cfg).keys())
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {cfg['_config_path']} ({config_name(cfg)})")
    print(f"Model: {args.model or '(none — baselines only)'} | episodes/scenario={n_eps}")
    print(f"env.sim={cfg['env'].get('sim', {})} reward.mode={cfg['env'].get('reward', {}).get('mode', 'shaped')}")

    agents = {}
    if args.model:
        dqn = load_dqn(args.model, device=args.device)
        env_dim = make_eval_env(cfg, scenarios[0], seed=0).observation_space.shape[0]
        if dqn.state_dim != env_dim:
            raise SystemExit(
                f"Model state_dim={dqn.state_dim} nhưng env (obs_version="
                f"{cfg['env'].get('sim', {}).get('obs_version', 1)}) cho {env_dim}. "
                f"Dùng đúng config mà model được train."
            )
        agents["DQN"] = (dqn, True)
    agents["Autoscaler-Spot"] = (AutoscalerAgent(use_spot=True), False)
    agents["Autoscaler-OD"] = (AutoscalerAgent(use_spot=False), False)
    if not args.no_legacy_baselines:
        agents.update({
            "OnDemand": (OnDemandAgent(target_capacity=8), False),
            "Spot": (SpotAgent(target_capacity=8), False),
            "Threshold": (ThresholdAgent(), False),
            "CheapestAZ": (CheapestAZAgent(), False),
            "CheapestType": (CheapestTypeAgent(), False),
            "Random": (RandomAgent(seed=0), False),
        })

    all_results = {}
    ac_dqn = np.zeros(N_ACTIONS, dtype=int)
    for scn in scenarios:
        print(f"\n{'=' * 60}\nScenario: {scn}\n{'=' * 60}")
        scn_res = {}
        for name, (agent, is_dqn) in agents.items():
            res = evaluate(agent, cfg, scn, n_eps, is_dqn)
            scn_res[name] = res
            print(f"  {name:<16} cost=${res['avg_cost']:8.1f}±{res['std_cost']:<4.0f} "
                  f"SLA={res['avg_sla']*100:5.1f}% (min {res['min_sla']*100:5.1f}%) "
                  f"interrupts/ep={res['avg_interruptions']:6.1f} HOLD={res['hold_pct']:4.1f}%")
        all_results[scn] = scn_res
        if "DQN" in scn_res:
            ac_dqn += scn_res["DQN"]["action_counts"]

    print_table(all_results, REF_AGENT)
    plot_comparison(all_results, out_dir, f"{config_name(cfg)} — DQN vs baselines")
    plot_savings_heatmap(all_results, out_dir, REF_AGENT)
    if ac_dqn.sum() > 0:
        plot_action_dist(ac_dqn, out_dir)

    summary = {"config": config_name(cfg), "config_path": cfg["_config_path"],
               "model": args.model, "episodes": n_eps, "reference": REF_AGENT, "scenarios": {}}
    for scn, res in all_results.items():
        ref_cost = res[REF_AGENT]["avg_cost"]
        summary["scenarios"][scn] = {
            name: {
                **{k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))},
                "savings_vs_autoscaler_od_pct": _savings(ref_cost, r["avg_cost"]),
            }
            for name, r in res.items()
        }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
