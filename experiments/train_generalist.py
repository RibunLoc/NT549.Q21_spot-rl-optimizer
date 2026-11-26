"""
Train DQN generalist (4 scenario) từ một file config dùng chung với evaluate_kaggle.py.

Vòng lặp chép từ notebooks/*_train.ipynb (eval-gated save, early stop, collapse guard,
checkpoint, MLflow) để notebook Kaggle/RunPod và chạy local dùng chung một code.

Usage:
    python experiments/train_generalist.py --config experiments/configs/generalist/s1_fix_lr_done.yaml \
        --save-dir results/s1/models
    # smoke test nhanh trên CPU:
    python experiments/train_generalist.py --config ... --episodes 3 --eval-start 2 --eval-every 1 --eval-episodes 1
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (  # noqa: E402
    SCENARIO_NAMES, config_name, flatten_config, load_config,
    make_agent, make_eval_env, make_train_env, scenario_paths,
)


def run_greedy_eval(agent, cfg: Dict[str, Any], episodes_per_scenario: int) -> Dict[str, Any]:
    """Greedy eval trên từng scenario, seed cố định 10000 + ep*97 (giống notebook)."""
    t = cfg["training"]
    sla_target = t.get("sla_target", 0.95)
    was_training = agent.q_network.training
    agent.q_network.eval()
    scenario_metrics = {}
    all_costs, all_slas, all_rewards = [], [], []
    for scn in scenario_paths(cfg):
        costs, slas, rewards = [], [], []
        for ep in range(episodes_per_scenario):
            env = make_eval_env(cfg, scn, seed=10_000 + ep * 97)
            obs, info = env.reset()
            total_reward = 0.0
            while True:
                action = agent.select_action(obs, training=False, action_mask=env.get_action_mask())
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                if term or trunc:
                    break
            costs.append(float(info.get("cost", 0.0)))
            slas.append(float(info.get("sla_compliance", 0.0)))
            rewards.append(float(total_reward))
        scenario_metrics[scn] = {
            "cost": float(np.mean(costs)), "sla": float(np.mean(slas)), "reward": float(np.mean(rewards)),
        }
        all_costs += costs
        all_slas += slas
        all_rewards += rewards
    if was_training:
        agent.q_network.train()
    cost, sla = float(np.mean(all_costs)), float(np.mean(all_slas))
    return {
        "cost": cost, "sla": sla, "reward": float(np.mean(all_rewards)),
        "score": -cost - 1000.0 * max(0.0, sla_target - sla),
        "scenarios": scenario_metrics,
    }


def train(
    cfg: Dict[str, Any],
    save_dir: str,
    mlflow_uri: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    start_episode: int = 0,
    resume_lr: Optional[float] = None,
    device: Optional[str] = None,
    platform: str = "local",
    tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t = cfg["training"]
    version = config_name(cfg)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)

    seed = t.get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = make_train_env(cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = make_agent(cfg, state_dim, action_dim, device)
    if resume_checkpoint:
        agent.load(resume_checkpoint)
        if resume_lr is not None:
            for g in agent.optimizer.param_groups:
                g["lr"] = resume_lr
        print(f"Resumed {resume_checkpoint}: eps={agent.epsilon:.3f} steps={agent.steps_done:,} lr={agent.current_lr():.2e}")

    # done lưu vào replay: cũ = term or trunc; đúng = chỉ term (env không bao giờ terminate)
    bootstrap_trunc = bool(t.get("bootstrap_on_truncation", False))

    num_episodes = int(t.get("num_episodes", 6000))
    min_episodes = int(t.get("min_episodes", 3000))
    warmup_steps = int(t.get("warmup_steps", agent.batch_size * 8))
    n_grad = int(t.get("n_gradient_steps", 1))
    eval_every = int(t.get("eval_every", 100))
    eval_start = int(t.get("eval_start", 500))
    eval_episodes = int(t.get("eval_episodes", 3))
    eval_sla_gate = float(t.get("eval_sla_gate", 0.945))
    eval_patience = int(t.get("eval_patience", 12))
    eps_gate = float(t.get("eps_gate", 0.08))
    collapse_window = int(t.get("collapse_window", 100))
    collapse_sla_stop = float(t.get("collapse_sla_stop", 0.88))
    checkpoint_every = int(t.get("checkpoint_every", 500))
    print_every = int(t.get("print_every", 10))
    log_every = int(t.get("log_every", 10))

    mlflow = None
    run_ctx = None
    if mlflow_uri:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(t.get("mlflow_experiment", "spot-rl-generalist"))
        run_ctx = mlflow.start_run(run_name=f"{version}-{platform}")
        params = flatten_config(cfg)
        params.update({"state_dim": state_dim, "platform": platform,
                       "resumed_from": resume_checkpoint or "scratch"})
        # MLflow giới hạn 100 params/lần gọi
        items = [(k, str(v)[:500]) for k, v in params.items()]
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i:i + 100]))
        mlflow.set_tag("version", version)
        if tags:  # vd. git_sha, run_config, trigger (từ notebook / CI)
            mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})
        mlflow.log_artifact(os.path.join(save_dir, "config.yaml"))

    hist = {
        "rewards": [], "loss": [], "epsilon": [], "lr": [], "cost": [], "sla": [],
        "scenario": [], "eval": [],
    }
    per_scn_cost, per_scn_sla = defaultdict(list), defaultdict(list)
    best_score = -float("inf")
    best_cost = best_sla = best_reward = None
    eval_no_improve = 0
    global_env_steps = global_grad_steps = 0
    start_time = time.time()

    def _dump_history(path):
        with open(path, "wb") as f:
            pickle.dump({
                **hist,
                "per_scn_cost": dict(per_scn_cost), "per_scn_sla": dict(per_scn_sla),
                "global_env_steps": global_env_steps, "global_grad_steps": global_grad_steps,
                "version": version,
            }, f)

    print(f"Starting {version} | state={state_dim} actions={action_dim} net={agent.network_type} "
          f"| lr={agent.current_lr():.1e} ({agent.lr_schedule}) | bootstrap_on_truncation={bootstrap_trunc} "
          f"| sim={cfg['env'].get('sim', {})} | reward.mode={cfg['env'].get('reward', {}).get('mode', 'shaped')}")

    try:
        for episode in range(start_episode, start_episode + num_episodes):
            ep_num = episode + 1
            completed = episode - start_episode + 1
            obs, info = env.reset()
            scn = info["scenario"]
            ep_reward, ep_losses = 0.0, []

            for _ in range(env.max_steps):
                mask = env.get_action_mask()
                action = agent.select_action(obs, training=True, action_mask=mask)
                next_obs, reward, term, trunc, info = env.step(action)
                stored_done = term if bootstrap_trunc else (term or trunc)
                agent.store_transition(obs, action, reward, next_obs, stored_done,
                                       next_action_mask=env.get_action_mask())
                global_env_steps += 1
                if global_env_steps >= warmup_steps:
                    for _ in range(n_grad):
                        loss = agent.train_step()
                        if loss is not None:
                            ep_losses.append(loss)
                            global_grad_steps += 1
                ep_reward += reward
                obs = next_obs
                if term or trunc:
                    break

            ep_cost = float(info.get("cost", 0.0))
            ep_sla = float(info.get("sla_compliance", 0.0))
            hist["rewards"].append(ep_reward)
            hist["loss"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)
            hist["epsilon"].append(agent.epsilon)
            hist["lr"].append(agent.current_lr())
            hist["cost"].append(ep_cost)
            hist["sla"].append(ep_sla)
            hist["scenario"].append(scn)
            per_scn_cost[scn].append(ep_cost)
            per_scn_sla[scn].append(ep_sla)

            if ep_num >= eval_start and ep_num % eval_every == 0:
                ev = run_greedy_eval(agent, cfg, eval_episodes)
                hist["eval"].append({"episode": ep_num, **ev})
                summary = " ".join(f"{n[:3]}:${m['cost']:.0f}/{m['sla']:.0%}" for n, m in ev["scenarios"].items())
                print(f"[EVAL ep {ep_num}] score={ev['score']:.1f} cost=${ev['cost']:.1f} "
                      f"SLA={ev['sla']:.1%} R={ev['reward']:+.1f} | {summary}")
                if run_ctx:
                    m = {"eval_score": ev["score"], "eval_cost": ev["cost"],
                         "eval_sla": ev["sla"], "eval_reward": ev["reward"]}
                    for n, sm in ev["scenarios"].items():
                        m[f"eval_cost_{n}"] = sm["cost"]
                        m[f"eval_sla_{n}"] = sm["sla"]
                    mlflow.log_metrics(m, step=ep_num)
                if ev["sla"] >= eval_sla_gate and ev["score"] > best_score + 0.5:
                    best_score, best_cost, best_sla, best_reward = ev["score"], ev["cost"], ev["sla"], ev["reward"]
                    eval_no_improve = 0
                    agent.save(f"{save_dir}/best_eval.pth")
                    agent.save(f"{save_dir}/best_mixed.pth")
                    with open(f"{save_dir}/best_eval_metrics.pkl", "wb") as f:
                        pickle.dump(ev, f)
                    print(f"  -> saved best_eval.pth cost=${best_cost:.1f} SLA={best_sla:.1%}")
                    if run_ctx:
                        mlflow.log_metrics({"best_cost": best_cost, "best_sla": best_sla,
                                            "best_reward": best_reward, "best_score": best_score}, step=ep_num)
                        try:
                            mlflow.log_artifact(f"{save_dir}/best_eval.pth", artifact_path="models")
                        except Exception as e:  # upload lỗi không được làm hỏng run
                            print(f"Artifact upload skipped: {e}")
                elif ep_num >= min_episodes:
                    eval_no_improve += 1

            if ep_num % checkpoint_every == 0:
                agent.save(f"{save_dir}/checkpoint_ep{ep_num}.pth")
                _dump_history(f"{save_dir}/history_latest.pkl")
                print(f"Saved checkpoint: {save_dir}/checkpoint_ep{ep_num}.pth")
                if run_ctx:
                    try:
                        mlflow.log_artifact(f"{save_dir}/checkpoint_ep{ep_num}.pth", artifact_path="checkpoints")
                    except Exception as e:
                        print(f"Checkpoint upload skipped: {e}")

            if run_ctx and ep_num % log_every == 0:
                m = {
                    "reward": float(np.mean(hist["rewards"][-log_every:])),
                    "cost": float(np.mean(hist["cost"][-log_every:])),
                    "sla": float(np.mean(hist["sla"][-log_every:])),
                    "loss": float(np.mean(hist["loss"][-log_every:])),
                    "epsilon": float(agent.epsilon), "lr": agent.current_lr(),
                    "grad_steps": global_grad_steps, "env_steps": global_env_steps,
                }
                for n in SCENARIO_NAMES:
                    if per_scn_cost[n]:
                        m[f"cost_{n}"] = float(np.mean(per_scn_cost[n][-20:]))
                        m[f"sla_{n}"] = float(np.mean(per_scn_sla[n][-20:]))
                mlflow.log_metrics(m, step=ep_num)

            if ep_num % print_every == 0:
                elapsed = (time.time() - start_time) / 60
                bc = f"${best_cost:.1f}" if best_cost is not None else "n/a"
                bs = f"{best_sla:.1%}" if best_sla is not None else "n/a"
                scn_sum = " ".join(
                    f"{n[:3]}:${np.mean(per_scn_cost[n][-10:]):.0f}/{np.mean(per_scn_sla[n][-10:]):.0%}"
                    for n in SCENARIO_NAMES if per_scn_cost[n]
                )
                warm = " [WARMUP]" if global_env_steps < warmup_steps else ""
                print(f"Ep {ep_num:4d}{warm} | R:{np.mean(hist['rewards'][-print_every:]):+7.1f} "
                      f"Cost:${np.mean(hist['cost'][-print_every:]):5.1f} SLA:{np.mean(hist['sla'][-print_every:]):.1%} "
                      f"Eps:{agent.epsilon:.3f} LR:{agent.current_lr():.1e} best:{bc}/{bs} | {scn_sum} | "
                      f"{elapsed:.1f}m ({completed / max(elapsed, 0.01):.1f} ep/min) grad:{global_grad_steps:,}")

            if (episode >= min_episodes and best_cost is not None
                    and agent.epsilon < eps_gate and eval_no_improve >= eval_patience):
                print(f"Early stop ep {ep_num}. Best cost=${best_cost:.1f} SLA={best_sla:.1%} eps={agent.epsilon:.3f}")
                break
            if (ep_num >= min_episodes and best_cost is not None
                    and len(hist["sla"]) >= collapse_window
                    and np.mean(hist["sla"][-collapse_window:]) < collapse_sla_stop):
                agent.save(f"{save_dir}/collapse_stop_ep{ep_num}.pth")
                print(f"Collapse guard stop ep {ep_num}: train SLA{collapse_window}="
                      f"{np.mean(hist['sla'][-collapse_window:]):.1%}. Keep best_eval.pth.")
                break
    finally:
        agent.save(f"{save_dir}/final_mixed.pth")
        _dump_history(f"{save_dir}/history_mixed.pkl")
        if run_ctx:
            try:
                mlflow.log_artifact(f"{save_dir}/final_mixed.pth", artifact_path="models")
                mlflow.log_artifact(f"{save_dir}/history_mixed.pkl", artifact_path="history")
            except Exception as e:
                print(f"Final artifact upload skipped: {e}")

    if run_ctx and best_cost is not None:
        try:
            mv = mlflow.register_model(f"runs:/{run_ctx.info.run_id}/models/best_eval.pth", "Spot-RL-Agent")
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(mv.name, mv.version, "avg_cost", f"{best_cost:.1f}")
            client.set_model_version_tag(mv.name, mv.version, "avg_sla", f"{best_sla:.3f}")
            client.set_model_version_tag(mv.name, mv.version, "version", version)
            print(f"Registered: Spot-RL-Agent v{mv.version}")
        except Exception as e:
            print(f"Model Registry skipped: {e}")
    if run_ctx:
        mlflow.end_run()

    total = (time.time() - start_time) / 60
    if best_cost is not None:
        print(f"Done. Best: cost=${best_cost:.1f} SLA={best_sla:.1%} R={best_reward:+.1f}")
    else:
        print("Done. No eval passed the SLA gate (no best_eval.pth).")
    print(f"Total: {total:.1f}m | env_steps={global_env_steps:,} | grad_steps={global_grad_steps:,} "
          f"| final lr={agent.current_lr():.2e}")
    return {**hist, "per_scn_cost": dict(per_scn_cost), "per_scn_sla": dict(per_scn_sla),
            "best_cost": best_cost, "best_sla": best_sla,
            "mlflow_run_id": run_ctx.info.run_id if run_ctx else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save-dir", default=None, help="mặc định results/<version>/models")
    p.add_argument("--resume", default=None)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--resume-lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    # Override cho smoke test
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-start", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    t = cfg["training"]
    for key, val in [("num_episodes", args.episodes), ("eval_start", args.eval_start),
                     ("eval_every", args.eval_every), ("eval_episodes", args.eval_episodes),
                     ("warmup_steps", args.warmup_steps)]:
        if val is not None:
            t[key] = val
    if args.max_steps is not None:
        cfg["env"]["max_steps"] = args.max_steps
    if args.episodes is not None:
        t["min_episodes"] = min(t.get("min_episodes", 0), args.episodes)

    save_dir = args.save_dir or f"results/{config_name(cfg)}/models"
    train(cfg, save_dir, mlflow_uri=args.mlflow_uri, resume_checkpoint=args.resume,
          start_episode=args.start_episode, resume_lr=args.resume_lr, device=args.device)


if __name__ == "__main__":
    main()
