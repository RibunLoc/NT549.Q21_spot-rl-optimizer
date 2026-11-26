"""
Dùng chung cho train và eval: đọc config YAML, dựng env và agent từ cùng một file.

Config hỗ trợ kế thừa: `extends: v41.yaml` (đường dẫn tương đối với file hiện tại),
các key ở file con ghi đè (deep merge) lên file cha.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCENARIO_NAMES = ("stable", "volatile", "spike", "az_divergence")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Đọc YAML, xử lý `extends` (đệ quy)."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(str((p.parent / parent).resolve()))
        base.pop("_config_path", None)
        cfg = _deep_merge(base, cfg)
    cfg["_config_path"] = str(p)
    return cfg


def scenario_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Tên scenario → đường dẫn tuyệt đối tới CSV giá."""
    out = {}
    for name, rel in cfg["env"]["scenarios"].items():
        rp = Path(rel)
        out[name] = str(rp if rp.is_absolute() else PROJECT_ROOT / rp)
    return out


def _env_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    e = cfg["env"]
    return dict(
        max_steps=e.get("max_steps", 672),
        sla_threshold=e.get("sla_threshold", 0.95),
        workload_config=e.get("workload", {}),
        reward_config=e.get("reward", {}),
        sim_config=e.get("sim", {}),
    )


def make_train_env(cfg: Dict[str, Any]):
    """Env train: MixedScenarioEnv bốc ngẫu nhiên 1 scenario mỗi episode."""
    from envs.mixed_scenario_env import MixedScenarioEnv
    e = cfg["env"]
    env = MixedScenarioEnv(
        scenario_paths=scenario_paths(cfg),
        weights=e.get("scenario_weights"),
        seed=e.get("seed", 42),
        **_env_kwargs(cfg),
    )
    env._loss_alpha = 0.0
    return env


def make_eval_env(cfg: Dict[str, Any], scenario: str, seed: Optional[int]):
    """Env eval: 1 scenario, seed cố định."""
    from envs.spot_orchestrator_env import SpotOrchestratorEnv
    return SpotOrchestratorEnv(
        data_path=scenario_paths(cfg)[scenario],
        seed=seed,
        **_env_kwargs(cfg),
    )


def make_agent(cfg: Dict[str, Any], state_dim: int, action_dim: int, device: str):
    from agents.dqn_agent import DQNAgent
    a = cfg["agent"]
    return DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=a.get("hidden_dim", 512),
        network_type=a.get("network_type", "factored"),
        learning_rate=float(a.get("learning_rate", 2e-5)),
        lr_schedule=a.get("lr_schedule", "constant"),
        lr_min=float(a.get("lr_min", 1e-6)),
        lr_total_steps=int(a.get("lr_total_steps", 2_000_000)),
        gamma=a.get("gamma", 0.99),
        batch_size=a.get("batch_size", 512),
        buffer_size=a.get("buffer_size", 500_000),
        epsilon_start=a.get("epsilon_start", 1.0),
        epsilon_end=a.get("epsilon_end", 0.08),
        epsilon_decay=a.get("epsilon_decay", 450_000),
        target_update_freq=a.get("target_update_freq", 8000),
        use_per=a.get("use_per", True),
        per_alpha=a.get("per_alpha", 0.5),
        per_beta_start=a.get("per_beta_start", 0.5),
        device=device,
    )


def flatten_config(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """{'env': {'reward': {'mode': 'simple'}}} → {'env.reward.mode': 'simple'} (cho MLflow params)."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_config(v, key + "."))
        else:
            out[key] = v
    return out


def config_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("version") or Path(cfg.get("_config_path", "config")).stem
