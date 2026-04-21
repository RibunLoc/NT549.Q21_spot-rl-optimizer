"""
Mixed-scenario wrapper: rotates through multiple price datasets per episode.

Trains one generalist agent that sees stable, volatile, spike, and az_divergence
markets in the same run. On reset(), samples one scenario uniformly (or by weights).

Each sub-env is a full SpotOrchestratorEnv with its own market_sim/workload_gen,
so state is kept isolated between scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import gymnasium as gym

from envs.spot_orchestrator_env import SpotOrchestratorEnv


class MixedScenarioEnv(gym.Env):
    """Wraps N SpotOrchestratorEnv instances, one per scenario.

    On reset(), picks a scenario (uniform by default) and delegates step() to it
    until the next reset. Action / observation spaces are identical across sub-envs.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        scenario_paths: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
        max_steps: int = 168,
        sla_threshold: float = 0.95,
        workload_config: Dict[str, Any] = None,
        reward_config: Dict[str, Any] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert len(scenario_paths) >= 1, "Need at least one scenario"

        self.scenario_names = list(scenario_paths.keys())
        self.rng = np.random.default_rng(seed)

        # Build one sub-env per scenario
        self.envs: Dict[str, SpotOrchestratorEnv] = {}
        for i, (name, path) in enumerate(scenario_paths.items()):
            sub_seed = None if seed is None else seed + i * 1000
            self.envs[name] = SpotOrchestratorEnv(
                data_path=path,
                max_steps=max_steps,
                sla_threshold=sla_threshold,
                workload_config=workload_config,
                reward_config=reward_config,
                seed=sub_seed,
            )

        # Sampling weights
        if weights is None:
            self.weights = np.ones(len(self.scenario_names)) / len(self.scenario_names)
        else:
            w = np.array([weights.get(n, 1.0) for n in self.scenario_names], dtype=float)
            self.weights = w / w.sum()

        # Spaces from first env (all identical)
        first = next(iter(self.envs.values()))
        self.observation_space = first.observation_space
        self.action_space = first.action_space

        # Active scenario
        self.active_name: str = self.scenario_names[0]
        self.active_env: SpotOrchestratorEnv = first

        # Prioritized sampling: track per-scenario TD-error proxy (avg loss)
        self._scn_loss: Dict[str, float] = {n: 1.0 for n in self.scenario_names}
        self._loss_alpha: float = 0.6   # exponent for priority
        self._loss_ema: float = 0.05    # EMA smoothing factor

    def update_scenario_loss(self, scenario: str, loss: float):
        """Call after each episode with mean TD-loss to update sampling priority."""
        old = self._scn_loss.get(scenario, 1.0)
        self._scn_loss[scenario] = (1 - self._loss_ema) * old + self._loss_ema * loss
        # Recompute weights: priority = loss^alpha, then normalize
        priorities = np.array(
            [self._scn_loss[n] ** self._loss_alpha for n in self.scenario_names],
            dtype=float,
        )
        self.weights = priorities / priorities.sum()

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample scenario for this episode
        idx = int(self.rng.choice(len(self.scenario_names), p=self.weights))
        self.active_name = self.scenario_names[idx]
        self.active_env = self.envs[self.active_name]

        obs, info = self.active_env.reset()
        info['scenario'] = self.active_name
        return obs, info

    def step(self, action: int):
        obs, reward, term, trunc, info = self.active_env.step(action)
        info['scenario'] = self.active_name
        return obs, reward, term, trunc, info

    def get_action_mask(self) -> np.ndarray:
        return self.active_env.get_action_mask()

    @property
    def current_step(self) -> int:
        return self.active_env.current_step

    @property
    def max_steps(self) -> int:
        return self.active_env.max_steps
