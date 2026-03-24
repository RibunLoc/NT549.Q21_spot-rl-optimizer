"""
Main Gymnasium environment for Spot Instance optimization.

MDP Formulation:
- State: spot price, workload, instances, time features
- Action: request spot/on-demand, terminate, migrate
- Reward: cost savings - SLA penalties - migration costs
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any
import pandas as pd

from envs.market_simulator import SpotMarketSimulator
from envs.workload_generator import WorkloadGenerator, Job
from envs.cost_calculator import CostCalculator


class SpotInstanceEnv(gym.Env):
    """
    Spot Instance Optimization Environment.

    Observation (15 features):
        [price, price_ma_6h, price_ma_24h, volatility, interr_prob,
         spot_instances, ondemand_instances, total_capacity,
         pending_jobs, running_jobs, workload_forecast, queue_wait_time,
         hour_of_day, day_of_week, progress]

    Actions (7 discrete):
        0: REQUEST_SPOT
        1: REQUEST_ONDEMAND
        2: TERMINATE_SPOT
        3: TERMINATE_ONDEMAND
        4: MIGRATE_TO_ONDEMAND
        5: MIGRATE_TO_SPOT
        6: DO_NOTHING
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Action constants
    ACTION_REQUEST_SPOT = 0
    ACTION_REQUEST_ONDEMAND = 1
    ACTION_TERMINATE_SPOT = 2
    ACTION_TERMINATE_ONDEMAND = 3
    ACTION_MIGRATE_TO_ONDEMAND = 4
    ACTION_MIGRATE_TO_SPOT = 5
    ACTION_DO_NOTHING = 6

    ACTION_NAMES = [
        "REQUEST_SPOT", "REQUEST_ONDEMAND", "TERMINATE_SPOT",
        "TERMINATE_ONDEMAND", "MIGRATE_TO_ONDEMAND", "MIGRATE_TO_SPOT",
        "DO_NOTHING",
    ]

    def __init__(
        self,
        data_path: str = None,
        price_data: pd.DataFrame = None,
        max_steps: int = 1000,
        sla_threshold: float = 0.95,
        spot_capacity: int = 10,
        ondemand_capacity: int = 5,
        workload_config: Dict[str, Any] = None,
        cost_config: Dict[str, Any] = None,
        render_mode: str = None,
    ):
        """
        Initialize environment.

        Args:
            data_path: Path to spot price CSV/pickle (optional if price_data given)
            price_data: DataFrame with spot prices (optional if data_path given)
            max_steps: Maximum steps per episode
            sla_threshold: Minimum SLA compliance (0-1)
            spot_capacity: Max number of spot instances
            ondemand_capacity: Max number of on-demand instances
            render_mode: Rendering mode
        """
        super().__init__()

        self.max_steps = max_steps
        self.sla_threshold = sla_threshold
        self.spot_capacity = spot_capacity
        self.ondemand_capacity = ondemand_capacity
        self.render_mode = render_mode

        # Load spot price data
        if price_data is not None:
            self.price_data = price_data
        elif data_path is not None:
            if data_path.endswith('.csv'):
                self.price_data = pd.read_csv(data_path)
            elif data_path.endswith('.pkl'):
                self.price_data = pd.read_pickle(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_path}")
        else:
            raise ValueError("Must provide either data_path or price_data")

        # Determine instance type and AZ from data
        self._instance_type = self.price_data['instance_type'].value_counts().index[0]
        self._az = self.price_data[
            self.price_data['instance_type'] == self._instance_type
        ]['availability_zone'].value_counts().index[0]

        # Initialize components
        workload_config = workload_config or {}
        cost_config = cost_config or {}

        self.market_sim = SpotMarketSimulator(
            historical_data=self.price_data,
            instance_type=self._instance_type,
            availability_zone=self._az,
            seed=None,
        )

        self.workload_gen = WorkloadGenerator(
            base_arrival_rate=workload_config.get('base_arrival_rate', 2.0),
            peak_multiplier=workload_config.get('peak_multiplier', 3.0),
            peak_hours=workload_config.get('peak_hours', [9, 10, 11, 14, 15, 16]),
            avg_job_duration=workload_config.get('avg_job_duration', 10),
            seed=None,
        )

        self.cost_calc = CostCalculator(
            ondemand_price=cost_config.get('ondemand_price', CostCalculator.ONDEMAND_PRICE_PER_HOUR),
            spot_price_avg=cost_config.get('spot_price_avg', CostCalculator.SPOT_PRICE_REFERENCE),
            sla_penalty=cost_config.get('sla_penalty', CostCalculator.SLA_PENALTY_PER_FAILED_JOB),
            migration_cost=cost_config.get('migration_cost', CostCalculator.MIGRATION_COST),
        )

        # Define action space (7 discrete actions)
        self.action_space = spaces.Discrete(7)

        # Define observation space (15 features, normalized to [0, 1])
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )

        # Normalization bounds for state features (min-max scaling to [0, 1])
        self._obs_min = np.array([
            0.0,    # current_price
            0.0,    # mean_6h
            0.0,    # mean_24h
            0.0,    # volatility
            0.0,    # interruption_prob
            0.0,    # spot_instances
            0.0,    # ondemand_instances
            0.0,    # total_capacity
            0.0,    # pending_jobs
            0.0,    # running_jobs
            0.0,    # workload_forecast
            0.0,    # avg_wait
            0.0,    # hour_of_day
            0.0,    # day_of_week
            0.0,    # progress
        ], dtype=np.float32)

        self._obs_max = np.array([
            0.20,                           # current_price (spot rarely > $0.20)
            0.20,                           # mean_6h
            0.20,                           # mean_24h
            0.05,                           # volatility
            1.0,                            # interruption_prob
            float(spot_capacity),           # spot_instances
            float(ondemand_capacity),       # ondemand_instances
            float(spot_capacity + ondemand_capacity),  # total_capacity
            100.0,                          # pending_jobs (tăng từ 50 vì peak hours tích lũy nhanh)
            float(spot_capacity + ondemand_capacity),  # running_jobs
            100.0,                          # workload_forecast
            100.0,                          # avg_wait (tăng từ 50 vì tích lũy theo thời gian)
            23.0,                           # hour_of_day
            6.0,                            # day_of_week
            1.0,                            # progress
        ], dtype=np.float32)

        # Internal state
        self._reset_state()

        # Episode history for analysis
        self.episode_history = []

    def _reset_state(self):
        """Reset all internal state variables."""
        self.current_step = 0
        self.num_spot_instances = 0
        self.num_ondemand_instances = 0
        self.pending_jobs = 0
        self.running_jobs_list: List[Job] = []  # Jobs currently executing
        self.running_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.total_cost = 0.0
        self.step_cost = 0.0
        self.num_migrations = 0
        self.num_migrations_to_spot = 0
        self.num_interruptions = 0
        self.queue_wait_accumulator = 0.0
        self.action_counts = np.zeros(7, dtype=int)

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self._reset_state()
        self.episode_history = []

        # Reset simulators
        self.market_sim.reset(seed=seed)
        self.workload_gen.reset(seed=seed)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and transition to next state."""
        # Track action
        self.action_counts[action] += 1

        # Reset per-step counters BEFORE action (migration counter set by action)
        self.num_migrations = 0
        self.num_migrations_to_spot = 0
        self.num_interruptions = 0
        self.step_failed_jobs = 0
        self.step_completed_jobs = 0

        # 1. Execute action
        self._execute_action(action)

        # 2. Simulate one time step
        self._simulate_timestep()

        # 3. Calculate reward
        reward = self._calculate_reward()

        # 4. Get observation
        observation = self._get_observation()

        # 5. Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = self._get_info()

        # Track history
        self.episode_history.append({
            'step': self.current_step,
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'reward': reward,
            'cost': self.step_cost,
            'total_cost': self.total_cost,
            'spot_instances': self.num_spot_instances,
            'ondemand_instances': self.num_ondemand_instances,
            'pending_jobs': self.pending_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'spot_price': self.market_sim.current_price,
            'sla': self._calculate_sla_compliance(),
        })

        return observation, reward, terminated, truncated, info

    # Số instance thay đổi mỗi action (cho agent scale nhanh hơn)
    SCALE_STEP = 3

    def _execute_action(self, action: int):
        """Execute the selected action."""
        if action == self.ACTION_REQUEST_SPOT:
            add = min(self.SCALE_STEP, self.spot_capacity - self.num_spot_instances)
            self.num_spot_instances += add

        elif action == self.ACTION_REQUEST_ONDEMAND:
            add = min(self.SCALE_STEP, self.ondemand_capacity - self.num_ondemand_instances)
            self.num_ondemand_instances += add

        elif action == self.ACTION_TERMINATE_SPOT:
            remove = min(self.SCALE_STEP, self.num_spot_instances)
            self.num_spot_instances -= remove

        elif action == self.ACTION_TERMINATE_ONDEMAND:
            remove = min(self.SCALE_STEP, self.num_ondemand_instances)
            self.num_ondemand_instances -= remove

        elif action == self.ACTION_MIGRATE_TO_ONDEMAND:
            migrate = min(self.SCALE_STEP, self.num_spot_instances,
                          self.ondemand_capacity - self.num_ondemand_instances)
            self.num_spot_instances -= migrate
            self.num_ondemand_instances += migrate
            self.num_migrations += migrate

        elif action == self.ACTION_MIGRATE_TO_SPOT:
            migrate = min(self.SCALE_STEP, self.num_ondemand_instances,
                          self.spot_capacity - self.num_spot_instances)
            self.num_ondemand_instances -= migrate
            self.num_spot_instances += migrate
            self.num_migrations_to_spot += migrate

        elif action == self.ACTION_DO_NOTHING:
            pass

    def _simulate_timestep(self):
        """Simulate one time step: market, workload, job execution."""
        # 1. Step market simulator
        price, interr_prob, interrupted = self.market_sim.step()

        # 2. Handle spot interruptions — kill running jobs on lost instances
        if interrupted and self.num_spot_instances > 0:
            self.num_spot_instances -= 1
            self.num_interruptions += 1
            # Kill one running job that was on the interrupted spot instance
            if self.running_jobs_list:
                self.running_jobs_list.pop()
                self.failed_jobs += 1
                self.step_failed_jobs += 1

        # 3. Generate new jobs
        new_jobs = self.workload_gen.step()
        self.pending_jobs += len(new_jobs)

        # 4. Tick running jobs — decrement remaining time, complete finished ones
        still_running: List[Job] = []
        for job in self.running_jobs_list:
            job.remaining_time -= 1
            if job.remaining_time <= 0:
                self.completed_jobs += 1
                self.step_completed_jobs += 1
            else:
                still_running.append(job)
        self.running_jobs_list = still_running

        # 5. Schedule new jobs onto free capacity
        total_capacity = self.num_spot_instances + self.num_ondemand_instances
        free_slots = total_capacity - len(self.running_jobs_list)

        if free_slots > 0 and self.pending_jobs > 0:
            jobs_to_start = min(self.pending_jobs, free_slots)
            # Pull jobs from the pending queue into running
            started = self.workload_gen.pending_jobs[:jobs_to_start]
            self.workload_gen.pending_jobs = self.workload_gen.pending_jobs[jobs_to_start:]
            self.running_jobs_list.extend(started)
            self.pending_jobs -= jobs_to_start

        self.running_jobs = len(self.running_jobs_list)

        # 6. SLA: jobs waiting too long fail (deadline exceeded)
        if self.pending_jobs > total_capacity * 3 and self.pending_jobs > 0:
            jobs_failed = max(1, self.pending_jobs // 10)
            self.pending_jobs -= jobs_failed
            # Also remove from workload generator's pending list
            self.workload_gen.pending_jobs = self.workload_gen.pending_jobs[:-jobs_failed] if jobs_failed <= len(self.workload_gen.pending_jobs) else []
            self.failed_jobs += jobs_failed
            self.step_failed_jobs += jobs_failed

        # 7. Track queue wait time
        self.queue_wait_accumulator += self.pending_jobs

    def _calculate_reward(self) -> float:
        """Calculate reward based on costs and penalties."""
        step_cost = self.cost_calc.compute_step_cost(
            num_spot=self.num_spot_instances,
            num_ondemand=self.num_ondemand_instances,
            spot_price=self.market_sim.current_price,
            timestep_duration=1.0,
        )
        self.step_cost = step_cost
        self.total_cost += step_cost

        # Savings so với baseline: nếu chạy toàn bộ running + pending jobs
        # bằng on-demand thì tốn bao nhiêu?
        jobs_needing_capacity = self.running_jobs + self.pending_jobs
        baseline_capacity = max(jobs_needing_capacity,
                                self.num_spot_instances + self.num_ondemand_instances)
        savings = self.cost_calc.compute_savings_vs_ondemand(
            actual_cost=step_cost,
            total_capacity=baseline_capacity,
            timestep_duration=1.0,
        )

        # SLA penalty: dùng số job thực tế trong step này
        step_total_jobs = self.step_failed_jobs + self.step_completed_jobs
        sla_penalty = self.cost_calc.compute_sla_penalty(
            failed_jobs=self.step_failed_jobs,
            total_jobs=step_total_jobs,
            sla_threshold=self.sla_threshold,
        )

        migration_penalty = self.cost_calc.compute_migration_penalty(
            num_migrations=self.num_migrations,
        ) + self.num_migrations_to_spot * self.cost_calc.migration_cost_to_spot

        interruption_penalty = self.cost_calc.compute_interruption_penalty(
            num_interruptions=self.num_interruptions,
        )

        reward = self.cost_calc.compute_total_reward(
            step_cost=step_cost,
            savings=savings,
            sla_penalty=sla_penalty,
            migration_penalty=migration_penalty,
            interruption_penalty=interruption_penalty,
        )

        # Phạt thêm cho pending jobs cao (khuyến khích agent giữ queue thấp)
        pending_penalty = self.pending_jobs * 0.1

        return reward - pending_penalty

    def _get_observation(self) -> np.ndarray:
        """Build state vector from current state."""
        price_stats = self.market_sim.get_price_statistics(window=24)
        workload_forecast = self.workload_gen.get_workload_forecast(horizon=10)

        hour_of_day = self.current_step % 24
        day_of_week = (self.current_step // 24) % 7
        avg_wait = self.queue_wait_accumulator / max(1, self.current_step)

        raw = np.array([
            self.market_sim.current_price,
            price_stats['mean_6h'],
            price_stats['mean_24h'],
            price_stats['volatility'],
            self.market_sim.interruption_prob,
            float(self.num_spot_instances),
            float(self.num_ondemand_instances),
            float(self.num_spot_instances + self.num_ondemand_instances),
            float(self.pending_jobs),
            float(self.running_jobs),
            workload_forecast,
            avg_wait,
            float(hour_of_day),
            float(day_of_week),
            self.current_step / self.max_steps,
        ], dtype=np.float32)

        # Min-max normalize to [0, 1]
        denom = self._obs_max - self._obs_min
        denom = np.where(denom == 0, 1.0, denom)  # avoid division by zero
        obs = np.clip((raw - self._obs_min) / denom, 0.0, 1.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return additional info for logging."""
        return {
            "step": self.current_step,
            "cost": self.total_cost,
            "step_cost": self.step_cost,
            "spot_instances": self.num_spot_instances,
            "ondemand_instances": self.num_ondemand_instances,
            "pending_jobs": self.pending_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "sla_compliance": self._calculate_sla_compliance(),
            "spot_price": self.market_sim.current_price,
            "interruption_prob": self.market_sim.interruption_prob,
            "action_counts": self.action_counts.copy(),
        }

    def _calculate_sla_compliance(self) -> float:
        """Calculate current SLA compliance rate."""
        total = self.completed_jobs + self.failed_jobs
        if total == 0:
            return 1.0
        return self.completed_jobs / total

    def get_episode_history(self) -> pd.DataFrame:
        """Return episode history as DataFrame for analysis."""
        if not self.episode_history:
            return pd.DataFrame()
        return pd.DataFrame(self.episode_history)

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"  Spot price: ${self.market_sim.current_price:.4f}/hr")
            print(f"  Instances - Spot: {self.num_spot_instances}, "
                  f"On-demand: {self.num_ondemand_instances}")
            print(f"  Jobs - Pending: {self.pending_jobs}, "
                  f"Completed: {self.completed_jobs}, Failed: {self.failed_jobs}")
            print(f"  SLA: {self._calculate_sla_compliance():.2%}")
            print(f"  Total cost: ${self.total_cost:.2f}")
            print("-" * 50)

    def close(self):
        """Cleanup resources."""
        pass
