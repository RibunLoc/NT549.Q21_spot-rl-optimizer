"""
Multi-Type × Multi-AZ Cost-Aware Spot Instance Orchestrator Environment.

MDP:
- State: 33 features (hybrid top-3 cheapest + aggregated + infra + workload + time)
- Action: 105 discrete (7 ops × 5 types × 3 AZs)
- Reward: savings-based (baseline_OD - actual_cost - penalties)

Agent learns to choose: which operation, which instance type, which AZ
to minimize cost while maintaining SLA for batch processing workloads.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from envs.instance_catalog import (
    INSTANCE_TYPES, AVAILABILITY_ZONES, N_TYPES, N_AZS, N_OPS, N_ACTIONS,
    STATE_DIM, MAX_INSTANCES, MAX_PER_AZ, MAX_VCPU, MAX_JOBS, MAX_WAIT,
    OP_REQUEST_SPOT, OP_REQUEST_ONDEMAND, OP_TERMINATE_SPOT,
    OP_TERMINATE_ONDEMAND, OP_MIGRATE_TO_ONDEMAND, OP_MIGRATE_TO_SPOT,
    OP_DO_NOTHING, OP_NAMES, decode_action, get_od_price, get_max_od_price,
)
from envs.market_simulator import MultiPoolMarketSimulator
from envs.workload_generator import WorkloadGenerator, Job


@dataclass
class PoolState:
    """State of a single (instance_type, az) pool."""
    spot_count: int = 0
    ondemand_count: int = 0


class SpotOrchestratorEnv(gym.Env):
    """
    Cost-Aware Resource Orchestrator Environment.

    Composite action: (operation, instance_type, az) encoded as flat index.
    State: 33 normalized features.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data_path: str = None,
        price_data: pd.DataFrame = None,
        max_steps: int = 168,
        sla_threshold: float = 0.95,
        workload_config: Dict[str, Any] = None,
        reward_config: Dict[str, Any] = None,
        seed: Optional[int] = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.sla_threshold = sla_threshold
        self.render_mode = render_mode

        # Load price data
        if price_data is not None:
            self.price_data = price_data
        elif data_path is not None:
            if data_path.endswith('.pkl'):
                try:
                    self.price_data = pd.read_pickle(data_path)
                except (NotImplementedError, Exception):
                    csv_path = data_path.replace('.pkl', '.csv')
                    self.price_data = pd.read_csv(csv_path)
            else:
                self.price_data = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide data_path or price_data")

        # On-demand prices per type
        self.od_prices = {t.name: t.ondemand_price for t in INSTANCE_TYPES}
        self.max_od_price = get_max_od_price()

        # Market simulator: 5 types × 3 AZs
        self.market_sim = MultiPoolMarketSimulator(
            historical_data=self.price_data,
            instance_types=[t.name for t in INSTANCE_TYPES],
            availability_zones=AVAILABILITY_ZONES,
            od_prices=self.od_prices,
            seed=seed,
        )

        # Workload generator
        wc = workload_config or {}
        self.workload_gen = WorkloadGenerator(
            base_arrival_rate=wc.get('base_arrival_rate', 5.0),
            peak_multiplier=wc.get('peak_multiplier', 3.0),
            avg_job_duration=wc.get('avg_job_duration', 3),
            seed=seed,
        )

        # Reward config
        rc = reward_config or {}
        self.sla_penalty_coeff = rc.get('sla_penalty', 10.0)
        self.interrupt_penalty_coeff = rc.get('interrupt_penalty', 0.5)
        self.migration_penalty_coeff = rc.get('migration_penalty', 1.0)
        self.concentration_penalty_coeff = rc.get('concentration_penalty', 0.2)
        self.concentration_threshold = rc.get('concentration_threshold', 0.8)

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Gymnasium RNG
        self.np_random = np.random.default_rng(seed)

        # Internal state
        self._reset_state()

    def _reset_state(self):
        """Reset all internal state variables."""
        self.current_step = 0

        # Pool states: (type_idx, az_idx) → PoolState
        self.pools: Dict[Tuple[int, int], PoolState] = {}
        for t in range(N_TYPES):
            for az in range(N_AZS):
                self.pools[(t, az)] = PoolState()

        # Job tracking
        self.pending_jobs = 0
        self.running_jobs_list: List[Job] = []
        self.running_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.step_completed_jobs = 0
        self.step_failed_jobs = 0
        self.queue_wait_accumulator = 0.0

        # Cost tracking
        self.total_cost = 0.0
        self.step_cost = 0.0

        # Migration/interruption tracking per step
        self.step_migrations = 0
        self.step_interruptions = 0
        self.step_wasted_action = False
        self.step_migrate_to_spot = False   # for migrate-back bonus

        # Idle streak: số steps liên tiếp mà pending=0 và running=0
        self.idle_streak = 0

        # Stability tracking: phát hiện request/terminate churn
        self.prev_instance_count = 0        # instance count bước trước
        self.instance_change_history = []   # +1=request, -1=terminate, 0=stable
        self.churn_streak = 0               # số steps liên tiếp flip request↔terminate

        # Migrate-back tracking: thưởng khi migrate về spot sau period giá cao
        self.steps_since_migrate_to_od = 0  # đếm steps kể từ lần migrate sang OD
        self.has_od_instances = False        # có OD instance nào không

        # SLA tracking (rolling window)
        self.sla_window: List[Tuple[int, int]] = []  # (completed, failed) per step

        # Action counts
        self.action_counts = np.zeros(N_ACTIONS, dtype=int)

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        self._reset_state()

        sub_seeds = self.np_random.integers(0, 2**31, size=2)
        self.market_sim.reset(seed=int(sub_seeds[0]))
        self.workload_gen.reset(seed=int(sub_seeds[1]))

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and advance one timestep."""
        self.action_counts[action] += 1

        # Reset per-step counters
        self.step_completed_jobs = 0
        self.step_failed_jobs = 0
        self.step_migrations = 0
        self.step_interruptions = 0

        # 1. Execute action
        self._execute_action(action)

        # 2. Simulate timestep
        self._simulate_timestep()

        # 3. Calculate reward
        reward = self._calculate_reward()

        # 4. Get observation
        observation = self._get_observation()

        # 5. Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return observation, reward, terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────
    #  Action execution
    # ──────────────────────────────────────────────

    def _execute_action(self, action: int):
        """Execute composite action (op, type, az)."""
        op, type_idx, az_idx = decode_action(action)
        pool = self.pools[(type_idx, az_idx)]

        self.step_wasted_action = False   # reset per action
        self.step_migrate_to_spot = False  # reset per action

        if op == OP_REQUEST_SPOT:
            # Block nếu hệ thống đã idle >= 3 steps liên tiếp (pending=0, running=0)
            if self.idle_streak >= 3:
                self.step_wasted_action = True
            elif self._total_instances() < MAX_INSTANCES and self._az_instances(az_idx) < MAX_PER_AZ:
                pool.spot_count += 1
            else:
                self.step_wasted_action = True  # at cap already

        elif op == OP_REQUEST_ONDEMAND:
            # Same guard: block request khi hoàn toàn idle
            if self.idle_streak >= 3:
                self.step_wasted_action = True
            elif self._total_instances() < MAX_INSTANCES and self._az_instances(az_idx) < MAX_PER_AZ:
                pool.ondemand_count += 1
            else:
                self.step_wasted_action = True

        elif op == OP_TERMINATE_SPOT:
            # Block nếu workload đang cao — utilization > 70% hoặc pending > 20% vcpu
            total_vcpu = max(self._total_vcpu(), 1)
            util = self.running_jobs / total_vcpu
            if util > 0.7 or self.pending_jobs > total_vcpu * 0.2:
                self.step_wasted_action = True  # đang cần capacity, không terminate
            elif pool.spot_count > 0:
                pool.spot_count -= 1
            else:
                self.step_wasted_action = True

        elif op == OP_TERMINATE_ONDEMAND:
            total_vcpu = max(self._total_vcpu(), 1)
            util = self.running_jobs / total_vcpu
            if util > 0.7 or self.pending_jobs > total_vcpu * 0.2:
                self.step_wasted_action = True  # đang cần capacity, không terminate
            elif pool.ondemand_count > 0:
                pool.ondemand_count -= 1
            else:
                self.step_wasted_action = True

        elif op == OP_MIGRATE_TO_ONDEMAND:
            if pool.spot_count > 0 and self._total_instances() < MAX_INSTANCES:
                pool.spot_count -= 1
                pool.ondemand_count += 1
                self.step_migrations += 1
            else:
                self.step_wasted_action = True

        elif op == OP_MIGRATE_TO_SPOT:
            # Block nếu workload đang quá cao — spot có thể bị interrupt lúc peak
            total_vcpu = max(self._total_vcpu(), 1)
            util = self.running_jobs / total_vcpu
            if util > 0.8 and self.pending_jobs > 0:
                self.step_wasted_action = True  # quá rủi ro khi migrate lúc peak
            elif pool.ondemand_count > 0:
                pool.ondemand_count -= 1
                pool.spot_count += 1
                self.step_migrations += 1
                self.step_migrate_to_spot = True
            else:
                self.step_wasted_action = True

        elif op == OP_DO_NOTHING:
            pass

    # ──────────────────────────────────────────────
    #  Timestep simulation
    # ──────────────────────────────────────────────

    def _simulate_timestep(self):
        """Simulate one timestep: prices, interruptions, workload, jobs."""
        # 1. Step all market simulators
        market_results = self.market_sim.step()

        # 2. Handle interruptions per pool
        for (t_idx, az_idx), (price, interr_prob, interrupted) in market_results.items():
            pool = self.pools[(t_idx, az_idx)]
            if interrupted and pool.spot_count > 0:
                pool.spot_count -= 1
                self.step_interruptions += 1
                # Kill a running job
                if self.running_jobs_list:
                    idx = self.np_random.integers(len(self.running_jobs_list))
                    self.running_jobs_list.pop(idx)
                    self.failed_jobs += 1
                    self.step_failed_jobs += 1

        # 3. Generate new jobs
        new_jobs = self.workload_gen.step()
        self.pending_jobs = len(self.workload_gen.pending_jobs)

        # 4. Tick running jobs
        still_running = []
        for job in self.running_jobs_list:
            job.remaining_time -= 1
            if job.remaining_time <= 0:
                self.completed_jobs += 1
                self.step_completed_jobs += 1
            else:
                still_running.append(job)
        self.running_jobs_list = still_running

        # 5. Schedule pending jobs onto free capacity
        total_vcpu = self._total_vcpu()
        used_vcpu = len(self.running_jobs_list)  # 1 job = 1 vCPU slot
        free_slots = total_vcpu - used_vcpu

        if free_slots > 0 and self.pending_jobs > 0:
            jobs_to_start = min(self.pending_jobs, free_slots)
            started = self.workload_gen.pending_jobs[:jobs_to_start]
            self.workload_gen.pending_jobs = self.workload_gen.pending_jobs[jobs_to_start:]
            self.running_jobs_list.extend(started)
            self.pending_jobs = len(self.workload_gen.pending_jobs)

        self.running_jobs = len(self.running_jobs_list)

        # Update idle streak (sau khi workload đã được update)
        if self.pending_jobs == 0 and self.running_jobs == 0:
            self.idle_streak += 1
        else:
            self.idle_streak = 0

        # 6. SLA: jobs waiting too long fail
        total_cap = max(1, total_vcpu)
        if self.pending_jobs > total_cap * 3:
            jobs_failed = max(1, self.pending_jobs // 10)
            actual_remove = min(jobs_failed, len(self.workload_gen.pending_jobs))
            self.workload_gen.pending_jobs = self.workload_gen.pending_jobs[actual_remove:]
            self.pending_jobs = len(self.workload_gen.pending_jobs)
            self.failed_jobs += actual_remove
            self.step_failed_jobs += actual_remove

        # 7. Queue wait tracking
        self.queue_wait_accumulator += self.pending_jobs

        # 8. SLA window
        self.sla_window.append((self.step_completed_jobs, self.step_failed_jobs))
        if len(self.sla_window) > 10:
            self.sla_window.pop(0)

        # 9. Compute step cost
        self.step_cost = 0.0
        for (t_idx, az_idx), pool in self.pools.items():
            spot_price = self.market_sim.get_pool_price(t_idx, az_idx)
            od_price = INSTANCE_TYPES[t_idx].ondemand_price
            self.step_cost += pool.spot_count * spot_price
            self.step_cost += pool.ondemand_count * od_price
        self.total_cost += self.step_cost

        # 10. Stability tracking — detect request/terminate churn
        current_count = self._total_instances()
        delta = current_count - self.prev_instance_count
        change = 1 if delta > 0 else (-1 if delta < 0 else 0)
        self.instance_change_history.append(change)
        if len(self.instance_change_history) > 4:
            self.instance_change_history.pop(0)
        # churn = alternating +1/-1 in recent history (e.g. [1,-1,1,-1])
        if len(self.instance_change_history) >= 3:
            flips = sum(
                1 for i in range(1, len(self.instance_change_history))
                if self.instance_change_history[i] != 0
                and self.instance_change_history[i] != self.instance_change_history[i-1]
                and self.instance_change_history[i-1] != 0
            )
            self.churn_streak = flips
        else:
            self.churn_streak = 0
        self.prev_instance_count = current_count

        # 11. Migrate-back tracking
        self.has_od_instances = any(p.ondemand_count > 0 for p in self.pools.values())
        if self.has_od_instances:
            self.steps_since_migrate_to_od += 1
        else:
            self.steps_since_migrate_to_od = 0

    # ──────────────────────────────────────────────
    #  Reward
    # ──────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        """
        Savings-based reward (no double-counting).

        R = savings - sla_penalty - interrupt_penalty - migration_cost - concentration_penalty
        """
        # Baseline: what it would cost if all current capacity were on-demand
        total_instances = self._total_instances()
        if total_instances > 0:
            # Baseline = each instance at its type's OD price
            baseline_cost = 0.0
            for (t_idx, az_idx), pool in self.pools.items():
                od_price = INSTANCE_TYPES[t_idx].ondemand_price
                baseline_cost += (pool.spot_count + pool.ondemand_count) * od_price
            savings = baseline_cost - self.step_cost
        else:
            savings = 0.0

        # SLA penalty
        sla_penalty = self.step_failed_jobs * self.sla_penalty_coeff

        # Interrupt penalty
        interrupt_penalty = self.step_interruptions * self.interrupt_penalty_coeff

        # Migration penalty — migrate về spot không bị phạt (khuyến khích)
        # chỉ phạt migrate sang OD (tốn tiền hơn)
        if self.step_migrate_to_spot:
            migration_penalty = 0.0   # không phạt migrate về spot
        else:
            migration_penalty = self.step_migrations * self.migration_penalty_coeff

        # Concentration penalty: only if >80% in single AZ
        concentration_penalty = 0.0
        if total_instances > 0:
            az_counts = np.zeros(N_AZS)
            for (t_idx, az_idx), pool in self.pools.items():
                az_counts[az_idx] += pool.spot_count + pool.ondemand_count
            concentration = float(np.max(az_counts)) / total_instances
            if concentration > self.concentration_threshold:
                concentration_penalty = self.concentration_penalty_coeff * (
                    concentration - self.concentration_threshold
                )

        # Pending penalty — encourage provisioning
        pending_penalty = 0.0
        total_cap = max(1, self._total_vcpu())
        if self.pending_jobs > 0:
            pending_penalty = min((self.pending_jobs / total_cap) * 2.0, 5.0)

        # Idle penalty — no instances but jobs waiting
        idle_penalty = 0.0
        if total_instances == 0 and self.pending_jobs > 0:
            idle_penalty = 5.0

        # Wasted action penalty — discourage no-op terminate/migrate
        # Phạt nặng hơn khi idle_streak cao (liên tục request khi không có job)
        if self.step_wasted_action:
            if self.idle_streak >= 6:
                wasted_penalty = 1.5   # idle lâu mà vẫn request → phạt nặng
            elif self.idle_streak >= 3:
                wasted_penalty = 0.8
            else:
                wasted_penalty = 0.3
        else:
            wasted_penalty = 0.0

        # Overprovisioning penalty — penalize excess idle capacity
        # Scaled by actual cost of idle instances, not just ratio
        overprov_penalty = 0.0
        if total_cap > 0 and self.running_jobs >= 0:
            idle_vcpu = total_cap - self.running_jobs
            idle_ratio = idle_vcpu / total_cap
            if idle_ratio > 0.5:  # >50% idle
                overprov_penalty = (idle_ratio - 0.5) * (idle_vcpu / 40.0) * 1.5

        # Workload-capacity mismatch penalty
        # Phạt khi số instance KHÔNG phù hợp với workload thực tế:
        # - Quá nhiều instance so với workload thấp (lãng phí tiền)
        # - Quá ít instance so với workload cao (SLA risk)
        mismatch_penalty = 0.0
        if total_cap > 0:
            util = self.running_jobs / total_cap
            ideal_vcpu = max(self.running_jobs + self.pending_jobs, 1) * 1.3  # buffer 30%
            actual_vcpu = float(total_cap)
            if actual_vcpu > ideal_vcpu * 2.0 and self.pending_jobs == 0:
                # Quá thừa: actual > 2x ideal và không có pending → phạt theo mức dư
                excess_ratio = (actual_vcpu - ideal_vcpu) / actual_vcpu
                mismatch_penalty = excess_ratio * 1.5
            elif actual_vcpu < ideal_vcpu * 0.5 and self.pending_jobs > 10:
                # Quá thiếu: actual < 50% ideal và pending nhiều → phạt nhẹ (pending_penalty đã cover)
                mismatch_penalty = 0.5

        # Stability reward — phạt khi request/terminate xen kẽ (churn)
        # churn_streak=2: 3 bước xen kẽ liên tiếp → phạt 0.4
        # churn_streak=3: 4 bước xen kẽ liên tiếp → phạt 0.8 (rất đau)
        churn_penalty = 0.0
        if self.churn_streak >= 2:
            churn_penalty = (self.churn_streak - 1) * 0.4

        # Migrate-back bonus — thưởng khi MIGRATE_TO_SPOT sau period có OD
        # Chỉ thưởng nếu: đang thực sự migrate về spot VÀ đã có OD >= 2 steps
        migrate_back_bonus = 0.0
        if self.step_migrate_to_spot and self.steps_since_migrate_to_od >= 2:
            avg_interr = np.mean([
                self.market_sim.get_pool_interrupt_prob(t_idx, az_idx)
                for (t_idx, az_idx) in self.pools
            ])
            if avg_interr < 0.15:   # spot đang ổn định → thưởng mạnh
                migrate_back_bonus = 2.0
            elif avg_interr < 0.25:
                migrate_back_bonus = 1.0
            else:
                migrate_back_bonus = 0.3  # vẫn thưởng nhẹ dù interrupt cao

        # Cost-efficiency bonus — thưởng khi savings/cost ratio cao
        # Khuyến khích dùng spot rẻ thay vì OD
        efficiency_bonus = 0.0
        if self.step_cost > 0 and savings > 0:
            efficiency_ratio = savings / self.step_cost  # spot savings vs actual cost
            if efficiency_ratio > 0.3:   # tiết kiệm > 30% so với full OD
                efficiency_bonus = min(efficiency_ratio * 0.5, 1.0)

        reward = (
            savings
            - sla_penalty
            - interrupt_penalty
            - migration_penalty
            - concentration_penalty
            - pending_penalty
            - idle_penalty
            - wasted_penalty
            - overprov_penalty
            - churn_penalty
            - mismatch_penalty
            + migrate_back_bonus
            + efficiency_bonus
        )

        return float(np.clip(reward, -20.0, 20.0))

    # ──────────────────────────────────────────────
    #  Observation (33 features)
    # ──────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build 33-feature state vector, normalized to [0,1]."""
        max_od = self.max_od_price
        total_instances = max(1, self._total_instances())

        # --- Per-pool data: (type_idx, az_idx) → (price_ratio, interr_prob, vcpu/$, az_idx) ---
        pool_data = []
        for t_idx in range(N_TYPES):
            for az_idx in range(N_AZS):
                price = self.market_sim.get_pool_price(t_idx, az_idx)
                interr = self.market_sim.get_pool_interrupt_prob(t_idx, az_idx)
                od_price = INSTANCE_TYPES[t_idx].ondemand_price
                vcpus = INSTANCE_TYPES[t_idx].vcpus

                price_ratio = price / od_price if od_price > 0 else 1.0
                vcpu_per_dollar = vcpus / price if price > 0 else 0.0

                pool_data.append({
                    'type_idx': t_idx,
                    'az_idx': az_idx,
                    'price': price,
                    'price_ratio': price_ratio,
                    'interr_prob': interr,
                    'vcpu_per_dollar': vcpu_per_dollar,
                })

        # Sort by price to find top-3 cheapest
        pool_data.sort(key=lambda x: x['price_ratio'])

        # Max vcpu/$ for normalization
        max_vcpu_dollar = max(p['vcpu_per_dollar'] for p in pool_data) if pool_data else 1.0

        # Feature vector
        features = []

        # [0-11] Top-3 cheapest combos (4 features each)
        for i in range(3):
            if i < len(pool_data):
                p = pool_data[i]
                features.append(np.clip(p['price_ratio'], 0, 2.0) / 2.0)
                features.append(np.clip(p['interr_prob'], 0, 1.0))
                features.append(np.clip(p['vcpu_per_dollar'] / max(max_vcpu_dollar, 1e-6), 0, 1.0))
                features.append(p['az_idx'] / max(N_AZS - 1, 1))
            else:
                features.extend([0.5, 0.1, 0.5, 0.0])

        # [12-14] Multi-AZ aggregated
        az_avg_prices = np.zeros(N_AZS)
        for p in pool_data:
            az_avg_prices[p['az_idx']] += p['price']
        az_avg_prices /= N_TYPES  # average across types

        if az_avg_prices.max() > 0:
            az_spread = (az_avg_prices.max() - az_avg_prices.min()) / max_od
        else:
            az_spread = 0.0
        features.append(np.clip(az_spread, 0, 1.0))  # [12] az_price_spread

        cheapest_az = int(np.argmin(az_avg_prices))
        features.append(cheapest_az / max(N_AZS - 1, 1))  # [13] cheapest_az_id

        # AZ concentration
        az_inst_counts = np.zeros(N_AZS)
        for (t_idx, az_idx), pool in self.pools.items():
            az_inst_counts[az_idx] += pool.spot_count + pool.ondemand_count
        if self._total_instances() > 0:
            az_concentration = float(np.max(az_inst_counts)) / self._total_instances()
        else:
            az_concentration = 0.0
        features.append(np.clip(az_concentration, 0, 1.0))  # [14]

        # [15-17] Multi-Type aggregated
        type_avg_prices = np.zeros(N_TYPES)
        type_avg_interr = np.zeros(N_TYPES)
        type_vcpu_ratios = np.zeros(N_TYPES)
        for p in pool_data:
            type_avg_prices[p['type_idx']] += p['price_ratio']
            type_avg_interr[p['type_idx']] += p['interr_prob']
            type_vcpu_ratios[p['type_idx']] += p['vcpu_per_dollar']
        type_avg_prices /= N_AZS
        type_avg_interr /= N_AZS
        type_vcpu_ratios /= N_AZS

        best_price_rank = float(np.argmin(type_avg_prices)) / max(N_TYPES - 1, 1)
        features.append(best_price_rank)  # [15]

        worst_interr_rank = float(np.argmax(type_avg_interr)) / max(N_TYPES - 1, 1)
        features.append(worst_interr_rank)  # [16]

        best_vcpu = float(np.max(type_vcpu_ratios))
        features.append(np.clip(best_vcpu / max(max_vcpu_dollar, 1e-6), 0, 1.0))  # [17]

        # [18-20] Infrastructure
        total_spot = sum(p.spot_count for p in self.pools.values())
        total_od = sum(p.ondemand_count for p in self.pools.values())
        total_vcpu = self._total_vcpu()

        features.append(total_spot / MAX_INSTANCES)  # [18]
        features.append(total_od / MAX_INSTANCES)  # [19]
        features.append(total_vcpu / MAX_VCPU)  # [20]

        # [21-24] Workload
        features.append(min(self.pending_jobs / MAX_JOBS, 1.0))  # [21]
        features.append(min(self.running_jobs / MAX_JOBS, 1.0))  # [22]

        forecast = self.workload_gen.get_workload_forecast(horizon=8)  # 8h ahead
        features.append(min(forecast / (MAX_JOBS * 3), 1.0))  # [23] normalize: horizon=8 → ~3x MAX_JOBS

        avg_wait = self.queue_wait_accumulator / max(1, self.current_step)
        features.append(min(avg_wait / MAX_WAIT, 1.0))  # [24] (capped)

        # [25-27] Time
        hour = self.current_step % 24
        day = (self.current_step // 24) % 7
        progress = self.current_step / max(self.max_steps, 1)

        features.append(hour / 23.0)  # [25]
        features.append(day / 6.0)  # [26]
        features.append(progress)  # [27]

        # [28-32] Current state
        # Avg spot price across running instances
        if total_spot > 0:
            avg_spot_price = 0.0
            for (t_idx, az_idx), pool in self.pools.items():
                if pool.spot_count > 0:
                    avg_spot_price += pool.spot_count * self.market_sim.get_pool_price(t_idx, az_idx)
            avg_spot_price /= total_spot
            features.append(np.clip(avg_spot_price / max_od, 0, 1.0))  # [28]
        else:
            features.append(0.0)

        # Avg interrupt prob across running spots
        if total_spot > 0:
            avg_interr = 0.0
            for (t_idx, az_idx), pool in self.pools.items():
                if pool.spot_count > 0:
                    avg_interr += pool.spot_count * self.market_sim.get_pool_interrupt_prob(t_idx, az_idx)
            avg_interr /= total_spot
            features.append(np.clip(avg_interr, 0, 1.0))  # [29]
        else:
            features.append(0.0)

        # Spot ratio
        total_all = total_spot + total_od
        features.append(total_spot / max(total_all, 1))  # [30]

        # Cost rate (current step cost normalized)
        max_cost_rate = sum(t.ondemand_price for t in INSTANCE_TYPES) * MAX_PER_AZ
        features.append(np.clip(self.step_cost / max(max_cost_rate, 1e-6), 0, 1.0))  # [31]

        # SLA health (rolling 10-step window)
        if self.sla_window:
            total_c = sum(c for c, f in self.sla_window)
            total_f = sum(f for c, f in self.sla_window)
            total = total_c + total_f
            sla_health = total_c / total if total > 0 else 1.0
        else:
            sla_health = 1.0
        features.append(np.clip(sla_health, 0, 1.0))  # [32]

        obs = np.array(features, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _total_instances(self) -> int:
        """Total instances across all pools."""
        return sum(p.spot_count + p.ondemand_count for p in self.pools.values())

    def _total_spot(self) -> int:
        return sum(p.spot_count for p in self.pools.values())

    def _total_ondemand(self) -> int:
        return sum(p.ondemand_count for p in self.pools.values())

    def _total_vcpu(self) -> int:
        """Total vCPU capacity."""
        total = 0
        for (t_idx, az_idx), pool in self.pools.items():
            vcpus = INSTANCE_TYPES[t_idx].vcpus
            total += (pool.spot_count + pool.ondemand_count) * vcpus
        return total

    def _az_instances(self, az_idx: int) -> int:
        """Total instances in a specific AZ."""
        return sum(
            p.spot_count + p.ondemand_count
            for (t, az), p in self.pools.items() if az == az_idx
        )

    def _calculate_sla(self) -> float:
        """Overall SLA compliance."""
        total = self.completed_jobs + self.failed_jobs
        return self.completed_jobs / total if total > 0 else 1.0

    def _get_info(self) -> Dict[str, Any]:
        """Return metrics for logging."""
        # Type distribution
        type_dist = {}
        for t_idx, itype in enumerate(INSTANCE_TYPES):
            count = sum(
                self.pools[(t_idx, az)].spot_count + self.pools[(t_idx, az)].ondemand_count
                for az in range(N_AZS)
            )
            type_dist[itype.name] = count

        # AZ distribution
        az_dist = {}
        for az_idx, az_name in enumerate(AVAILABILITY_ZONES):
            count = self._az_instances(az_idx)
            az_dist[az_name] = count

        return {
            "step": self.current_step,
            "cost": self.total_cost,
            "step_cost": self.step_cost,
            "spot_instances": self._total_spot(),
            "ondemand_instances": self._total_ondemand(),
            "total_instances": self._total_instances(),
            "total_vcpu": self._total_vcpu(),
            "pending_jobs": self.pending_jobs,
            "running_jobs": self.running_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "sla_compliance": self._calculate_sla(),
            "type_distribution": type_dist,
            "az_distribution": az_dist,
        }

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step {self.current_step}/{self.max_steps} | "
                  f"Spot:{info['spot_instances']} OD:{info['ondemand_instances']} "
                  f"vCPU:{info['total_vcpu']} | "
                  f"Pending:{self.pending_jobs} Running:{self.running_jobs} | "
                  f"SLA:{info['sla_compliance']:.1%} Cost:${self.total_cost:.2f}")

    def close(self):
        pass
