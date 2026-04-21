"""
Multi-Type × Multi-AZ Cost-Aware Spot Instance Orchestrator Environment.

MDP:
- State: 42 features (hybrid top-3 cheapest + aggregated + infra + workload + time + extra context)
- Action: 135 discrete (9 ops × 5 types × 3 AZs)
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
    INSTANCE_TYPES, AVAILABILITY_ZONES, N_TYPES, N_AZS,
    STATE_DIM, MAX_INSTANCES, MAX_PER_AZ, MAX_VCPU, MAX_JOBS, MAX_WAIT,
    get_od_price, get_max_od_price,
)
from envs.action_schema import (
    Operation, N_ACTIONS, HOLD_ACTION, OPERATION_NAMES,
    decode_action, encode_action,
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

        # Extra state tracking (for new features [33-41])
        self.interrupt_history: List[int] = []      # interrupt count per step (last 10)
        self.pending_history: List[int] = []        # pending jobs per step (last 5)
        self.step_migrate_spot_to_spot = False      # for reward bonus
        self.step_unnecessary_od = False            # for unnecessary OD penalty
        self.step_reserved_capacity = False         # v2: proactive RESERVE_CAPACITY bonus
        self.baseline_od_cost_per_step = sum(       # full OD cost baseline (per step)
            t.ondemand_price for t in INSTANCE_TYPES
        )

        # Per-pool resource tracking (for state features [42-71])
        self.pool_running_jobs: Dict[Tuple[int, int], int] = {
            (t, az): 0 for t in range(N_TYPES) for az in range(N_AZS)
        }
        self.pool_cpu_util: Dict[Tuple[int, int], float] = {
            (t, az): 0.0 for t in range(N_TYPES) for az in range(N_AZS)
        }
        self.pool_ram_util: Dict[Tuple[int, int], float] = {
            (t, az): 0.0 for t in range(N_TYPES) for az in range(N_AZS)
        }

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
        """Execute pool-targeted operation using v2 action schema.

        Operations:
          PROVISION_SPOT / PROVISION_ONDEMAND  — add capacity
          RELEASE_SPOT / RELEASE_ONDEMAND      — remove capacity
          CONVERT_TO_SPOT / CONVERT_TO_OD      — swap billing (same pool)
          REBALANCE_SPOT                       — move spot from expensive pool here
          RESERVE_CAPACITY                     — preemptive OD when interrupt risk high
          HOLD                                 — no-op
        """
        op, type_idx, az_idx = decode_action(action)
        pool = self.pools[(type_idx, az_idx)]

        # Reset per-step action flags
        self.step_wasted_action = False
        self.step_migrate_to_spot = False
        self.step_migrate_spot_to_spot = False
        self.step_unnecessary_od = False
        self.step_reserved_capacity = False  # NEW in v2

        # Shortcuts
        total_instances = self._total_instances()
        total_vcpu = max(self._total_vcpu(), 1)
        at_capacity = total_instances >= MAX_INSTANCES or self._az_instances(az_idx) >= MAX_PER_AZ

        if op == Operation.PROVISION_SPOT:
            # Block khi đã idle >= 3 steps liên tiếp (không có workload)
            if self.idle_streak >= 3 or at_capacity:
                self.step_wasted_action = True
            else:
                pool.spot_count += 1

        elif op == Operation.PROVISION_ONDEMAND:
            if self.idle_streak >= 3 or at_capacity:
                self.step_wasted_action = True
            else:
                pool.ondemand_count += 1

        elif op == Operation.RELEASE_SPOT:
            # Block khi pending > 80% capacity (agent đang overload)
            if self.pending_jobs > total_vcpu * 0.8:
                self.step_wasted_action = True
            elif pool.spot_count > 0:
                pool.spot_count -= 1
            else:
                self.step_wasted_action = True

        elif op == Operation.RELEASE_ONDEMAND:
            if self.pending_jobs > total_vcpu * 0.8:
                self.step_wasted_action = True
            elif pool.ondemand_count > 0:
                pool.ondemand_count -= 1
            else:
                self.step_wasted_action = True

        elif op == Operation.CONVERT_TO_ONDEMAND:
            # Spot → OD: stabilize when interrupt risk high. Always counts as migration.
            if pool.spot_count > 0 and total_instances < MAX_INSTANCES:
                pool.spot_count -= 1
                pool.ondemand_count += 1
                self.step_migrations += 1
            else:
                self.step_wasted_action = True

        elif op == Operation.CONVERT_TO_SPOT:
            # OD → spot: opportunistic savings. Block khi peak workload.
            util = self.running_jobs / total_vcpu
            if util > 0.8 and self.pending_jobs > 0:
                self.step_wasted_action = True  # unsafe at peak
            elif pool.ondemand_count > 0:
                pool.ondemand_count -= 1
                pool.spot_count += 1
                self.step_migrations += 1
                self.step_migrate_to_spot = True
            else:
                self.step_wasted_action = True

        elif op == Operation.REBALANCE_SPOT:
            # Move 1 spot from most-expensive pool → (type_idx, az_idx).
            # Requires: dst ≥15% cheaper than src, capacity available.
            dst_price = self.market_sim.get_pool_price(type_idx, az_idx)
            src_key, src_price = None, -1.0
            for (t, az), p in self.pools.items():
                if p.spot_count > 0 and (t, az) != (type_idx, az_idx):
                    price = self.market_sim.get_pool_price(t, az)
                    if price > src_price:
                        src_price, src_key = price, (t, az)

            if src_key is None or src_price <= 0:
                self.step_wasted_action = True  # no source
            elif dst_price >= src_price * 0.85:
                self.step_wasted_action = True  # savings < 15%
            elif at_capacity:
                self.step_wasted_action = True
            else:
                self.pools[src_key].spot_count -= 1
                pool.spot_count += 1
                self.step_migrations += 1
                self.step_migrate_spot_to_spot = True

        elif op == Operation.RESERVE_CAPACITY:
            # Preemptive OD: allowed only when P(interrupt) high OR SLA at risk.
            # Replaces old OP_REQUEST_OD_EMERGENCY with a cleaner intent-based semantic:
            # "I anticipate spot loss → reserve OD backup NOW".
            pool_interrupt = self.market_sim.get_pool_interrupt_prob(type_idx, az_idx)
            avg_interrupt = float(np.mean([
                self.market_sim.get_pool_interrupt_prob(t, az)
                for (t, az) in self.pools
            ]))
            sla_risk = min(self.pending_jobs / (total_vcpu * 0.5), 1.0)
            spot_at_risk = sum(
                p.spot_count for (t, az), p in self.pools.items()
                if self.market_sim.get_pool_interrupt_prob(t, az) > 0.5
            )

            # Justification: reserve only if (high interrupt pressure) OR (SLA at risk)
            # OR (agent holds spot in high-risk pools)
            justified = (
                avg_interrupt >= 0.35
                or sla_risk >= 0.5
                or spot_at_risk >= 2
                or pool_interrupt >= 0.5
            )
            if not justified:
                self.step_wasted_action = True
                self.step_unnecessary_od = True
            elif at_capacity:
                self.step_wasted_action = True
            else:
                pool.ondemand_count += 1
                self.step_reserved_capacity = True

        elif op == Operation.HOLD:
            pass  # no-op

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
                # Kill a running job — prefer job on this pool
                if self.running_jobs_list:
                    pool_jobs = [i for i, j in enumerate(self.running_jobs_list)
                                 if j.pool == (t_idx, az_idx)]
                    idx = pool_jobs[0] if pool_jobs else int(self.np_random.integers(len(self.running_jobs_list)))
                    killed_job = self.running_jobs_list.pop(idx)
                    if killed_job.pool is not None:
                        self.pool_running_jobs[killed_job.pool] = max(
                            0, self.pool_running_jobs[killed_job.pool] - 1
                        )
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
                if job.pool is not None:
                    self.pool_running_jobs[job.pool] = max(
                        0, self.pool_running_jobs[job.pool] - 1
                    )
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

            # Assign each job to pool with most free capacity
            pool_free = {}
            for (t, az), pool in self.pools.items():
                pool_vcpu = (pool.spot_count + pool.ondemand_count) * INSTANCE_TYPES[t].vcpus
                pool_free[(t, az)] = max(0, pool_vcpu - self.pool_running_jobs[(t, az)])

            for job in started:
                # Pick pool with most free slots
                best_pool = max(pool_free, key=lambda k: pool_free[k]) if pool_free else None
                if best_pool and pool_free[best_pool] > 0:
                    job.pool = best_pool
                    self.pool_running_jobs[best_pool] += 1
                    pool_free[best_pool] -= 1

            self.running_jobs_list.extend(started)
            self.pending_jobs = len(self.workload_gen.pending_jobs)

        self.running_jobs = len(self.running_jobs_list)

        # Recompute per-pool CPU + RAM utilization
        pool_cpu_sum = {k: 0.0 for k in self.pool_running_jobs}
        pool_ram_sum = {k: 0.0 for k in self.pool_running_jobs}
        for job in self.running_jobs_list:
            if job.pool is not None:
                pool_cpu_sum[job.pool] += job.cpu_demand
                pool_ram_sum[job.pool] += job.ram_demand

        for (t, az) in self.pools:
            pool = self.pools[(t, az)]
            pool_vcpu = (pool.spot_count + pool.ondemand_count) * INSTANCE_TYPES[t].vcpus
            pool_ram = (pool.spot_count + pool.ondemand_count) * INSTANCE_TYPES[t].memory_gb
            self.pool_cpu_util[(t, az)] = (
                pool_cpu_sum[(t, az)] / max(pool_vcpu, 1e-6)
            )
            self.pool_ram_util[(t, az)] = (
                pool_ram_sum[(t, az)] / max(pool_ram, 1e-6)
            )

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

        # Track interrupt and pending history for new state features
        self.interrupt_history.append(self.step_interruptions)
        if len(self.interrupt_history) > 10:
            self.interrupt_history.pop(0)
        self.pending_history.append(self.pending_jobs)
        if len(self.pending_history) > 5:
            self.pending_history.pop(0)

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
        Clean reward function — 5 terms chính, không trùng lặp mục đích.

        Mục tiêu: tiết kiệm cost + SLA >= 95%

        R = cost_efficiency     # reward chính: dùng ít tiền hơn baseline OD lý thuyết
            - sla_penalty       # phạt khi job failed
            - interrupt_penalty # phạt khi spot bị interrupt
            - stability_penalty # phạt churn/wasted action/concentration (gộp)
            + smart_bonus       # thưởng migrate_to_spot + spot_to_spot rebalance
        """
        total_instances = self._total_instances()
        total_cap = max(1, self._total_vcpu())

        # ═══ 1. COST EFFICIENCY (reward chính) ═══
        # Baseline = chi phí OD tối thiểu cho workload hiện tại
        # dùng cheapest type (m5.large $0.096/h) × số job cần serve
        cheapest_od = min(t.ondemand_price for t in INSTANCE_TYPES)
        jobs_to_serve = max(self.running_jobs + self.pending_jobs, 1)
        jobs_to_serve = min(jobs_to_serve, MAX_INSTANCES)  # cap
        baseline_cost = jobs_to_serve * cheapest_od

        # Nếu không có job → không cần instance → baseline = 0 → step_cost = penalty thuần
        if self.running_jobs == 0 and self.pending_jobs == 0:
            baseline_cost = 0.0

        cost_efficiency = baseline_cost - self.step_cost  # có thể âm nếu overprov

        # ═══ 2. SLA PENALTY ═══
        sla_penalty = self.step_failed_jobs * self.sla_penalty_coeff

        # Pending penalty (underprovisioning): scale theo số job đang chờ
        if self.pending_jobs > 0 and total_instances == 0:
            pending_penalty = 3.0  # idle khi có job → rất tệ
        elif self.pending_jobs > total_cap * 0.3:
            pending_penalty = 1.0  # over capacity by >30% → thiếu
        else:
            pending_penalty = 0.0

        # ═══ 3. INTERRUPT PENALTY ═══
        interrupt_penalty = self.step_interruptions * self.interrupt_penalty_coeff

        # ═══ 4. STABILITY PENALTY (gộp: wasted + churn + concentration) ═══
        stability_penalty = 0.0

        # Wasted action (request vô ích khi idle, terminate pool rỗng, etc.)
        if self.step_wasted_action:
            if self.idle_streak >= 6:
                stability_penalty += 2.0
            elif self.idle_streak >= 3:
                stability_penalty += 1.0
            else:
                stability_penalty += 0.3

        # Churn (flip-flop request/terminate)
        if self.churn_streak >= 2:
            stability_penalty += (self.churn_streak - 1) * 0.3

        # Concentration (dồn quá nhiều vào 1 AZ)
        if total_instances > 3:  # chỉ phạt khi có >3 instances
            az_counts = np.zeros(N_AZS)
            for (_, az_idx), pool in self.pools.items():
                az_counts[az_idx] += pool.spot_count + pool.ondemand_count
            concentration = float(np.max(az_counts)) / total_instances
            if concentration > self.concentration_threshold:
                stability_penalty += (concentration - self.concentration_threshold) * 2.0

        # Migration cost (trừ khi migrate về spot — đó là hành vi tốt)
        if self.step_migrations > 0 and not self.step_migrate_to_spot \
                and not self.step_migrate_spot_to_spot:
            stability_penalty += self.step_migrations * self.migration_penalty_coeff

        # Unnecessary OD emergency
        if self.step_unnecessary_od:
            stability_penalty += 1.5

        # ═══ 5. SMART BONUS (proactive good behaviors) ═══
        smart_bonus = 0.0
        # a) CONVERT_TO_SPOT khi market ổn (migrate back sau phase OD)
        if self.step_migrate_to_spot and self.steps_since_migrate_to_od >= 2:
            avg_interr = float(np.mean([
                self.market_sim.get_pool_interrupt_prob(t, az)
                for (t, az) in self.pools
            ]))
            if avg_interr < 0.2:
                smart_bonus += 1.0
        # b) REBALANCE_SPOT: thưởng khi move sang pool rẻ hơn
        if self.step_migrate_spot_to_spot:
            smart_bonus += 0.5
        # c) RESERVE_CAPACITY justified: thưởng nhỏ vì reserve đúng lúc (phòng interrupt).
        # Agent học dùng action này proactively thay vì reactive sau khi SLA đã fail.
        if self.step_reserved_capacity:
            smart_bonus += 0.7

        # ═══ TỔNG ═══
        reward = (
            cost_efficiency
            - sla_penalty
            - pending_penalty
            - interrupt_penalty
            - stability_penalty
            + smart_bonus
        )

        # Clip nhẹ để tránh outlier (baseline có thể up to 20 × 0.096 = 1.92)
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

        # [33-41] Extra context features
        # [33] budget_spent_ratio: tổng chi phí so với baseline full OD
        baseline_od_total = self.baseline_od_cost_per_step * max(self.current_step, 1)
        budget_spent_ratio = np.clip(self.total_cost / max(baseline_od_total, 1e-6), 0.0, 2.0) / 2.0
        features.append(float(budget_spent_ratio))

        # [34] idle_spot_vcpu_ratio: spot vCPU đang bị lãng phí
        total_spot_vcpu = sum(
            self.pools[(t, az)].spot_count * INSTANCE_TYPES[t].vcpus
            for t in range(N_TYPES) for az in range(N_AZS)
        )
        idle_spot_vcpu = max(0, total_spot_vcpu - self.running_jobs)
        idle_spot_ratio = idle_spot_vcpu / max(MAX_VCPU, 1)
        features.append(np.clip(idle_spot_ratio, 0.0, 1.0))

        # [35] workload_trend: pending jobs đang tăng hay giảm (so với 5 steps trước)
        if len(self.pending_history) >= 2:
            trend = (self.pending_jobs - self.pending_history[0]) / max(MAX_JOBS, 1)
        else:
            trend = 0.0
        features.append(np.clip((trend + 1.0) / 2.0, 0.0, 1.0))  # shift to [0,1]

        # [36] interrupt_streak_rate: tần suất interrupt trong 10 steps gần nhất
        interrupt_rate = (
            sum(self.interrupt_history[-10:]) / max(len(self.interrupt_history), 1)
        )
        features.append(np.clip(interrupt_rate, 0.0, 1.0))

        # [37] current_pool_price_ratio: giá trung bình của các pool đang chạy / max OD
        running_spot_pools = [
            (t, az) for (t, az), p in self.pools.items() if p.spot_count > 0
        ]
        if running_spot_pools:
            avg_running_price = np.mean([
                self.market_sim.get_pool_price(t, az) for (t, az) in running_spot_pools
            ])
            features.append(np.clip(avg_running_price / max(max_od, 1e-6), 0.0, 1.0))
        else:
            features.append(0.0)

        # [38] price_trend_top1: giá pool rẻ nhất đang tăng hay giảm
        # trend ∈ [-1, 1]: dương = giá đang tăng, âm = đang giảm → shift về [0, 1]
        if pool_data:
            top1 = pool_data[0]
            sim = self.market_sim.sims[(top1['type_idx'], top1['az_idx'])]
            trend_val = sim._compute_trend()  # ∈ [-1, 1]
            features.append(np.clip((trend_val + 1.0) / 2.0, 0.0, 1.0))
        else:
            features.append(0.5)

        # [39] cheaper_spot_available: có pool spot rẻ hơn ≥15% pool đang chạy nhiều nhất?
        most_expensive_running = 0.0
        for (t, az), p in self.pools.items():
            if p.spot_count > 0:
                most_expensive_running = max(
                    most_expensive_running, self.market_sim.get_pool_price(t, az)
                )
        cheaper_available = 0.0
        if most_expensive_running > 0:
            for p_data in pool_data:
                if p_data['price'] < most_expensive_running * 0.85:
                    cheaper_available = 1.0
                    break
        features.append(cheaper_available)

        # [40] od_spot_price_gap: OD đắt hơn spot bao nhiêu % (trung bình)
        if total_spot > 0:
            avg_spot = sum(
                self.market_sim.get_pool_price(t, az) * self.pools[(t, az)].spot_count
                for (t, az) in self.pools if self.pools[(t, az)].spot_count > 0
            ) / total_spot
            avg_od_price = np.mean([t.ondemand_price for t in INSTANCE_TYPES])
            od_gap = np.clip((avg_od_price - avg_spot) / max(avg_od_price, 1e-6), 0.0, 1.0)
        else:
            od_gap = 0.5  # neutral khi không có spot
        features.append(float(od_gap))

        # [41] sla_risk_score: nguy cơ SLA vi phạm sắp tới (nhìn về tương lai gần)
        total_vcpu_cap = max(self._total_vcpu(), 1)
        sla_risk = min(self.pending_jobs / (total_vcpu_cap * 0.5), 1.0)
        features.append(np.clip(sla_risk, 0.0, 1.0))

        # [42-56] Per-pool CPU utilization (15 features: N_TYPES × N_AZS)
        for t in range(N_TYPES):
            for az in range(N_AZS):
                features.append(np.clip(self.pool_cpu_util.get((t, az), 0.0), 0.0, 1.0))

        # [57-71] Per-pool RAM utilization (15 features: N_TYPES × N_AZS)
        for t in range(N_TYPES):
            for az in range(N_AZS):
                features.append(np.clip(self.pool_ram_util.get((t, az), 0.0), 0.0, 1.0))

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

    def get_action_mask(self) -> np.ndarray:
        """Boolean mask [N_ACTIONS] — True = action valid in current state.

        Mask encodes hard constraints (pool empty, at capacity, price invalid).
        Soft preferences (churn, idle) are handled via reward.

        Invalid rules per op:
          PROVISION_SPOT/OD:      pool_full OR capacity_saturated
          RELEASE_SPOT/OD:        corresponding count == 0
          CONVERT_TO_ONDEMAND:    pool has no spot
          CONVERT_TO_SPOT:        pool has no OD, OR spot_price >= od_price
          REBALANCE_SPOT:         pool has no spot (source), OR at_capacity
          RESERVE_CAPACITY:       at_capacity, OR no interrupt/SLA justification
          HOLD:                   always valid
        """
        mask = np.ones(N_ACTIONS, dtype=bool)
        total_instances = self._total_instances()
        total_cap = max(self._total_vcpu(), 1)
        needed = max(self.running_jobs + self.pending_jobs, 0)

        # Capacity guard: block PROVISION when we already have 1.5× headroom
        capacity_saturated = (total_cap >= needed * 1.5 and needed > 0) or \
                             (total_instances >= 3 and needed == 0)

        # RESERVE_CAPACITY global justification
        avg_interrupt = float(np.mean([
            self.market_sim.get_pool_interrupt_prob(t, az)
            for t in range(N_TYPES) for az in range(N_AZS)
        ]))
        sla_risk = min(self.pending_jobs / (total_cap * 0.5), 1.0)
        spot_at_risk = sum(
            p.spot_count for (t, az), p in self.pools.items()
            if self.market_sim.get_pool_interrupt_prob(t, az) > 0.5
        )
        reserve_justified_globally = (
            avg_interrupt >= 0.35 or sla_risk >= 0.5 or spot_at_risk >= 2
        )

        for t in range(N_TYPES):
            od_price = INSTANCE_TYPES[t].ondemand_price
            for az in range(N_AZS):
                pool = self.pools[(t, az)]
                az_count = self._az_instances(az)
                pool_full = (total_instances >= MAX_INSTANCES or az_count >= MAX_PER_AZ)
                spot_price = self.market_sim.get_pool_price(t, az)
                pool_interrupt = self.market_sim.get_pool_interrupt_prob(t, az)

                # PROVISION ops — block khi pool full hoặc capacity đã dư
                if pool_full or capacity_saturated:
                    mask[encode_action(Operation.PROVISION_SPOT, t, az)] = False
                    mask[encode_action(Operation.PROVISION_ONDEMAND, t, az)] = False

                # RELEASE ops — cần count > 0
                if pool.spot_count == 0:
                    mask[encode_action(Operation.RELEASE_SPOT, t, az)] = False
                if pool.ondemand_count == 0:
                    mask[encode_action(Operation.RELEASE_ONDEMAND, t, az)] = False

                # CONVERT_TO_ONDEMAND — cần có spot ở pool
                if pool.spot_count == 0:
                    mask[encode_action(Operation.CONVERT_TO_ONDEMAND, t, az)] = False

                # CONVERT_TO_SPOT — cần có OD + spot rẻ hơn OD
                if pool.ondemand_count == 0 or spot_price >= od_price:
                    mask[encode_action(Operation.CONVERT_TO_SPOT, t, az)] = False

                # REBALANCE_SPOT — target pool phải còn slot
                if pool_full:
                    mask[encode_action(Operation.REBALANCE_SPOT, t, az)] = False

                # RESERVE_CAPACITY — cần justification (global OR pool-local high risk)
                # Khác PROVISION_ONDEMAND: bypass capacity_saturated guard vì đây là
                # proactive backup, không phải scale-up
                reserve_valid = (
                    not pool_full
                    and (reserve_justified_globally or pool_interrupt >= 0.5)
                )
                if not reserve_valid:
                    mask[encode_action(Operation.RESERVE_CAPACITY, t, az)] = False

        # HOLD always valid
        mask[HOLD_ACTION] = True
        return mask

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
