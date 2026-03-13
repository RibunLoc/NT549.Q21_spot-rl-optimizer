"""
Cost calculator for spot and on-demand instances.

Computes costs, savings, and SLA penalties.
"""

from typing import Dict


class CostCalculator:
    """
    Calculates costs and penalties for the environment.

    TODO: Implement cost calculation logic:
    - Spot vs on-demand pricing
    - SLA penalties for failed jobs
    - Migration costs
    - Startup/teardown costs
    """

    # AWS EC2 pricing (example for m5.large in us-east-1)
    ONDEMAND_PRICE_PER_HOUR = 0.096  # $/hour
    SPOT_PRICE_REFERENCE = 0.030  # $/hour (average spot price)

    # Penalty parameters
    SLA_PENALTY_PER_FAILED_JOB = 10.0  # $ per failed job
    MIGRATION_COST = 1.0  # $ per migration (downtime cost)
    INTERRUPTION_PENALTY = 5.0  # $ per interruption event

    def __init__(
        self,
        ondemand_price: float = ONDEMAND_PRICE_PER_HOUR,
        spot_price_avg: float = SPOT_PRICE_REFERENCE,
        sla_penalty: float = SLA_PENALTY_PER_FAILED_JOB,
        migration_cost: float = MIGRATION_COST,
    ):
        """
        Initialize cost calculator.

        Args:
            ondemand_price: On-demand price ($/hour)
            spot_price_avg: Average spot price ($/hour, for reference)
            sla_penalty: Penalty per failed job ($)
            migration_cost: Cost per migration event ($)
        """
        self.ondemand_price = ondemand_price
        self.spot_price_avg = spot_price_avg
        self.sla_penalty = sla_penalty
        self.migration_cost = migration_cost

    def compute_step_cost(
        self,
        num_spot: int,
        num_ondemand: int,
        spot_price: float,
        timestep_duration: float = 1.0,  # hours
    ) -> float:
        """
        Compute infrastructure cost for one timestep.

        Args:
            num_spot: Number of spot instances running
            num_ondemand: Number of on-demand instances running
            spot_price: Current spot price ($/hour)
            timestep_duration: Duration of timestep in hours

        Returns:
            Total cost for this timestep ($)
        """
        spot_cost = num_spot * spot_price * timestep_duration
        ondemand_cost = num_ondemand * self.ondemand_price * timestep_duration

        return spot_cost + ondemand_cost

    def compute_savings_vs_ondemand(
        self,
        actual_cost: float,
        total_capacity: int,
        timestep_duration: float = 1.0,
    ) -> float:
        """
        Compute cost savings compared to all on-demand baseline.

        Args:
            actual_cost: Actual cost incurred ($)
            total_capacity: Total number of instances (spot + on-demand)
            timestep_duration: Duration in hours

        Returns:
            Savings compared to on-demand ($, positive means saved)
        """
        ondemand_baseline_cost = (
            total_capacity * self.ondemand_price * timestep_duration
        )

        return ondemand_baseline_cost - actual_cost

    def compute_sla_penalty(
        self,
        failed_jobs: int,
        total_jobs: int,
        sla_threshold: float = 0.95,
    ) -> float:
        """
        Compute SLA penalty if compliance is below threshold.

        Args:
            failed_jobs: Number of failed jobs
            total_jobs: Total number of jobs (completed + failed)
            sla_threshold: Minimum SLA compliance (0-1)

        Returns:
            Penalty ($, always non-negative)
        """
        if total_jobs == 0:
            return 0.0

        compliance = 1.0 - (failed_jobs / total_jobs)

        if compliance < sla_threshold:
            # Penalty proportional to violation magnitude
            violation = sla_threshold - compliance
            penalty = failed_jobs * self.sla_penalty * (1 + violation)
            return penalty

        return 0.0

    def compute_migration_penalty(self, num_migrations: int) -> float:
        """
        Compute cost of migrating jobs (downtime, overhead).

        Args:
            num_migrations: Number of migration events in this timestep

        Returns:
            Migration cost ($)
        """
        return num_migrations * self.migration_cost

    def compute_interruption_penalty(self, num_interruptions: int) -> float:
        """
        Compute penalty for spot interruptions.

        Args:
            num_interruptions: Number of interruption events

        Returns:
            Interruption penalty ($)
        """
        return num_interruptions * self.INTERRUPTION_PENALTY

    def compute_total_reward(
        self,
        step_cost: float,
        savings: float,
        sla_penalty: float,
        migration_penalty: float,
        interruption_penalty: float,
    ) -> float:
        """
        Compute total reward for the timestep.

        Reward = savings - cost - penalties

        Args:
            step_cost: Infrastructure cost ($)
            savings: Cost savings vs baseline ($)
            sla_penalty: SLA violation penalty ($)
            migration_penalty: Migration cost ($)
            interruption_penalty: Interruption penalty ($)

        Returns:
            Total reward (can be negative)
        """
        reward = (
            savings
            - step_cost
            - sla_penalty
            - migration_penalty
            - interruption_penalty
        )

        return reward

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Return cost parameters for logging."""
        return {
            "ondemand_price": self.ondemand_price,
            "spot_price_avg": self.spot_price_avg,
            "sla_penalty": self.sla_penalty,
            "migration_cost": self.migration_cost,
        }
