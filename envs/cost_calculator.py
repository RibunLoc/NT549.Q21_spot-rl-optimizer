"""
Bộ tính chi phí cho spot và on-demand instance.

Tính toán chi phí, tiết kiệm, và các khoản phạt SLA.
"""

from typing import Dict


class CostCalculator:
    """
    Tính toán chi phí và các khoản phạt cho môi trường.

    TODO: Cài đặt logic tính chi phí:
    - So sánh giá Spot vs On-demand
    - Phạt SLA cho các job thất bại
    - Chi phí migration
    - Chi phí khởi động/dừng instance
    """

    # Bảng giá AWS EC2 (ví dụ cho m5.large tại us-east-1)
    ONDEMAND_PRICE_PER_HOUR = 0.096  # $/giờ
    SPOT_PRICE_REFERENCE = 0.030  # $/giờ (giá spot trung bình)

    # Tham số phạt — scale nhỏ để reward nằm trong khoảng [-5, +5] mỗi step
    # Giúp Q-network học ổn định, tránh loss bùng nổ
    SLA_PENALTY_PER_FAILED_JOB = 2.0  # $ cho mỗi job thất bại (giảm từ 10)
    MIGRATION_COST = 0.3  # $ cho mỗi lần migration spot→OD (giảm từ 1.0)
    MIGRATION_COST_TO_SPOT = 0.15  # $ cho mỗi lần migration OD→spot (giảm từ 0.5)
    INTERRUPTION_PENALTY = 1.0  # $ cho mỗi sự kiện bị gián đoạn (giảm từ 5.0)

    def __init__(
        self,
        ondemand_price: float = ONDEMAND_PRICE_PER_HOUR,
        spot_price_avg: float = SPOT_PRICE_REFERENCE,
        sla_penalty: float = SLA_PENALTY_PER_FAILED_JOB,
        migration_cost: float = MIGRATION_COST,
        migration_cost_to_spot: float = MIGRATION_COST_TO_SPOT,
    ):
        """
        Khởi tạo bộ tính chi phí.

        Args:
            ondemand_price: Giá on-demand ($/giờ)
            spot_price_avg: Giá spot trung bình ($/giờ, để tham chiếu)
            sla_penalty: Tiền phạt cho mỗi job thất bại ($)
            migration_cost: Chi phí migration spot→OD ($)
            migration_cost_to_spot: Chi phí migration OD→spot ($)
        """
        self.ondemand_price = ondemand_price
        self.spot_price_avg = spot_price_avg
        self.sla_penalty = sla_penalty
        self.migration_cost = migration_cost
        self.migration_cost_to_spot = migration_cost_to_spot

    def compute_step_cost(
        self,
        num_spot: int,
        num_ondemand: int,
        spot_price: float,
        timestep_duration: float = 1.0,  # giờ
    ) -> float:
        """
        Tính chi phí hạ tầng cho một timestep.

        Args:
            num_spot: Số spot instance đang chạy
            num_ondemand: Số on-demand instance đang chạy
            spot_price: Giá spot hiện tại ($/giờ)
            timestep_duration: Thời lượng timestep tính bằng giờ

        Returns:
            Tổng chi phí cho timestep này ($)
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
        Tính mức tiết kiệm so với baseline toàn bộ on-demand.

        Args:
            actual_cost: Chi phí thực tế phát sinh ($)
            total_capacity: Tổng số instance (spot + on-demand)
            timestep_duration: Thời lượng tính bằng giờ

        Returns:
            Mức tiết kiệm so với on-demand ($, dương = tiết kiệm được)
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
        Tính tiền phạt SLA nếu mức tuân thủ dưới ngưỡng.

        Args:
            failed_jobs: Số job thất bại
            total_jobs: Tổng số job (hoàn thành + thất bại)
            sla_threshold: Mức tuân thủ SLA tối thiểu (0-1)

        Returns:
            Tiền phạt ($, luôn không âm)
        """
        if total_jobs == 0 or failed_jobs == 0:
            return 0.0

        compliance = 1.0 - (failed_jobs / total_jobs)

        if compliance < sla_threshold:
            # Phạt tuyến tính: mỗi job fail bị phạt, cộng thêm bonus phạt theo mức vi phạm
            # Giữ penalty bounded — tránh spike quá lớn trong 1 step
            violation = sla_threshold - compliance  # 0 → 0.95
            penalty = failed_jobs * self.sla_penalty * (1.0 + violation * 2.0)
            return min(penalty, 10.0)  # Cap tối đa $10/step

        return 0.0

    def compute_migration_penalty(self, num_migrations: int) -> float:
        """
        Tính chi phí migration job (downtime, overhead).

        Args:
            num_migrations: Số sự kiện migration trong timestep này

        Returns:
            Chi phí migration ($)
        """
        return num_migrations * self.migration_cost

    def compute_interruption_penalty(self, num_interruptions: int) -> float:
        """
        Tính tiền phạt cho các lần spot bị gián đoạn.

        Args:
            num_interruptions: Số sự kiện bị gián đoạn

        Returns:
            Tiền phạt gián đoạn ($)
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
        Tính tổng reward cho timestep.

        Reward = savings - penalties

        Args:
            step_cost: Chi phí hạ tầng ($)
            savings: Mức tiết kiệm so với baseline ($)
            sla_penalty: Tiền phạt vi phạm SLA ($)
            migration_penalty: Chi phí migration ($)
            interruption_penalty: Tiền phạt gián đoạn ($)

        Returns:
            Tổng reward (có thể âm)
        """
        # savings = baseline - actual_cost, nên nếu trừ thêm step_cost
        # sẽ bị phạt chi phí hạ tầng 2 lần.
        reward = (
            savings
            - sla_penalty
            - migration_penalty
            - interruption_penalty
        )

        return reward

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Trả về các tham số chi phí để ghi log."""
        return {
            "ondemand_price": self.ondemand_price,
            "spot_price_avg": self.spot_price_avg,
            "sla_penalty": self.sla_penalty,
            "migration_cost": self.migration_cost,
            "migration_cost_to_spot": self.migration_cost_to_spot,
        }
