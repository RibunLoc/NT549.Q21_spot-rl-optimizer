"""
Bộ sinh workload cho các batch processing job.

Sinh ra các mẫu job arrival theo phân phối Poisson, có hiệu ứng theo giờ trong ngày.

Cải tiến so với bản cũ:
- Weekend effect: workload giảm cuối tuần
- Workload spikes: đột biến ngẫu nhiên
- Smoother peak transitions: dùng sine profile thay vì step function
- Forecast thông minh: dự báo dựa trên time-of-day pattern thực tế
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Job:
    """Đại diện cho một batch processing job."""
    job_id: int
    arrival_time: int  # Timestep khi job đến
    duration: int  # Thời gian thực thi dự kiến ban đầu (số timestep)
    priority: int = 1  # Độ ưu tiên (cao hơn = quan trọng hơn)
    deadline: Optional[int] = None  # Hạn chót (timestep)
    size: float = 1.0  # Yêu cầu tài nguyên (vd: số vCPU)
    remaining_time: int = -1  # Thời gian còn lại (-1 = chưa bắt đầu)
    pool: Optional[tuple] = None  # (type_idx, az_idx) pool đang chạy job này
    cpu_demand: float = 0.2  # 0-1: mức CPU job cần (0.1=nhẹ, 0.9=nặng)
    ram_demand: float = 1.0  # GB: RAM job cần

    def __post_init__(self):
        if self.remaining_time == -1:
            self.remaining_time = self.duration


# Hourly arrival rate profile (normalized multiplier, index = hour 0-23)
# Shape: thấp ban đêm, ramp up sáng, peak chiều, giảm tối
HOURLY_PROFILE = np.array([
    0.3, 0.2, 0.2, 0.2, 0.3, 0.5,   # 00-05: đêm khuya, rất ít
    0.7, 0.9, 1.2, 1.5, 1.6, 1.4,   # 06-11: sáng, tăng dần
    1.2, 1.5, 1.8, 2.0, 1.8, 1.5,   # 12-17: chiều, peak
    1.2, 1.0, 0.8, 0.6, 0.5, 0.4,   # 18-23: tối, giảm dần
])


class WorkloadGenerator:
    """
    Sinh workload batch processing với các mẫu thực tế.

    Features:
    - Arrival theo Poisson với rate thay đổi theo giờ (smooth profile)
    - Weekend discount (workload giảm ~40% cuối tuần)
    - Random spikes (đột biến workload bất ngờ)
    - Forecast dựa trên time-of-day pattern
    """

    def __init__(
        self,
        base_arrival_rate: float = 2.0,
        peak_multiplier: float = 3.0,
        peak_hours: List[int] = None,  # Kept for backward compat but uses HOURLY_PROFILE
        avg_job_duration: int = 10,
        weekend_factor: float = 0.6,
        spike_probability: float = 0.02,
        spike_multiplier: float = 3.0,
        seed: Optional[int] = None,
    ):
        """
        Khởi tạo bộ sinh workload.

        Args:
            base_arrival_rate: Số job trung bình mỗi giờ (baseline, nhân với hourly profile)
            peak_multiplier: Hệ số scale cho toàn bộ hourly profile
            peak_hours: (Legacy, ignored) — dùng HOURLY_PROFILE thay thế
            avg_job_duration: Thời gian thực thi trung bình của job
            weekend_factor: Hệ số nhân cuối tuần (0.6 = giảm 40%)
            spike_probability: Xác suất xảy ra workload spike mỗi step
            spike_multiplier: Hệ số nhân khi có spike
            seed: Seed ngẫu nhiên
        """
        self.base_arrival_rate = base_arrival_rate
        self.peak_multiplier = peak_multiplier
        self.avg_job_duration = avg_job_duration
        self.weekend_factor = weekend_factor
        self.spike_probability = spike_probability
        self.spike_multiplier = spike_multiplier
        self.rng = np.random.default_rng(seed)

        # Job resource profile — mix of light/medium/heavy jobs
        # Mô phỏng thực tế: 60% light, 30% medium, 10% heavy
        self._job_profiles = [
            # (prob, cpu_low, cpu_high, ram_low_gb, ram_high_gb)
            (0.60, 0.05, 0.25, 0.5,  2.0),   # light: ETL, log processing
            (0.30, 0.30, 0.65, 2.0,  8.0),   # medium: ML inference, data transform
            (0.10, 0.70, 0.95, 8.0, 16.0),   # heavy: model training, large joins
        ]

        # Normalize hourly profile so mean = 1.0, then scale by peak_multiplier
        self.hourly_profile = HOURLY_PROFILE / HOURLY_PROFILE.mean()

        self.current_timestep = 0
        self.job_counter = 0
        self.pending_jobs: List[Job] = []

        # Spike state: còn bao nhiêu steps trong spike hiện tại
        self._spike_remaining = 0

    def reset(self, seed: Optional[int] = None):
        """Đặt lại bộ sinh về trạng thái ban đầu."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.job_counter = 0
        self.pending_jobs = []
        self._spike_remaining = 0

    def _get_current_rate(self) -> float:
        """
        Tính arrival rate hiện tại dựa trên giờ, ngày, và spike.

        Returns:
            Arrival rate (jobs/step)
        """
        hour_of_day = self.current_timestep % 24
        day_of_week = (self.current_timestep // 24) % 7  # 0=Mon ... 6=Sun

        # Base rate × hourly profile (smooth transitions)
        rate = self.base_arrival_rate * self.hourly_profile[hour_of_day]

        # Weekend discount (Sat=5, Sun=6)
        if day_of_week >= 5:
            rate *= self.weekend_factor

        # Spike handling
        if self._spike_remaining > 0:
            rate *= self.spike_multiplier
            self._spike_remaining -= 1
        elif self.rng.random() < self.spike_probability:
            # Start new spike: lasts 3-12 steps
            self._spike_remaining = self.rng.integers(3, 13)
            rate *= self.spike_multiplier

        return rate

    def step(self) -> List[Job]:
        """
        Sinh các job mới cho timestep hiện tại.

        Returns:
            Danh sách các job mới đến
        """
        self.current_timestep += 1

        # Tính rate và sample arrivals
        current_rate = self._get_current_rate()
        num_arrivals = self.rng.poisson(current_rate)

        # Sinh các job
        new_jobs = []
        for _ in range(num_arrivals):
            job = self._generate_job()
            new_jobs.append(job)
            self.pending_jobs.append(job)

        return new_jobs

    def _generate_job(self) -> Job:
        """Sinh một job đơn lẻ với các tham số ngẫu nhiên."""
        duration = max(1, int(self.rng.lognormal(
            mean=np.log(self.avg_job_duration),
            sigma=0.5
        )))

        deadline = self.current_timestep + duration * 3  # 3x buffer

        # Sample job profile (light/medium/heavy)
        probs = [p[0] for p in self._job_profiles]
        profile_idx = self.rng.choice(len(self._job_profiles), p=probs)
        _, cpu_low, cpu_high, ram_low, ram_high = self._job_profiles[profile_idx]

        cpu_demand = float(self.rng.uniform(cpu_low, cpu_high))
        ram_demand = float(self.rng.uniform(ram_low, ram_high))

        job = Job(
            job_id=self.job_counter,
            arrival_time=self.current_timestep,
            duration=duration,
            priority=1,
            deadline=deadline,
            size=1.0,
            cpu_demand=cpu_demand,
            ram_demand=ram_demand,
        )

        self.job_counter += 1
        return job

    def get_pending_jobs(self) -> List[Job]:
        """Lấy danh sách các job đang chờ (chưa được lên lịch)."""
        return self.pending_jobs

    def remove_job(self, job_id: int):
        """Xóa job khỏi hàng đợi chờ (sau khi đã lên lịch)."""
        self.pending_jobs = [j for j in self.pending_jobs if j.job_id != job_id]

    def get_workload_forecast(self, horizon: int = 10) -> float:
        """
        Dự báo workload kỳ vọng cho `horizon` timestep tiếp theo.

        Dự báo dựa trên hourly profile và day-of-week, cho agent biết
        sắp tới workload sẽ tăng hay giảm.

        Returns:
            Số lượng job kỳ vọng trong `horizon` timestep tiếp theo
        """
        total_expected = 0.0
        for dt in range(1, horizon + 1):
            future_step = self.current_timestep + dt
            hour = future_step % 24
            day = (future_step // 24) % 7

            rate = self.base_arrival_rate * self.hourly_profile[hour]
            if day >= 5:
                rate *= self.weekend_factor

            total_expected += rate

        return total_expected
