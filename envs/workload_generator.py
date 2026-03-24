"""
Bộ sinh workload cho các batch processing job.

Sinh ra các mẫu job arrival theo phân phối Poisson, có hiệu ứng theo giờ trong ngày.
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

    def __post_init__(self):
        if self.remaining_time == -1:
            self.remaining_time = self.duration


class WorkloadGenerator:
    """
    Sinh workload batch processing với các mẫu thực tế.

    TODO: Cài đặt sinh workload:
    - Quá trình arrival theo Poisson với tốc độ thay đổi theo thời gian (λ)
    - Phân phối thời gian job (vd: log-normal)
    - Giờ cao điểm (λ cao hơn trong giờ làm việc)
    - Mẫu cuối tuần (λ thấp hơn)
    """

    def __init__(
        self,
        base_arrival_rate: float = 2.0,  # Số job trung bình mỗi giờ
        peak_multiplier: float = 3.0,  # Hệ số nhân trong giờ cao điểm
        peak_hours: List[int] = [9, 10, 11, 14, 15, 16],  # Giờ làm việc
        avg_job_duration: int = 10,  # Thời gian job trung bình (số timestep)
        seed: Optional[int] = None,
    ):
        """
        Khởi tạo bộ sinh workload.

        Args:
            base_arrival_rate: Số job trung bình mỗi giờ (baseline)
            peak_multiplier: Hệ số nhân cho giờ cao điểm
            peak_hours: Danh sách các giờ có tải cao (0-23)
            avg_job_duration: Thời gian thực thi trung bình của job
            seed: Seed ngẫu nhiên
        """
        self.base_arrival_rate = base_arrival_rate
        self.peak_multiplier = peak_multiplier
        self.peak_hours = peak_hours
        self.avg_job_duration = avg_job_duration
        self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.job_counter = 0
        self.pending_jobs: List[Job] = []

    def reset(self, seed: Optional[int] = None):
        """Đặt lại bộ sinh về trạng thái ban đầu."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.job_counter = 0
        self.pending_jobs = []

    def step(self) -> List[Job]:
        """
        Sinh các job mới cho timestep hiện tại.

        Returns:
            Danh sách các job mới đến
        """
        # TODO: Cài đặt sinh job
        # 1. Tính tốc độ arrival hiện tại dựa trên giờ trong ngày
        # 2. Lấy mẫu số lượng arrival từ Poisson(λ)
        # 3. Sinh các tham số job (duration, deadline)

        self.current_timestep += 1

        # Tính tốc độ arrival thay đổi theo thời gian
        hour_of_day = self.current_timestep % 24
        if hour_of_day in self.peak_hours:
            current_rate = self.base_arrival_rate * self.peak_multiplier
        else:
            current_rate = self.base_arrival_rate

        # Lấy mẫu số lượng arrival (Poisson)
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
        # TODO: Tùy chỉnh sinh job
        # - Duration: phân phối log-normal
        # - Priority: đồng đều hoặc có trọng số
        # - Deadline: duration + thời gian dự phòng

        duration = max(1, int(self.rng.lognormal(
            mean=np.log(self.avg_job_duration),
            sigma=0.5
        )))

        deadline = self.current_timestep + duration * 3  # 3 lần thời gian dự phòng

        job = Job(
            job_id=self.job_counter,
            arrival_time=self.current_timestep,
            duration=duration,
            priority=1,
            deadline=deadline,
            size=1.0,
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

        Returns:
            Số lượng job kỳ vọng trong `horizon` timestep tiếp theo
        """
        # TODO: Cài đặt dự báo dựa trên mẫu thời gian
        # Phiên bản đơn giản: trả về tốc độ arrival trung bình
        return self.base_arrival_rate * horizon
