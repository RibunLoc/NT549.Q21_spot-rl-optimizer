"""
Workload generator for batch processing jobs.

Generates job arrival patterns (Poisson process, time-of-day effects).
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Job:
    """Represents a batch processing job."""
    job_id: int
    arrival_time: int  # Timestep when job arrived
    duration: int  # Expected execution time (timesteps)
    priority: int = 1  # Priority (higher = more important)
    deadline: Optional[int] = None  # Deadline (timestep)
    size: float = 1.0  # Resource requirement (e.g., vCPUs)


class WorkloadGenerator:
    """
    Generates batch processing workload with realistic patterns.

    TODO: Implement workload generation:
    - Poisson arrival process with time-varying rate (λ)
    - Job duration distribution (e.g., log-normal)
    - Peak hours (higher λ during business hours)
    - Weekend patterns (lower λ)
    """

    def __init__(
        self,
        base_arrival_rate: float = 2.0,  # Average jobs per hour
        peak_multiplier: float = 3.0,  # Multiplier during peak hours
        peak_hours: List[int] = [9, 10, 11, 14, 15, 16],  # Business hours
        avg_job_duration: int = 10,  # Average job duration (timesteps)
        seed: Optional[int] = None,
    ):
        """
        Initialize workload generator.

        Args:
            base_arrival_rate: Average jobs per hour (baseline)
            peak_multiplier: Multiplier for peak hours
            peak_hours: List of hours with higher load (0-23)
            avg_job_duration: Average job execution time
            seed: Random seed
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
        """Reset generator to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.job_counter = 0
        self.pending_jobs = []

    def step(self) -> List[Job]:
        """
        Generate new jobs for current timestep.

        Returns:
            List of newly arrived jobs
        """
        # TODO: Implement job generation
        # 1. Compute current arrival rate based on time of day
        # 2. Sample number of arrivals from Poisson(λ)
        # 3. Generate job parameters (duration, deadline)

        self.current_timestep += 1

        # Compute time-varying arrival rate
        hour_of_day = self.current_timestep % 24
        if hour_of_day in self.peak_hours:
            current_rate = self.base_arrival_rate * self.peak_multiplier
        else:
            current_rate = self.base_arrival_rate

        # Sample number of arrivals (Poisson)
        num_arrivals = self.rng.poisson(current_rate)

        # Generate jobs
        new_jobs = []
        for _ in range(num_arrivals):
            job = self._generate_job()
            new_jobs.append(job)
            self.pending_jobs.append(job)

        return new_jobs

    def _generate_job(self) -> Job:
        """Generate a single job with random parameters."""
        # TODO: Customize job generation
        # - Duration: log-normal distribution
        # - Priority: uniform or weighted
        # - Deadline: duration + slack time

        duration = max(1, int(self.rng.lognormal(
            mean=np.log(self.avg_job_duration),
            sigma=0.5
        )))

        deadline = self.current_timestep + duration * 3  # 3x slack

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
        """Get list of pending (not yet scheduled) jobs."""
        return self.pending_jobs

    def remove_job(self, job_id: int):
        """Remove job from pending queue (after scheduling)."""
        self.pending_jobs = [j for j in self.pending_jobs if j.job_id != job_id]

    def get_workload_forecast(self, horizon: int = 10) -> float:
        """
        Forecast expected workload for next `horizon` timesteps.

        Returns:
            Expected number of jobs in the next `horizon` timesteps
        """
        # TODO: Implement forecast based on time patterns
        # Simple version: return average arrival rate
        return self.base_arrival_rate * horizon
