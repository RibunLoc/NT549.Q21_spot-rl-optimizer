"""
Tạo các mẫu khối lượng công việc tổng hợp để xử lý hàng loạt

Tạo dấu vết việc làm thực tế với:
- Mẫu thời gian trong ngày (tải cao hơn trong giờ hành chính)
- Mẫu ngày trong tuần (tải thấp hơn trong cuối tuần)
- Thỉnh thoảng có tăng đột biến
"""

import pandas as pd # tạo bảng dữ liệu
import numpy as np # sinh số ngẫu nhiên
from datetime import datetime, timedelta # tạo ra các timeline theo giờ
import argparse # chạy script bằng CLI
from pathlib import Path # lưu file
import logging # in log cho dễ debug 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo ra một workload từng giờ trong duration_days
def generate_workload(
    duration_days: int = 30,
    base_rate: float = 2.0,
    peak_multiplier: float = 3.0,
    weekend_multiplier: float = 0.5,
    spike_prob: float = 0.05,
    spike_multiplier: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo dấu vết khối lượng công việc tổng hợp.

    Args:
        duration_days: Số ngày để mô phỏng
        base_rate: Số job/giờ trung bình
        peak_multiplier: Hệ số nhân tải cho giờ cao điểm
        weekend_multiplier: Hệ số giảm tải cho giờ cuối tuần
        spike_prob: Xác suất có spike (đột biến)
        spike_multiplier: độ mạnh của spike
        seed: Để kết quả reproducible (tái sử dụng được)

    Returns:
        DataFrame with columns: [timestamp, num_jobs, avg_duration]
    """
    logger.info(f"Generating {duration_days} days of workload...")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate hourly timestamps
    """
        timestamps[0] = 2026-01-01 00:00:00
        timestamps[1] = 2026-01-01 01:00:00
        ...
    """
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(duration_days * 24)]

    workload_data = []

    for timestamp in timestamps:
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday

        # Base rate
        current_rate = base_rate

        # Time-of-day effect (higher during 9am-5pm) giờ hành chính 
        if 9 <= hour_of_day <= 17:
            current_rate *= peak_multiplier

        # Day-of-week effect (lower on weekends) giờ cuối tuần
        if day_of_week >= 5:  # Saturday, Sunday
            current_rate *= weekend_multiplier

        # Random spikes 
        if rng.random() < spike_prob:
            current_rate *= spike_multiplier

        # Sample number of jobs (Poisson distribution) - sinh số jobs
        num_jobs = rng.poisson(current_rate)

        # Sample average job duration (log-normal) - sinh thời gian chạy job
        avg_duration = max(1, int(rng.lognormal(mean=np.log(10), sigma=0.5)))

        workload_data.append({
            'timestamp': timestamp,
            'num_jobs': num_jobs,
            'avg_duration': avg_duration,
        })

    df = pd.DataFrame(workload_data)

    logger.info(f"Generated {len(df)} hourly workload records")
    logger.info(f"Total jobs: {df['num_jobs'].sum()}")
    logger.info(f"Peak hour: {df['num_jobs'].max()} jobs")
    logger.info(f"Average: {df['num_jobs'].mean():.2f} jobs/hour")

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic workload trace')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration in days (default: 30)')
    parser.add_argument('--pattern', type=str, default='batch_ml_training',
                        choices=['batch_ml_training', 'web_service', 'mixed'],
                        help='Workload pattern type')
    parser.add_argument('--output', type=str, default='data/raw/workload_traces.csv',
                        help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Pattern-specific parameters
    if args.pattern == 'batch_ml_training':
        # ML training: moderate base rate, high peak, low weekend
        params = {
            'base_rate': 2.0,
            'peak_multiplier': 3.0,
            'weekend_multiplier': 0.3,
            'spike_prob': 0.05,
            'spike_multiplier': 5.0,
        }
    elif args.pattern == 'web_service':
        # Web service: high base rate, moderate peak, no weekend reduction
        params = {
            'base_rate': 5.0,
            'peak_multiplier': 1.5,
            'weekend_multiplier': 1.0,
            'spike_prob': 0.1,
            'spike_multiplier': 3.0,
        }
    else:  # mixed
        params = {
            'base_rate': 3.0,
            'peak_multiplier': 2.0,
            'weekend_multiplier': 0.6,
            'spike_prob': 0.08,
            'spike_multiplier': 4.0,
        }

    # Generate workload
    df = generate_workload(
        duration_days=args.duration,
        seed=args.seed,
        **params,
    )

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved workload trace to {output_path}")


if __name__ == '__main__':
    main()


# Example usage:
# python generate_workload.py --duration 30 --pattern batch_ml_training --output data/raw/workload.csv
