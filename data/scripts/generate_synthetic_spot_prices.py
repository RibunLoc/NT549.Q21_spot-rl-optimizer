"""
Tạo dữ liệu spot price mô phỏng dựa trên pattern thực tế của AWS.

Mô phỏng:
- Giá spot dao động theo giờ (higher during peak hours 9am-5pm)
- Biến động theo ngày trong tuần (thấp hơn cuối tuần)
- Random price spikes (mô phỏng supply/demand shocks)
- Different patterns per instance type
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_spot_prices(
    instance_type: str,
    region: str,
    availability_zones: list,
    duration_days: int = 30,
    base_price: float = 0.03,  # Base price ($/hour)
    volatility: float = 0.15,  # Price volatility (std dev ratio)
    peak_hour_multiplier: float = 1.3,  # Price multiplier for peak hours
    spike_prob: float = 0.02,  # Probability of price spike per hour
    spike_multiplier: float = 3.0,  # Max spike multiplier
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo dữ liệu spot price mô phỏng cho một instance type.

    Args:
        instance_type: Instance type (e.g., 'm5.large')
        region: AWS region
        availability_zones: List of AZs (e.g., ['us-east-1a', 'us-east-1b'])
        duration_days: Số ngày để mô phỏng
        base_price: Giá cơ bản ($/giờ)
        volatility: Độ biến động giá (tỷ lệ std dev)
        peak_hour_multiplier: Hệ số nhân giá giờ cao điểm
        spike_prob: Xác suất có price spike
        spike_multiplier: Độ mạnh của spike
        seed: Random seed cho reproducibility

    Returns:
        DataFrame với columns: [timestamp, instance_type, availability_zone, spot_price, product_description]
    """
    logger.info(f"Generating spot prices for {instance_type} in {region}...")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate timestamps (mỗi 5 phút - giống AWS spot price history resolution)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    num_points = duration_days * 24 * 12  # 12 points per hour (5-minute intervals)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(num_points)]

    all_prices = []

    for az in availability_zones:
        # Mỗi AZ có base price hơi khác nhau
        az_base_price = base_price * rng.uniform(0.95, 1.05)

        # Generate price series với Geometric Brownian Motion (GBM)
        # Đây là mô hình phổ biến cho giá tài chính
        prices = [az_base_price]

        for i, timestamp in enumerate(timestamps[1:], 1):
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()

            # Drift: giá có xu hướng tăng ở giờ cao điểm
            drift = 0.0
            if 9 <= hour_of_day <= 17:  # Business hours
                drift = 0.0001 * peak_hour_multiplier
            if day_of_week >= 5:  # Weekend - giá thấp hơn
                drift -= 0.0001

            # Random walk với volatility
            shock = rng.normal(0, volatility / np.sqrt(24 * 12))  # Scaled by time interval

            # Price spike events
            if rng.random() < spike_prob:
                spike_factor = rng.uniform(1.0, spike_multiplier)
                shock += np.log(spike_factor)

            # Geometric Brownian Motion: S_{t+1} = S_t * exp(drift + shock)
            new_price = prices[-1] * np.exp(drift + shock)

            # Clamp price to reasonable bounds (0.5x - 5x base price)
            new_price = np.clip(new_price, az_base_price * 0.5, az_base_price * 5.0)

            prices.append(new_price)

        # Create DataFrame for this AZ
        for timestamp, price in zip(timestamps, prices):
            all_prices.append({
                'timestamp': timestamp,
                'instance_type': instance_type,
                'availability_zone': az,
                'spot_price': round(price, 4),
                'product_description': 'Linux/UNIX',
            })

    df = pd.DataFrame(all_prices)

    logger.info(f"  Generated {len(df)} price records")
    logger.info(f"  Mean price: ${df['spot_price'].mean():.4f}/hour")
    logger.info(f"  Min price: ${df['spot_price'].min():.4f}/hour")
    logger.info(f"  Max price: ${df['spot_price'].max():.4f}/hour")
    logger.info(f"  Std dev: ${df['spot_price'].std():.4f}/hour")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic AWS spot price data'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region (default: us-east-1)',
    )
    parser.add_argument(
        '--instance-types',
        type=str,
        default='m5.large,c5.large',
        help='Comma-separated instance types (default: m5.large,c5.large)',
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to generate (default: 30)',
    )
    parser.add_argument(
        '--volatility',
        type=float,
        default=None,
        help='Override volatility for all instance types (default: None)',
    )
    parser.add_argument(
        '--spike-prob',
        type=float,
        default=None,
        help='Override spike probability per hour (default: None)',
    )
    parser.add_argument(
        '--spike-multiplier',
        type=float,
        default=None,
        help='Override spike multiplier (default: None)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/spot_prices/',
        help='Output directory (default: data/raw/spot_prices/)',
    )
    parser.add_argument(
        '--tag',
        type=str,
        default='synthetic',
        help='Tag suffix for output filename (default: synthetic)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )

    args = parser.parse_args()

    # Parse instance types
    instance_types = [t.strip() for t in args.instance_types.split(',')]

    # Define AZs based on region
    if args.region == 'us-east-1':
        azs = ['us-east-1a', 'us-east-1b', 'us-east-1c']
    elif args.region == 'us-west-2':
        azs = ['us-west-2a', 'us-west-2b', 'us-west-2c']
    else:
        azs = [f"{args.region}a", f"{args.region}b", f"{args.region}c"]

    # Instance type specific parameters
    instance_params = {
        'm5.large': {'base_price': 0.030, 'volatility': 0.15},
        'm5.xlarge': {'base_price': 0.060, 'volatility': 0.18},
        'c5.large': {'base_price': 0.035, 'volatility': 0.12},
        'c5.xlarge': {'base_price': 0.070, 'volatility': 0.14},
        'r5.large': {'base_price': 0.040, 'volatility': 0.16},
    }

    all_data = []

    for instance_type in instance_types:
        # Get params or use defaults
        params = instance_params.get(
            instance_type, {'base_price': 0.03, 'volatility': 0.15}
        )
        if args.volatility is not None:
            params = {**params, 'volatility': args.volatility}

        # Generate data
        df = generate_spot_prices(
            instance_type=instance_type,
            region=args.region,
            availability_zones=azs,
            duration_days=args.days,
            seed=args.seed + hash(instance_type) % 1000,  # Different seed per instance
            spike_prob=args.spike_prob if args.spike_prob is not None else 0.02,
            spike_multiplier=args.spike_multiplier if args.spike_multiplier is not None else 3.0,
            **params,
        )

        all_data.append(df)

    # Combine all instance types
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')

    # Save to CSV
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"spot_prices_{args.region}_{args.tag}.csv"
    combined_df.to_csv(output_file, index=False)

    logger.info(f"\n✅ Saved {len(combined_df)} total records to {output_file}")

    # Print summary per instance type
    logger.info("\n📊 Summary by instance type:")
    for instance_type in instance_types:
        subset = combined_df[combined_df['instance_type'] == instance_type]
        logger.info(f"  {instance_type}:")
        logger.info(f"    Records: {len(subset)}")
        logger.info(f"    Mean: ${subset['spot_price'].mean():.4f}/hour")
        logger.info(f"    Min: ${subset['spot_price'].min():.4f}/hour")
        logger.info(f"    Max: ${subset['spot_price'].max():.4f}/hour")


if __name__ == '__main__':
    main()


# Example usage:
# python generate_synthetic_spot_prices.py --region us-east-1 --instance-types m5.large --days 30 --tag stable
