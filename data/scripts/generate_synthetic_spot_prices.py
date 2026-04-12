"""
Tao du lieu spot price mo phong dua tren pattern thuc te cua AWS.

Mo hinh: Ornstein-Uhlenbeck (mean-reverting) + daily seasonality + spikes.

Dac diem:
- Gia quay ve muc base (mean-reversion) — giong AWS thuc te
- Pattern gia theo gio: re dem (0-7h), dat ngay (9-17h), giam chieu (18-23h)
- Pattern theo tuan: cuoi tuan re hon 15-20%
- Price spikes xay ra roi TU GIAM ve base sau vai gio
- Moi AZ co gia hoi khac nhau (correlated nhung khong giong het)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Daily price profile (24h) — dang U-shape nhu AWS thuc te
#  Index = gio trong ngay, value = he so nhan voi base price
# ──────────────────────────────────────────────
HOURLY_PROFILE = np.array([
    # 0h    1h    2h    3h    4h    5h    6h    7h
    0.90, 0.88, 0.86, 0.85, 0.85, 0.87, 0.90, 0.95,
    # 8h    9h   10h   11h   12h   13h   14h   15h
    1.00, 1.08, 1.12, 1.15, 1.13, 1.10, 1.12, 1.15,
    # 16h   17h   18h   19h   20h   21h   22h   23h
    1.12, 1.08, 1.02, 0.98, 0.95, 0.93, 0.92, 0.91,
])


def generate_spot_prices(
    instance_type: str,
    region: str,
    availability_zones: list,
    duration_days: int = 90,
    base_price: float = 0.03,
    volatility: float = 0.10,
    mean_reversion_speed: float = 0.15,
    peak_hour_strength: float = 1.0,
    weekend_discount: float = 0.15,
    spike_prob_per_hour: float = 0.01,
    spike_multiplier: float = 2.0,
    spike_decay_hours: float = 3.0,
    az_correlation: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tao du lieu spot price mo phong cho mot instance type.

    Mo hinh: Ornstein-Uhlenbeck process voi seasonal component.
        dX = theta * (mu(t) - X) * dt + sigma * dW + spike_jump

    Args:
        instance_type: Instance type (e.g., 'm5.large')
        region: AWS region
        availability_zones: List of AZs
        duration_days: So ngay mo phong
        base_price: Gia co ban ($/gio)
        volatility: Do bien dong gia (annual std dev ratio)
        mean_reversion_speed: Toc do quay ve gia trung binh (theta, 0-1)
            0.05 = cham, 0.3 = nhanh
        peak_hour_strength: Do manh cua pattern gio cao diem (0=khong co, 1=binh thuong, 2=manh)
        weekend_discount: Giam gia cuoi tuan (0.15 = giam 15%)
        spike_prob_per_hour: Xac suat spike moi gio
        spike_multiplier: Gia tang bao nhieu khi spike (2.0 = x2)
        spike_decay_hours: Spike giam ve base sau bao nhieu gio
        az_correlation: Tuong quan gia giua cac AZ (0-1)
        seed: Random seed

    Returns:
        DataFrame [timestamp, instance_type, availability_zone, spot_price, product_description]
    """
    logger.info(f"Generating {instance_type} in {region} "
                f"(vol={volatility:.0%}, spike={spike_prob_per_hour:.1%}/hr, "
                f"mr_speed={mean_reversion_speed})")

    rng = np.random.default_rng(seed)

    # Timestamps: moi 5 phut (12 points/gio)
    dt = 1.0 / (24 * 12)  # 5 phut tinh theo ngay
    num_points = duration_days * 24 * 12
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(num_points)]

    # Tinh seasonal target price cho moi timestamp
    seasonal_targets = np.zeros(num_points)
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        minute = ts.minute
        day_of_week = ts.weekday()

        # Noi suy hourly profile (smooth giua cac gio)
        hour_frac = hour + minute / 60.0
        h0 = int(hour_frac) % 24
        h1 = (h0 + 1) % 24
        frac = hour_frac - int(hour_frac)
        hourly_factor = HOURLY_PROFILE[h0] * (1 - frac) + HOURLY_PROFILE[h1] * frac

        # Dieu chinh do manh cua peak (blend voi 1.0)
        hourly_factor = 1.0 + (hourly_factor - 1.0) * peak_hour_strength

        # Giam gia cuoi tuan
        if day_of_week >= 5:
            hourly_factor *= (1.0 - weekend_discount)

        seasonal_targets[i] = base_price * hourly_factor

    # Generate spike events (shared across AZs, voi noise rieng)
    # Spike la event cap gio, anh huong nhieu 5-min intervals
    num_hours = duration_days * 24
    spike_events = rng.random(num_hours) < spike_prob_per_hour
    spike_magnitudes = np.zeros(num_hours)
    for h in range(num_hours):
        if spike_events[h]:
            spike_magnitudes[h] = rng.uniform(1.3, spike_multiplier)

    # Expand spikes to 5-min resolution voi exponential decay
    spike_overlay = np.zeros(num_points)
    for h in range(num_hours):
        if spike_magnitudes[h] > 0:
            # Spike bat dau tai diem nay va decay theo exponential
            start_idx = h * 12
            decay_points = int(spike_decay_hours * 12)
            for j in range(decay_points):
                idx = start_idx + j
                if idx >= num_points:
                    break
                decay = np.exp(-j / (spike_decay_hours * 12 / 3))  # tau = decay_hours/3
                spike_overlay[idx] = max(
                    spike_overlay[idx],
                    (spike_magnitudes[h] - 1.0) * decay
                )

    # Generate shared noise (common shock cho tat ca AZ)
    common_noise = rng.normal(0, 1, num_points)

    all_prices = []

    for az in availability_zones:
        # Moi AZ co base price hoi khac (+/- 3%)
        az_offset = rng.uniform(-0.03, 0.03)
        az_base = base_price * (1.0 + az_offset)

        # AZ-specific noise (phan khong correlated)
        az_noise = rng.normal(0, 1, num_points)

        # Ket hop common + AZ noise theo correlation
        combined_noise = (
            az_correlation * common_noise +
            np.sqrt(1 - az_correlation ** 2) * az_noise
        )

        # Ornstein-Uhlenbeck process
        # X[t+1] = X[t] + theta * (target[t] - X[t]) * dt + sigma * sqrt(dt) * noise
        sigma = volatility * az_base  # Absolute volatility

        prices = np.zeros(num_points)
        prices[0] = az_base * HOURLY_PROFILE[0]

        for i in range(1, num_points):
            target = seasonal_targets[i] * (az_base / base_price)

            # Mean-reversion pull
            pull = mean_reversion_speed * (target - prices[i - 1])

            # Stochastic shock
            shock = sigma * np.sqrt(dt) * combined_noise[i]

            # Spike effect (nhan voi target, khong phai gia hien tai)
            spike_factor = 1.0 + spike_overlay[i]

            # Update price
            new_price = prices[i - 1] + pull + shock
            new_price *= spike_factor if spike_overlay[i] > spike_overlay[i - 1] else 1.0

            # Khi spike dang decay, keo gia ve target * spike_factor
            if spike_overlay[i] > 0 and spike_overlay[i] <= spike_overlay[max(0, i - 1)]:
                spike_target = target * (1.0 + spike_overlay[i] * 0.3)
                new_price += 0.3 * (spike_target - new_price)

            # Hard bounds: 60% - 300% base price
            new_price = np.clip(new_price, az_base * 0.6, az_base * 3.0)

            prices[i] = new_price

        # Build records
        for i in range(num_points):
            all_prices.append({
                'timestamp': timestamps[i],
                'instance_type': instance_type,
                'availability_zone': az,
                'spot_price': round(float(prices[i]), 4),
                'product_description': 'Linux/UNIX',
            })

    df = pd.DataFrame(all_prices)

    logger.info(f"  Records: {len(df)}")
    logger.info(f"  Mean: ${df['spot_price'].mean():.4f}/hr")
    logger.info(f"  Std:  ${df['spot_price'].std():.4f}/hr")
    logger.info(f"  Min:  ${df['spot_price'].min():.4f}/hr")
    logger.info(f"  Max:  ${df['spot_price'].max():.4f}/hr")
    n_spikes = int(spike_events.sum())
    logger.info(f"  Spikes: {n_spikes} events in {num_hours} hours "
                f"({n_spikes / num_hours:.1%})")

    return df


# ──────────────────────────────────────────────
#  Preset scenarios
# ──────────────────────────────────────────────
SCENARIOS = {
    'stable': {
        'volatility': 0.08,
        'mean_reversion_speed': 0.20,
        'peak_hour_strength': 0.8,
        'weekend_discount': 0.10,
        'spike_prob_per_hour': 0.003,
        'spike_multiplier': 1.5,
        'spike_decay_hours': 2.0,
        'seed': 42,
    },
    'volatile': {
        'volatility': 0.25,
        'mean_reversion_speed': 0.08,
        'peak_hour_strength': 1.5,
        'weekend_discount': 0.20,
        'spike_prob_per_hour': 0.03,
        'spike_multiplier': 2.5,
        'spike_decay_hours': 4.0,
        'seed': 123,
    },
    'spike': {
        'volatility': 0.15,
        'mean_reversion_speed': 0.12,
        'peak_hour_strength': 2.0,
        'weekend_discount': 0.05,
        'spike_prob_per_hour': 0.06,
        'spike_multiplier': 3.5,
        'spike_decay_hours': 6.0,
        'seed': 456,
    },
    'az_divergence': {
        'volatility': 0.10,
        'mean_reversion_speed': 0.15,
        'peak_hour_strength': 1.0,
        'weekend_discount': 0.15,
        'spike_prob_per_hour': 0.01,
        'spike_multiplier': 2.0,
        'spike_decay_hours': 3.0,
        'az_correlation': 0.3,  # low correlation → AZs diverge more
        'seed': 789,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic AWS spot price data'
    )
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--instance-types', type=str, default='m5.large')
    parser.add_argument('--days', type=int, default=90,
                        help='Number of days (default: 90)')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['stable', 'volatile', 'spike', 'az_divergence'],
                        help='Use preset scenario (overrides vol/spike params)')
    parser.add_argument('--volatility', type=float, default=None)
    parser.add_argument('--mean-reversion-speed', type=float, default=None)
    parser.add_argument('--peak-hour-strength', type=float, default=None)
    parser.add_argument('--weekend-discount', type=float, default=None)
    parser.add_argument('--spike-prob', type=float, default=None)
    parser.add_argument('--spike-multiplier', type=float, default=None)
    parser.add_argument('--spike-decay-hours', type=float, default=None)
    parser.add_argument('--output', type=str, default='data/raw/spot_prices/')
    parser.add_argument('--tag', type=str, default='synthetic')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-scenarios', action='store_true',
                        help='Generate all 3 scenarios (stable, volatile, spike)')

    args = parser.parse_args()

    instance_types = [t.strip() for t in args.instance_types.split(',')]

    # AZs theo region
    region_azs = {
        'us-east-1': ['us-east-1a', 'us-east-1b', 'us-east-1c'],
        'us-west-2': ['us-west-2a', 'us-west-2b', 'us-west-2c'],
        'ap-southeast-1': ['ap-southeast-1a', 'ap-southeast-1b', 'ap-southeast-1c'],
    }
    azs = region_azs.get(args.region,
                         [f"{args.region}a", f"{args.region}b", f"{args.region}c"])

    # Instance type defaults — base_price ≈ 30-40% of on-demand price
    instance_defaults = {
        'm5.large':   {'base_price': 0.034},   # OD=0.096
        'c5.xlarge':  {'base_price': 0.060},   # OD=0.170
        'r5.large':   {'base_price': 0.044},   # OD=0.126
        'm5.xlarge':  {'base_price': 0.067},   # OD=0.192
        'c5.2xlarge': {'base_price': 0.119},   # OD=0.340
        'r6i.xlarge': {'base_price': 0.140},   # OD=0.252
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Xac dinh scenarios can generate
    if args.all_scenarios:
        scenarios_to_run = list(SCENARIOS.keys())
    elif args.scenario:
        scenarios_to_run = [args.scenario]
    else:
        scenarios_to_run = [args.tag]  # custom params

    for scenario_name in scenarios_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Scenario: {scenario_name}")
        logger.info(f"{'=' * 60}")

        # Lay params tu preset hoac CLI
        if scenario_name in SCENARIOS:
            params = SCENARIOS[scenario_name].copy()
        else:
            params = {
                'volatility': args.volatility or 0.10,
                'mean_reversion_speed': args.mean_reversion_speed or 0.15,
                'peak_hour_strength': args.peak_hour_strength or 1.0,
                'weekend_discount': args.weekend_discount or 0.15,
                'spike_prob_per_hour': args.spike_prob or 0.01,
                'spike_multiplier': args.spike_multiplier or 2.0,
                'spike_decay_hours': args.spike_decay_hours or 3.0,
                'seed': args.seed,
            }

        # CLI overrides (cho ca preset)
        if args.volatility is not None:
            params['volatility'] = args.volatility
        if args.mean_reversion_speed is not None:
            params['mean_reversion_speed'] = args.mean_reversion_speed
        if args.spike_prob is not None:
            params['spike_prob_per_hour'] = args.spike_prob
        if args.spike_multiplier is not None:
            params['spike_multiplier'] = args.spike_multiplier

        seed = params.pop('seed', args.seed)

        all_data = []
        for inst_type in instance_types:
            inst_defaults = instance_defaults.get(
                inst_type, {'base_price': 0.03}
            )

            df = generate_spot_prices(
                instance_type=inst_type,
                region=args.region,
                availability_zones=azs,
                duration_days=args.days,
                base_price=inst_defaults['base_price'],
                seed=seed + hash(inst_type) % 1000,
                **params,
            )
            all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True).sort_values('timestamp')

        tag = scenario_name if scenario_name in SCENARIOS else args.tag
        output_file = output_dir / f"spot_prices_{args.region}_{tag}.csv"
        combined.to_csv(output_file, index=False)

        logger.info(f"\nSaved {len(combined)} records to {output_file}")

        # Summary
        for inst_type in instance_types:
            sub = combined[combined['instance_type'] == inst_type]
            logger.info(f"  {inst_type}: mean=${sub['spot_price'].mean():.4f} "
                        f"std=${sub['spot_price'].std():.4f} "
                        f"range=[${sub['spot_price'].min():.4f}, "
                        f"${sub['spot_price'].max():.4f}]")


if __name__ == '__main__':
    main()


# Example usage:
#
# Generate tat ca 3 scenarios (stable, volatile, spike) cho 90 ngay:
#   python generate_synthetic_spot_prices.py --all-scenarios --days 90
#
# Generate 1 scenario cu the:
#   python generate_synthetic_spot_prices.py --scenario stable --days 90
#
# Custom params:
#   python generate_synthetic_spot_prices.py --volatility 0.20 --spike-prob 0.05 --tag custom
