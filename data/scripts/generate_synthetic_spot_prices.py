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

# On-demand prices per type ($/hr) — mirrors envs/instance_catalog.py
_OD_PRICES = {
    "m5.large":   0.096,
    "c5.xlarge":  0.170,
    "r5.large":   0.126,
    "m5.xlarge":  0.192,
    "c5.2xlarge": 0.340,
}


# ──────────────────────────────────────────────
#  Daily price profile (24h) — AWS thực tế: cực kỳ flat
#  Spot prices trên AWS hầu như không có daily seasonality rõ ràng
#  (không như on-demand; spot driven bởi capacity availability, không phải demand)
#  Calibrated từ AWS console chart: c5.2xlarge ap-southeast-1, 3 months
# ──────────────────────────────────────────────
HOURLY_PROFILE = np.array([
    # 0h    1h    2h    3h    4h    5h    6h    7h
    1.00, 1.00, 0.99, 0.99, 0.99, 1.00, 1.00, 1.00,
    # 8h    9h   10h   11h   12h   13h   14h   15h
    1.01, 1.01, 1.02, 1.02, 1.01, 1.01, 1.02, 1.02,
    # 16h   17h   18h   19h   20h   21h   22h   23h
    1.01, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])


def generate_capacity_signal(
    instance_types: list,
    availability_zones: list,
    num_points: int,
    dt_minutes: int = 5,
    az_shock_rate_per_week: float = 2.0,
    shock_depth_range: tuple = (0.3, 0.6),
    shock_decay_hours: float = 12.0,
    baseline: float = 0.80,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict:
    """Generate pool capacity depth signal INDEPENDENT from price.

    Models AWS's real spot interruption driver: OD/Reserved capacity pressure
    per-AZ. When enterprise customers book reserved capacity, all spot pools in
    that AZ lose depth simultaneously.

    Returns:
        dict[(instance_type, az)] → np.ndarray[num_points], values ∈ [0, 1]
        1 = pool full, 0 = sắp cạn → interrupt imminent
    """
    rng = np.random.default_rng(seed)

    # Step 1: AZ-wide shock schedule (shared across all types trong AZ)
    # Duration in points: num_points * dt_minutes / (60*24*7) weeks
    weeks = num_points * dt_minutes / (60 * 24 * 7)
    n_shocks_per_az = int(weeks * az_shock_rate_per_week)
    decay_points = int(shock_decay_hours * 60 / dt_minutes)

    az_shock_overlay = {az: np.zeros(num_points) for az in availability_zones}
    for az in availability_zones:
        shock_times = rng.integers(0, num_points, size=n_shocks_per_az)
        for t0 in shock_times:
            depth = rng.uniform(*shock_depth_range)
            for j in range(decay_points):
                idx = t0 + j
                if idx >= num_points:
                    break
                decay = np.exp(-j / (decay_points / 3.0))
                az_shock_overlay[az][idx] = max(az_shock_overlay[az][idx], depth * decay)

    # Step 2: Per-(type, AZ) random walk + AZ shock + peak-hour effect
    capacity = {}
    for itype in instance_types:
        for az in availability_zones:
            # OU random walk around baseline
            walk = np.zeros(num_points)
            walk[0] = baseline
            theta = 0.05  # mean-reversion speed
            for i in range(1, num_points):
                pull = theta * (baseline - walk[i-1])
                shock = rng.normal(0, noise_std)
                walk[i] = walk[i-1] + pull + shock

            # Peak-hour effect: 9-17h drop 0.05
            peak_boost = np.zeros(num_points)
            for i in range(num_points):
                hour = (i * dt_minutes // 60) % 24
                if 9 <= hour <= 17:
                    peak_boost[i] = -0.05

            depth = walk + peak_boost - az_shock_overlay[az]
            depth = np.clip(depth, 0.0, 1.0)
            capacity[(itype, az)] = depth

    return capacity


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
            low = min(1.1, spike_multiplier * 0.85)
            spike_magnitudes[h] = rng.uniform(low, spike_multiplier)

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
        # Mỗi AZ có base price hơi khác (~2-4% total spread)
        # AWS thực: c5.2xlarge ap-southeast-1 AZ spread ~$0.004/$0.175 = ~2.3%
        az_offset = rng.uniform(-0.02, 0.02)
        az_base = base_price * (1.0 + az_offset)

        # Per-AZ slow random walk on the mean-reversion target
        # Mô phỏng AZ-level capacity drift độc lập với common market signal
        # Walk speed: ~0.5% per day → tạo AZ divergence tự nhiên theo thời gian
        az_drift = np.zeros(num_points)
        drift_sigma = az_base * 0.003 * np.sqrt(dt)  # 0.3% daily drift per AZ
        for i in range(1, num_points):
            az_drift[i] = az_drift[i-1] * 0.998 + rng.normal(0, drift_sigma)  # slow-decay walk
        az_drift = np.clip(az_drift, -az_base * 0.05, az_base * 0.05)

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
            # Per-AZ target includes slow independent drift → breaks AZ correlation
            target = seasonal_targets[i] * (az_base / base_price) + az_drift[i]

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

            # Hard bounds: 85% - 200% base price
            # AWS thực: spot ít khi xuống dưới 85% long-term mean hoặc vượt 2x
            new_price = np.clip(new_price, az_base * 0.85, az_base * 2.0)

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
    # Calibrated từ AWS console: c5.2xlarge ap-southeast-1, Feb-May 2026
    # Chart đặc điểm: cực kỳ flat (std/mean ~2%), AZ spread ~3%, ~0 spikes/3 months
    # Đây là behavior thực tế của most spot instances ở ap-southeast-1
    'stable': {
        'volatility': 0.02,             # AWS thực: std/mean ~1-2% cho stable types
        'mean_reversion_speed': 0.30,   # Strong reversion → giữ giá gần flat
        'peak_hour_strength': 0.1,      # Gần 0: không có daily pattern rõ
        'weekend_discount': 0.02,       # AWS spot: weekend effect rất nhỏ (<2%)
        'spike_prob_per_hour': 0.0003,  # ~0.6 spikes/90 days ≈ 0 (chart: 0 spikes)
        'spike_multiplier': 1.2,        # Micro-spike chỉ +20%
        'spike_decay_hours': 1.0,
        'seed': 42,
    },
    # Volatile: instance family có demand cao hơn (GPU-adjacent, burstable)
    # Vẫn realistic nhưng có thể xảy ra ở busy regions/AZs
    # Ví dụ: c5n, g4dn trong us-east-1 busy periods
    'volatile': {
        'volatility': 0.08,             # 3-4x stable, vẫn trong khoảng AWS observe
        'mean_reversion_speed': 0.15,
        'peak_hour_strength': 0.3,      # Rất nhỏ
        'weekend_discount': 0.05,
        'spike_prob_per_hour': 0.005,   # ~11 spikes/90d — "volatile" nhưng plausible
        'spike_multiplier': 1.6,        # +60% max spike
        'spike_decay_hours': 2.0,
        'seed': 123,
    },
    # Spike scenario: mô phỏng re:Invent week / EC2 capacity crunch
    # Xảy ra vài lần/năm — agent cần biết handle nhưng không overfit
    'spike': {
        'volatility': 0.10,
        'mean_reversion_speed': 0.10,
        'peak_hour_strength': 0.2,
        'weekend_discount': 0.03,
        'spike_prob_per_hour': 0.012,   # ~26 spikes/90d = ~1 spike/3.5 days
        'spike_multiplier': 2.2,        # Spike lên tới +120% — dramatic but real
        'spike_decay_hours': 4.0,
        'seed': 456,
    },
    # AZ divergence: một AZ bị capacity pressure do Reserved/OD booking
    # Giá AZ bị ảnh hưởng tăng nhẹ, nhưng AZ khác vẫn flat
    # az_correlation=0.25: AZs hầu như độc lập (thực tế khi AZ bị isolate)
    'az_divergence': {
        'volatility': 0.05,
        'mean_reversion_speed': 0.20,
        'peak_hour_strength': 0.1,
        'weekend_discount': 0.02,
        'spike_prob_per_hour': 0.003,
        'spike_multiplier': 1.5,
        'spike_decay_hours': 3.0,
        'az_correlation': 0.25,         # Low correlation → AZ prices diverge
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

    # Instance type defaults — base_price calibrated từ AWS console thực tế
    # ap-southeast-1, 3 months (Feb-May 2026): spot/OD ratio ~55-57%
    # c5.2xlarge: OD=$0.392, spot avg=$0.175 → ratio=44.6% (cheapest AZ)
    # m5.large:   OD=$0.096, spot avg=$0.054 → ratio=56% (estimated)
    # Note: ap-southeast-1 thường cao hơn us-east-1 ~15-20%
    instance_defaults = {
        'm5.large':   {'base_price': 0.054},   # OD=0.096 → 56% ratio
        'c5.xlarge':  {'base_price': 0.094},   # OD=0.170 → 55% ratio
        'r5.large':   {'base_price': 0.069},   # OD=0.126 → 55% ratio
        'm5.xlarge':  {'base_price': 0.106},   # OD=0.192 → 55% ratio
        'c5.2xlarge': {'base_price': 0.175},   # OD=0.340 → 51% ratio (AWS actual avg)
        'r6i.xlarge': {'base_price': 0.140},   # OD=0.252 → 56% ratio
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

        # Per-scenario capacity signal params
        _cap_params = {
            'stable':        {'az_shock_rate_per_week': 1.0, 'shock_depth_range': (0.2, 0.4)},
            'volatile':      {'az_shock_rate_per_week': 3.0, 'shock_depth_range': (0.3, 0.6)},
            'spike':         {'az_shock_rate_per_week': 5.0, 'shock_depth_range': (0.4, 0.7)},
            'az_divergence': {'az_shock_rate_per_week': 3.0, 'shock_depth_range': (0.3, 0.6)},
        }.get(scenario_name, {'az_shock_rate_per_week': 2.0, 'shock_depth_range': (0.3, 0.5)})

        # Generate capacity signal (same time grid as prices)
        num_points = args.days * 24 * 12  # 5-min resolution
        capacity = generate_capacity_signal(
            instance_types=instance_types,
            availability_zones=azs,
            num_points=num_points,
            dt_minutes=5,
            seed=seed + 777,
            **_cap_params,
        )

        # Merge capacity column into combined by (timestamp, type, AZ) position
        # Simpler: both arrays share same ordering (by type → az → time), append column
        combined = combined.reset_index(drop=True)
        cap_col = np.zeros(len(combined))
        # Build lookup: for each row, find capacity[(type, az)][time_index]
        # Rows are sorted by timestamp asc, so time_index per (type, az) = cumulative count
        cap_counters = {}
        for i in range(len(combined)):
            key = (combined.iloc[i]['instance_type'], combined.iloc[i]['availability_zone'])
            idx = cap_counters.get(key, 0)
            if idx < num_points:
                cap_col[i] = capacity[key][idx]
            cap_counters[key] = idx + 1
        combined['capacity_depth'] = cap_col

        # SPS score: correlated với capacity_depth nhưng có noise độc lập
        # Mô hình: sps = clip(capacity * 0.7 + price_signal * 0.3 + noise, 0, 1)
        # Ý nghĩa: capacity thấp → SPS thấp (khó launch), nhưng SPS còn phụ thuộc
        # demand pressure (giá cao = nhiều người cũng đang request → SPS thấp hơn)
        rng_sps = np.random.default_rng(seed + 999)
        price_col = combined['spot_price'].values
        price_ratios = np.zeros(len(combined))
        for itype in instance_types:
            mask = combined['instance_type'] == itype
            od_price = _OD_PRICES.get(itype, float(price_col[mask].mean()) * 3.0)
            price_ratios[mask] = np.clip(price_col[mask] / od_price, 0.0, 1.5)

        # price_pressure: giá cao → nhiều demand → SPS giảm
        price_pressure = np.clip(price_ratios / 1.5, 0.0, 1.0)  # normalize to [0,1]

        # SPS = 70% capacity + 30% inverse price pressure + small noise
        sps_noise = rng_sps.normal(0, 0.04, size=len(combined))
        sps_col = 0.70 * cap_col + 0.30 * (1.0 - price_pressure) + sps_noise
        sps_col = np.clip(sps_col, 0.0, 1.0)
        combined['sps_score'] = sps_col

        logger.info(f"  Capacity signal: mean={cap_col.mean():.3f} "
                    f"min={cap_col.min():.3f} p5={np.percentile(cap_col, 5):.3f}")
        logger.info(f"  SPS score:       mean={sps_col.mean():.3f} "
                    f"min={sps_col.min():.3f} p5={np.percentile(sps_col, 5):.3f} "
                    f"corr_with_cap={np.corrcoef(cap_col, sps_col)[0,1]:.3f}")

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
