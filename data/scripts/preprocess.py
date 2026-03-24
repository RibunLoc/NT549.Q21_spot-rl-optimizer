"""
Preprocess raw data for training.

Feature engineering:
- Resample spot prices to hourly frequency
- Rolling averages (1h, 6h, 24h)
- Price volatility (rolling std)
- Price trend indicator
- Time features (hour of day, day of week)
- Interruption probability estimation

Usage:
    python preprocess.py --input data/raw/ --output data/processed/ --output-name price_features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_spot_prices(
    df: pd.DataFrame,
    instance_type: str = "m5.large",
    availability_zone: str = None,
) -> pd.DataFrame:
    """
    Preprocess spot price data into hourly features.

    Args:
        df: Raw spot price DataFrame with columns
            [timestamp, instance_type, availability_zone, spot_price]
        instance_type: Instance type to filter
        availability_zone: AZ to filter (None = first available)

    Returns:
        Processed DataFrame with hourly features
    """
    logger.info("Preprocessing spot prices...")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp')

    # Filter by instance type
    mask = df['instance_type'] == instance_type
    if mask.sum() == 0:
        available = df['instance_type'].unique().tolist()
        raise ValueError(
            f"No data for instance_type={instance_type}. Available: {available}"
        )
    df = df[mask]

    # Pick one AZ
    if availability_zone is None:
        availability_zone = df['availability_zone'].value_counts().index[0]
    df = df[df['availability_zone'] == availability_zone].copy()

    logger.info(f"  Filtered: {instance_type} in {availability_zone} ({len(df)} records)")

    # Set timestamp as index and resample to hourly
    df = df.set_index('timestamp')
    hourly = df['spot_price'].resample('1h').mean().interpolate(method='linear')
    hourly = hourly.dropna()

    # Build feature DataFrame
    features = pd.DataFrame(index=hourly.index)
    features['spot_price'] = hourly
    features['instance_type'] = instance_type
    features['availability_zone'] = availability_zone

    # Rolling averages
    features['price_ma_1h'] = hourly.rolling(window=1, min_periods=1).mean()
    features['price_ma_6h'] = hourly.rolling(window=6, min_periods=1).mean()
    features['price_ma_24h'] = hourly.rolling(window=24, min_periods=1).mean()

    # Volatility (rolling std)
    features['price_volatility_6h'] = hourly.rolling(window=6, min_periods=2).std().fillna(0)
    features['price_volatility_24h'] = hourly.rolling(window=24, min_periods=2).std().fillna(0)

    # Price trend: ratio of current price to 24h MA
    features['price_trend'] = (hourly / features['price_ma_24h']).fillna(1.0)

    # Estimated interruption probability:
    # Higher when price is above average (demand > supply)
    features['est_interruption_prob'] = np.clip(
        (features['price_trend'] - 1.0) * 0.5, 0.01, 0.30
    )

    # Time features
    features['hour_of_day'] = features.index.hour
    features['day_of_week'] = features.index.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(float)
    features['is_peak_hour'] = features['hour_of_day'].apply(
        lambda h: 1.0 if 9 <= h <= 17 else 0.0
    )

    # Reset index
    features = features.reset_index()
    features = features.rename(columns={'timestamp': 'timestamp'})

    logger.info(f"  Generated {len(features)} hourly feature records")
    logger.info(f"  Price range: ${features['spot_price'].min():.4f} - ${features['spot_price'].max():.4f}")
    logger.info(f"  Mean volatility (24h): {features['price_volatility_24h'].mean():.6f}")

    return features


def main():
    parser = argparse.ArgumentParser(description='Preprocess spot price and workload data')
    parser.add_argument('--input', type=str, default='data/raw/',
                        help='Input directory with raw data')
    parser.add_argument('--output', type=str, default='data/processed/',
                        help='Output directory for processed data')
    parser.add_argument('--input-glob', type=str, default='spot_prices/*.csv',
                        help='Glob pattern under input dir (default: spot_prices/*.csv)')
    parser.add_argument('--output-name', type=str, default='price_features',
                        help='Base name for output files (default: price_features)')
    parser.add_argument('--instance-type', type=str, default='m5.large',
                        help='Instance type to preprocess')
    parser.add_argument('--az', type=str, default=None,
                        help='Availability zone (default: auto-select)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load spot price CSVs
    spot_files = list(input_dir.glob(args.input_glob))
    if not spot_files:
        logger.error("No spot price CSV files found in data/raw/spot_prices/")
        return

    logger.info(f"Found {len(spot_files)} spot price files")
    spot_df = pd.concat([pd.read_csv(f) for f in spot_files], ignore_index=True)
    logger.info(f"Loaded {len(spot_df)} total spot price records")

    # Preprocess
    features_df = preprocess_spot_prices(
        spot_df,
        instance_type=args.instance_type,
        availability_zone=args.az,
    )

    # Save as both CSV and pickle
    base_name = args.output_name
    csv_output = output_dir / f'{base_name}.csv'
    pkl_output = output_dir / f'{base_name}.pkl'

    features_df.to_csv(csv_output, index=False)
    features_df.to_pickle(pkl_output)

    logger.info(f"Saved processed features to {csv_output} and {pkl_output}")
    logger.info(f"Total records: {len(features_df)}")


if __name__ == '__main__':
    main()
