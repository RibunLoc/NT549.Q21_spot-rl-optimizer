"""
Script to fetch AWS EC2 spot price history using boto3.

Usage:
    python fetch_spot_prices.py --region us-east-1 --instance-types m5.large,c5.large --days 30
"""

import boto3
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_spot_prices(
    region: str,
    instance_types: list,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
):
    """
    Fetch spot price history from AWS.

    Args:
        region: AWS region (e.g., 'us-east-1')
        instance_types: List of instance types (e.g., ['m5.large', 'c5.large'])
        start_date: Start date for history
        end_date: End date for history
        output_dir: Directory to save CSV files
    """
    # TODO: Implement AWS API call
    # 1. Initialize boto3 client
    # 2. Call describe_spot_price_history() for each instance type
    # 3. Parse response and convert to DataFrame
    # 4. Save to CSV

    logger.info(f"Fetching spot prices from {region}")
    logger.info(f"Instance types: {instance_types}")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize EC2 client
    ec2_client = boto3.client('ec2', region_name=region)

    all_prices = []

    for instance_type in instance_types:
        logger.info(f"Fetching {instance_type}...")

        try:
            # Paginate through all results
            paginator = ec2_client.get_paginator('describe_spot_price_history')
            page_iterator = paginator.paginate(
                InstanceTypes=[instance_type],
                StartTime=start_date,
                EndTime=end_date,
                ProductDescriptions=['Linux/UNIX'],
            )

            count = 0
            for page in page_iterator:
                for item in page['SpotPriceHistory']:
                    all_prices.append({
                        'timestamp': item['Timestamp'],
                        'instance_type': item['InstanceType'],
                        'availability_zone': item['AvailabilityZone'],
                        'spot_price': float(item['SpotPrice']),
                        'product_description': item['ProductDescription'],
                    })
                    count += 1

            logger.info(f"  Fetched {count} records")

        except Exception as e:
            logger.error(f"Error fetching {instance_type}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_prices)

    if df.empty:
        logger.warning("No data fetched!")
        return

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"spot_prices_{region}_{start_date.date()}_to_{end_date.date()}.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"Saved {len(df)} records to {output_file}")

    # Print summary statistics
    logger.info("\nSummary statistics:")
    for instance_type in instance_types:
        subset = df[df['instance_type'] == instance_type]
        if not subset.empty:
            logger.info(f"  {instance_type}:")
            logger.info(f"    Mean price: ${subset['spot_price'].mean():.4f}/hour")
            logger.info(f"    Min price: ${subset['spot_price'].min():.4f}/hour")
            logger.info(f"    Max price: ${subset['spot_price'].max():.4f}/hour")
            logger.info(f"    Std dev: ${subset['spot_price'].std():.4f}/hour")


def main():
    parser = argparse.ArgumentParser(description='Fetch AWS EC2 spot price history')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region (default: us-east-1)')
    parser.add_argument('--instance-types', type=str, default='m5.large',
                        help='Comma-separated instance types (default: m5.large)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to fetch (default: 30)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD), overrides --days')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--output', type=str, default='data/raw/spot_prices/',
                        help='Output directory (default: data/raw/spot_prices/)')

    args = parser.parse_args()

    # Parse instance types
    instance_types = [t.strip() for t in args.instance_types.split(',')]

    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days)

    # Fetch prices
    output_dir = Path(args.output)
    fetch_spot_prices(args.region, instance_types, start_date, end_date, output_dir)


if __name__ == '__main__':
    main()


# Example usage:
# python fetch_spot_prices.py --region us-east-1 --instance-types m5.large,m5.xlarge,c5.large --days 30
