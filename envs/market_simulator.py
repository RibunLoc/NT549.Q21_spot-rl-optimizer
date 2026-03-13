"""
Spot market price simulator.

Simulates spot price dynamics and interruption events based on historical data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SpotMarketSimulator:
    """
    Simulates AWS EC2 spot market dynamics.

    TODO: Implement realistic spot price simulation:
    - Use historical data to model price patterns
    - Add time-of-day and day-of-week effects
    - Simulate interruption events based on price levels
    - Support multiple instance types
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        instance_type: str = "m5.large",
        availability_zone: str = "us-east-1a",
        seed: Optional[int] = None,
    ):
        """
        Initialize market simulator.

        Args:
            historical_data: DataFrame with columns [timestamp, spot_price, instance_type, availability_zone]
            instance_type: EC2 instance type
            availability_zone: AWS availability zone
            seed: Random seed for reproducibility
        """
        self.historical_data = historical_data
        self.instance_type = instance_type
        self.availability_zone = availability_zone
        self.rng = np.random.default_rng(seed)

        # Filter data for selected instance_type and AZ
        mask = (
            (historical_data['instance_type'] == instance_type) &
            (historical_data['availability_zone'] == availability_zone)
        )
        filtered_data = historical_data[mask].copy()

        if len(filtered_data) == 0:
            # Fallback: use any data for this instance type
            mask = historical_data['instance_type'] == instance_type
            filtered_data = historical_data[mask].copy()

            if len(filtered_data) == 0:
                raise ValueError(f"No data found for instance type {instance_type}")

        # Sort by timestamp and extract prices
        filtered_data = filtered_data.sort_values('timestamp')
        self.prices = filtered_data['spot_price'].values
        self.timestamps = filtered_data['timestamp'].values

        # Current state
        self.current_timestep = 0
        self.current_price = 0.0
        self.interruption_prob = 0.0
        self.price_history = []  # Track price history for statistics

    def reset(self, seed: Optional[int] = None):
        """Reset simulator to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.price_history = []

        # Set initial price from historical data
        if len(self.prices) > 0:
            self.current_price = float(self.prices[0])
            self.price_history.append(self.current_price)
        else:
            self.current_price = 0.03  # Fallback

        # Initial interruption probability (low)
        self.interruption_prob = 0.05

    def step(self) -> Tuple[float, float, bool]:
        """
        Advance market by one timestep.

        Returns:
            current_price: Spot price at current timestep ($/hour)
            interruption_prob: Probability of interruption (0-1)
            interrupted: Whether interruption occurred
        """
        # Replay historical prices (cycle if we reach the end)
        price_index = self.current_timestep % len(self.prices)
        self.current_price = float(self.prices[price_index])
        self.price_history.append(self.current_price)

        # Calculate interruption probability based on price volatility
        # Higher price relative to recent average = higher interruption risk
        if len(self.price_history) >= 24:
            recent_avg = np.mean(self.price_history[-24:])
            price_ratio = self.current_price / recent_avg if recent_avg > 0 else 1.0

            # Interruption prob increases when price is high
            self.interruption_prob = min(0.3, max(0.01, (price_ratio - 1.0) * 0.5))
        else:
            self.interruption_prob = 0.05  # Default 5%

        # Sample interruption event
        interrupted = self.rng.random() < self.interruption_prob

        self.current_timestep += 1

        return self.current_price, self.interruption_prob, interrupted

    def get_price_statistics(self, window: int = 24) -> dict:
        """
        Get price statistics for the past window timesteps.

        Returns:
            dict with keys: mean_1h, mean_24h, volatility, trend
        """
        history = self.price_history[-window:] if len(self.price_history) > 0 else [self.current_price]

        mean_price = np.mean(history)
        volatility = np.std(history) if len(history) > 1 else 0.0

        # Calculate trend (simple: compare first half vs second half)
        if len(history) >= 4:
            mid = len(history) // 2
            first_half_mean = np.mean(history[:mid])
            second_half_mean = np.mean(history[mid:])
            trend = (second_half_mean - first_half_mean) / first_half_mean if first_half_mean > 0 else 0.0
            trend = np.clip(trend, -1.0, 1.0)  # Normalize to [-1, 1]
        else:
            trend = 0.0

        return {
            "mean_1h": np.mean(self.price_history[-1:]) if len(self.price_history) > 0 else self.current_price,
            "mean_24h": mean_price,
            "volatility": volatility,
            "trend": trend,
        }

    def get_interruption_frequency(self, lookback: int = 168) -> float:
        """
        Get historical interruption frequency (past lookback hours).

        Args:
            lookback: Number of hours to look back (default 168 = 1 week)

        Returns:
            Interruption frequency (0-1)
        """
        # TODO: Calculate from historical data
        return 0.1  # Placeholder: 10% interruption rate
