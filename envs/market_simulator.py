"""
Bộ mô phỏng giá thị trường Spot.

Mô phỏng biến động giá spot và các sự kiện gián đoạn dựa trên dữ liệu lịch sử.

Cải tiến so với bản cũ:
- Random start position mỗi episode (không luôn bắt đầu từ index 0)
- Noise overlay trên giá replay → cùng timestep cho giá hơi khác nhau mỗi episode
- Sử dụng pre-computed features (est_interruption_prob, volatility) nếu có
- Interruption probability tính từ data thực, không hardcode
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SpotMarketSimulator:
    """
    Mô phỏng động lực thị trường spot AWS EC2.

    Replay giá lịch sử với random offset + noise để tạo diversity giữa các episode.
    Tính interruption probability từ biến động giá thực tế.
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        instance_type: str = "m5.large",
        availability_zone: str = "us-east-1a",
        noise_std: float = 0.005,
        seed: Optional[int] = None,
    ):
        """
        Khởi tạo bộ mô phỏng thị trường.

        Args:
            historical_data: DataFrame với các cột [timestamp, spot_price, instance_type, availability_zone]
                             Có thể có thêm: est_interruption_prob, volatility_24h, price_ma_6h, price_ma_24h
            instance_type: Loại EC2 instance
            availability_zone: Vùng khả dụng AWS
            noise_std: Độ lệch chuẩn của noise overlay (fraction of price)
            seed: Seed ngẫu nhiên để tái tạo kết quả
        """
        self.instance_type = instance_type
        self.availability_zone = availability_zone
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Lọc dữ liệu theo instance_type và AZ đã chọn
        mask = (
            (historical_data['instance_type'] == instance_type) &
            (historical_data['availability_zone'] == availability_zone)
        )
        filtered_data = historical_data[mask].copy()

        if len(filtered_data) == 0:
            # Dự phòng: dùng bất kỳ dữ liệu nào cho instance type này
            mask = historical_data['instance_type'] == instance_type
            filtered_data = historical_data[mask].copy()

            if len(filtered_data) == 0:
                raise ValueError(f"Không tìm thấy dữ liệu cho instance type {instance_type}")

        # Sắp xếp theo timestamp
        filtered_data = filtered_data.sort_values('timestamp').reset_index(drop=True)

        # Trích xuất giá và các features đã pre-compute (nếu có)
        self.prices = filtered_data['spot_price'].values.astype(np.float64)
        self.timestamps = filtered_data['timestamp'].values
        self.data_len = len(self.prices)

        # Pre-computed features từ preprocess.py (nếu có)
        self.has_features = 'est_interruption_prob' in filtered_data.columns
        if self.has_features:
            self.precomputed_interr_prob = filtered_data['est_interruption_prob'].values
            # preprocess.py outputs: price_volatility_24h, price_ma_6h, price_ma_24h
            self.precomputed_volatility = filtered_data.get(
                'price_volatility_24h', filtered_data.get(
                    'volatility_24h', pd.Series(np.zeros(self.data_len))
                )
            ).values
            self.precomputed_ma_6h = filtered_data.get(
                'price_ma_6h', pd.Series(self.prices)
            ).values
            self.precomputed_ma_24h = filtered_data.get(
                'price_ma_24h', pd.Series(self.prices)
            ).values

        # Tính thống kê toàn cục để bound noise
        self.global_mean = float(np.mean(self.prices))
        self.global_min = float(np.min(self.prices)) * 0.6
        self.global_max = float(np.max(self.prices)) * 1.5

        # Trạng thái hiện tại
        self.current_timestep = 0
        self.start_offset = 0
        self.current_price = 0.0
        self.interruption_prob = 0.0
        self.price_history = []
        self.interruption_history = []  # Track interruptions cho frequency calc

    def reset(self, seed: Optional[int] = None):
        """
        Đặt lại bộ mô phỏng.

        Random start position để agent không memorize thứ tự giá.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.price_history = []
        self.interruption_history = []

        # Random start offset — mỗi episode bắt đầu từ vị trí khác nhau trong data
        self.start_offset = self.rng.integers(0, self.data_len)

        # Đặt giá ban đầu (có noise)
        base_price = float(self.prices[self.start_offset])
        self.current_price = self._add_noise(base_price)
        self.price_history.append(self.current_price)

        # Interruption prob ban đầu
        if self.has_features:
            self.interruption_prob = float(self.precomputed_interr_prob[self.start_offset])
        else:
            self.interruption_prob = 0.05

    def step(self) -> Tuple[float, float, bool]:
        """
        Tiến thị trường thêm một timestep.

        Returns:
            current_price: Giá spot tại timestep hiện tại ($/giờ)
            interruption_prob: Xác suất bị gián đoạn (0-1)
            interrupted: Có xảy ra gián đoạn không
        """
        self.current_timestep += 1

        # Replay giá lịch sử với random offset (wrap around)
        price_index = (self.start_offset + self.current_timestep) % self.data_len
        base_price = float(self.prices[price_index])

        # Thêm noise để cùng timestep → giá hơi khác mỗi episode
        self.current_price = self._add_noise(base_price)
        self.price_history.append(self.current_price)

        # Tính xác suất gián đoạn
        if self.has_features:
            # Dùng pre-computed interruption probability
            self.interruption_prob = float(self.precomputed_interr_prob[price_index])
            # Thêm một chút random variation
            noise = self.rng.normal(0, 0.02)
            self.interruption_prob = float(np.clip(
                self.interruption_prob + noise, 0.01, 0.5
            ))
        else:
            # Fallback: tính từ price history
            self.interruption_prob = self._compute_interruption_prob()

        # Lấy mẫu sự kiện gián đoạn
        interrupted = bool(self.rng.random() < self.interruption_prob)
        self.interruption_history.append(interrupted)

        return self.current_price, self.interruption_prob, interrupted

    def _add_noise(self, base_price: float) -> float:
        """Thêm multiplicative noise vào giá, giữ trong bound hợp lý."""
        noise_factor = 1.0 + self.rng.normal(0, self.noise_std)
        noisy_price = base_price * noise_factor
        return float(np.clip(noisy_price, self.global_min, self.global_max))

    def _compute_interruption_prob(self) -> float:
        """Tính interruption probability từ price history khi không có pre-computed features."""
        if len(self.price_history) < 6:
            return 0.05

        recent_prices = self.price_history[-24:] if len(self.price_history) >= 24 else self.price_history
        recent_avg = np.mean(recent_prices)

        if recent_avg <= 0:
            return 0.05

        # Giá cao hơn trung bình → interrupt probability cao hơn
        price_ratio = self.current_price / recent_avg

        # Biến động cao → interrupt probability cao hơn
        volatility = np.std(recent_prices) / recent_avg if recent_avg > 0 else 0

        # Kết hợp cả 2 tín hiệu
        prob = 0.02 + max(0, (price_ratio - 1.0)) * 0.3 + volatility * 0.5
        return float(np.clip(prob, 0.01, 0.4))

    def get_price_statistics(self, window: int = 24) -> dict:
        """
        Lấy thống kê giá.

        Ưu tiên dùng pre-computed features nếu có, fallback sang tính từ history.
        """
        if self.has_features:
            price_index = (self.start_offset + self.current_timestep) % self.data_len
            return {
                "mean_6h": float(self.precomputed_ma_6h[price_index]),
                "mean_24h": float(self.precomputed_ma_24h[price_index]),
                "volatility": float(self.precomputed_volatility[price_index]),
                "trend": self._compute_trend(),
            }

        # Fallback: tính từ price_history
        history = self.price_history[-window:] if self.price_history else [self.current_price]

        return {
            "mean_6h": float(np.mean(self.price_history[-6:])) if len(self.price_history) >= 6 else float(np.mean(self.price_history)) if self.price_history else self.current_price,
            "mean_24h": float(np.mean(history)),
            "volatility": float(np.std(history)) if len(history) > 1 else 0.0,
            "trend": self._compute_trend(),
        }

    def _compute_trend(self) -> float:
        """Tính xu hướng giá: dương = giá đang tăng."""
        if len(self.price_history) < 4:
            return 0.0
        recent = self.price_history[-24:] if len(self.price_history) >= 24 else self.price_history
        mid = len(recent) // 2
        first_half = np.mean(recent[:mid])
        second_half = np.mean(recent[mid:])
        if first_half <= 0:
            return 0.0
        trend = (second_half - first_half) / first_half
        return float(np.clip(trend, -1.0, 1.0))

    def get_interruption_frequency(self, lookback: int = 168) -> float:
        """
        Tần suất gián đoạn thực tế trong lookback steps gần nhất.

        Args:
            lookback: Số steps nhìn lại

        Returns:
            Tần suất gián đoạn (0-1)
        """
        if not self.interruption_history:
            return 0.05  # Default trước khi có data

        recent = self.interruption_history[-lookback:]
        return float(np.mean(recent))


class MultiPoolMarketSimulator:
    """
    Manages N_TYPES × N_AZS parallel SpotMarketSimulator instances.

    Each pool (instance_type, AZ) has its own price stream from historical data.
    Provides unified step/reset across all pools.
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        instance_types: list,
        availability_zones: list,
        od_prices: dict,
        noise_std: float = 0.005,
        seed: Optional[int] = None,
    ):
        """
        Args:
            historical_data: Multi-pool DataFrame with columns
                [timestamp, spot_price, instance_type, availability_zone, ...]
            instance_types: List of instance type names
            availability_zones: List of AZ names
            od_prices: Dict mapping instance_type name → on-demand price
            noise_std: Noise for price replay
            seed: Random seed
        """
        self.instance_types = instance_types
        self.availability_zones = availability_zones
        self.od_prices = od_prices
        self.n_types = len(instance_types)
        self.n_azs = len(availability_zones)
        self.rng = np.random.default_rng(seed)

        # Create one SpotMarketSimulator per (type, AZ)
        self.sims = {}
        for t_idx, itype in enumerate(instance_types):
            for az_idx, az in enumerate(availability_zones):
                sub_seed = None
                if seed is not None:
                    sub_seed = seed + t_idx * 100 + az_idx
                self.sims[(t_idx, az_idx)] = SpotMarketSimulator(
                    historical_data=historical_data,
                    instance_type=itype,
                    availability_zone=az,
                    noise_std=noise_std,
                    seed=sub_seed,
                )

    def reset(self, seed: Optional[int] = None):
        """Reset all pool simulators."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for (t_idx, az_idx), sim in self.sims.items():
            sub_seed = None
            if seed is not None:
                sub_seed = seed + t_idx * 100 + az_idx
            sim.reset(seed=sub_seed)

    def step(self) -> dict:
        """
        Step all pool simulators.

        Returns:
            Dict of (type_idx, az_idx) → (price, interr_prob, interrupted)
        """
        results = {}
        for key, sim in self.sims.items():
            price, interr_prob, interrupted = sim.step()
            results[key] = (price, interr_prob, interrupted)
        return results

    def get_pool_price(self, type_idx: int, az_idx: int) -> float:
        """Get current spot price for a pool."""
        return self.sims[(type_idx, az_idx)].current_price

    def get_pool_interrupt_prob(self, type_idx: int, az_idx: int) -> float:
        """Get current interruption probability for a pool."""
        return self.sims[(type_idx, az_idx)].interruption_prob

    def get_all_prices(self) -> np.ndarray:
        """Get price matrix (n_types, n_azs)."""
        prices = np.zeros((self.n_types, self.n_azs))
        for (t, az), sim in self.sims.items():
            prices[t, az] = sim.current_price
        return prices

    def get_all_interrupt_probs(self) -> np.ndarray:
        """Get interrupt prob matrix (n_types, n_azs)."""
        probs = np.zeros((self.n_types, self.n_azs))
        for (t, az), sim in self.sims.items():
            probs[t, az] = sim.interruption_prob
        return probs
