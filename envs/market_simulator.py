"""
Bộ mô phỏng giá thị trường Spot.

Mô phỏng biến động giá spot và các sự kiện gián đoạn dựa trên dữ liệu lịch sử.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SpotMarketSimulator:
    """
    Mô phỏng động lực thị trường spot AWS EC2.

    TODO: Cài đặt mô phỏng giá spot thực tế hơn:
    - Dùng dữ liệu lịch sử để mô hình hóa các mẫu giá
    - Thêm hiệu ứng theo giờ trong ngày và ngày trong tuần
    - Mô phỏng sự kiện gián đoạn dựa trên mức giá
    - Hỗ trợ nhiều loại instance
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        instance_type: str = "m5.large",
        availability_zone: str = "us-east-1a",
        seed: Optional[int] = None,
    ):
        """
        Khởi tạo bộ mô phỏng thị trường.

        Args:
            historical_data: DataFrame với các cột [timestamp, spot_price, instance_type, availability_zone]
            instance_type: Loại EC2 instance
            availability_zone: Vùng khả dụng AWS
            seed: Seed ngẫu nhiên để tái tạo kết quả
        """
        self.historical_data = historical_data
        self.instance_type = instance_type
        self.availability_zone = availability_zone
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

        # Sắp xếp theo timestamp và trích xuất giá
        filtered_data = filtered_data.sort_values('timestamp')
        self.prices = filtered_data['spot_price'].values
        self.timestamps = filtered_data['timestamp'].values

        # Trạng thái hiện tại
        self.current_timestep = 0
        self.current_price = 0.0
        self.interruption_prob = 0.0
        self.price_history = []  # Lưu lịch sử giá để tính thống kê

    def reset(self, seed: Optional[int] = None):
        """Đặt lại bộ mô phỏng về trạng thái ban đầu."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_timestep = 0
        self.price_history = []

        # Đặt giá ban đầu từ dữ liệu lịch sử
        if len(self.prices) > 0:
            self.current_price = float(self.prices[0])
            self.price_history.append(self.current_price)
        else:
            self.current_price = 0.03  # Giá dự phòng

        # Xác suất gián đoạn ban đầu (thấp)
        self.interruption_prob = 0.05

    def step(self) -> Tuple[float, float, bool]:
        """
        Tiến thị trường thêm một timestep.

        Returns:
            current_price: Giá spot tại timestep hiện tại ($/giờ)
            interruption_prob: Xác suất bị gián đoạn (0-1)
            interrupted: Có xảy ra gián đoạn không
        """
        # Phát lại giá lịch sử (lặp vòng nếu đến cuối)
        price_index = self.current_timestep % len(self.prices)
        self.current_price = float(self.prices[price_index])
        self.price_history.append(self.current_price)

        # Tính xác suất gián đoạn dựa trên biến động giá
        # Giá cao hơn so với trung bình gần đây = rủi ro gián đoạn cao hơn
        if len(self.price_history) >= 24:
            recent_avg = np.mean(self.price_history[-24:])
            price_ratio = self.current_price / recent_avg if recent_avg > 0 else 1.0

            # Xác suất gián đoạn tăng khi giá cao
            self.interruption_prob = min(0.3, max(0.01, (price_ratio - 1.0) * 0.5))
        else:
            self.interruption_prob = 0.05  # Mặc định 5%

        # Lấy mẫu sự kiện gián đoạn
        interrupted = self.rng.random() < self.interruption_prob

        self.current_timestep += 1

        return self.current_price, self.interruption_prob, interrupted

    def get_price_statistics(self, window: int = 24) -> dict:
        """
        Lấy thống kê giá cho window timestep vừa qua.

        Returns:
            dict với các khóa: mean_6h, mean_24h, volatility, trend
        """
        history = self.price_history[-window:] if len(self.price_history) > 0 else [self.current_price]

        mean_price = np.mean(history)
        volatility = np.std(history) if len(history) > 1 else 0.0

        # Tính xu hướng (đơn giản: so sánh nửa đầu với nửa sau)
        if len(history) >= 4:
            mid = len(history) // 2
            first_half_mean = np.mean(history[:mid])
            second_half_mean = np.mean(history[mid:])
            trend = (second_half_mean - first_half_mean) / first_half_mean if first_half_mean > 0 else 0.0
            trend = np.clip(trend, -1.0, 1.0)  # Chuẩn hóa về [-1, 1]
        else:
            trend = 0.0

        return {
            "mean_6h": np.mean(self.price_history[-6:]) if len(self.price_history) >= 6 else np.mean(self.price_history) if len(self.price_history) > 0 else self.current_price,
            "mean_24h": mean_price,
            "volatility": volatility,
            "trend": trend,
        }

    def get_interruption_frequency(self, lookback: int = 168) -> float:
        """
        Lấy tần suất gián đoạn lịch sử (trong lookback giờ vừa qua).

        Args:
            lookback: Số giờ nhìn lại (mặc định 168 = 1 tuần)

        Returns:
            Tần suất gián đoạn (0-1)
        """
        # TODO: Tính toán từ dữ liệu lịch sử
        return 0.1  # Placeholder: tỷ lệ gián đoạn 10%
