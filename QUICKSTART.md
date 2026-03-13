# Quick Start Guide

Hướng dẫn nhanh để bắt đầu project Spot RL.

## 1. Setup môi trường (5 phút)

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt
```

## 2. Thu thập dữ liệu (15-30 phút)

### Option A: Dùng dữ liệu thực từ AWS (khuyên dùng)

**Yêu cầu:** AWS account + boto3 credentials

```bash
# Configure AWS credentials (chỉ làm 1 lần)
aws configure

# Fetch spot price history (30 ngày)
python data/scripts/fetch_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --output data/raw/spot_prices/
```

### Option B: Generate synthetic data (nhanh hơn, cho testing)

```bash
# Generate synthetic spot prices (stable scenario)
python data/scripts/generate_synthetic_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --volatility 0.10 \
    --spike-prob 0.005 \
    --spike-multiplier 1.5 \
    --tag stable \
    --output data/raw/spot_prices/
```

### Preprocess data

```bash
python data/scripts/preprocess.py \
    --input data/raw/ \
    --input-glob "spot_prices/*stable*.csv" \
    --output data/processed/ \
    --output-name price_features_stable \
    --instance-type m5.large
```

## 3. Test Gym environment (5 phút)

```bash
# Test với random policy
python -c "
import sys
sys.path.append('.')
from envs.spot_env import SpotInstanceEnv

env = SpotInstanceEnv(
    data_path='data/processed/price_features_stable.pkl',
    max_steps=100
)

obs, info = env.reset()
print('Initial state:', obs)
print('Initial info:', info)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step {i}: action={action}, reward={reward:.2f}, cost={info.get(\"cost\", 0):.2f}')

    if terminated or truncated:
        break

print('Test passed!')
"
```

## 4. Train DQN agent (1-2 giờ)

```bash
# Training với config mặc định
python experiments/train.py \
    --config experiments/configs/dqn_default.yaml \
    --experiment-name dqn_first_run

# Monitor training (mở terminal khác)
tensorboard --logdir results/dqn_first_run/
# Mở browser: http://localhost:6006
```

**Lưu ý:** Training 5000 episodes có thể mất 1-2 giờ. Để test nhanh, sửa `num_episodes: 100` trong config.

## 5. Evaluate agent (5 phút)

```bash
python experiments/evaluate.py \
    --model results/models/dqn_first_run_best.pth \
    --config experiments/configs/stable_price.yaml \
    --episodes 100 \
    --seeds 5 \
    --output-dir results
```

## 6. Compare với baselines (10 phút)

```bash
python experiments/compare_baselines.py \
    --dqn-model results/models/dqn_first_run_best.pth \
    --scenarios stable,volatile,spike \
    --episodes 50 \
    --output-dir results

python experiments/generate_report.py --results-dir results
```

---

## Troubleshooting

### Lỗi: boto3 không tìm thấy credentials

```bash
# Configure AWS credentials
aws configure
# Nhập: Access Key ID, Secret Access Key, Region (us-east-1)
```

### Lỗi: CUDA out of memory

Sửa trong `experiments/configs/dqn_default.yaml`:
```yaml
agent:
  batch_size: 32  # Giảm từ 64
```

Hoặc force dùng CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
# hoặc: set CUDA_VISIBLE_DEVICES=  # Windows
```

### Training quá chậm

- Giảm `num_episodes` trong config (test với 100-500 episodes trước)
- Giảm `max_steps_per_episode` (test với 500 steps)
- Dùng synthetic data thay vì real data

### Environment không chạy

- Kiểm tra file data có tồn tại: `ls -la data/processed/`
- Kiểm tra format dữ liệu trong `data/processed/price_features.pkl`
- Đọc error message chi tiết và debug step by step

---

## Next Steps

Sau khi chạy xong Quick Start, cậu có thể:

1. **Tune hyperparameters**: Thử các learning rate, epsilon decay khác nhau
2. **Implement missing TODOs**: Hoàn thiện các phần TODO trong code
3. **Add scenarios**: Tạo configs cho volatile price, workload spike
4. **Implement baselines**: Hoàn thiện các baseline agents
5. **Visualization**: Tạo plots so sánh DQN vs baselines
6. **Write report**: Phân tích kết quả, viết báo cáo

Đọc [README.md](README.md) để biết chi tiết hơn!
