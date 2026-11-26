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
# Test multi-pool env với random policy
python -c "
import sys
sys.path.append('.')
from envs.spot_orchestrator_env import SpotOrchestratorEnv

env = SpotOrchestratorEnv(
    data_path='data/processed/multipool_stable.csv',
    max_steps=100,
)

obs, info = env.reset()
print('State dim:', obs.shape)

for i in range(10):
    mask = env.get_action_mask()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step {i}: action={action}, reward={reward:.2f}, cost={info.get(\"cost\", 0):.2f}')
    if terminated or truncated:
        break

print('Test passed!')
"
```

## 4. Train DQN agent

Training thực tế chạy trên notebook (Kaggle / RunPod) — xem `notebooks/kaggle_train.ipynb`
hoặc `notebooks/runpod_train.ipynb`. Để train nhanh local qua CLI:

```bash
python experiments/train.py \
    --config experiments/configs/multi_pool_stable.yaml \
    --experiment-name mp_stable
```

## 5. Evaluate agent + so sánh baselines (10 phút)

```bash
# Đánh giá DQN + 6 baseline trên 4 scenario
python experiments/evaluate_kaggle.py \
    --model results/kaggle/best_eval.pth \
    --episodes 30 \
    --output-dir results/kaggle/eval

# Sinh report Markdown
python experiments/generate_report.py --results-dir results/kaggle/eval
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

Sửa trong `experiments/configs/multi_pool_stable.yaml`:
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
