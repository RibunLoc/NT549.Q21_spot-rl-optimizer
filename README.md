# Tối ưu chiến lược sử dụng Spot Instance bằng Học tăng cường (Deep RL)

## Tổng quan

Dự án nghiên cứu áp dụng Deep Reinforcement Learning (DQN) để tối ưu hóa chiến lược sử dụng AWS EC2 Spot Instance, giảm chi phí cloud 20-40% trong khi vẫn đảm bảo availability cho batch processing workloads.

**Mục tiêu chính:**
- Giảm chi phí cloud so với on-demand instances
- Thích nghi với biến động giá spot theo thời gian thực
- Đảm bảo SLA: >95% tỷ lệ hoàn thành jobs
- Giảm thiểu ảnh hưởng từ spot interruption

**Kết quả dự kiến:**
- 20-40% tiết kiệm chi phí vs always on-demand
- Agent học được pattern giá theo giờ/ngày
- Tự động migrate khi xác suất interruption cao
- SLA compliance >95%

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                     DQN Agent (PyTorch)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Q-Network   │  │ Target       │  │ Replay Buffer    │  │
│  │ (MLP/LSTM)  │  │ Network      │  │ (Experience)     │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │ Action: [Spot/OnDemand/Terminate/Migrate]
             ↓
┌─────────────────────────────────────────────────────────────┐
│         SpotInstanceEnv (Gymnasium Environment)             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ State:       │  │ Reward:      │  │ Dynamics:       │  │
│  │ - spot price │  │ - cost saved │  │ - price model   │  │
│  │ - workload   │  │ - SLA penalty│  │ - interruption  │  │
│  │ - instances  │  │ - migration  │  │ - job scheduler │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│            Data Sources (AWS Spot History)                  │
│  • Historical spot pricing (boto3 API)                      │
│  • Interruption frequency per AZ/instance-type              │
│  • Synthetic workload patterns (batch jobs)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
spot-rl-optimiztion/
├── README.md                  # File này
├── requirements.txt           # Dependencies
├── .gitignore
│
├── data/                      # Dữ liệu và datasets
│   ├── raw/                   # Dữ liệu thô từ AWS API
│   │   ├── spot_prices/       # Historical spot prices (CSV/parquet)
│   │   └── interruptions/     # Interruption frequency data
│   ├── processed/             # Dữ liệu đã xử lý
│   │   ├── price_features_*.pkl # Engineered features (rolling avg, volatility)
│   │   └── workload_traces.pkl# Workload patterns
│   └── scripts/               # Scripts thu thập dữ liệu
│       ├── fetch_spot_prices.py   # Crawl AWS spot price history
│       ├── generate_workload.py   # Generate synthetic workload
│       └── preprocess.py          # Data cleaning & feature engineering
│
├── envs/                      # Gymnasium environments
│   ├── __init__.py
│   ├── spot_env.py            # Main SpotInstanceEnv (MDP formulation)
│   ├── market_simulator.py   # Spot price simulator (time-series model)
│   ├── workload_generator.py # Batch job arrival & execution model
│   └── cost_calculator.py    # Chi phí spot/on-demand, SLA penalty
│
├── agents/                    # RL agents
│   ├── __init__.py
│   ├── dqn_agent.py           # DQN implementation (PyTorch)
│   ├── networks.py            # Neural network architectures (MLP, LSTM)
│   ├── replay_buffer.py       # Experience replay buffer
│   └── baselines.py           # Baseline strategies (always on-demand, threshold)
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── config.py              # Config parser (YAML)
│   ├── logger.py              # Logging & TensorBoard
│   ├── metrics.py             # Evaluation metrics (cost, availability, SLA)
│   └── visualization.py       # Plotting (price curves, action heatmaps)
│
├── experiments/               # Experiment scripts
│   ├── configs/               # YAML configs cho các thí nghiệm
│   │   ├── dqn_default.yaml
│   │   ├── stable_price.yaml  # Scenario: giá ổn định
│   │   ├── volatile_price.yaml# Scenario: giá biến động mạnh
│   │   └── workload_spike.yaml# Scenario: workload đột biến
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── compare_baselines.py  # So sánh với baselines
│   └── generate_report.py    # Sinh báo cáo tĩnh
│
├── results/                   # Kết quả thực nghiệm
│   ├── models/                # Trained models (.pth)
│   ├── logs/                  # Training logs (TensorBoard)
│   ├── plots/                 # Biểu đồ (PNG/PDF)
│   └── reports/               # Báo cáo chi tiết (Markdown/LaTeX)
│
└── notebooks/                 # Jupyter notebooks (EDA, demo)
    ├── 01_data_exploration.ipynb    # Phân tích dữ liệu spot price
    ├── 02_env_testing.ipynb         # Test Gym environment
    ├── 03_training_analysis.ipynb   # Phân tích quá trình training
    └── 04_results_visualization.ipynb # Visualize kết quả
```

---

## Phương pháp luận

### 1. Mô hình hóa bài toán (MDP)

#### State Space (Observation)
```python
state = {
    # Market information
    'current_spot_price': float,           # Giá spot hiện tại ($/hour)
    'spot_price_ma_1h': float,             # Moving avg 1 hour
    'spot_price_ma_24h': float,            # Moving avg 24 hours
    'price_volatility': float,             # Độ biến động giá (std dev)
    'interruption_prob': float,            # Xác suất bị interrupt (0-1)

    # Infrastructure state
    'num_spot_instances': int,             # Số spot instances đang chạy
    'num_ondemand_instances': int,         # Số on-demand instances
    'total_capacity': int,                 # Tổng capacity (vCPUs)

    # Workload state
    'pending_jobs': int,                   # Jobs đang chờ
    'running_jobs': int,                   # Jobs đang chạy
    'workload_forecast_1h': float,         # Dự đoán workload 1h tới
    'queue_wait_time': float,              # Thời gian chờ trung bình (minutes)

    # Time features
    'hour_of_day': int,                    # 0-23 (giá thường cao giờ cao điểm)
    'day_of_week': int,                    # 0-6 (pattern khác cuối tuần)
}
```

#### Action Space (Discrete)
```python
actions = {
    0: 'REQUEST_SPOT',        # Request thêm spot instance
    1: 'REQUEST_ONDEMAND',    # Request on-demand instance
    2: 'TERMINATE_SPOT',      # Terminate spot instance
    3: 'TERMINATE_ONDEMAND',  # Terminate on-demand
    4: 'MIGRATE_TO_ONDEMAND', # Migrate job từ spot sang on-demand
    5: 'DO_NOTHING',          # Giữ nguyên
}
```

#### Reward Function
```python
reward = (
    - cost_incurred              # Âm: chi phí spot/on-demand trong timestep
    + cost_saved_vs_ondemand     # Dương: tiền tiết kiệm được
    - sla_penalty                # Âm: phạt khi vi phạm SLA (job timeout)
    - migration_cost             # Âm: chi phí migrate (downtime)
    - interruption_penalty       # Âm: phạt khi spot bị interrupt
)

# Ví dụ:
# - On-demand m5.large: $0.096/hour
# - Spot m5.large: $0.030/hour (average)
# - SLA penalty: -$10 per failed job
# - Migration cost: -$1 per migration
```

### 2. Deep Q-Network (DQN)

**Thuật toán:** DQN với Experience Replay và Target Network

**Kiến trúc neural network:**
```python
class QNetwork(nn.Module):
    # Input: state vector (dimension ~15)
    # Hidden: 2-3 fully connected layers (256, 128 units)
    # Output: Q-values cho 6 actions
    # Activation: ReLU
    # Optional: LSTM layer để capture temporal dependencies
```

**Hyperparameters:**
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Epsilon (exploration): 1.0 → 0.01 (decay over 100k steps)
- Replay buffer size: 100k transitions
- Batch size: 64
- Target network update frequency: 1000 steps

### 3. Baselines để so sánh

1. **Always On-Demand**: Chỉ dùng on-demand (chi phí cao nhất, no risk)
2. **Always Spot**: Chỉ dùng spot (rẻ nhưng rủi ro cao)
3. **Threshold-based**: Request spot khi giá < threshold (e.g., 30% on-demand price)
4. **Random**: Random action (sanity check)

---

## Hướng dẫn sử dụng

### 1. Setup môi trường

```bash
# Clone repo (nếu có git)
git clone <repo-url>
cd spot-rl-optimiztion

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu AWS Spot Price

**Yêu cầu:** AWS account + boto3 credentials configured (`~/.aws/credentials`)

```bash
# Fetch spot price history (30 ngày gần nhất)
python data/scripts/fetch_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --output data/raw/spot_prices/

# Generate synthetic price scenarios (stable/volatile/spike)
python data/scripts/generate_synthetic_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --volatility 0.10 \
    --spike-prob 0.005 \
    --spike-multiplier 1.5 \
    --tag stable \
    --output data/raw/spot_prices/

python data/scripts/generate_synthetic_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --volatility 0.25 \
    --spike-prob 0.05 \
    --spike-multiplier 4.0 \
    --tag volatile \
    --output data/raw/spot_prices/

python data/scripts/generate_synthetic_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --volatility 0.18 \
    --spike-prob 0.08 \
    --spike-multiplier 5.0 \
    --tag spike \
    --output data/raw/spot_prices/

# Preprocess data
python data/scripts/preprocess.py \
    --input data/raw/ \
    --input-glob "spot_prices/*stable*.csv" \
    --output data/processed/ \
    --output-name price_features_stable \
    --instance-type m5.large

python data/scripts/preprocess.py \
    --input data/raw/ \
    --input-glob "spot_prices/*volatile*.csv" \
    --output data/processed/ \
    --output-name price_features_volatile \
    --instance-type m5.large

python data/scripts/preprocess.py \
    --input data/raw/ \
    --input-glob "spot_prices/*spike*.csv" \
    --output data/processed/ \
    --output-name price_features_spike \
    --instance-type m5.large
```

### 3. Test Gym environment

```bash
# Chạy random policy để test env
python -c "
import gymnasium as gym
from envs.spot_env import SpotInstanceEnv

env = SpotInstanceEnv(
    data_path='data/processed/price_features_stable.pkl',
    max_steps=1000
)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Action: {action}, Reward: {reward:.2f}, Cost: {info[\"cost\"]:.2f}')
    if terminated or truncated:
        break
"
```

### 4. Training DQN agent

```bash
# Train với config mặc định
python experiments/train.py --config experiments/configs/dqn_default.yaml

# Train với custom config
python experiments/train.py \
    --config experiments/configs/volatile_price.yaml \
    --experiment-name "dqn_volatile_price"

# Best model được export ra:
# results/models/dqn_volatile_price_best.pth
```

**Config file example** (`dqn_default.yaml`):
```yaml
env:
  data_path: "data/processed/price_features_stable.pkl"
  max_steps: 1000
  sla_threshold: 0.95
  workload:
    base_arrival_rate: 2.0
    peak_multiplier: 3.0
    peak_hours: [9, 10, 11, 14, 15, 16]
    avg_job_duration: 10
  cost:
    ondemand_price: 0.096
    sla_penalty: 10.0
    migration_cost: 1.0

agent:
  type: "DQN"
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 100000
  batch_size: 64
  replay_buffer_size: 100000
  target_update_freq: 1000

training:
  num_episodes: 5000
  max_steps_per_episode: 1000
  log_interval: 10
  save_interval: 100
```

### 5. Evaluation và so sánh

```bash
# Evaluate trained model
python experiments/evaluate.py \
    --config experiments/configs/volatile_price.yaml \
    --model results/models/dqn_volatile_price_best.pth \
    --episodes 100 \
    --seeds 5 \
    --output-dir results

# So sánh với baselines
python experiments/compare_baselines.py \
    --dqn-model results/models/dqn_default_best.pth \
    --scenarios stable,volatile,spike \
    --output-dir results

# Sinh báo cáo tĩnh
python experiments/generate_report.py --results-dir results
```

### 6. Visualization

```bash
# Plot training curves
python utils/visualization.py \
    --tensorboard-log results/logs/dqn_default \
    --output results/plots/training_curves.png

# Hoặc dùng Jupyter notebook
jupyter notebook notebooks/04_results_visualization.ipynb
```

---

## Kịch bản thử nghiệm

### Scenario 1: Giá ổn định (Stable Price)
- **Mô tả:** Giá spot biến động thấp (±10% xung quanh mean)
- **Mục tiêu:** Agent nên maximize spot usage
- **Config:** `experiments/configs/stable_price.yaml`

### Scenario 2: Giá biến động mạnh (Volatile Price)
- **Mô tả:** Giá spike 2-3x đột ngột, xác suất interrupt cao
- **Mục tiêu:** Agent học được khi nào switch sang on-demand
- **Config:** `experiments/configs/volatile_price.yaml`

### Scenario 3: Workload spike
- **Mô tả:** Workload đột ngột tăng 5x (giờ cao điểm)
- **Mục tiêu:** Đảm bảo SLA khi demand cao
- **Config:** `experiments/configs/workload_spike.yaml`

---

## Metrics đánh giá

### Cost Metrics
- **Total Cost**: Tổng chi phí trong episode
- **Cost per Job**: Chi phí trung bình/job
- **Savings vs On-Demand**: % tiết kiệm so với baseline
- **Spot Utilization Rate**: % thời gian dùng spot vs on-demand

### Performance Metrics
- **SLA Compliance**: % jobs hoàn thành đúng hạn
- **Job Completion Rate**: % jobs hoàn thành (không timeout)
- **Average Queue Wait Time**: Thời gian chờ trung bình
- **Interruption Impact**: % jobs bị ảnh hưởng bởi spot interrupt

### Agent Metrics
- **Cumulative Reward**: Tổng reward trong episode
- **Q-value Convergence**: Stability của Q-values
- **Action Distribution**: Tần suất các actions được chọn

---

## Deliverables

### Code deliverables
- [x] Gymnasium environment (`envs/spot_env.py`)
- [x] DQN agent implementation (`agents/dqn_agent.py`)
- [x] Data collection scripts (`data/scripts/`)
- [ ] Trained model checkpoints (`results/models/`)
- [x] Baseline implementations (`agents/baselines.py`)

### Documentation
- [x] README.md (file này)
- [ ] API documentation (docstrings + Sphinx)
- [ ] Architecture diagram
- [ ] Experiment report (LaTeX/Markdown)

### Results
- [ ] Training curves (loss, reward, epsilon)
- [ ] Comparison tables (DQN vs baselines)
- [ ] Cost-performance plots
- [ ] Action heatmaps (action distribution over price levels)
- [ ] Case study: specific episodes with annotations

---

## Roadmap & Timeline

### Phase 1: Data & Environment (Tuần 1-2)
- [ ] Crawl AWS spot price history (30-60 ngày)
- [ ] Generate synthetic workload patterns
- [ ] Implement Gymnasium environment
- [ ] Test environment với random policy

### Phase 2: Agent Implementation (Tuần 3-4)
- [ ] Implement DQN agent (PyTorch)
- [ ] Implement baselines (always on-demand, threshold, random)
- [ ] Setup training pipeline & logging

### Phase 3: Experiments (Tuần 5-6)
- [ ] Train DQN trên 3 scenarios (stable, volatile, spike)
- [ ] Hyperparameter tuning
- [ ] Evaluate và so sánh với baselines

### Phase 4: Analysis & Report (Tuần 7-8)
- [ ] Phân tích kết quả (cost savings, SLA compliance)
- [ ] Tạo visualizations (plots, heatmaps)
- [ ] Viết báo cáo cuối kỳ
- [ ] Chuẩn bị slide thuyết trình

---

## Tech Stack

- **Python:** 3.9+
- **RL Framework:** Gymnasium 0.29+
- **Deep Learning:** PyTorch 2.0+
- **Data:** pandas, numpy, boto3 (AWS SDK)
- **Visualization:** matplotlib, seaborn, TensorBoard
- **Config:** YAML, Hydra (optional)

---

## Tài liệu tham khảo

### Papers
1. **DQN:** Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. **Spot Instance Optimization:**
   - "SpotOn: On-Demand Spot Instances" (NSDI 2020)
   - "Tributary: Spot-Dancing for Elastic Services" (ATC 2018)

### AWS Documentation
- [EC2 Spot Instances Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
- [Boto3 EC2 Spot Price API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2.html#EC2.Client.describe_spot_price_history)

### Tutorials
- [Stable Baselines3 DQN Tutorial](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [Custom Gymnasium Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

---

## Liên hệ & Đóng góp

**Tác giả:** [Tên cậu]
**Email:** [Email]
**Giảng viên hướng dẫn:** [Tên giảng viên]
**Seminar:** MLOps - [Tên trường]

**Contribution guidelines:**
- Fork repo và tạo feature branch
- Follow PEP8 coding style
- Add docstrings cho functions/classes
- Test code trước khi commit

---

## License

MIT License (hoặc Academic Use Only - tùy yêu cầu)

---

## Ghi chú kỹ thuật

### Lý do chọn DQN thay vì policy gradient
- Action space discrete, DQN sample-efficient hơn
- Off-policy learning: tái sử dụng experience
- Đơn giản hơn để debug và tune

### Challenges dự kiến
1. **State representation:** Cần feature engineering tốt (price patterns, time features)
2. **Reward shaping:** Cân bằng giữa cost saving và SLA penalty
3. **Non-stationary environment:** Giá spot thay đổi theo thời gian thực
4. **Exploration:** Epsilon-greedy có thể không đủ, cân nhắc thêm noise

### Extensions có thể làm thêm (nếu có thời gian)
- [ ] Multi-region optimization (chọn AZ có giá tốt nhất)
- [ ] LSTM-based Q-network để capture temporal patterns
- [ ] Prioritized Experience Replay
- [ ] Double DQN / Dueling DQN
- [ ] Transfer learning: pre-train trên synthetic data, fine-tune trên real data
