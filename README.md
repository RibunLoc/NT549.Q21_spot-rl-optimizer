# Tối ưu chi phí AWS Spot Instance bằng Deep Reinforcement Learning

## Dự án này làm gì?

Khi chạy workload trên AWS, bạn có 2 lựa chọn:

| | EC2 On-Demand | EC2 Spot Instance |
|---|---|---|
| **Giá** | Cố định ($0.096/hr cho m5.large) | Rẻ hơn 60-90%, nhưng biến động theo thị trường |
| **Rủi ro** | Không có | AWS có thể **thu hồi bất kỳ lúc nào** (interruption) |
| **Phù hợp** | Workload quan trọng, cần ổn định | Batch jobs, có thể retry |

**Vấn đề:** Dùng toàn On-Demand thì đắt. Dùng toàn Spot thì hay bị gián đoạn. Làm sao chọn đúng lúc dùng Spot, đúng lúc dùng On-Demand?

**Giải pháp:** Train một **DQN agent** (Deep Q-Network) để tự động quyết định:
- Khi nào thêm/bớt Spot instance
- Khi nào thêm/bớt On-Demand instance
- Khi nào migrate workload từ Spot sang On-Demand (tránh bị interrupt)
- Khi nào giữ nguyên

Agent học từ dữ liệu giá Spot thực tế của AWS và tối ưu đồng thời **giảm chi phí** và **đảm bảo SLA** (>95% jobs hoàn thành đúng hạn).

---

## Kết quả đã đạt được

### Các lần training đã chạy

| Run | Scenario | Episodes | Mô tả |
|-----|----------|----------|--------|
| `dqn_first_run` | Stable price | 3000 | Lần chạy đầu tiên, giá ổn định |
| `dqn_v2` | Stable price | 3000 | Tuning hyperparameters |
| `dqn_v3_stable` | Stable price | 3000 | Version ổn định nhất |
| `dqn_v4_volatile` | Volatile price | — | Test với giá biến động mạnh |

### Models đã train

```
results/models/
├── dqn_v3_stable_best.pth     # Best model trên stable price
├── dqn_v4_volatile_best.pth   # Best model trên volatile price
├── dqn_v2_best.pth            # Version 2
├── dqn_first_run_best.pth     # Lần chạy đầu
└── training_results.json      # Kết quả tổng hợp
```

### Plots đã generate

```
results/plots/
├── action_distribution_dqn_default.png   # Agent chọn action nào nhiều nhất
├── cost_sla_comparison_stable.png        # So sánh DQN vs baselines
└── cost_sla_dqn_default.png              # Cost vs SLA trade-off
```

---

## Bài toán RL (MDP Formulation)

### State — Agent nhìn thấy gì? (15 features)

```
┌─────────────────────────────────────────────────────────┐
│  MARKET (5)          │  INFRA (3)       │  WORKLOAD (4)  │  TIME (3)          │
│  ─────────           │  ─────────       │  ─────────     │  ─────────         │
│  Giá spot hiện tại   │  Số spot inst    │  Jobs đang chờ │  Giờ trong ngày    │
│  MA giá 6h           │  Số OD inst      │  Jobs đang chạy│  Ngày trong tuần   │
│  MA giá 24h          │  Tổng capacity   │  Dự báo load   │  % episode đã qua  │
│  Độ biến động giá    │                  │  Thời gian chờ │                    │
│  Xác suất interrupt  │                  │                │                    │
└─────────────────────────────────────────────────────────┘
```

Tất cả được normalize về [0, 1].

### Action — Agent có thể làm gì? (7 actions)

| # | Action | Ý nghĩa | Mỗi lần thay đổi |
|---|--------|---------|-------------------|
| 0 | `REQUEST_SPOT` | Thêm spot instances | +3 instances |
| 1 | `REQUEST_ONDEMAND` | Thêm on-demand instances | +3 instances |
| 2 | `TERMINATE_SPOT` | Giảm spot instances | -3 instances |
| 3 | `TERMINATE_ONDEMAND` | Giảm on-demand instances | -3 instances |
| 4 | `MIGRATE_TO_ONDEMAND` | Chuyển spot → on-demand (tránh interrupt) | 3 instances, $1/inst |
| 5 | `MIGRATE_TO_SPOT` | Chuyển on-demand → spot (tiết kiệm chi phí) | 3 instances, $0.5/inst |
| 6 | `DO_NOTHING` | Giữ nguyên | — |

### Reward — Agent được thưởng/phạt thế nào?

```
reward = savings - sla_penalty - migration_penalty - interruption_penalty - pending_penalty

│  savings            =  (chi phí nếu dùng toàn OD) - (chi phí thực tế)    → khuyến khích dùng Spot
│  sla_penalty        =  số job fail × $10 × (1 + tỷ lệ vi phạm)          → phạt nặng nếu SLA < 95%
│  migration_penalty  =  migrate→OD: $1/inst, migrate→Spot: $0.5/inst       → migrate tốn tiền
│  interruption_penalty = số lần bị interrupt × $5                          → bị AWS thu hồi
│  pending_penalty    =  số job đang chờ × $0.1                             → queue dài = chậm
```

**Mục tiêu:** Agent cần cân bằng giữa tiết kiệm (dùng Spot) và an toàn (dùng On-Demand).

---

## Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────┐
│                   DQN Agent (PyTorch)                     │
│                                                          │
│   Q-Network ──────► Action ◄────── ε-greedy exploration  │
│   (MLP: 15→256→128→6)                                    │
│                                                          │
│   Target Network    Replay Buffer (100K transitions)     │
│   (sync mỗi 500 steps)                                   │
└─────────────┬────────────────────────────────────────────┘
              │ action
              ▼
┌──────────────────────────────────────────────────────────┐
│              SpotInstanceEnv (Gymnasium)                   │
│                                                          │
│   Market Simulator ─── replay giá spot thực từ AWS       │
│   Workload Generator ─ Poisson arrival, peak hours       │
│   Job Scheduler ────── FIFO queue, deadline-based SLA    │
│   Cost Calculator ──── spot/OD pricing, penalties        │
└─────────────┬────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│                 Dữ liệu AWS Spot Price                    │
│                                                          │
│   ap-southeast-1, us-east-1, us-west-2                   │
│   m5.large — 90 ngày lịch sử (12/2025 → 03/2026)        │
│   + synthetic scenarios (stable, volatile, spike)        │
└──────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
spot-rl-optimiztion/
│
├── agents/                         # RL Agents
│   ├── dqn_agent.py                # DQN agent chính (ε-greedy, experience replay)
│   ├── networks.py                 # Neural networks (MLP, LSTM, Dueling)
│   ├── replay_buffer.py            # Experience replay buffer
│   └── baselines.py                # 4 baselines để so sánh
│
├── envs/                           # Gymnasium Environment
│   ├── spot_env.py                 # SpotInstanceEnv — environment chính
│   ├── market_simulator.py         # Replay giá spot + tính interruption probability
│   ├── workload_generator.py       # Sinh workload theo Poisson process
│   └── cost_calculator.py          # Tính chi phí spot/OD và penalties
│
├── experiments/                    # Training & Evaluation
│   ├── train.py                    # Script training chính
│   ├── evaluate.py                 # Đánh giá model
│   ├── compare_baselines.py        # So sánh DQN vs baselines
│   └── configs/                    # YAML configs cho từng scenario
│       ├── dqn_default.yaml        # Config chính (3000 ep, stable price)
│       ├── dqn_quick_test.yaml     # Test nhanh (100 ep)
│       ├── stable_price.yaml       # Scenario giá ổn định
│       ├── volatile_price.yaml     # Scenario giá biến động
│       └── workload_spike.yaml     # Scenario workload đột biến
│
├── data/
│   ├── raw/spot_prices/            # CSV giá spot thực từ AWS (3 regions)
│   ├── processed/                  # Features đã xử lý (.pkl)
│   │   ├── price_features_stable.pkl
│   │   ├── price_features_volatile.pkl
│   │   └── price_features_spike.pkl
│   └── scripts/                    # Scripts thu thập & xử lý data
│       ├── fetch_spot_prices.py    # Crawl giá từ AWS API (boto3)
│       └── preprocess.py           # Feature engineering
│
├── utils/                          # Tiện ích
│   ├── config.py                   # Đọc YAML config
│   ├── logger.py                   # Logging + TensorBoard
│   ├── metrics.py                  # MetricsTracker (save/load metrics.pkl)
│   └── visualization.py           # Plotting functions
│
├── results/                        # Kết quả training
│   ├── dqn_first_run/             # Run 1: metrics.pkl + checkpoints + logs
│   ├── dqn_v2/                    # Run 2
│   ├── dqn_v3_stable/             # Run 3 (stable scenario)
│   ├── dqn_v4_volatile/           # Run 4 (volatile scenario)
│   ├── models/                    # Best & final models (.pth)
│   ├── plots/                     # Biểu đồ PNG
│   └── reports/                   # Báo cáo
│
├── notebooks/
│   └── visualize_training.ipynb   # Notebook phân tích kết quả training
│
├── app.py                         # Web interface
├── dashboard.py                   # Dashboard visualization
├── Makefile                       # Build commands
├── Dockerfile                     # Container
├── docker-compose.yml
└── requirements.txt               # Dependencies
```

---

## Hướng dẫn sử dụng

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu

```bash
# Lấy giá spot thực từ AWS (cần AWS credentials)
python data/scripts/fetch_spot_prices.py \
    --region ap-southeast-1 \
    --instance-types m5.large \
    --days 30 \
    --output data/raw/spot_prices/

# Xử lý data → features
python data/scripts/preprocess.py \
    --input data/raw/ \
    --input-glob "spot_prices/*stable*.csv" \
    --output data/processed/ \
    --output-name price_features_stable \
    --instance-type m5.large
```

### 3. Training

```bash
# Train DQN agent (3000 episodes, ~vài giờ)
python experiments/train.py \
    --config experiments/configs/dqn_default.yaml \
    --experiment-name dqn_my_run

# Test nhanh (100 episodes)
python experiments/train.py \
    --config experiments/configs/dqn_quick_test.yaml \
    --experiment-name quick_test

# Monitor bằng TensorBoard
tensorboard --logdir results/dqn_my_run/
```

### 4. Đánh giá & So sánh

```bash
# Evaluate model
python experiments/evaluate.py \
    --model results/models/dqn_my_run_best.pth \
    --config experiments/configs/stable_price.yaml \
    --episodes 100

# So sánh với baselines
python experiments/compare_baselines.py \
    --dqn-model results/models/dqn_my_run_best.pth \
    --scenarios stable,volatile,spike \
    --output-dir results
```

### 5. Xem kết quả

Mở notebook `notebooks/visualize_training.ipynb` để xem:
- Learning curves (reward, cost, SLA)
- So sánh các lần chạy
- Action distribution
- Cost vs SLA trade-off

---

## Kịch bản thử nghiệm

### Scenario 1: Giá ổn định (Stable)
- Giá spot biến động thấp (~±10%)
- Agent nên tối đa dùng Spot → tiết kiệm nhiều
- Config: `experiments/configs/stable_price.yaml`

### Scenario 2: Giá biến động mạnh (Volatile)
- Giá spike 2-4x đột ngột, interruption probability cao
- Agent cần học khi nào rút về On-Demand
- Config: `experiments/configs/volatile_price.yaml`

### Scenario 3: Workload đột biến (Spike)
- Workload tăng 3-5x vào giờ cao điểm
- Agent cần đảm bảo đủ capacity, không để job fail
- Config: `experiments/configs/workload_spike.yaml`

---

## So sánh với Baselines

| Strategy | Mô tả | Chi phí | SLA | Rủi ro |
|----------|--------|---------|-----|--------|
| **Always On-Demand** | Chỉ dùng OD | Cao nhất | ~100% | Không |
| **Always Spot** | Chỉ dùng Spot | Thấp nhất | 70-80% | Cao |
| **Threshold** | Dùng Spot khi giá < ngưỡng | Trung bình | 85-90% | Trung bình |
| **Random** | Chọn ngẫu nhiên | Ngẫu nhiên | ~50% | Cao |
| **DQN Agent** | Học từ data | **Thấp** | **>95%** | **Thấp** |

---

## Hyperparameters chính

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| Learning rate | 0.0003 | Tốc độ học của Q-network |
| Gamma (γ) | 0.99 | Discount factor — agent nhìn xa |
| Epsilon | 1.0 → 0.01 | Exploration: ban đầu random, dần dần exploit |
| Epsilon decay | 50,000 steps | Tốc độ giảm exploration |
| Batch size | 128 | Số transitions mỗi lần update |
| Replay buffer | 100,000 | Lưu bao nhiêu experience |
| Target update | 500 steps | Sync target network mỗi 500 steps |
| Episodes | 3,000 | Số episode training |
| Max steps/ep | 500 | Số bước tối đa mỗi episode |

---

## Tech Stack

- **Python** 3.12
- **PyTorch** — DQN agent, neural networks
- **Gymnasium** — RL environment
- **pandas / numpy** — Data processing
- **matplotlib** — Visualization
- **TensorBoard** — Training monitoring
- **boto3** — AWS Spot Price API
- **YAML** — Configuration

---

## Metrics đánh giá

| Metric | Mô tả | Mục tiêu |
|--------|--------|----------|
| **Episode Reward** | Tổng reward mỗi episode | Càng cao càng tốt |
| **Episode Cost** | Chi phí ($) mỗi episode | Càng thấp càng tốt |
| **SLA Compliance** | % jobs hoàn thành đúng hạn | ≥ 95% |
| **Spot Usage** | % dùng spot vs tổng | Cao = tiết kiệm |
| **Cost Savings** | % tiết kiệm so với toàn On-Demand | Mục tiêu 20-40% |

---

## Tham khảo

**Papers:**
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- SpotOn (NSDI 2020) — On-Demand Spot Instances
- Tributary (ATC 2018) — Elastic Services trên Spot

**AWS:**
- [EC2 Spot Instances Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)

**Tutorials:**
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Custom Gymnasium Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
