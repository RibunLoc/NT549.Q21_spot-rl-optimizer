# Tối ưu chi phí AWS Spot Instance bằng Deep Reinforcement Learning

## Dự án này làm gì?

Khi chạy batch workload trên AWS, bạn có thể chọn:

| | EC2 On-Demand | EC2 Spot Instance |
|---|---|---|
| **Giá** | Cố định (ví dụ $0.096/hr cho m5.large) | Rẻ hơn 60–90%, biến động theo thị trường |
| **Rủi ro** | Không có | AWS có thể **thu hồi bất kỳ lúc nào** (interruption) |
| **Phù hợp** | Workload quan trọng, cần ổn định | Batch jobs, có thể tolerate interruption |

**Vấn đề:** Dùng toàn On-Demand thì đắt. Dùng toàn Spot thì hay bị gián đoạn. Thực tế có **5 loại instance** và **3 Availability Zone** — tức là 15 pools khác nhau, mỗi pool có giá và xác suất interruption riêng.

**Giải pháp:** Train một **DQN agent** để tự động quyết định:
- Loại instance nào nên dùng (m5.large, c5.xlarge, r5.large, m5.xlarge, c5.2xlarge)
- AZ nào có giá rẻ nhất và ít interruption nhất
- Khi nào thêm/bớt instance, khi nào migrate
- Cân bằng giữa **giảm chi phí** và **đảm bảo SLA ≥ 95%**

---

## Bài toán RL (MDP Formulation)

### State — 33 features (normalized [0, 1])

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TOP-3 CHEAPEST POOLS (12)   │  MULTI-AZ AGG (6)  │  INFRA (6)             │
│  ────────────────────        │  ──────────────     │  ──────                │
│  Mỗi pool: price_ratio,      │  AZ price spread    │  Total spot / OD       │
│  interrupt_prob,             │  Cheapest AZ id     │  Total vCPU            │
│  vCPU/$, az_id               │  AZ concentration   │                        │
│                              │  Best type rank     │                        │
│                              │  Worst interr rank  │                        │
│                              │  Best vCPU/$        │                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  WORKLOAD (4)                │  TIME (3)           │  CURRENT STATE (5)     │
│  ────────────                │  ──────             │  ──────────────        │
│  Pending jobs                │  Hour of day        │  Avg spot price        │
│  Running jobs                │  Day of week        │  Avg interrupt prob    │
│  Workload forecast           │  Episode progress   │  Spot ratio            │
│  Avg queue wait              │                     │  Cost rate             │
│                              │                     │  SLA health (10-step)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Action — 105 discrete actions

```
Action = op × type × az  (7 ops × 5 types × 3 AZs = 105)
```

| # | Operation | Ý nghĩa |
|---|-----------|---------|
| 0 | `REQUEST_SPOT` | Thêm 1 spot instance vào pool (type, az) |
| 1 | `REQUEST_ONDEMAND` | Thêm 1 on-demand instance vào pool (type, az) |
| 2 | `TERMINATE_SPOT` | Bỏ 1 spot instance khỏi pool |
| 3 | `TERMINATE_ONDEMAND` | Bỏ 1 on-demand instance khỏi pool |
| 4 | `MIGRATE_TO_ONDEMAND` | Chuyển spot → OD trong pool (tránh interrupt) |
| 5 | `MIGRATE_TO_SPOT` | Chuyển OD → spot trong pool (tiết kiệm) |
| 6 | `DO_NOTHING` | Giữ nguyên trạng thái |

**Instance types:**

| Type | vCPU | OD Price ($/hr) |
|------|------|-----------------|
| m5.large | 2 | $0.096 |
| c5.xlarge | 4 | $0.170 |
| r5.large | 2 | $0.126 |
| m5.xlarge | 4 | $0.192 |
| c5.2xlarge | 8 | $0.340 |

**Availability Zones:** ap-southeast-1a, ap-southeast-1b, ap-southeast-1c

### Reward — Savings-based

```
R = savings − sla_penalty − interrupt_penalty − migration_penalty − concentration_penalty − pending_penalty

  savings              = Σ (OD_price × instances) − Σ (spot_price × instances)   ← khuyến khích dùng Spot
  sla_penalty          = failed_jobs × 10.0                                       ← phạt nặng job fail
  interrupt_penalty    = interruptions × 0.5                                      ← bị AWS thu hồi
  migration_penalty    = migrations × 1.0                                         ← migrate tốn overhead
  concentration_penalty = nếu >80% instances trong 1 AZ                          ← khuyến khích đa AZ
  pending_penalty      = f(queue_length / capacity)                               ← queue dài = không đủ tài nguyên
```

Clipped vào `[-10, +10]` mỗi step để ổn định training.

---

## Kiến trúc hệ thống

```
┌────────────────────────────────────────────────────────────────┐
│                   DQN Agent (PyTorch)                          │
│                                                                │
│   Q-Network ──────► action ◄────── ε-greedy exploration       │
│   (MLP: 33 → 512 → 256 → 128 → 105)                          │
│                                                                │
│   Target Network     Replay Buffer (200K transitions)          │
│   (sync mỗi 500 steps)   Huber Loss + grad clip               │
└─────────────────────────┬──────────────────────────────────────┘
                          │ (op, type, az) decoded from action index
                          ▼
┌────────────────────────────────────────────────────────────────┐
│           SpotOrchestratorEnv (Gymnasium, 33-dim, 105 actions) │
│                                                                │
│   MultiPoolMarketSimulator ── 5 types × 3 AZs price replay    │
│   WorkloadGenerator ───────── Poisson arrival, daily profile  │
│   PoolState tracking ──────── spot/OD count per (type, az)    │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                 Dữ liệu Spot Price (Synthetic)                  │
│                                                                │
│   4 scenarios: stable, volatile, spike, az_divergence          │
│   5 types × 3 AZs = 15 pools × ~720 timesteps mỗi file        │
└────────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
spot-rl-optimiztion/
│
├── agents/
│   ├── dqn_agent.py            # DQN (ε-greedy, experience replay, target net)
│   ├── networks.py             # QNetwork, DuelingQNetwork, BranchingQNetwork, LSTMQNetwork
│   ├── replay_buffer.py        # Experience replay buffer
│   └── baselines.py            # 6 baselines: OnDemand, Spot, Threshold,
│                               #   CheapestAZ, CheapestType, Random
│
├── envs/
│   ├── spot_orchestrator_env.py # SpotOrchestratorEnv — multi-pool env chính
│   ├── instance_catalog.py      # 5 types × 3 AZs, constants, encode/decode action
│   ├── market_simulator.py      # MultiPoolMarketSimulator (15 pools)
│   ├── workload_generator.py    # WorkloadGenerator (Poisson, daily profile, spikes)
│   ├── spot_env.py              # SpotInstanceEnv (single-pool, legacy)
│   └── cost_calculator.py       # CostCalculator utility
│
├── experiments/
│   ├── train.py                 # Script training: auto-dispatch single/multi_pool
│   └── configs/
│       ├── multi_pool_stable.yaml       # Giá ổn định
│       ├── multi_pool_volatile.yaml     # Giá biến động mạnh
│       ├── multi_pool_spike.yaml        # Workload đột biến
│       ├── multi_pool_az_divergence.yaml# Giá lệch mạnh giữa các AZ
│       ├── dqn_default.yaml             # Single-pool default
│       ├── stable_price.yaml            # Single-pool stable
│       ├── volatile_price.yaml          # Single-pool volatile
│       └── workload_spike.yaml          # Single-pool spike
│
├── data/
│   ├── processed/               # Dữ liệu đã xử lý (.pkl, .csv)
│   │   ├── multipool_stable.pkl
│   │   ├── multipool_volatile.pkl
│   │   ├── multipool_spike.pkl
│   │   ├── multipool_az_divergence.pkl
│   │   └── price_features_*.pkl (single-pool)
│   └── scripts/
│       ├── generate_synthetic_spot_prices.py
│       └── preprocess.py
│
├── utils/
│   ├── logger.py               # setup_logger, TensorBoardLogger
│   ├── metrics.py              # MetricsTracker
│   ├── mlflow_logger.py        # MLflow integration
│   └── visualization.py       # Plotting
│
├── results/                   # Training outputs (gitignored binaries)
│   └── models/                # best_model.pth, final_model.pth per experiment
│
├── notebooks/
│   ├── visualize_training.ipynb
│   └── dqn_spot_demo.ipynb
│
├── app.py
├── dashboard.py
├── docker-compose.yml
└── requirements.txt
```

---

## Hướng dẫn sử dụng

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Generate dữ liệu

```bash
# Tạo synthetic spot price data cho 4 scenarios
python data/scripts/generate_synthetic_spot_prices.py

# Preprocess → features (multi-pool)
python data/scripts/preprocess.py --mode multi_pool
```

### 3. Training

```bash
cd experiments

# Multi-pool (khuyến nghị)
python train.py \
    --config configs/multi_pool_stable.yaml \
    --experiment-name mp_stable

python train.py \
    --config configs/multi_pool_volatile.yaml \
    --experiment-name mp_volatile

# Monitor bằng TensorBoard
tensorboard --logdir ../results/mp_stable/

# Monitor bằng MLflow
mlflow ui --backend-store-uri ../mlruns/
```

### 4. Resume training

```bash
python train.py \
    --config configs/multi_pool_stable.yaml \
    --experiment-name mp_stable_resume \
    --resume ../results/models/mp_stable_best.pth \
    --resume-episode 1000
```

---

## Kịch bản thử nghiệm

| Scenario | Data | Đặc điểm | Mục tiêu agent |
|----------|------|----------|----------------|
| **Stable** | `multipool_stable.pkl` | Giá biến động thấp (~±10%) | Tối đa dùng Spot, tiết kiệm nhiều nhất |
| **Volatile** | `multipool_volatile.pkl` | Giá spike 2–4x, interrupt cao | Biết khi nào rút về On-Demand |
| **Spike** | `multipool_spike.pkl` | Workload đột biến 4–5x | Scale nhanh, không để job fail |
| **AZ Divergence** | `multipool_az_divergence.pkl` | Giá lệch lớn giữa AZ | Học chọn đúng AZ rẻ nhất |

---

## So sánh với Baselines

| Strategy | Mô tả | Chi phí (stable) | SLA | Nhận xét |
|----------|--------|------------------|-----|----------|
| **OnDemand** | Luôn dùng OD, round-robin type/AZ | Cao nhất (~$422/168 steps) | ~100% | Không rủi ro, tốn tiền |
| **Spot** | Luôn dùng Spot, round-robin | Thấp (~$100) | ~96% | Rẻ nhưng rủi ro interrupt |
| **Threshold** | Spot khi price_ratio < 0.5 | Thấp (~$96) | ~98% | Đơn giản, hiệu quả |
| **CheapestAZ** | Spot tại AZ rẻ nhất | Thấp (~$110) | ~98% | Khai thác thông tin AZ |
| **CheapestType** | Spot với type rẻ nhất, đổi AZ | Thấp (~$94) | ~96% | Tập trung 1 type |
| **Random** | Ngẫu nhiên | ~$194 | ~98% | Sanity check |
| **DQN Agent** | Học từ data | **Thấp + ổn định** | **≥95%** | Cân bằng tốt nhất |

---

## Hyperparameters (multi-pool)

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| State dim | 33 | 33 features normalized [0,1] |
| Action dim | 105 | 7 ops × 5 types × 3 AZs |
| Learning rate | 0.0003 | Adam optimizer |
| Gamma (γ) | 0.99 | Discount factor |
| Epsilon | 1.0 → 0.01 | Exploration decay |
| Epsilon decay | 80,000 steps | ~25 episodes × 168 steps |
| Batch size | 128 | Số transitions mỗi update |
| Replay buffer | 200,000 | Lưu nhiều hơn do action space lớn |
| Target update | 500 steps | Sync target network |
| Max steps/ep | 168 | 1 tuần (hourly timestep) |
| Episodes | 3,000 | Tổng training |

---

## Tech Stack

- **Python** 3.12
- **PyTorch** — DQN, neural networks
- **Gymnasium** — RL environment
- **pandas / numpy** — Data processing
- **MLflow** — Experiment tracking
- **TensorBoard** — Training monitoring
- **matplotlib** — Visualization
- **YAML** — Configuration

---

## Metrics đánh giá

| Metric | Mô tả | Mục tiêu |
|--------|--------|----------|
| **Episode Reward** | Tổng reward mỗi episode | Tối đa |
| **Total Cost** | Chi phí ($) mỗi episode 168 steps | Tối thiểu |
| **SLA Compliance** | % jobs hoàn thành không fail | ≥ 95% |
| **Spot Usage** | % capacity là spot | Cao = tiết kiệm |
| **Cost Savings** | % tiết kiệm so với toàn On-Demand | Mục tiêu 40–70% |
| **Interruptions** | Số lần spot bị AWS thu hồi | Tối thiểu |

---

## Tham khảo

**Papers:**
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Kurth-Nelson et al. — Branching Dueling Q-Network (BDQ)
- SpotOn (NSDI 2020) — On-Demand Spot Instances
- Tributary (ATC 2018) — Elastic Services trên Spot

**AWS:**
- [EC2 Spot Instances Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)

**Tutorials:**
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Custom Gymnasium Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
