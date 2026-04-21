# Spot RL Optimization -- Tối ưu Chi Phí AWS Spot Instance bằng Deep RL

Tối ưu chi phí AWS Spot Instance bằng một **Generalist DQN Agent** được train trên 4 market scenarios khác nhau.

## Kết quả

| Agent | Cost (stable) | Cost (volatile) | Cost (spike) | Cost (az_div) | SLA |
|---|---|---|---|---|---|
| **DQN (ours)** | **$209** | **$196** | **$204** | **$202** | **~98.9%** |
| OnDemand | $575 | $575 | $575 | $575 | 96.4% |
| Spot (pure) | $148 | $129 | $129 | $139 | 85-92% |
| Threshold | $156 | $147 | $153 | $154 | 90-91% |
| Random | $394 | $408 | $399 | $391 | 83-84% |

**DQN tiết kiệm ~65% chi phí so với OnDemand trong khi vẫn đảm bảo SLA >= 95%.**

---

## Bài toán RL (MDP Formulation)

### State -- 72 features (normalized [0, 1])

```
+------------------------------------------------------------------+
|  TOP-3 CHEAPEST POOLS (12)  |  POOL STATS (45)                  |
|  price_ratio, interrupt,    |  Mỗi pool trong 15 pools:          |
|  vcpu_per_dollar, az_id     |  spot_price, od_price,            |
|                             |  interrupt_rate, vcpu_per_dollar  |
+------------------------------------------------------------------+
|  INFRA (3)                  |  WORKLOAD (4)                     |
|  num_spot, num_od,          |  pending_jobs, running_jobs,      |
|  total_vcpu                 |  forecast_1h, queue_wait          |
+------------------------------------------------------------------+
|  TIME (3)                   |  CURRENT (5)                      |
|  hour_of_day, day_of_week,  |  avg_spot_price, avg_interrupt,   |
|  episode_progress           |  spot_ratio, cost_rate, sla_health|
+------------------------------------------------------------------+
```

### Action -- 121 discrete actions

```
Action = op x pool + HOLD   (8 ops x 15 pools + 1 HOLD = 121)
```

| # | Operation | Mô tả |
|---|-----------|-------|
| 0 | PROVISION_SPOT | Thêm 1 spot instance vào pool (type, az) |
| 1 | PROVISION_ONDEMAND | Thêm 1 on-demand instance vào pool |
| 2 | RELEASE_SPOT | Bỏ 1 spot instance khỏi pool |
| 3 | RELEASE_ONDEMAND | Bỏ 1 on-demand instance khỏi pool |
| 4 | CONVERT_TO_ONDEMAND | Chuyển spot -> OD (tránh interrupt) |
| 5 | CONVERT_TO_SPOT | Chuyển OD -> spot (tiết kiệm) |
| 6 | REBALANCE_SPOT | Rebalance spot across AZs |
| 7 | RESERVE_CAPACITY | Reserve capacity cho workload spike |
| 120 | HOLD | Giữ nguyên trạng thái |

**15 pools = 5 instance types x 3 Availability Zones:**

| Type | vCPU | OD Price ($/hr) |
|------|------|-----------------|
| m5.large | 2 | $0.096 |
| c5.xlarge | 4 | $0.170 |
| r5.large | 2 | $0.126 |
| m5.xlarge | 4 | $0.192 |
| c5.2xlarge | 8 | $0.340 |

AZs: ap-southeast-1a, ap-southeast-1b, ap-southeast-1c

### Reward -- Savings-based

```
R = savings - sla_penalty - interrupt_penalty - migration_penalty - concentration_penalty

savings            = baseline_ondemand_cost - actual_cost
sla_penalty        = failed_jobs x 10.0
interrupt_penalty  = interruptions x 0.5
migration_penalty  = migrations x 1.0
concentration_penalty = max(0, az_concentration - 0.8) x 5.0
```

---

## Kiến trúc Agent

### Factored Dueling DQN

```
State (72-dim)
      |
   Trunk (512 -> 256)
      |
  +---+---+----------+
  |       |          |
op_head  pool_head  hold_head
(8)      (15)       (1)
  |       |          |
  +---+---+----------+
        |
  Q(s,a) = op_score[op] + pool_score[pool]   for pool actions
  Q(s,a) = hold_score                         for HOLD
        |
   121 Q-values
```

**236K parameters** -- nhỏ nhưng hiệu quả nhờ cấu trúc factored.

### Training Setup

| Parameter | Giá trị |
|-----------|---------|
| State dim | 72 |
| Action dim | 121 |
| Hidden dim | 512 |
| Learning rate | 3e-4 (Adam) |
| Gamma | 0.99 |
| Epsilon | 1.0 -> 0.05 (decay 300K steps) |
| Batch size | 256 |
| Replay buffer | 500K (PER) |
| Target update | 500 steps |
| Max steps/ep | 168 (1 tuần, hourly) |
| Episodes | 5,000 |

**Cải tiến so với vanilla DQN:**
- **Double DQN** -- giảm overestimation bias
- **Dueling Network** -- tách Value vs Advantage stream
- **Prioritized Experience Replay (PER)** -- ưu tiên transition khó
- **Action Masking** -- chỉ cho phép actions hợp lệ
- **Prioritized Scenario Sampling** -- sample scenario theo TD-loss

---

## Mixed Training (Generalist)

Thay vì train 4 specialist models riêng biệt, train **1 generalist agent** trên `MixedScenarioEnv`:

```python
env = MixedScenarioEnv(scenario_paths={
    "stable":        "data/processed/multipool_stable.csv",
    "volatile":      "data/processed/multipool_volatile.csv",
    "spike":         "data/processed/multipool_spike.csv",
    "az_divergence": "data/processed/multipool_az_divergence.csv",
})
# Mỗi episode: sample 1 scenario ngẫu nhiên (có ưu tiên theo TD-loss)
```

**Kết quả:** Agent generalize tốt trên cả 4 scenarios với cost chỉ lệch ~$10 giữa các scenarios.

---

## 4 Training Scenarios

| Scenario | Đặc điểm | Mục tiêu agent |
|----------|----------|----------------|
| **Stable** | Giá biến động thấp (~+-10%) | Tối đa dùng Spot |
| **Volatile** | Giá spike 2-4x, interrupt cao | Biết khi nào rút về OD |
| **Spike** | Workload đột biến 4-5x | Scale nhanh, không để job fail |
| **AZ Divergence** | Giá lệch lớn giữa AZ | Chọn đúng AZ rẻ nhất |

---

## Cấu trúc thư mục

```
spot-rl-optimiztion/
|
+-- agents/
|   +-- dqn_agent.py            # DQN: Double DQN + Dueling + PER + action masking
|   +-- networks.py             # FactoredDuelingQNetwork, DuelingQNetwork
|   +-- replay_buffer.py        # ReplayBuffer + PrioritizedReplayBuffer
|   +-- baselines.py            # 6 baselines: OnDemand, Spot, Threshold,
|                               #   CheapestAZ, CheapestType, Random
|
+-- envs/
|   +-- spot_orchestrator_env.py # SpotOrchestratorEnv -- multi-pool env chính
|   +-- mixed_scenario_env.py    # MixedScenarioEnv -- generalist training wrapper
|   +-- action_schema.py         # N_ACTIONS=121, decode_action, OPERATION_NAMES
|   +-- instance_catalog.py      # 5 types x 3 AZs, STATE_DIM=72, constants
|   +-- workload_generator.py    # WorkloadGenerator (Poisson, daily profile)
|
+-- experiments/
|   +-- evaluate_kaggle.py       # Evaluate DQN vs 6 baselines trên 4 scenarios
|
+-- data/
|   +-- processed/               # 4 x multipool_*.csv (synthetic spot prices)
|   +-- scripts/
|       +-- generate_synthetic_spot_prices.py
|       +-- preprocess.py
|
+-- notebooks/
|   +-- kaggle_train.ipynb       # Training notebook cho Kaggle GPU
|
+-- scripts/
|   +-- rebuild_notebook_clean.py # Rebuild notebook từ scratch
|   +-- fix_notebook_encoding.py
|
+-- results/
|   +-- kaggle/                  # Model checkpoints (gitignored)
|       +-- eval/                # Evaluation results: plots + JSON
|
+-- pack_kaggle.ps1              # Pack zip để upload lên Kaggle Dataset
+-- requirements.txt
```

---

## Hướng dẫn sử dụng

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Generate data

```bash
python data/scripts/generate_synthetic_spot_prices.py
python data/scripts/preprocess.py --mode multi_pool
```

### 3. Train trên Kaggle

```bash
# Pack code + data thành zip
.\pack_kaggle.ps1

# Upload spot_rl_kaggle.zip lên Kaggle Datasets
# Chạy notebooks/kaggle_train.ipynb trên Kaggle GPU T4
```

**Kaggle Secrets cần thêm:**
```
MLFLOW_USERNAME       -- MLflow basic auth username
MLFLOW_PASSWORD       -- MLflow basic auth password
AWS_ACCESS_KEY_ID     -- AWS S3 artifact store
AWS_SECRET_ACCESS_KEY -- AWS S3 artifact store
```

### 4. Evaluate

```bash
python experiments/evaluate_kaggle.py \
    --model results/kaggle/best_mixed_v1.pth \
    --episodes 30 \
    --output-dir results/kaggle/eval
```

Output: `comparison_all_scenarios.png`, `savings_heatmap.png`, `dqn_action_dist.png`, `eval_summary.json`

### 5. Resume training từ checkpoint

Trong `notebooks/kaggle_train.ipynb`, cell Step 5c:
```python
RESUME_CHECKPOINT = '/kaggle/input/spot-rl-models/best_mixed_v5.pth'
```

---

## MLflow Tracking

- **Tracking server:** mlflow.holoc.id.vn
- **Experiment:** spot-rl-generalist
- **Model Registry:** Spot-RL-Agent

Metrics được log mỗi 10 episodes:
- `reward`, `cost`, `sla`, `loss`, `epsilon`
- `cost_stable`, `cost_volatile`, `cost_spike`, `cost_az_divergence`
- `sla_stable`, `sla_volatile`, `sla_spike`, `sla_az_divergence`
- `weight_<scenario>` -- prioritized sampling weights
- System metrics: CPU, RAM, GPU utilization (tự động)

---

## Tech Stack

- **Python** 3.12
- **PyTorch** -- DQN, neural networks
- **Gymnasium** -- RL environment
- **pandas / numpy** -- Data processing
- **MLflow** -- Experiment tracking + Model Registry
- **AWS S3** -- Artifact store
- **Kaggle** -- GPU training (T4)
- **matplotlib** -- Visualization

---

## Metrics đánh giá

| Metric | Mô tả | Mục tiêu |
|--------|--------|----------|
| **Episode Reward** | Tổng reward mỗi episode | Tối đa |
| **Total Cost** | Chi phí ($) mỗi 168 steps | Tối thiểu |
| **SLA Compliance** | % jobs hoàn thành | >= 95% |
| **Cost Savings** | % tiết kiệm vs OnDemand | ~65% |
| **Spot Usage** | % capacity là spot | Cao = tiết kiệm |

---

## Tham khảo

**Papers:**
- Mnih et al. (2015) -- *Human-level control through deep reinforcement learning* (DQN)
- Wang et al. (2016) -- *Dueling Network Architectures for Deep RL*
- Schaul et al. (2016) -- *Prioritized Experience Replay*
- SpotOn (NSDI 2020) -- On-Demand Spot Instances
- Tributary (ATC 2018) -- Elastic Services trên Spot

**AWS:**
- [EC2 Spot Instances Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
