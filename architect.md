# Spot RL Optimization — System Architecture

Tài liệu này mô tả kiến trúc tổng thể của hệ thống tối ưu hóa AWS Spot Instance bằng học máy tăng cường. Mục đích: cung cấp cái nhìn end-to-end từ dữ liệu thô → huấn luyện → triển khai → vận hành.

---

## 1. Tổng quan kiến trúc

Hệ thống được chia thành **5 lớp** vận hành độc lập nhưng phối hợp qua các artifact (model checkpoint, ONNX, MLflow runs, DVC data):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Lớp 1: DATA                                                             │
│  raw spot prices  →  synthetic scenarios  →  processed CSV (DVC tracked) │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Lớp 2: SIMULATION                                                       │
│  MarketSimulator + WorkloadGenerator + CostCalculator                    │
│        ↓                                                                 │
│  SpotOrchestratorEnv  ←  wrapped by  ←  MixedScenarioEnv                 │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Lớp 3: TRAINING                                                         │
│  DQN Agent (Factored Dueling + Double DQN + PER)                         │
│      ↑                                                                   │
│      └─ trained on RunPod GPU / Kaggle / Jenkins (Spot worker)           │
│      → checkpoint .pth + ONNX export → MLflow Tracker                    │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Lớp 4: EVALUATION                                                       │
│  evaluate_kaggle.py: DQN vs 6 baselines × 4 scenarios × 30 episodes      │
│  → eval_summary.json + comparison plots                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Lớp 5: DEPLOYMENT                                                       │
│  Terraform AWS infrastructure                                            │
│  → Controller (EC2 t3.small) loads ONNX, polls SQS, manages Spot fleet   │
└──────────────────────────────────────────────────────────────────────────┘
```

```

---

## 3. Lớp 1 — Data Pipeline

### 3.1 Nguồn dữ liệu

| Loại | Mô tả | Tracked by |
|------|-------|------------|
| Raw spot prices | AWS Spot price history (`describe-spot-price-history`) | DVC + S3 |
| Synthetic scenarios | 4 kịch bản tổng hợp 1 tuần | DVC |
| Workload pattern | Poisson arrivals theo hour/day pattern | Sinh on-the-fly |

### 3.2 Scenarios

| Scenario | Đặc trưng | File |
|----------|----------|------|
| **stable** | Giá ổn định, ít spike | `multipool_stable.csv` |
| **volatile** | Biến động cao, $\sigma_{price}$ lớn | `multipool_volatile.csv` |
| **spike** | Đột biến giá tức thì (3-5x) | `multipool_spike.csv` |
| **az_divergence** | Giá phân kỳ giữa các AZ | `multipool_az_divergence.csv` |

### 3.3 Schema CSV

```
timestamp, instance_type, availability_zone, spot_price, interruption_prob, capacity_depth
2024-01-01 00:00, m5.large, ap-southeast-1a, 0.042, 0.018, 0.85
...
```

---

## 4. Lớp 2 — Simulation

### 4.1 Sơ đồ tương tác

```
┌─────────────────┐     prices       ┌──────────────────────┐
│ MarketSimulator │─────────────────▶│                      │
└─────────────────┘                  │                      │
                                     │                      │
┌─────────────────┐     jobs         │  SpotOrchestratorEnv │
│WorkloadGenerator│─────────────────▶│  (gym.Env)           │
└─────────────────┘                  │                      │
                                     │  - reset()           │
┌─────────────────┐     cost         │  - step(action)      │
│ CostCalculator  │◀────────────────▶│  - get_action_mask() │
└─────────────────┘                  └──────────────────────┘
                                              ▲
                                              │ wraps + rotates
                                     ┌────────┴─────────┐
                                     │ MixedScenarioEnv │
                                     └──────────────────┘
```

### 4.2 State space (92 chiều)

| Group | Dim | Mô tả |
|-------|----:|-------|
| Top-3 pool features | 30 | 3 pool rẻ nhất × 10 đặc trưng (price, interrupt, vCPU/$, AZ, SPS, ...) |
| Multi-AZ signals | 3 | Spread, cheapest AZ, concentration |
| Multi-type signals | 3 | Price rank, interrupt rank, vCPU ratio |
| Infrastructure | 3 | #Spot, #OD, total capacity |
| Workload | 6 | pending, running, forecast, queue wait, avg CPU/RAM demand |
| Time | 3 | hour, day, episode progress |
| Current metrics | 5 | avg price, avg interrupt, spot ratio, cost rate, SLA health |
| Extended context | 9 | budget, idle ratio, trend, interrupt streak, ... |
| Per-pool CPU util | 15 | 1 chiều mỗi pool |
| Per-pool RAM util | 15 | 1 chiều mỗi pool |
| **Total** | **92** | |

### 4.3 Action space (121 hành động)

```
8 operations × 15 pools = 120 pool-targeted actions
                       + 1 HOLD (no-op)
                       = 121 discrete actions
```

8 operations: `PROVISION_SPOT`, `PROVISION_ONDEMAND`, `RELEASE_SPOT`, `RELEASE_ONDEMAND`, `CONVERT_TO_ONDEMAND`, `CONVERT_TO_SPOT`, `REBALANCE_SPOT`, `RESERVE_CAPACITY`.

### 4.4 Reward function

```
R_t = savings_t + Σ bonus_i − Σ penalty_j

savings_t = baseline_OD_cost_t − actual_cost_t
```

Coefficient v41 (xem `envs/spot_orchestrator_env.py:130-170`):

| Component | Value |
|-----------|------:|
| SLA penalty (per failed job) | −3.50 |
| Rebalance penalty | −0.80 |
| Provision Spot bonus | +0.80 |
| Empty fleet / Convert-to-Spot penalty | −0.50 |
| SLA bonus (SLA ≥ 0.93) | +0.50 |
| Convert-to-OD penalty | −0.40 |
| Over-provision penalty | −0.30 |
| Interrupt penalty | −0.10 |
| Concentration penalty | −0.05 |
| Idle fleet penalty | −0.02 |
| Fleet size penalty | −0.01 |

### 4.5 Action masking

Khi `env.get_action_mask()` trả về mask, các action không hợp lệ được set Q-value = `-inf` trước khi argmax:
- `RELEASE_SPOT` nếu pool không có Spot instance
- `PROVISION_*` nếu đã đạt `MAX_INSTANCES=20` hoặc `MAX_PER_AZ=10`
- `CONVERT_*` nếu loại instance tương ứng không có

---

## 5. Lớp 3 — Training

### 5.1 DQN Agent components

```
agents/dqn_agent.py
└── DQNAgent
    ├── select_action(obs, training, action_mask) → int
    ├── store_transition(s, a, r, s', done, mask') → None
    ├── train_step() → loss
    ├── save(path), load(path)
    └── components:
        ├── online_net:  FactoredDuelingQNetwork  ──┐
        ├── target_net:  FactoredDuelingQNetwork ←──┤ soft update
        ├── optimizer:   AdamW                       │
        ├── buffer:      PrioritizedReplayBuffer (PER)
        └── epsilon:     1.0 → 0.08 (exponential decay)
```

### 5.2 Network: Factored Dueling DQN

```
                    ┌──────────────────────────┐
                    │  Input: state ∈ ℝ⁹²      │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   Shared trunk (MLP)     │
                    │   92 → 512 → 512 (ReLU)  │
                    └────────────┬─────────────┘
                                 │
        ┌────────────┬───────────┼───────────┬────────────┐
        ▼            ▼           ▼           ▼            ▼
   ┌─────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  V(s)   │  │A_op(s,·)│ │A_pool   │ │A_hold   │ │ (heads) │
   │ scalar  │  │  ∈ ℝ⁸   │ │  ∈ ℝ¹⁵  │ │  scalar │ │         │
   └─────────┘  └─────────┘ └─────────┘ └─────────┘ └─────────┘

Q(s, op, pool) = V(s) + A_op(s, op) + A_pool(s, pool)
Q(s, HOLD)     = V(s) + A_hold(s)
```

So với DQN naive (1 head × 121 outputs), factored network giảm số tham số và cho phép chia sẻ tri thức giữa các pool/op tương tự.

### 5.3 Training loop

```python
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    for step in range(MAX_STEPS):
        mask = env.get_action_mask()
        action = agent.select_action(obs, training=True, action_mask=mask)  # ε-greedy
        next_obs, reward, term, trunc, info = env.step(action)
        next_mask = env.get_action_mask()
        agent.store_transition(obs, action, reward, next_obs, done, next_mask)

        if global_env_steps >= WARMUP:                  # warmup = 512 × 4
            loss = agent.train_step()                   # 1 gradient step
            if global_grad_steps % TARGET_UPDATE == 0:  # default 24k
                agent.update_target()
        obs = next_obs
```

### 5.4 Hyperparameters (v41 default)

| Param | Value |
|-------|------:|
| Hidden dim | 512 |
| Learning rate | 3e-5 |
| Batch size | 512 |
| Buffer size | 300k transitions |
| Gamma (discount) | 0.99 |
| Epsilon start → end | 1.0 → 0.08 |
| Epsilon decay | 1.35M env steps |
| Target update | every 24k grad steps |
| PER alpha / beta start | 0.5 / 0.5 |
| Loss | Huber |
| Optimizer | AdamW |

### 5.5 Compute platforms

| Platform | Use case | Notebook |
|----------|----------|----------|
| **RunPod RTX2000-Ada** | Main training (cloud GPU) | `notebooks/runpod_train.ipynb` |
| **Kaggle GPU** | Backup training (free tier) | `notebooks/kaggle_train.ipynb` |
| **Jenkins Spot worker** | CI/CD demo runs (50 ep resume) | `Jenkinsfile.demo` |
| **Local CPU** | Smoke tests, debugging | `smoke_v42_train.py` |

### 5.6 MLflow tracking

Mỗi run log vào `mlruns/`:
- **Params**: hyperparameters từ config
- **Metrics**: `train/reward`, `train/sla`, `train/cost`, `train/loss`, `eval/sla_{scenario}`, `eval/cost_{scenario}`, `eval/reward`, `epsilon`, `env_steps`, `grad_steps`
- **Artifacts**: `best.pth`, `last.pth`, `config.yaml`, `actions_log.csv`

---

## 6. Lớp 4 — Evaluation

### 6.1 Pipeline đánh giá

```
experiments/evaluate_kaggle.py
│
├── Load checkpoint best_mixed_v41.pth
├── For scenario in [stable, volatile, spike, az_divergence]:
│   For agent in [DQN, OnDemand, Spot, Threshold, CheapestAZ, CheapestType, Random]:
│     For ep in range(30):
│       Roll out 1 episode → log (reward, cost, sla, savings)
│
├── Aggregate → eval_summary.json
└── Plots:
    ├── comparison_all_scenarios.png  (cost + SLA bars)
    ├── savings_heatmap.png           (savings % heatmap)
    └── dqn_action_dist.png           (action histogram)
```

### 6.2 Baselines

| Baseline | Logic | Code |
|----------|-------|------|
| **OnDemand** | Luôn provision OD theo demand | `baselines.OnDemandStrategy` |
| **Spot** | Luôn provision Spot, không fallback | `baselines.SpotPureStrategy` |
| **Threshold** | Spot khi giá < ngưỡng, OD khi vượt | `baselines.ThresholdStrategy` |
| **CheapestAZ** | Spot tại AZ rẻ nhất hiện tại | `baselines.CheapestAZStrategy` |
| **CheapestType** | Spot tại type rẻ nhất hiện tại | `baselines.CheapestTypeStrategy` |
| **Random** | Action ngẫu nhiên (mask) | `baselines.RandomStrategy` |

### 6.3 Metrics

```python
savings_vs_od_pct = (cost_OD - cost_agent) / cost_OD × 100
sla_compliance    = jobs_completed_on_time / total_jobs
migration_count   = #CONVERT_* + #REBALANCE_*
hold_ratio        = #HOLD / total_steps
```

### 6.4 Kết quả v41 (30 episodes × 4 scenarios)

| Strategy | Avg Cost | Avg SLA | Avg Savings |
|----------|---------:|--------:|------------:|
| **DQN** | **$400** | **94.9%** | **37.1%** |
| OnDemand | $636 | 89.1% | 0% |
| Spot Pure | $98 | 39.6% | 84.5% |
| Threshold | $187 | 66.6% | 70.6% |
| CheapestAZ | $202 | 70.6% | 68.3% |
| CheapestType | $208 | 72.4% | 67.3% |
| Random | $496 | 75.4% | 22.0% |

---

## 7. Lớp 5 — Deployment

### 7.1 AWS Infrastructure (Terraform)

```
                          ┌──────────────────────┐
                          │  AWS Budget (alarm)  │
                          └──────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │   VPC ap-southeast-1│
                          │   3 public subnets  │
                          │   (1 mỗi AZ)        │
                          └──┬───────────────┬──┘
                             │               │
                ┌────────────▼──────┐        │
                │ Controller EC2    │        │
                │ t2.micro          │        │
                │                   │        │
                │ ┌───────────────┐ │        │
                │ │ Go controller │ │        │
                │ │ + ONNX model  │ │        │
                │ └───────┬───────┘ │        │
                │         │         │        │
                │  ┌──────▼──────┐  │        │
                │  │ poll SQS    │  │        │
                │  │ → inference │  │        │
                │  │ → AWS API   │  │        │
                │  └─────────────┘  │        │
                └──────┬────────────┘        │
                       │                     │
            ┌──────────▼───────┐    ┌────────▼─────────┐
            │ SQS workload     │    │ Spot fleet       │
            │ queue            │    │ (managed by      │
            │                  │    │  controller)     │
            └──────────────────┘    └──────────────────┘
                       ▲
                       │ push jobs
            ┌──────────┴───────┐
            │ scripts/         │
            │ trigger_workload │
            └──────────────────┘
```

### 7.2 Controller flow

Controller (Go binary, repo riêng `spot-rl-controller`) chạy trong container trên EC2 t3.small:

```
1. Boot: pull Docker image, load ONNX model
2. Loop every step_minutes (5 phút):
   a. Build state vector từ AWS API
        - DescribeSpotPriceHistory (giá hiện tại)
        - DescribeInstances (fleet hiện tại)
        - GetQueueAttributes (#jobs pending từ SQS)
        - GetSpotPlacementScores (SPS)
   b. Apply action mask (kiểm tra capacity, cooldown)
   c. ONNX inference → action ∈ [0..120]
   d. Translate action → AWS API call:
        - PROVISION_SPOT     → RunInstances --market-type spot
        - PROVISION_OD       → RunInstances
        - RELEASE_*          → TerminateInstances
        - CONVERT_*          → Terminate old + Run new
        - REBALANCE_SPOT     → Terminate + Run in target pool
        - HOLD               → no-op
3. Log decision → CloudWatch / S3
```

### 7.3 Safety guards

| Guard | Mô tả | Where |
|-------|-------|-------|
| Budget cap | AWS Budget alert tại $X/tháng | `Terraform/budget.tf` |
| Fleet cap | `MAX_INSTANCES=20`, `MAX_PER_AZ=10` | `instance_catalog.py` |
| Cooldown | Mỗi pool locked 2-12 steps sau PROVISION/CONVERT/REBALANCE | `spot_orchestrator_env.py` |
| Action mask | Loại bỏ action không hợp lệ trước inference | Controller mask layer |
| Shadow mode | (optional) Chỉ log quyết định, không execute | Controller flag |
| Kill switch | Manual stop qua SSM Run Command | Ops procedure |

---

## 8. CI/CD — Jenkins Pipeline

```
Jenkinsfile (production):
┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 1. Setup     │→ │ 2. DVC   │→ │ 3. Env   │→ │ 4. Train │→ │ 5. ONNX  │→ │ 6. Push  │
│ pip install  │  │ Pull S3  │  │ smoke    │  │ resume   │  │ export   │  │ artifact │
│ venv         │  │ data+ckpt│  │ test     │  │ +50 ep   │  │          │  │ DVC      │
└──────────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                                    │
                                                    └─▶ MLflow run logged
```

Agent label: `spot-worker` (Jenkins agent chạy trên AWS Spot, ~$0.02/h).

---

## 9. Local development stack (docker-compose)

```
docker-compose up:
├── api          (port 8000)  — FastAPI serve inference
├── dashboard    (port 8501)  — Streamlit dashboard.py
└── mlflow       (port 5000)  — MLflow tracking UI (sqlite backend)
```

Dùng cho demo + so sánh runs trước khi deploy.

---

## 10. Data flow tổng kết

```
                ┌─────────────────────────────────────────────────────────┐
                │                    DEVELOPMENT                          │
                │                                                         │
       (DVC)    │   raw spot prices ──▶ preprocess ──▶ scenario CSVs      │
                │                                          │              │
                │                                          ▼              │
                │   Gym Env (SpotOrchestratorEnv + MixedScenarioEnv)      │
                │                          │                              │
                │                          ▼                              │
                │   ┌──────────────────────────────────────────┐          │
                │   │   DQN training loop (RunPod / Kaggle)    │          │
                │   │   - Factored Dueling DQN + Double + PER  │          │
                │   │   - logged to MLflow                     │          │
                │   └──────────────────┬───────────────────────┘          │
                │                      ▼                                  │
                │            .pth checkpoint                              │
                │                      │                                  │
                │                      ▼                                  │
                │            evaluate_kaggle.py                           │
                │            (DQN vs 6 baselines × 4 scenarios)           │
                │                      │                                  │
                │                      ▼                                  │
                │            ONNX export (scripts/export_model.py)        │
                └──────────────────────┼──────────────────────────────────┘
                                       │
                                       ▼  (Docker image + ONNX)
                ┌────────────────────────────────────────────────────────┐
                │                    PRODUCTION (AWS)                     │
                │                                                         │
                │   Terraform apply ──▶ VPC + EC2 controller + SQS        │
                │                              │                          │
                │                              ▼                          │
                │   Controller (Go) loads ONNX, polls SQS every 5min      │
                │   ↓                                                     │
                │   AWS APIs: DescribeSpotPriceHistory, DescribeInstances │
                │   ↓                                                     │
                │   state vector ──▶ ONNX inference ──▶ action            │
                │   ↓                                                     │
                │   RunInstances / TerminateInstances on Spot fleet       │
                │   ↓                                                     │
                │   metrics → CloudWatch + S3                             │
                └────────────────────────────────────────────────────────┘
```

---

## 11. Phiên bản & lịch sử quan trọng

| Version | Thay đổi | File |
|---------|----------|------|
| v34 | sla_penalty 2.0 baseline | spot_orchestrator_env.py |
| v35 | sla_penalty 2.0→3.5, provision_spot_bonus 0.4→0.8, sla_bonus_threshold 0.95→0.93 | |
| v39 | fleet_size_penalty 0.03→0.01, empty_fleet capped by pending=10 | |
| **v41** | **Current stable**: 37.1% savings, 94.9% SLA, model `best_mixed_v41.pth` | |
| v42 (WIP) | step_minutes=5 (thay vì 15), 2016 steps/episode, train trên RunPod | `feat/step-5min` branch |

---

## 12. References

- **Code entry points**: `experiments/train.py`, `experiments/evaluate_kaggle.py`, `scripts/export_model.py`
- **Configs**: `experiments/configs/*.yaml`
- **Main env class**: `envs/spot_orchestrator_env.py`
- **Controller repo** (separate): `spot-rl-controller` (Go)
- **Report (LaTeX)**: `Học_máy_tăng_cường_cho_các_hệ_thống_mạng___NT549_Q21/`
- **Diagrams (drawio)**: `architecture.drawio`, `excalidraw.excalidraw`
