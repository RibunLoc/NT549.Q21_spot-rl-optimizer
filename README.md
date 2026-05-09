# Spot RL Optimization

**Dạy một AI agent biết khi nào nên mua AWS Spot, khi nào mua On-Demand, dùng instance type nào và đặt ở Availability Zone nào — để chi phí thấp nhất mà vẫn hoàn thành công việc đúng hạn.**

---

## 1. Bài toán

AWS bán compute theo 2 kiểu:
- **On-Demand (OD)**: đắt, ổn định, không bao giờ bị cắt.
- **Spot**: rẻ hơn ~50-60%, nhưng AWS có thể "rút lại" bất kỳ lúc nào (interrupt) → job đang chạy bị giết.

Nếu dùng 100% OD thì đắt. Nếu dùng 100% Spot thì rẻ nhưng job hay fail → vi phạm SLA (cam kết % job phải hoàn thành đúng hạn). Đây là bài toán **đánh đổi cost ↔ reliability** dưới môi trường giá biến động.

**Phức tạp hơn nữa:** có 15 "ô chợ" khác nhau = 5 loại instance × 3 Availability Zone. Giá mỗi ô biến động độc lập theo thời gian, và xác suất interrupt phụ thuộc vào capacity depth của từng pool.

→ Không có heuristic đơn giản nào đủ tốt. Dùng **Reinforcement Learning**.

---

## 2. Cách agent hoạt động (Intuition)

Tưởng tượng agent là một "nhân viên vận hành" ngồi trước 15 màn hình giá, mỗi 15 phút ra một quyết định:

> *"Giờ này cluster đang có 8 spot + 2 OD, queue còn 12 job pending, giá m5.large ở AZ-1a đang rất rẻ nhưng capacity_depth thấp — có nên mua thêm không? Hay nên chuyển 2 spot sang OD vì sắp có đợt spike giá?"*

Agent học bằng **thử-sai hàng nghìn episode** trong simulator, dần dần phát hiện pattern:
- Khi giá m5 AZ-1a rẻ hơn 40% OD và interrupt rate thấp → mua spot
- Khi queue wait > 5h và workload spike → mua OD để đảm bảo SLA
- Khi một AZ bị capacity pressure → phân tán sang AZ khác

---

## 3. MDP Formulation

RL chạy trên Markov Decision Process: (State, Action, Reward, Transition).

### State: 90 chiều số

Mỗi bước agent nhận vector `s ∈ ℝ⁹⁰`, tất cả normalize về [0, 1]:

| Nhóm | Chiều | Nội dung |
|---|---|---|
| Top-3 pools rẻ nhất | 30 | price_ratio, interrupt, vcpu/$, az_id, capacity_depth (×3 pools × 10 features) |
| Multi-AZ aggregated | 3 | az_price_spread, cheapest_az_id, az_concentration |
| Multi-Type aggregated | 3 | best_type_rank, worst_interrupt_rank, best_vcpu_ratio |
| Infrastructure | 3 | num_spot, num_ondemand, total_vcpu |
| Workload | 4 | pending_jobs, running_jobs, forecast_1h, queue_wait |
| Time | 3 | hour_of_day, day_of_week, episode_progress |
| Current metrics | 5 | avg_spot_price, avg_interrupt, spot_ratio, cost_rate, sla_health |
| Extended context | 9 | budget_used, idle_ratio, price_trend, interrupt_streak, ... |
| Per-pool CPU util | 15 | utilization mỗi pool (15 pools) |
| Per-pool RAM util | 15 | utilization mỗi pool (15 pools) |

### Action: 121 lựa chọn rời rạc

```
Action = op × 15_pools + pool_id     (op ∈ {0..7}, pool ∈ {0..14})
Action = 120                          (HOLD — giữ nguyên)
→ Tổng: 8 × 15 + 1 = 121
```

| # | Operation | Nghĩa |
|---|---|---|
| 0 | PROVISION_SPOT | Mua 1 spot instance ở (type, AZ) đã chọn |
| 1 | PROVISION_ONDEMAND | Mua 1 on-demand instance |
| 2 | RELEASE_SPOT | Trả 1 spot instance |
| 3 | RELEASE_ONDEMAND | Trả 1 on-demand instance |
| 4 | CONVERT_TO_ONDEMAND | Chuyển spot → OD (né interrupt sắp xảy ra) |
| 5 | CONVERT_TO_SPOT | Chuyển OD → spot (tiết kiệm chi phí) |
| 6 | REBALANCE_SPOT | Phân bổ lại spot giữa các AZ |
| 7 | RESERVE_CAPACITY | Dự trữ OD trước khi workload spike |
| 120 | HOLD | Không làm gì |

### Reward: savings-based

```
R = savings - Σ penalties

savings              = baseline_ondemand_cost - actual_cost
sla_penalty          = số_job_fail    × 3.5   # phạt mỗi job trễ deadline
interrupt_penalty    = số_interrupt   × 0.1   # phạt khi spot bị AWS thu hồi
migration_penalty    = số_migrate     × 0.4   # phạt khi chuyển OD ↔ spot
rebalance_penalty    = số_rebalance   × 0.8   # phạt khi rebalance fleet
concentration_penalty = max(0, az_conc - 0.8) × 0.05  # phạt tập trung 1 AZ
fleet_size_penalty   = max(0, fleet - needed) × 0.01   # phạt instance dư thừa
```

### Transition: mô phỏng thị trường

Sau mỗi action, simulator:
1. Áp dụng thay đổi fleet (mua/bán/chuyển instance)
2. Sample giá spot mới từ dữ liệu (Ornstein-Uhlenbeck mean-reverting process)
3. Sample interrupt event theo capacity_depth của từng pool
4. Schedule job từ queue → instance đang rảnh
5. Tick 15 phút, tính cost & SLA
6. Trả về `(next_state, reward, done, info)`

**1 episode = 168 steps = 1 tuần mô phỏng (15 phút/step).  
5000 episodes ≈ 840K quyết định.**

---

## 4. Kiến trúc Agent: Factored Dueling DQN

### Tại sao không dùng DQN vanilla?

DQN vanilla học `Q(s, a)` với 121 output neurons độc lập. Khi thấy một combination (op, pool) mới, phải học lại từ đầu — không tận dụng được kiến thức "PROVISION_SPOT ở pool rẻ thường tốt cho mọi pool".

### Factored DQN: tách op × pool

```
              State (90-dim)
                   │
            Trunk [512 → 256]          # feature extractor chung
                   │
      ┌────────────┼────────────┐
      │            │            │
  op_head(8)  pool_head(15)  hold_head(1)
      │            │            │
      └────────────┼────────────┘
                   │
   Q(s, action) = op_score[op] + pool_score[pool]
   Q(s, HOLD)   = hold_score
```

→ **24 output neurons** thay vì 121. Chia sẻ kiến thức giữa các op/pool, học nhanh hơn và generalize tốt hơn. **~236K params** tổng.

### Kỹ thuật bổ sung

| Kỹ thuật | Giải quyết gì |
|---|---|
| **Double DQN** | Giảm overestimation bias khi chọn action |
| **Dueling head** | Tách `V(s)` (state value) khỏi `A(s,a)` (action advantage) |
| **Prioritized Experience Replay** | Ưu tiên replay các transition khó (TD-error lớn) |
| **Action Masking** | Vô hiệu hoá action không hợp lệ (vd: RELEASE khi pool trống) |

---

## 5. Data: Synthetic Spot Prices

Dữ liệu giá được **tổng hợp** (synthetic) dựa trên hành vi thực tế của AWS, calibrated từ AWS console:

- **Spot/OD ratio** ~51-56% (c5.2xlarge ap-southeast-1 thực tế: ~55%)
- **Volatility** cực thấp ở stable (~std/mean = 1-2%) — match AWS chart thực
- **AZ spread** ~2-4% (AZ-level independent drift)
- **Spike frequency** realistic: 0-1 spike/90 ngày (stable) đến ~26/90 ngày (spike scenario)

### 4 Scenarios training

| Scenario | Đặc điểm | Bài học |
|---|---|---|
| **Stable** | Giá flat, gần không có spike, AZ spread nhỏ | Tối đa spot usage |
| **Volatile** | Volatility 3-4x stable, spike nhẹ (~0.5%/h) | Khi nào rút về OD |
| **Spike** | Spike dramatic (+120%), ~1 spike/3.5 ngày | Handle price shock, không overreact |
| **AZ Divergence** | AZ prices độc lập (az_correlation=0.25) | Chọn AZ tối ưu liên tục |

### Data pipeline

```bash
# 1. Generate raw data (5 types × 3 AZs × 4 scenarios × 90 ngày × 5min granularity)
python data/scripts/generate_synthetic_spot_prices.py \
    --all-scenarios --days 90 \
    --instance-types m5.large,c5.xlarge,r5.large,m5.xlarge,c5.2xlarge \
    --region ap-southeast-1 --output data/raw/spot_prices/

# 2. Preprocess → multi-pool features (15 pools, 8640 steps/pool/scenario)
for scn in stable volatile spike az_divergence; do
    python data/scripts/preprocess.py \
        --input data/raw/ --output data/processed/ \
        --input-glob "spot_prices/spot_prices_ap-southeast-1_${scn}.csv" \
        --output-name multipool_${scn} \
        --multi-pool \
        --instance-types m5.large,c5.xlarge,r5.large,m5.xlarge,c5.2xlarge
done
# → data/processed/multipool_{stable,volatile,spike,az_divergence}.csv
# → 129,600 rows × 4 scenarios (15 pools × 8,640 timesteps)
```

---

## 6. Generalist Training

Thay vì train 4 model chuyên biệt, dùng **1 agent tổng quát** qua `MixedScenarioEnv`:

```python
env = MixedScenarioEnv(
    scenario_paths={
        'stable':        'data/processed/multipool_stable.csv',
        'volatile':      'data/processed/multipool_volatile.csv',
        'spike':         'data/processed/multipool_spike.csv',
        'az_divergence': 'data/processed/multipool_az_divergence.csv',
    },
    weights={
        'stable': 1.0, 'volatile': 1.0,   # equal coverage
        'spike':  1.0, 'az_divergence': 1.0,
    },
    max_steps=168,
)
```

Mỗi episode: wrapper sample 1 scenario ngẫu nhiên theo weights → delegate tới `SpotOrchestratorEnv` tương ứng.

---

## 7. Training Pipeline (Kaggle)

Model được train trên **Kaggle GPU T4** (free tier, ~9h/session):

```
Local                           Kaggle
─────                           ──────
pack_kaggle.ps1                 kaggle_train.ipynb
  │                               │
  ├─ envs/                        ├─ Cell 1: pip install
  ├─ agents/                      ├─ Cell 2: unzip dataset
  ├─ data/processed/*.csv         ├─ Cell 3: verify imports
  └─ → spot_rl_kaggle.zip         ├─ Cell 4: check GPU
       │ upload                   ├─ Cell 5: init env + agent (v20)
       ▼                          ├─ Cell 6: MLflow setup
  Kaggle Dataset                  ├─ Cell 7: load checkpoint (auto-detect)
  "spot-rl-code"                  └─ Cell 8: training loop
                                       │
                                       ▼ download
                               results/kaggle/best_mixed_v20.pth
```

**Kaggle Secrets cần cấu hình:**
```
MLFLOW_USERNAME, MLFLOW_PASSWORD    # MLflow basic auth
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  # S3 artifact store
```

**Session management:** Kaggle timeout ~9h = ~4000-4500 ep/session. Upload `.pth` vào Kaggle Dataset để auto-resume session tiếp theo.

---

## 8. Kết quả (v20)

Model **v20** train 5000+ episodes với `sla_penalty=3.5` và soft model scoring:

| Agent | Cost/ep | SLA | Savings vs OD |
|---|---|---|---|
| **DQN v20 (best)** | **~$345** | **~95%** | **~40%** |
| On-Demand pure | ~$575 | 96%+ | 0% |
| Spot pure | ~$148 | 85-92% | 75% (fail SLA) |
| Threshold | ~$155 | 90-91% | 73% (fail SLA) |
| Random | ~$394 | 83-84% | 30% |

**Key insight:** Spot/Threshold rẻ hơn DQN nhưng không đạt SLA ≥ 95%. DQN là baseline duy nhất vừa tiết kiệm vừa đảm bảo reliability.

**Training history v20 (Kaggle T4, ~500 phút):**
- Reward: 0 → ~950 (converged)
- Cost: ~$450 → ~$345
- SLA: 83% → ~95% (cả 4 scenarios hội tụ sát ngưỡng)

---

## 9. Cấu trúc thư mục

```
spot-rl-optimiztion/
│
├── agents/
│   ├── dqn_agent.py              # DQN (Double + Dueling + PER + masking)
│   ├── networks.py               # FactoredDuelingQNetwork (24 outputs)
│   ├── replay_buffer.py          # ReplayBuffer + PrioritizedReplayBuffer
│   └── baselines.py              # 6 baselines để so sánh
│
├── envs/
│   ├── spot_orchestrator_env.py  # env chính: 15 pools, 121 actions
│   ├── mixed_scenario_env.py     # wrapper: sample scenario theo weights
│   ├── action_schema.py          # encode/decode 121 actions
│   ├── instance_catalog.py       # 5 types × 3 AZs, STATE_DIM=90
│   ├── market_simulator.py       # mô phỏng giá spot + interrupt
│   └── workload_generator.py     # sinh job (Poisson + daily profile)
│
├── experiments/
│   └── evaluate_kaggle.py        # DQN vs 6 baselines × 4 scenarios
│
├── data/
│   ├── processed/                # multipool_{stable,volatile,spike,az_divergence}.csv
│   └── scripts/
│       ├── generate_synthetic_spot_prices.py  # calibrated từ AWS thực tế
│       └── preprocess.py                      # multi-pool feature engineering
│
├── notebooks/
│   ├── kaggle_train.ipynb        # training notebook (Kaggle T4)
│   └── data_analysis.ipynb       # kiểm tra chất lượng data trước khi train
│
├── results/kaggle/               # checkpoints (.pth) + history (.pkl)
│
├── pack_kaggle.ps1               # đóng gói zip để upload Kaggle
└── requirements.txt
```

---

## 10. Go Controller (Production)

Sau khi train xong, model được export sang ONNX và chạy bởi **spot-rl-controller** — một Go service triển khai trên AWS EC2:

```
spot-rl-controller/
├── cmd/controller/main.go        # vòng lặp chính: collect → infer → guard → execute
├── cmd/sender/main.go            # gửi test jobs vào SQS
├── pkg/types/types.go            # shared types (State 90-dim, Action 121)
├── internal/
│   ├── state/collector.go        # build state vector từ AWS APIs
│   ├── inference/model.go        # ONNX Runtime wrapper (CGO)
│   ├── action/executor.go        # decode action → gọi EC2 API
│   ├── aws/
│   │   ├── ec2.go                # RunInstances, TerminateInstances (multi-AZ)
│   │   ├── pricing.go            # DescribeSpotPriceHistory
│   │   └── sqs.go                # GetQueueAttributes (workload signal)
│   └── safety/limits.go          # guardrails: fleet cap, cost cap, rate limit
└── Terraform/                    # IaC: VPC, IAM, SQS, Budget alerts, EC2
```

**Control loop (mỗi 5 phút):**
```
AWS EC2/SQS APIs
      ↓
State Collector (90 features)
      ↓
ONNX Runtime → Q-values [121]
      ↓
argmax → Action decode (op, type, AZ)
      ↓
Safety Guard (cost cap, fleet cap, rate limit, kill switch)
      ↓
EC2 Executor (provisioning / termination)
```

**Shadow mode** (mặc định ON): controller log decisions mà không thực thi EC2 calls — dùng để validate trước khi deploy thật.

---

## 11. Hyperparameters (v20)

| Parameter | Giá trị | Ghi chú |
|---|---|---|
| STATE_DIM | 90 | 10 nhóm features |
| ACTION_DIM | 121 | 8 ops × 15 pools + HOLD |
| Hidden dim | 512 → 256 | Factored DQN trunk |
| Learning rate | 1e-4 | Adam |
| Gamma | 0.99 | discount factor |
| Epsilon | 1.0 → 0.05 | exponential decay, 150K steps |
| Batch size | 256 | |
| Replay buffer | 500K | PER α=0.6, β 0.4→1.0 |
| Target update | 2000 steps | |
| Episodes/session | ~4000-4500 | Kaggle 9h timeout |
| Steps/episode | 168 | 1 tuần × 15 phút/step |
| SLA penalty | 3.5 | per failed job |
| Model score | Soft: `-cost - 500×sla_deficit` | không hard-cliff tại 95% |

---

## 12. MLflow Tracking

- **Server:** `mlflow.holoc.id.vn`
- **Experiment:** `spot-rl-generalist`
- **Model Registry:** `Spot-RL-Agent`

Metrics log mỗi 10 episodes:
- `reward`, `cost`, `sla`, `loss`, `epsilon`
- `cost_{scenario}`, `sla_{scenario}` — per-scenario breakdown

Model tags: `avg_cost`, `avg_sla`, `epsilon_end`, `sampling`, `session`.

---

## 13. Tech Stack

| Layer | Technology |
|---|---|
| **RL Training** | Python 3.12, PyTorch, Gymnasium |
| **GPU Training** | Kaggle T4 (free tier) |
| **Experiment Tracking** | MLflow + AWS S3 artifact store |
| **Data** | pandas, numpy — synthetic Ornstein-Uhlenbeck process |
| **Production Controller** | Go 1.24, ONNX Runtime 1.20.1 (CGO) |
| **Infrastructure** | AWS EC2, SQS, Terraform |
| **Containerization** | Docker (multi-stage build) |

---

## 14. Tham khảo

**Papers:**
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*
- Schaul et al. (2016) — *Prioritized Experience Replay*
- He et al. (2020, SpotOn, NSDI) — *SpotOn: A Batch Computing Service for the Spot Market*
- Harlap et al. (2018, Tributary, ATC) — *Tributary: spot-dancing for elastic services*

**AWS:**
- [EC2 Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
- [EC2 Spot Best Practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html)
