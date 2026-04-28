# Spot RL Optimization

**Dạy một AI agent biết khi nào nên mua AWS Spot, khi nào mua On-Demand, dùng instance type nào và đặt ở Availability Zone nào -- để chi phí thấp nhất mà vẫn hoàn thành công việc đúng hạn.**

---

## 1. Bài toán

AWS bán compute theo 2 kiểu:
- **On-Demand (OD)**: đắt, ổn định, không bao giờ bị cắt.
- **Spot**: rẻ 60-90%, nhưng AWS có thể "rút lại" bất kỳ lúc nào (interrupt) → job đang chạy trên instance đó bị giết.

Nếu ta dùng 100% OD thì đắt. Nếu dùng 100% Spot thì rẻ nhưng job hay fail → vi phạm SLA (cam kết % job phải hoàn thành). Đây là bài toán **đánh đổi cost ↔ reliability** dưới môi trường giá biến động.

**Phức tạp hơn nữa:** có 15 "ô chợ" khác nhau = 5 loại instance × 3 AZ. Giá mỗi ô biến động độc lập theo thời gian. Xác suất interrupt phụ thuộc vào giá (giá tăng → AWS càng muốn lấy lại spot).

→ Không có heuristic đơn giản nào đủ tốt. Dùng **Reinforcement Learning**.

---

## 2. Cách agent hoạt động (Intuition)

Tưởng tượng agent là một "nhân viên vận hành" ngồi trước 15 màn hình giá, mỗi giờ ra một quyết định:

> *"Giờ này cluster đang có 8 spot + 2 OD đang chạy, queue còn 12 job pending, giá Spot ở m5.large AZ-1a đang rất rẻ, nhưng interrupt rate cao -- có nên mua thêm không? Hay nên chuyển 2 spot đang chạy sang OD vì sắp có đợt spike giá?"*

Quyết định đó ảnh hưởng tới:
- **Chi phí** giờ tiếp theo (dùng spot rẻ vs OD đắt)
- **SLA** (nếu interrupt → job fail → mất điểm)
- **Trạng thái tương lai** (quá tập trung 1 AZ thì dễ bị interrupt cùng lúc)

Agent học bằng **thử-sai hàng triệu lần** trong simulator, dần dần phát hiện pattern: *"khi giá m5 ở AZ-1a rẻ hơn 40% OD mà interrupt rate < 3%, mua spot đáng giá; nhưng nếu đang có workload spike và queue wait > 5 giờ thì phải mua OD để đảm bảo SLA."*

---

## 3. MDP Formulation (công thức toán)

RL chạy trên Markov Decision Process: (State, Action, Reward, Transition).

### State: 72 chiều số

Mỗi bước, agent thấy một vector `s ∈ R^72` gói 6 nhóm tin:

| Nhóm | Số chiều | Nội dung |
|---|---|---|
| Top-3 pools rẻ nhất | 12 | price_ratio, interrupt_rate, vcpu/$, az_id (×3 pools) |
| Thống kê 15 pools | 45 | spot_price, od_price, interrupt_rate, vcpu/$ (×15 pools) |
| Infrastructure | 3 | num_spot, num_ondemand, total_vcpu hiện có |
| Workload | 4 | pending_jobs, running_jobs, forecast_1h, queue_wait |
| Time | 3 | hour_of_day, day_of_week, episode_progress |
| Current metrics | 5 | avg_spot_price, avg_interrupt, spot_ratio, cost_rate, sla_health |

Tất cả normalize về [0, 1] để neural net dễ học.

### Action: 121 lựa chọn rời rạc

```
Action = op × 15_pools + pool_id     (op ∈ {0..7})
Action = 120                          (HOLD, giữ nguyên)
→ Tổng 8 × 15 + 1 = 121
```

8 operations:

| # | Op | Nghĩa |
|---|---|---|
| 0 | PROVISION_SPOT | Mua 1 spot ở (type, az) đã chọn |
| 1 | PROVISION_ONDEMAND | Mua 1 OD |
| 2 | RELEASE_SPOT | Trả 1 spot |
| 3 | RELEASE_ONDEMAND | Trả 1 OD |
| 4 | CONVERT_TO_ONDEMAND | Chuyển spot → OD (né interrupt) |
| 5 | CONVERT_TO_SPOT | Chuyển OD → spot (tiết kiệm) |
| 6 | REBALANCE_SPOT | Phân bổ lại spot giữa các AZ |
| 7 | RESERVE_CAPACITY | Dự trữ trước khi có spike |
| HOLD | -- | Không làm gì |

### Reward: thưởng khi tiết kiệm

Mỗi bước (1 giờ mô phỏng) agent nhận:

```
R = savings - penalties

savings              = baseline_ondemand_cost - actual_cost
sla_penalty          = số_job_fail × 10.0
interrupt_penalty    = số_interrupt × 0.5
migration_penalty    = số_chuyển_spot_OD × 1.0
concentration_penalty = max(0, az_concentration - 0.8) × 5.0
```

→ Agent thấy **"tiết kiệm được càng nhiều vs. nếu toàn OD → reward càng cao"**, nhưng sẽ bị phạt nặng nếu để job fail.

### Transition: chuyện gì xảy ra sau action

Sau khi agent chọn action, simulator:
1. Áp dụng thay đổi đội hình (mua/bán instance)
2. Sample giá spot mới (từ dataset CSV hoặc noise)
3. Sample interrupt event (xác suất phụ thuộc giá)
4. Schedule job từ queue → running instances
5. Tick 1 giờ, tính cost & SLA cho giờ đó
6. Trả về `(next_state, reward, done, info)`

Một episode = 168 giờ = 1 tuần. 5000 episodes ≈ **840K quyết định** agent phải học để tối ưu.

---

## 4. Kiến trúc Agent: Factored Dueling DQN

### Vấn đề với DQN vanilla

DQN học hàm `Q(s, a)` -- "ở trạng thái s, chọn action a sẽ thu về bao nhiêu reward tổng". Với 121 actions, vanilla DQN dùng output layer có **121 neurons độc lập**. Khi thấy một action mới, phải học lại từ đầu.

### Factored: chia action thành op + pool

Observation: action 121 thực chất có cấu trúc `op × pool`. Nếu agent học được "PROVISION_SPOT ở pool rẻ thường tốt", kiến thức này nên áp dụng cho **mọi pool**, không chỉ pool đang thấy.

```
                  State (72-dim)
                       |
                  Trunk (512 -> 256)    # feature extractor chung
                       |
        +--------------+---------------+
        |              |               |
    op_head(8)    pool_head(15)    hold_head(1)
        |              |               |
        +--------------+---------------+
                       |
   Q(s, action) = op_score[op] + pool_score[pool]    # composite action
   Q(s, HOLD)   = hold_score                          # special action
```

→ 8 + 15 + 1 = **24 output neurons** thay vì 121. Chia sẻ thông tin giữa các action, học nhanh hơn. **236K params** tổng.

### Các kỹ thuật bổ sung

| Kỹ thuật | Giải quyết gì |
|---|---|
| **Double DQN** | Giảm overestimation bias (Q-value thường bị phóng đại) |
| **Dueling head** | Tách `V(s)` (state quan trọng thế nào) khỏi `A(s,a)` (action nào tốt hơn) |
| **Prioritized Experience Replay (PER)** | Ưu tiên replay transition "khó" (TD-error lớn) để học nhanh hơn |
| **Action Masking** | Invalidate action vô lý (vd: RELEASE_SPOT khi pool trống) |
| **Prioritized Scenario Sampling** | Sample scenario theo TD-loss → agent tập trung scenario đang khó |

---

## 5. Generalist Training trên 4 Scenarios

Thay vì train 4 model chuyên biệt (stable/volatile/spike/az_divergence), dùng **1 agent tổng quát** train trên `MixedScenarioEnv`:

```python
env = MixedScenarioEnv(scenario_paths={
    "stable":        "data/processed/multipool_stable.csv",
    "volatile":      "data/processed/multipool_volatile.csv",
    "spike":         "data/processed/multipool_spike.csv",
    "az_divergence": "data/processed/multipool_az_divergence.csv",
})

# Mỗi episode:
#   1. Wrapper sample 1 scenario (weighted theo TD-loss EMA)
#   2. Delegate step() tới SpotOrchestratorEnv tương ứng
#   3. Cuối episode: update_scenario_loss(scn, td_loss) để điều chỉnh weight
```

**4 scenarios trong dataset:**

| Scenario | Mô phỏng gì | Bài học cho agent |
|---|---|---|
| Stable | Giá biến động thấp (~±10%) | Tối đa hóa spot usage |
| Volatile | Giá spike 2-4x, interrupt cao | Khi nào rút về OD |
| Spike | Workload đột biến 4-5x | Scale nhanh, tránh job fail |
| AZ Divergence | Giá lệch lớn giữa các AZ | Chọn AZ rẻ nhất |

→ Agent generalize tốt cả 4 (cost lệch nhau ~$10 giữa scenarios).

---

## 6. Flow end-to-end

```
  ┌─────────────────────────┐
  │ 1. Generate synthetic   │  data/scripts/generate_synthetic_spot_prices.py
  │    price data (CSV)     │  → 4 × multipool_*.csv
  └───────────┬─────────────┘
              │
  ┌───────────▼─────────────┐
  │ 2. Preprocess           │  data/scripts/preprocess.py --mode multi_pool
  └───────────┬─────────────┘
              │
  ┌───────────▼─────────────┐
  │ 3. Pack Kaggle zip      │  pack_kaggle.ps1
  │    (code + data)        │  → spot_rl_kaggle.zip (~40MB)
  └───────────┬─────────────┘
              │  upload to Kaggle Dataset
  ┌───────────▼─────────────┐
  │ 4. Train on Kaggle GPU  │  notebooks/kaggle_train.ipynb
  │    (T4, 5-8 hours)      │  → best_mixed.pth (on Kaggle /working/)
  │                         │  → log metrics to MLflow
  └───────────┬─────────────┘
              │  download .pth to results/kaggle/
  ┌───────────▼─────────────┐
  │ 5. Evaluate vs baselines│  experiments/evaluate_kaggle.py
  │    (30 ep × 4 scenarios)│  → comparison plots + eval_summary.json
  └───────────┬─────────────┘
              │
  ┌───────────▼─────────────┐
  │ 6. (future) Deploy      │  spot-rl-controller (Go k8s operator)
  │    as k8s controller    │  consumes ONNX export of model
  └─────────────────────────┘
```

---

## 7. Kết quả

| Agent | stable | volatile | spike | az_div | SLA | Savings vs OD |
|---|---|---|---|---|---|---|
| **DQN (v1)** | **$209** | **$196** | **$204** | **$202** | **~98.9%** | **~65%** |
| OnDemand pure | $575 | $575 | $575 | $575 | 96.4% | 0% |
| Spot pure | $148 | $129 | $129 | $139 | 85-92% | 75% (nhưng fail SLA) |
| Threshold | $156 | $147 | $153 | $154 | 90-91% | 73% (nhưng fail SLA) |
| Random | $394 | $408 | $399 | $391 | 83-84% | 30% |

**Điểm mấu chốt:** Spot/Threshold rẻ hơn DQN nhưng **không đạt SLA 95%**. DQN là baseline duy nhất vừa rẻ vừa đảm bảo SLA.

---

## 8. Cấu trúc thư mục

```
spot-rl-optimiztion/
│
├── agents/                    # RL agent
│   ├── dqn_agent.py          # DQN (Double + Dueling + PER + mask)
│   ├── networks.py           # FactoredDuelingQNetwork
│   ├── replay_buffer.py      # ReplayBuffer + PrioritizedReplayBuffer
│   └── baselines.py          # 6 baselines để so sánh
│
├── envs/                     # Environment (MDP)
│   ├── spot_orchestrator_env.py  # env chính: 15 pools, 121 actions
│   ├── mixed_scenario_env.py     # wrapper sample scenario theo TD-loss
│   ├── action_schema.py          # encode/decode action 121
│   ├── instance_catalog.py       # 5 types × 3 AZs, STATE_DIM=72
│   ├── market_simulator.py       # mô phỏng giá spot + interrupt
│   └── workload_generator.py     # sinh job (Poisson, daily profile)
│
├── experiments/
│   └── evaluate_kaggle.py    # so sánh DQN vs 6 baselines × 4 scenarios
│
├── data/
│   ├── processed/            # multipool_{stable,volatile,spike,az_div}.csv
│   └── scripts/              # generate + preprocess
│
├── notebooks/
│   └── kaggle_train.ipynb    # training notebook cho Kaggle T4
│
├── scripts/                  # helper: repack notebook, update cells
├── results/kaggle/           # checkpoints (.pth) + eval outputs
│
├── pack_kaggle.ps1           # tạo zip để upload lên Kaggle
└── requirements.txt
```

---

## 9. Hướng dẫn chạy

### Local setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Generate data
```bash
python data/scripts/generate_synthetic_spot_prices.py
python data/scripts/preprocess.py --mode multi_pool
```

### Train trên Kaggle
```bash
.\pack_kaggle.ps1              # tạo spot_rl_kaggle.zip
# → Upload lên Kaggle Dataset "spot-rl-code"
# → Chạy notebooks/kaggle_train.ipynb với GPU T4
```

**Kaggle Secrets cần cấu hình:**
```
MLFLOW_USERNAME, MLFLOW_PASSWORD    # MLflow basic auth
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  # S3 artifact store
```

### Evaluate
```bash
python experiments/evaluate_kaggle.py \
    --model results/kaggle/best_mixed_v1.pth \
    --episodes 30 \
    --output-dir results/kaggle/eval
```

### Resume training (không khuyến khích)
Trong cell Step 5c của notebook:
```python
RESUME_CHECKPOINT = '/kaggle/input/spot-rl-models/best_mixed_v1.pth'
```

**Lưu ý:** Replay buffer KHÔNG được save -- resume sẽ làm policy drift. Ưu tiên train from scratch.

---

## 10. MLflow Tracking

- **Server:** mlflow.holoc.id.vn
- **Experiment:** `spot-rl-generalist`
- **Model Registry:** `Spot-RL-Agent`

Metrics log mỗi 10 episodes:
- `reward`, `cost`, `sla`, `loss`, `epsilon`
- `cost_{scenario}`, `sla_{scenario}` (per-scenario breakdown)
- `weight_{scenario}` -- prioritized sampling weight
- System metrics: CPU, RAM, GPU util (tự động)

Model được tag với: `avg_cost`, `avg_sla`, `epsilon_end`, `steps_done`, `resumed_from`.

---

## 11. Hyperparameters

| Param | Giá trị | Ghi chú |
|---|---|---|
| State dim | 72 | |
| Action dim | 121 | 8 ops × 15 pools + HOLD |
| Hidden dim | 512 | trunk của Q-network |
| Learning rate | 3e-4 | Adam |
| Gamma | 0.99 | discount factor |
| Epsilon | 1.0 → 0.05 | decay linear 600K steps |
| Batch size | 256 | |
| Replay buffer | 500K | PER với alpha=0.6, beta 0.4→1.0 |
| Target update | 500 steps | |
| Episodes | 8000 | 168 step/ep = 1 tuần mô phỏng |

---

## 12. Tech stack

- **Python 3.12**, **PyTorch** -- DQN + neural net
- **Gymnasium** -- RL env interface
- **MLflow** -- experiment tracking + model registry
- **AWS S3** -- artifact store
- **Kaggle T4** -- GPU training (free tier)
- **pandas / numpy / matplotlib**

---

## 13. Tham khảo

**Papers:**
- Mnih et al. (2015) -- *Human-level control through deep reinforcement learning* (DQN)
- Wang et al. (2016) -- *Dueling Network Architectures*
- Schaul et al. (2016) -- *Prioritized Experience Replay*
- SpotOn (NSDI 2020) -- On-Demand Spot Instances
- Tributary (ATC 2018) -- Elastic Services on Spot

**AWS docs:**
- [EC2 Spot Instances Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
