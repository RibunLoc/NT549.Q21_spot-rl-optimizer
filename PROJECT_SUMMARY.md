# Project Summary - Spot Instance Optimization với Deep RL

## Tổng quan dự án

**Mục tiêu:** Áp dụng Deep Reinforcement Learning (DQN) để tối ưu hóa chiến lược sử dụng AWS EC2 Spot Instance, giảm 20-40% chi phí cloud trong khi đảm bảo SLA >95%.

**Thời gian dự kiến:** 7-8 tuần

**Công nghệ:** Python, PyTorch, Gymnasium, AWS boto3, TensorBoard

---

## Cấu trúc project đã tạo

```
spot-rl-optimiztion/
├── README.md                      # ✅ Documentation chính (chi tiết)
├── QUICKSTART.md                  # ✅ Hướng dẫn nhanh
├── TODO.md                        # ✅ Danh sách tasks
├── requirements.txt               # ✅ Dependencies
├── .gitignore                     # ✅ Git ignore rules
│
├── data/                          # 📁 Dữ liệu
│   └── scripts/                   # ✅ Scripts thu thập & xử lý data
│       ├── fetch_spot_prices.py   # Crawl AWS spot price history
│       ├── generate_workload.py   # Generate synthetic workload
│       └── preprocess.py          # Feature engineering
│
├── envs/                          # 📁 Gymnasium environments
│   ├── __init__.py                # ✅
│   ├── spot_env.py                # ⚠️ Main environment (có TODOs)
│   ├── market_simulator.py       # ⚠️ Spot price simulator (có TODOs)
│   ├── workload_generator.py     # ✅ Batch job generator
│   └── cost_calculator.py        # ✅ Cost & reward calculator
│
├── agents/                        # 📁 RL agents
│   ├── __init__.py                # ✅
│   ├── dqn_agent.py               # ✅ DQN implementation (complete)
│   ├── networks.py                # ✅ Neural networks (MLP, LSTM, Dueling)
│   ├── replay_buffer.py           # ✅ Experience replay
│   └── baselines.py               # ✅ 4 baseline strategies
│
├── utils/                         # 📁 Utilities
│   ├── __init__.py                # ✅
│   ├── config.py                  # ✅ Config management
│   ├── logger.py                  # ✅ Logging & TensorBoard
│   └── metrics.py                 # ✅ Metrics tracking
│
└── experiments/                   # 📁 Training & evaluation
    ├── configs/
    │   └── dqn_default.yaml       # ✅ Default config
    └── train.py                   # ✅ Training script
```

**Trạng thái:**
- ✅ = Đã implement đầy đủ
- ⚠️ = Có template với TODOs cần hoàn thiện
- ❌ = Chưa có

---

## Files quan trọng cần đọc

1. **[README.md](README.md)** - Đọc đầu tiên!
   - Kiến trúc hệ thống
   - Mô hình MDP (state, action, reward)
   - Hướng dẫn sử dụng chi tiết
   - Tech stack & tài liệu tham khảo

2. **[QUICKSTART.md](QUICKSTART.md)** - Để bắt đầu nhanh
   - Setup môi trường (5 phút)
   - Thu thập data (15-30 phút)
   - Train model đầu tiên (1-2 giờ)
   - Troubleshooting

3. **[TODO.md](TODO.md)** - Danh sách tasks
   - Phase-by-phase breakdown
   - Priority tasks
   - Checklist để track progress

---

## Workflow dự kiến

### Week 1-2: Data & Environment
1. Setup environment: `pip install -r requirements.txt`
2. Thu thập data AWS spot price (hoặc generate synthetic)
3. Hoàn thiện `SpotInstanceEnv` (implement TODOs)
4. Test environment với random policy

### Week 3-4: Training
5. Train DQN agent với config default
6. Monitor với TensorBoard
7. Tune hyperparameters
8. Train trên 3 scenarios (stable, volatile, spike)

### Week 5-6: Evaluation
9. Implement evaluation script
10. So sánh DQN vs baselines
11. Tạo visualizations (plots, heatmaps)
12. Phân tích kết quả

### Week 7-8: Report & Presentation
13. Viết báo cáo (Introduction → Results)
14. Tạo slide deck
15. Chuẩn bị demo
16. Final review & submission

---

## Key TODOs cần làm ngay

### 🔥 Critical (blocking)

1. **Thu thập dữ liệu** (Week 1)
   ```bash
   # Option 1: AWS real data
   python data/scripts/fetch_spot_prices.py --region us-east-1 --days 30

   # Option 2: Synthetic data (nhanh hơn)
   python data/scripts/generate_workload.py --duration 30
   ```

2. **Hoàn thiện SpotInstanceEnv** (Week 1-2)
   - File: `envs/spot_env.py`
   - Search "TODO" trong file
   - Implement: `__init__`, `reset()`, `step()`, `_simulate_timestep()`, `_calculate_reward()`

3. **Hoàn thiện SpotMarketSimulator** (Week 1-2)
   - File: `envs/market_simulator.py`
   - Implement price evolution model
   - Implement interruption sampling

### ⚠️ Important (để train được)

4. **Test environment** (Week 2)
   ```python
   # Test script trong QUICKSTART.md
   python -c "from envs.spot_env import SpotInstanceEnv; ..."
   ```

5. **First training run** (Week 3)
   ```bash
   python experiments/train.py --config experiments/configs/dqn_default.yaml
   ```

### 📊 Can do later

6. Implement `experiments/evaluate.py`
7. Implement `experiments/compare_baselines.py`
8. Create visualization script `utils/visualization.py`
9. Write report

---

## Expected Results

Nếu implement đúng, cậu sẽ có:

### Cost Savings
- **Baseline (always on-demand):** $0.096/hour × 5 instances × 720 hours = $345.6/month
- **DQN optimized:** ~$200-250/month (**30-40% savings**)
- **Always spot (risky):** ~$150/month (nhưng SLA < 80%)

### SLA Compliance
- **Target:** >95% jobs completed successfully
- **DQN:** 95-98% (adaptive: dùng on-demand khi giá spot cao)
- **Always spot:** 70-80% (bị interrupt nhiều)

### Deliverables
1. Trained DQN model (`.pth` file)
2. Evaluation results (CSV/JSON)
3. Comparison plots (DQN vs baselines)
4. Báo cáo chi tiết (10-15 pages)
5. Slide deck (10-15 slides)

---

## Debugging Tips

### Environment không chạy
- Kiểm tra data file exists: `ls data/processed/`
- Check observation space dimension matches state vector
- Print state/reward/info ở mỗi step để debug

### Training không converge
- Check reward function: reward phải có signal rõ ràng
- Giảm learning rate: thử 1e-5 thay vì 1e-4
- Tăng epsilon decay: explore lâu hơn
- Check replay buffer: phải có đủ diverse experiences

### CUDA out of memory
- Giảm batch_size: 32 thay vì 64
- Force CPU: `export CUDA_VISIBLE_DEVICES=""`
- Giảm network size: [128, 64] thay vì [256, 128]

---

## Resources

### Code References
- **DQN paper:** [Mnih et al. 2015](https://www.nature.com/articles/nature14236)
- **Gymnasium docs:** https://gymnasium.farama.org/
- **PyTorch DQN tutorial:** https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

### AWS Documentation
- **Spot Instance Pricing:** https://aws.amazon.com/ec2/spot/pricing/
- **Boto3 API:** https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2.html

### Similar Projects
- SpotOn (NSDI 2020) - academic paper on spot optimization
- Tributary (ATC 2018) - elastic services on spot

---

## Contact & Support

Nếu gặp vấn đề:
1. Đọc lại README.md và QUICKSTART.md
2. Check TODO.md để xem task đã làm đúng chưa
3. Debug với print statements
4. Google error messages
5. Hỏi giảng viên hướng dẫn

Good luck với đề tài! 🚀
