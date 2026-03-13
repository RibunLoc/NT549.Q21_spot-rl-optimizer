# TODO List - Spot RL Project

Danh sách các tasks cần hoàn thành để có project hoàn chỉnh.

## Phase 1: Data Collection & Preprocessing ⏳

- [ ] **Data Collection**
  - [ ] Test `fetch_spot_prices.py` với AWS account
  - [ ] Crawl 30-60 ngày spot price history
  - [ ] Download interruption frequency data (nếu có API)
  - [ ] Generate synthetic workload với `generate_workload.py`

- [ ] **Data Preprocessing**
  - [ ] Implement `preprocess_spot_prices()` đầy đủ
    - [ ] Resample to hourly frequency
    - [ ] Calculate rolling averages (1h, 24h, 168h)
    - [ ] Calculate volatility (rolling std)
    - [ ] Extract time features (hour, day, month)
  - [ ] Implement `preprocess_workload()` đầy đủ
    - [ ] Smooth arrival rates
    - [ ] Calculate forecasted load
  - [ ] Implement `merge_datasets()` để join price + workload

## Phase 2: Environment Implementation 🚧

- [ ] **SpotInstanceEnv** (`envs/spot_env.py`)
  - [ ] Hoàn thiện `__init__`: load data, init simulators
  - [ ] Hoàn thiện `reset()`: reset all state variables
  - [ ] Hoàn thiện `step()`: implement full transition logic
  - [ ] Hoàn thiện `_execute_action()`: handle all 6 actions
  - [ ] Hoàn thiện `_simulate_timestep()`: market + workload + jobs
  - [ ] Hoàn thiện `_calculate_reward()`: cost + savings + penalties
  - [ ] Hoàn thiện `_get_observation()`: build state vector correctly
  - [ ] Test environment với random policy

- [ ] **SpotMarketSimulator** (`envs/market_simulator.py`)
  - [ ] Load và preprocess historical data
  - [ ] Implement `step()`: price evolution (ARIMA hoặc replay)
  - [ ] Implement interruption event sampling
  - [ ] Implement `get_price_statistics()`: rolling stats

- [ ] **WorkloadGenerator** (`envs/workload_generator.py`)
  - [ ] Implement `step()`: Poisson arrival với time-varying λ
  - [ ] Implement `_generate_job()`: random job parameters
  - [ ] Add time-of-day patterns
  - [ ] Add day-of-week patterns
  - [ ] Add workload spikes

- [ ] **Job Scheduler** (new file: `envs/job_scheduler.py`)
  - [ ] Implement job queue management
  - [ ] Implement job execution on instances
  - [ ] Handle job completion / failure
  - [ ] Handle spot interruptions (kill running jobs)

## Phase 3: Agent Implementation ✅

- [x] **DQN Agent** (`agents/dqn_agent.py`) - Đã implement cơ bản
- [x] **Neural Networks** (`agents/networks.py`) - Đã có QNetwork
- [x] **Replay Buffer** (`agents/replay_buffer.py`) - Đã có ReplayBuffer
- [x] **Baselines** (`agents/baselines.py`) - Đã có 4 baselines

### Optional Improvements:
- [ ] Implement Prioritized Experience Replay
- [ ] Implement LSTM Q-Network (cho time-series)
- [ ] Implement Dueling DQN
- [ ] Implement Double DQN

## Phase 4: Training Pipeline 🚧

- [ ] **Training Script** (`experiments/train.py`)
  - [ ] Test training loop end-to-end
  - [ ] Fix any bugs trong integration
  - [ ] Add validation loop (eval every N episodes)
  - [ ] Add early stopping (nếu performance plateau)
  - [ ] Add model checkpointing (save best model)

- [ ] **Configs**
  - [x] `dqn_default.yaml` - Done
  - [ ] `stable_price.yaml` - Scenario giá ổn định
  - [ ] `volatile_price.yaml` - Scenario giá biến động
  - [ ] `workload_spike.yaml` - Scenario workload cao điểm

## Phase 5: Evaluation & Comparison 📊

- [ ] **Evaluation Script** (`experiments/evaluate.py`)
  - [ ] Load trained model
  - [ ] Run evaluation episodes (no exploration)
  - [ ] Compute metrics (cost, SLA, savings)
  - [ ] Save results to JSON/CSV

- [ ] **Baseline Comparison** (`experiments/compare_baselines.py`)
  - [ ] Implement comparison framework
  - [ ] Run DQN vs all baselines
  - [ ] Generate comparison table
  - [ ] Statistical significance tests (optional)

- [ ] **Visualization** (`utils/visualization.py`)
  - [ ] Plot training curves (reward, loss, epsilon)
  - [ ] Plot cost comparison (DQN vs baselines)
  - [ ] Plot SLA compliance over time
  - [ ] Plot action distribution heatmap
  - [ ] Plot spot price vs action taken
  - [ ] Plot spot usage ratio over time

## Phase 6: Experiments & Results 🔬

- [ ] **Run Experiments**
  - [ ] Experiment 1: Stable price scenario
  - [ ] Experiment 2: Volatile price scenario
  - [ ] Experiment 3: Workload spike scenario
  - [ ] Hyperparameter tuning (learning rate, epsilon decay)
  - [ ] Ablation studies (reward components, network architecture)

- [ ] **Analysis**
  - [ ] Cost savings analysis (20-40% target)
  - [ ] SLA compliance analysis (>95% target)
  - [ ] Failure case analysis (when does DQN fail?)
  - [ ] Interpretability (which features matter most?)

## Phase 7: Documentation & Presentation 📝

- [x] **README.md** - Done
- [x] **QUICKSTART.md** - Done
- [ ] **API Documentation**
  - [ ] Add docstrings to all classes/functions
  - [ ] Generate Sphinx docs (optional)

- [ ] **Report**
  - [ ] Introduction & motivation
  - [ ] Problem formulation (MDP)
  - [ ] Methodology (DQN, baselines)
  - [ ] Experimental setup
  - [ ] Results & analysis
  - [ ] Conclusion & future work

- [ ] **Presentation**
  - [ ] Slide deck (10-15 slides)
  - [ ] Demo video (optional)

## Phase 8: Extensions (Optional) 🚀

- [ ] **Multi-region optimization**
  - [ ] Extend state space với multiple AZs
  - [ ] Action: chọn AZ có giá tốt nhất

- [ ] **Advanced RL algorithms**
  - [ ] Implement PPO (policy gradient)
  - [ ] Implement A3C (async training)
  - [ ] Compare with DQN

- [ ] **Real-world integration**
  - [ ] API wrapper để deploy model
  - [ ] Integration với AWS Auto Scaling
  - [ ] Real-time monitoring dashboard

- [ ] **Simulation improvements**
  - [ ] More realistic price model (GBM, volatility clustering)
  - [ ] Instance startup/shutdown delays
  - [ ] Data transfer costs
  - [ ] Reserved instance integration

---

## Priority Tasks (Làm trước)

1. ✅ Setup project structure
2. ✅ Create README & documentation
3. **🔥 Collect/generate data** (blocking)
4. **🔥 Implement SpotInstanceEnv fully** (blocking)
5. Test environment với random policy
6. Run first training experiment
7. Implement evaluation script
8. Run baseline comparison
9. Create visualizations
10. Write report

---

## Notes

- Ưu tiên làm đủ để có 1 end-to-end pipeline trước (data → train → eval)
- Sau đó mới optimize và add features
- Commit code thường xuyên với Git
- Document mọi thứ trong quá trình làm (dễ viết báo cáo sau)
