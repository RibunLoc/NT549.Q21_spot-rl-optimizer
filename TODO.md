# TODO List - Spot RL Project

Checklist này phản ánh trạng thái code hiện tại. Pipeline chính hiện là multi-pool generalist agent: `MixedScenarioEnv` + Factored Dueling DQN + action masking + Kaggle/RunPod training notebook.

## Done

- [x] Multi-pool action schema: 8 pool-targeted ops x 15 pools + `HOLD` = 121 actions
- [x] State vector 92 features trong `envs/instance_catalog.py`
- [x] `SpotOrchestratorEnv`: fleet lifecycle, workload scheduling, interruptions, cost, SLA, reward shaping
- [x] `MixedScenarioEnv`: sample stable/volatile/spike/az_divergence theo episode
- [x] Synthetic multi-pool spot data generator và preprocess pipeline
- [x] Processed scenario data tracked with DVC: `multipool_*.csv`
- [x] DQN agent with Double DQN, Dueling network, Prioritized Experience Replay
- [x] Factored Dueling Q-network for op/pool/hold action structure
- [x] Action masking support in agent + environment
- [x] Baseline agents for comparison
- [x] Kaggle/RunPod training notebook with MLflow logging
- [x] Evaluation scripts for trained checkpoints and baseline comparison
- [x] Model checkpoint metadata: `state_dim`, `action_dim`, `network_type`, `hidden_dim`
- [x] Docker, MLflow, DVC, Terraform scaffolding

## High Priority

- [ ] Sync `experiments/train.py` with Kaggle v36 pipeline
  - [ ] Support `env.type: mixed_scenario`
  - [ ] Pass `network_type`, `hidden_dim`, `use_per`, PER params, and `device` from config into `DQNAgent`
  - [ ] Use `env.get_action_mask()` during train/eval when available
  - [ ] Save best model by soft score: `-cost - 500 * sla_deficit`, not only episode reward

- [ ] Update multi-pool config files
  - [ ] Change `data_path` from `.pkl` to `.csv`
  - [ ] Add a mixed-scenario config matching notebook v36
  - [ ] Clarify `max_steps=168`: current training window, not a full 1-week 15-minute episode
  - [ ] Move reward/workload hyperparameters from notebook into YAML

- [ ] Clean notebook/script version drift
  - [ ] Make `notebooks/kaggle_train.ipynb` the canonical notebook
  - [ ] Remove or update stale helper scripts that still mention `STATE_DIM=90` or `.pkl`
  - [ ] Keep one documented path for Kaggle packaging and one for RunPod

- [ ] Add smoke tests
  - [ ] Env reset returns observation shape `(92,)`
  - [ ] Action mask shape is `(121,)` and always includes `HOLD`
  - [ ] Factored DQN forward returns `[batch, 121]`
  - [ ] One short train step works on CPU with synthetic data
  - [ ] Checkpoint load auto-detects `factored` vs `dueling`

## Evaluation

- [ ] Re-run `experiments/evaluate_kaggle.py` on the latest best checkpoint
- [ ] Save latest results table to `results/reports/`
- [ ] Report per-scenario cost/SLA for stable, volatile, spike, az_divergence
- [ ] Include action distribution plot for interpretability
- [ ] Compare with OnDemand, Spot, Threshold, CheapestAZ, CheapestType, Random
- [ ] Add confidence intervals or seed-level variance for final thesis numbers

## Documentation

- [x] Update README to current state/action dimensions and Kaggle v36 hyperparameters
- [ ] Update `QUICKSTART.md` to avoid old single-pool `.pkl` workflow as the default path
- [ ] Add a short "Current canonical training path" section
- [ ] Document how to reproduce DVC pull from S3
- [ ] Document checkpoint naming/versioning convention
- [ ] Add ONNX export instructions and expected controller schema: state 92, action 121

## Production / MLOps

- [ ] Verify `scripts/export_model.py` exports the latest factored model correctly
- [ ] Validate ONNX output shape `[1, 121]`
- [ ] Align Go controller state collector with `STATE_DIM=92`
- [ ] Add controller-side schema/version check before loading a model
- [ ] Keep shadow mode as default for first real AWS deployment
- [ ] Add cost/fleet kill-switch documentation

## Thesis / Report

- [ ] Update MDP formulation with 92-dimensional state
- [ ] Explain why action masking is necessary for operational constraints
- [ ] Explain reward shaping changes: SLA bonus, empty fleet penalty, provision spot bonus
- [ ] Add ablation discussion: flat dueling vs factored dueling, mask vs no mask
- [ ] Add failure analysis for SLA drift / Q overestimation in v35-v36
- [ ] Include architecture diagram: simulator, agent, MLflow, controller

## Optional Extensions

- [ ] Train with 672 steps/episode for full 1-week at 15-minute granularity
- [ ] Add startup/shutdown delays for EC2 realism
- [ ] Add checkpoint/retry behavior for interrupted jobs
- [ ] Add real AWS spot history fetch path as a reproducibility appendix
- [ ] Try PPO or discrete SAC as a comparison, after DQN pipeline is stable
- [ ] Add feature importance / saliency analysis for state features

## Notes

- Canonical training path today: `notebooks/kaggle_train.ipynb`.
- Canonical processed data format today: `data/processed/multipool_*.csv`.
- Canonical model schema today: `STATE_DIM=92`, `ACTION_DIM=121`.
