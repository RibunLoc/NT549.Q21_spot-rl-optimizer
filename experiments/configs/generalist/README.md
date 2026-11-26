# Config train / eval generalist

Mỗi file là **một bước**, kế thừa file trước qua `extends`. Train lần lượt từng file rồi eval bằng
**chính file đó** thì thấy được mỗi thay đổi tác động bao nhiêu.

| File | Thay đổi so với file cha | obs | Chạy với controller hiện tại? |
|---|---|---|---|
| `v41.yaml` | Tái hiện v41, giữ nguyên các lỗi (làm mốc) | 92 | Có |
| `s1_fix_lr_done.yaml` | `lr_schedule: constant`, `bootstrap_on_truncation: true` | 92 | Có |
| `s3_interrupt_realistic.yaml` | Interrupt theo từng instance, ~10%/tháng | 92 | Có |
| `s3_interrupt_stress.yaml` | Như trên nhưng ~1 lần/instance/ngày (không dùng để báo cáo % tiết kiệm) | 92 | Có |
| `s4_pool_obs.yaml` | obs v2 (feature từng pool, bỏ progress/budget/avg_wait) + `network_type: poolwise` | 254 | **Không**, phải sửa `collector.go` |
| `s5_simple_reward.yaml` | `reward.mode: simple`, `cost_step_hours: 0.25` ($ thật) | 254 | **Không** |

## Train

Notebook: sửa `CONFIG = ...` ở Step 5 của `notebooks/kaggle_train.ipynb` / `runpod_train.ipynb`.

Local / script:

```bash
python experiments/train_generalist.py --config experiments/configs/generalist/s1_fix_lr_done.yaml
# smoke test vài phút trên CPU
python experiments/train_generalist.py --config experiments/configs/generalist/s1_fix_lr_done.yaml \
    --episodes 3 --max-steps 300 --warmup-steps 200 --eval-start 2 --eval-every 1 --eval-episodes 1
```

## Eval

Luôn dùng đúng config mà model được train. `--model` nhận `.pth` hoặc `.onnx`; bỏ trống thì chỉ chạy baseline.

```bash
python experiments/evaluate_kaggle.py --config experiments/configs/generalist/s1_fix_lr_done.yaml \
    --model results/s1_fix_lr_done/models/best_eval.pth --output-dir results/eval/s1
```

Mốc so sánh chính là **Autoscaler-OD** (autoscaler chỉ dùng on-demand, đạt SLA). **Autoscaler-Spot** là
heuristic mà RL cần thắng. Các baseline cũ (OnDemand, Spot, ...) không scale-in và không đạt SLA 95%.

## Chọn `failed_job_cost` (bước 5)

Đây là núm chỉnh đánh đổi cost–SLA. Train vài giá trị (ví dụ 0.2 / 0.5 / 1 / 2), eval từng model,
vẽ cost theo SLA, rồi chọn điểm đạt SLA ≥ 95% với cost thấp nhất.
