# Research Plan - Spot Instance Optimization with Reinforcement Learning

## 1) Mục tiêu đề tài

Xây dựng và đánh giá chiến lược tự động chọn Spot/On-demand instance bằng RL để:
- Giảm chi phí hạ tầng cloud so với chiến lược chỉ dùng On-demand.
- Duy trì SLA hoàn thành job >= 95%.
- Thích nghi với biến động giá Spot và rủi ro interruption.

## 2) Câu hỏi nghiên cứu

- RQ1: RL (DQN) có giúp giảm chi phí đáng kể so với các baseline heuristic không?
- RQ2: RL có duy trì SLA tốt hơn so với chiến lược “Always Spot” không?
- RQ3: RL có ổn định khi giá Spot biến động mạnh hoặc workload tăng đột biến không?

## 3) Giả thuyết

- H1: DQN giảm tổng chi phí >= 20% so với Always On-demand.
- H2: DQN giữ SLA >= 95% trong phần lớn kịch bản.
- H3: DQN đạt trade-off cost/SLA tốt hơn Threshold-based policy.

## 4) Thiết kế thực nghiệm

## 4.1 Môi trường và tác tử
- Environment: `envs/spot_env.py`
- Market simulator: `envs/market_simulator.py`
- Workload generator: `envs/workload_generator.py`
- RL agent: `agents/dqn_agent.py`
- Baselines: `agents/baselines.py`

## 4.2 Dữ liệu
- Synthetic Spot price: `data/scripts/generate_synthetic_spot_prices.py`
- AWS real Spot price (nếu có credentials): `data/scripts/fetch_spot_prices.py`
- Workload synthetic: `data/scripts/generate_workload.py`

## 4.3 Các kịch bản cần chạy
- Scenario A (Stable): volatility thấp, spike ít.
- Scenario B (Volatile): volatility cao, spike nhiều.
- Scenario C (Workload spike): arrival rate tăng mạnh theo giờ cao điểm.

## 4.4 Phương pháp so sánh
- DQN (mô hình chính).
- Always On-demand.
- Always Spot.
- Threshold-based.
- Random (sanity check).

## 4.5 Chỉ số đánh giá
- `Total Cost` ($/episode)
- `Cost Savings vs On-demand` (%)
- `SLA Compliance` (%)
- `Failed Jobs` (count/rate)
- `Interruption Events` (count/rate)
- `Action Mix` (% spot/on-demand/migrate)

## 4.6 Thiết kế thống kê
- Mỗi cấu hình chạy >= 5 seeds.
- Báo cáo Mean ± Std cho các metric.
- Dùng kiểm định đơn giản (Mann-Whitney U hoặc t-test) cho metric chính (cost, SLA).

## 5) Quy trình chạy thực nghiệm (từ repo hiện tại)

## 5.1 Chuẩn bị dữ liệu
```bash
make data-synthetic
make data-workload
make data-preprocess
```

## 5.2 Train mô hình
```bash
python experiments/train.py \
  --config experiments/configs/dqn_quick_test.yaml \
  --experiment-name quick_test
```

Sau khi ổn định, dùng `experiments/configs/dqn_default.yaml` cho run chính.

## 5.3 Khoảng trống cần hoàn thiện để làm nghiên cứu chuẩn
Hiện repo chưa có:
- `experiments/evaluate.py`
- `experiments/compare_baselines.py`

Hai script này là phần bắt buộc để tạo bảng kết quả cho báo cáo.

## 6) Kế hoạch triển khai đề tài (4 tuần gọn)

- Tuần 1: Chuẩn hóa dữ liệu + kiểm tra môi trường + chạy quick training.
- Tuần 2: Hoàn thiện evaluate + compare baselines, chạy scenario A.
- Tuần 3: Chạy scenario B/C + multi-seed + tổng hợp bảng kết quả.
- Tuần 4: Phân tích, viết báo cáo, chuẩn bị slide/demo.

## 7) Cấu trúc báo cáo gợi ý

- Introduction (vấn đề Spot cost-risk trade-off)
- Related Work (Spot optimization + RL scheduling)
- Problem Formulation (MDP: state/action/reward)
- Method (DQN + baseline policies)
- Experimental Setup (data/scenarios/metrics)
- Results (cost-SLA trade-off, robustness)
- Discussion (failure cases, limitations)
- Conclusion & Future Work

## 8) Rủi ro và cách giảm thiểu

- Reward shaping chưa ổn định: tách thành nhiều thành phần và theo dõi từng thành phần.
- Overfit synthetic data: thêm run với dữ liệu thật từ AWS.
- Kết quả dao động theo seed: bắt buộc multi-seed và report độ lệch chuẩn.

## 9) Kết quả đầu ra kỳ vọng

- Model DQN tốt nhất (`.pth`)
- Bảng so sánh DQN vs baselines
- Biểu đồ cost/SLA/action distribution
- Báo cáo hoàn chỉnh và slide bảo vệ
