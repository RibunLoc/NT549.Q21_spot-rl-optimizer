# III. THIẾT KẾ HỆ THỐNG — 3.1. Mô hình bài toán

> **Đề tài:** Học tăng cường cho tối ưu chi phí AWS Spot Instance
> **Phạm vi slide:** chỉ phần môi trường mô phỏng + agent training trong repo `spot-rl-optimiztion`
> **Style:** mỗi slide 1 chủ đề, có bảng/công thức/khối ngắn

---

## Slide 1 — MÔ TẢ HỆ THỐNG

**Bài toán:**
> Cấp phát compute resource (Spot/On-Demand) cho workload batch trên AWS sao cho **tối thiểu chi phí** đồng thời **đảm bảo SLA**.

**Bối cảnh mô phỏng:**
- Thị trường: 15 pool = **5 instance type × 3 AZ** (ap-southeast-1)
- Giá Spot biến động theo thời gian thực, có thể bị thu hồi với 2 phút thông báo
- Workload: arrival rate có pattern theo giờ/ngày + spike ngẫu nhiên
- Agent ra quyết định mỗi **15 phút** (672 step / 1 tuần)

**Thông điệp:** Bài toán quyết định tuần tự — RL phù hợp.

**Speaker notes:**
> Bài toán em mô hình hóa: agent cần cấp phát compute cho workload sao cho rẻ nhất nhưng vẫn đảm bảo SLA. Môi trường có 15 pool khác nhau, giá Spot biến động theo thị trường thực và có thể bị thu hồi bất ngờ. Workload mô phỏng có pattern theo giờ giống Jenkins thực tế. Agent ra quyết định mỗi 15 phút — đủ nhanh để phản ứng, không quá nhanh để gây ra việc thay đổi tài nguyên quá thường xuyên, chẳng hạn như liên tục scale/down hay chuyển đổi spot - OD khiến cho hệ thống mất ổn định và tốn overhead nhiều khi lợi ích đạt được.
Trong thực tế với Amazon web service thì điều này rất nguy hiểm vì: EC2 boot mất thời gian, jenkins worker reconnect liên tục, API provisioning tăng nhiều gây lãng phí ngân sách, đôi khi còn tốn tiền hơn do launch quá nhiều instance ngắn hạn.

---

## Slide 2 — STATE SPACE — Vector 92 chiều

**Hiển thị:**

> **Không gian trạng thái — Vector 92 chiều, chuẩn hóa về `[0, 1]`**

| Nhóm dữ liệu | Dim | Câu hỏi mà nhóm này trả lời |
|---|:---:|---|
| **Top-3 pool rẻ nhất** | 30 | *Pool nào tiềm năng nhất? Giá / rủi ro / hiệu quả vCPU ra sao?* |
| **Tổng hợp theo AZ** | 3 | *Chênh lệch giá giữa các AZ? AZ nào rẻ nhất?* |
| **Tổng hợp theo type** | 3 | *Loại máy nào tốt nhất về giá / rủi ro / vCPU?* |
| **Hạ tầng hiện tại** | 3 | *Đang có bao nhiêu Spot / OD / tổng vCPU?* |
| **Workload** | 6 | *Bao nhiêu job đang chờ / chạy? 1 giờ tới dự báo bao nhiêu?* |
| **Thời gian** | 3 | *Đang là mấy giờ? Ngày trong tuần? Tiến độ episode?* |
| **Tình trạng hệ thống** | 5 | *Giá trung bình, interrupt, tỉ lệ Spot, chi phí, SLA hiện tại?* |
| **Bối cảnh nâng cao** | 9 | *Ngân sách còn? Workload tăng hay giảm? SLA có rủi ro?* |
| **Tải từng pool** | 30 | *CPU/RAM của 15 pool — pool nào đang bận?* |

> **Mục tiêu:** giúp agent phân biệt khi nào nên **tiết kiệm** (giá rẻ, ít rủi ro) — khi nào cần **bảo vệ SLA** (workload tăng, interrupt cao).

**Speaker notes:**
> State 92 chiều, chia 9 nhóm. Mỗi nhóm trả lời một câu hỏi cụ thể cho agent. Quan trọng nhất là nhóm "Top-3 pool rẻ nhất" với 30 feature — em chỉ đưa 3 pool tiềm năng nhất thay vì tất cả 15, để agent tập trung vào quyết định có ý nghĩa. Workload và thời gian giúp agent biết bối cảnh: đang lúc cao điểm hay thấp điểm. Tình trạng hệ thống cho biết SLA có rủi ro hay không. Tải từng pool tránh quyết định mù — như release pool đang bận xử lý job. Em sẽ zoom vào chi tiết Top-3 pool ở slide kế tiếp.

---

## Slide 3 — STATE SPACE — Zoom: Top-3 Pool rẻ nhất (30 feature)

**Hiển thị:**

> Trong 15 pool, em chỉ chọn **3 pool có giá Spot rẻ nhất** đưa vào state — mỗi pool **10 feature**, tổng cộng **30 feature** (chiếm 1/3 state vector).

**Vì sao chỉ Top-3?**
- Tránh đưa vào noise của 12 pool "không liên quan"
- Agent học nhanh hơn — chỉ tập trung vào quyết định có ý nghĩa
- Pool rẻ là cơ hội tiết kiệm chính

**10 feature mỗi pool:**

| # | Feature | Câu hỏi cho agent |
|:---:|---|---|
| 1 | **price_ratio** | Giá Spot bằng bao nhiêu % giá OD? |
| 2 | **interrupt_prob** | Rủi ro bị thu hồi bao nhiêu? |
| 3 | **vCPU per $** | Mua 1$ được bao nhiêu vCPU? |
| 4 | **az_idx** | Pool này ở AZ nào? (1a / 1b / 1c) |
| 5 | **base_rate** | Tỉ lệ interrupt nền của loại máy |
| 6 | **baseline_savings** | Tiết kiệm tiềm năng so với OD |
| 7 | **SPS score** | Điểm AWS đánh giá khả năng cấp phát |
| 8 | **price_CV** | Giá có biến động mạnh trong 24h không? |
| 9 | **rebalance_flag** | Có signal AWS đề xuất di chuyển không? |
| 10 | **termination_flag** | Có signal AWS sắp thu hồi không? |

→ Tất cả normalize về `[0, 1]`. Pool sắp xếp theo `price_ratio` tăng dần.

**Speaker notes:**
> Slide này em giải thích chi tiết nhóm quan trọng nhất — Top-3 pool rẻ nhất chiếm 30 trong tổng 92 feature. Em chỉ lấy 3 pool rẻ nhất vì agent không cần biết hết 15 pool, chỉ cần biết các cơ hội tiết kiệm hàng đầu. Mỗi pool có 10 feature trả lời 10 câu hỏi: giá so với OD bao nhiêu, rủi ro thế nào, hiệu quả vCPU/$ ra sao, ở AZ nào, biến động giá có cao không. Hai flag cuối là tín hiệu từ AWS — rebalance và termination — báo trước khi spot có vấn đề.

---

## Slide 4 — ACTION SPACE — Cấu trúc

**Hiển thị:**

> **Không gian hành động là tập rời rạc gồm 121 hành động:**
>
> $$A = 8 \text{ operations} \times 15 \text{ pools} + 1 \text{ HOLD} = 121$$

**Vì sao chọn cấu trúc này?**

| Yêu cầu | Cách giải quyết |
|---|---|
| Quyết định **rời rạc** (không liên tục) | Tập hữu hạn 121 action → phù hợp DQN (Q-learning) |
| Mỗi quyết định cần **3 thông tin**: làm gì, loại máy, AZ | Tích Descartes 8 × 5 × 3 = đủ phủ mọi tổ hợp |
| Giá Spot và rủi ro **khác nhau theo pool** | Mỗi `(type, AZ)` là 1 pool riêng → agent chọn được pool tối ưu |
| Đôi khi **không nên can thiệp** | Thêm `HOLD` đặc biệt → agent có thể "giữ nguyên" |
| Tránh **action vô nghĩa** (vd: release pool rỗng) | Action mask chặn trước khi argmax |

**Lý do không gộp/tách khác:**

- ❌ **Continuous action** (ví dụ "scale ±N instances"): không phù hợp DQN, mất khả năng chọn pool cụ thể
- ❌ **2 agent riêng** (1 chọn op, 1 chọn pool): mất tương quan giữa op và pool, học khó
- ✅ **1 flat 121**: đơn giản, ổn định, áp dụng được action mask dễ dàng

**Cách quy đổi số → hành động cụ thể:**

> **Bước 1:** Chia số cho **15** để biết **thao tác** (kết quả phép chia nguyên).
> **Bước 2:** Lấy **số dư** khi chia cho 15 để biết **pool** (vị trí 0-14 trong bảng pool).

```
                ┌─────────────────────────┐
   action  ──→  │   0  → 14  : PROVISION_SPOT       │
   (0-120)      │  15  → 29  : PROVISION_ONDEMAND   │
                │  30  → 44  : RELEASE_SPOT         │
                │  45  → 59  : RELEASE_ONDEMAND     │
                │  60  → 74  : CONVERT_TO_OD        │
                │  75  → 89  : CONVERT_TO_SPOT      │
                │  90  → 104 : REBALANCE_SPOT       │
                │ 105  → 119 : RESERVE_CAPACITY     │
                │ 120        : HOLD                 │
                └─────────────────────────┘
```

**Bảng 15 pool (chỉ số 0-14):**

| Pool # | Instance type | AZ |  | Pool # | Instance type | AZ |
|:---:|---|---|---|:---:|---|---|
| **0** | m5.large | 1a |  | **8** | r5.large | 1c |
| **1** | m5.large | 1b |  | **9** | m5.xlarge | 1a |
| **2** | m5.large | 1c |  | **10** | m5.xlarge | 1b |
| **3** | c5.xlarge | 1a |  | **11** | m5.xlarge | 1c |
| **4** | c5.xlarge | 1b |  | **12** | c5.2xlarge | 1a |
| **5** | c5.xlarge | 1c |  | **13** | c5.2xlarge | 1b |
| **6** | r5.large | 1a |  | **14** | c5.2xlarge | 1c |
| **7** | r5.large | 1b |  |  |  |  |

**Ví dụ — Log thật `action=12`:**

```
  12 ÷ 15  =  0  dư 12
     │            │
     │            └──► Pool #12 → c5.2xlarge / 1a
     └──► Thao tác #0 → PROVISION_SPOT

→ Action 12 = "PROVISION_SPOT  c5.2xlarge  ap-southeast-1a"
            = "Tạo 1 Spot c5.2xlarge ở Singapore zone 1a"
```

**Ví dụ thứ 2 — `action=72`:**

```
  72 ÷ 15  =  4  dư 12
     │            │
     │            └──► Pool #12 → c5.2xlarge / 1a
     └──► Thao tác #4 → CONVERT_TO_OD

→ Action 72 = "CONVERT_TO_OD  c5.2xlarge  ap-southeast-1a"
```

**Speaker notes:**
> Em chọn cấu trúc 121 action rời rạc vì 4 lý do. Thứ nhất, bài toán quyết định của em là rời rạc, phù hợp DQN. Thứ hai, mỗi quyết định cần 3 thông tin (làm gì, trên máy nào, ở đâu) nên dùng tích 8×5×3. Thứ ba, giá Spot khác nhau theo pool nên không thể gộp. Thứ tư, thêm HOLD để agent có thể không can thiệp.
>
> Về cách quy đổi số ra hành động — em dùng phép chia có dư. Lấy số chia 15: kết quả là thao tác, số dư là vị trí pool. Ví dụ action 12: 12 chia 15 được 0 dư 12, nghĩa là thao tác số 0 (PROVISION_SPOT) trên pool số 12 (c5.2xlarge/1a). Action 72: 72 chia 15 được 4 dư 12, nghĩa là thao tác số 4 (CONVERT_TO_OD) cũng trên pool 12. Cách đánh số này đảm bảo mọi số từ 0 đến 120 đều quy ra đúng 1 hành động cụ thể, không trùng lặp.

---

## Slide 5 — ACTION SPACE — 9 loại Operation & Action Mask

**Hiển thị:**

**9 thao tác:**

| # | Operation | Ý nghĩa |
|:---:|---|---|
| 0 | `PROVISION_SPOT` | Tạo thêm 1 Spot worker |
| 1 | `PROVISION_ONDEMAND` | Tạo thêm 1 On-Demand worker |
| 2 | `RELEASE_SPOT` | Terminate 1 Spot worker |
| 3 | `RELEASE_ONDEMAND` | Terminate 1 On-Demand worker |
| 4 | `CONVERT_TO_OD` | Chuyển Spot → On-Demand (an toàn) |
| 5 | `CONVERT_TO_SPOT` | Chuyển On-Demand → Spot (tiết kiệm) |
| 6 | `REBALANCE_SPOT` | Di chuyển Spot sang pool khác |
| 7 | `RESERVE_CAPACITY` | Giữ OD dự phòng |
| 8 | `HOLD` | Không thay đổi |

---

**Action Mask — vì sao cần?**

> Trong 121 action, ở mỗi state **không phải action nào cũng hợp lệ**.
> Nếu để agent tự học né, sẽ phí hàng triệu transition để học các luật cơ bản.
> → **Filter trước**, chỉ cho agent chọn trong tập action hợp lệ.

**Luồng quyết định mỗi step:**

```
   Trạng thái hệ thống (state)
       │
       ▼
   Mạng neural cho điểm 121 action
   (action càng tốt → điểm càng cao)
       │
       ▼
   ┌──────────────────┐
   │  Action Mask     │ ◄── Loại các action không hợp lệ
   │  (gán điểm = -∞) │     (vd: release pool rỗng)
   └──────────────────┘
       │
       ▼
   Chọn action có điểm CAO NHẤT
   trong các action còn lại
       │
       ▼
   Hành động được thực thi
```

**Ví dụ minh họa — Pool m5.large/1a có: 0 Spot, 2 OD**

| Action thử | Hợp lệ? | Lý do |
|---|:---:|---|
| `PROVISION_SPOT` | ✅ | Có thể tạo thêm |
| `RELEASE_SPOT` | ❌ | Pool có 0 Spot — không có gì để xóa |
| `CONVERT_TO_OD` | ❌ | Pool có 0 Spot — không có Spot để convert |
| `CONVERT_TO_SPOT` | ✅ | Có 2 OD, có thể chuyển |
| `HOLD` | ✅ | Luôn hợp lệ |

→ Agent **không bao giờ chọn được** `RELEASE_SPOT` hay `CONVERT_TO_OD` ở pool này.

**Speaker notes:**
> Có 9 thao tác từ tạo mới, giải phóng, chuyển đổi đến HOLD. Mỗi thao tác có ý nghĩa cụ thể trong việc điều phối fleet.
>
> Về Action Mask — đây là kỹ thuật quan trọng giúp agent học nhanh hơn nhiều. Quy trình ra quyết định gồm 4 bước. Bước 1: agent nhìn vào trạng thái hệ thống. Bước 2: mạng neural chấm điểm cho từng action — action nào được dự đoán là tốt thì điểm cao, action xấu thì điểm thấp. Bước 3: action mask loại bỏ các action không hợp lệ — em gán điểm của chúng bằng âm vô cực để chắc chắn không được chọn. Bước 4: agent chọn action còn lại có điểm cao nhất. Ví dụ pool có 0 Spot và 2 OD: RELEASE_SPOT và CONVERT_TO_OD bị loại vì không có Spot để xóa hay chuyển. Agent chỉ chọn được trong các action hợp lệ. HOLD luôn hợp lệ để đảm bảo agent không bị kẹt khi tất cả action khác đều bị mask.

---

## Slide 6 — REWARD FUNCTION

**Hiển thị:**

**Hàm phần thưởng tại mỗi bước:**

$$R_t = \text{savings}_t + B_\text{spot} + B_\text{SLA} - P_\text{SLA} - P_\text{interrupt} - P_\text{migration} - P_\text{concentration} - P_\text{fleet} - P_\text{empty}$$

Trong đó $B_\text{SLA}$ là bonus được trao khi SLA health vượt ngưỡng (mặc định **0.95**). Các hệ số $\lambda$ được tinh chỉnh qua nhiều phiên bản huấn luyện (v1 đến v41) để đảm bảo agent không nghiêng hẳn về một thái cực: chi phí thấp - SLA kém, hoặc SLA tốt - chi phí cao.

**Bảng hệ số (v41):**

| Thành phần | Hệ số | Ý nghĩa |
|---|:---:|---|
| `savings_t` | `+1.0` | Chênh lệch chi phí so với baseline OD-only |
| `B_spot` (provision_spot_bonus) | `+0.9` | Khuyến khích agent dùng Spot |
| `B_SLA` (sla_bonus) | `+1.0` | Thưởng khi SLA health ≥ 0.95 |
| `P_SLA` (sla_penalty) | `-5.5` | Phạt theo số job miss deadline |
| `P_empty` (empty_fleet_penalty) | `-1.0` | Phạt khi fleet=0 mà có pending jobs |
| `P_migration` (convert / rebalance) | `-0.4 / -0.8` | Phạt convert OD ↔ Spot và rebalance AZ |
| `P_interrupt` (interrupt_penalty) | `-0.1` | Phạt khi Spot bị thu hồi |
| `P_concentration` | `-0.05` | Phạt khi tập trung quá nhiều vào 1 AZ |
| `P_fleet` (size + idle) | `-0.0025 / -0.015` | Phạt over-provision và idle |

→ **`P_SLA` lớn gấp 5.5× các penalty còn lại** → tín hiệu rõ: *ưu tiên SLA trước, tiết kiệm sau*.

**Speaker notes:**
> Reward gồm 3 phần dương — savings là chính, cộng bonus khi dùng Spot và đạt SLA — và 6 phần âm là các penalty. Quan trọng nhất là hệ số P_SLA bằng 5.5, gấp khoảng 5 lần các penalty khác. Cách thiết kế này dạy agent rằng: thà spend thêm để giữ SLA còn hơn tiết kiệm rồi job fail. Các hệ số lambda em đã tinh chỉnh qua khoảng 40 phiên bản training, từ v1 đến v41, để cân bằng được giữa cost và SLA — tránh việc agent học theo một thái cực, ví dụ chỉ tối ưu cost mà bỏ SLA, hoặc ngược lại giữ toàn OD đắt tiền.

---

## Slide 7 — ĐỘNG HỌC TRẠNG THÁI (State Transition)

**Hiển thị:**

> **Sau mỗi hành động `a_t`, hệ thống chuyển từ trạng thái `s_t` sang `s_{t+1}` qua 4 nhóm cập nhật:**

**Sơ đồ chuyển trạng thái:**

```
   s_t  +  a_t
     │
     ▼
┌────────────────────────────────────────────┐
│ 1. CẬP NHẬT FLEET                          │
│    Áp dụng hành động lên pool:             │
│      PROVISION  → +1 instance              │
│      RELEASE    → -1 instance              │
│      CONVERT    → swap Spot ↔ OD           │
│      REBALANCE  → di chuyển sang pool khác │
│      HOLD       → giữ nguyên               │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ 2. SINH WORKLOAD MỚI                       │
│    Workload generator tạo job mới theo:    │
│      base_arrival_rate × hourly_profile    │
│      + spike ngẫu nhiên (xác suất 2%)      │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ 3. UPDATE THỊ TRƯỜNG                       │
│    Market simulator update:                │
│      • Giá Spot mỗi pool (đọc từ CSV)      │
│      • Sự kiện interrupt (random theo prob)│
│      • Rebalance / Termination flag        │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ 4. XỬ LÝ JOB & TÍNH METRIC                 │
│    • Schedule pending → running pool       │
│    • Hoàn thành job đủ thời lượng          │
│    • Đếm job miss deadline (SLA)           │
│    • Tính chi phí step + reward            │
└────────────────────────────────────────────┘
     │
     ▼
   s_{t+1}  (state vector 92 chiều mới)
```

**Đặc điểm động học:**

| Nguồn ngẫu nhiên | Ảnh hưởng đến |
|---|---|
| Workload arrival (Poisson + spike) | pending, running, forecast |
| Giá Spot (chuỗi thời gian từ CSV) | price_ratio, price_CV, savings |
| Sự kiện interrupt (Bernoulli theo `interrupt_prob`) | num_spot ↓, sla_health ↓ |
| Hourly profile thời gian | base_arrival_rate × hệ số giờ |

→ **MDP không tất định** — agent phải học chính sách robust trước biến động thị trường + workload.

**Speaker notes:**
> Sau mỗi hành động, môi trường chuyển sang trạng thái mới qua 4 nhóm cập nhật. Đầu tiên áp dụng action lên fleet — provision tạo thêm instance, release xóa, convert hoán đổi, rebalance di chuyển, HOLD giữ nguyên. Bước hai sinh workload mới theo arrival rate có pattern giờ trong ngày, kèm spike ngẫu nhiên. Bước ba update thị trường: giá Spot mỗi pool đọc từ CSV scenario, đồng thời tung xúc xắc xem có interrupt nào không dựa trên xác suất tương ứng. Bước bốn xử lý job: schedule pending sang running, hoàn thành job, đếm job miss SLA, tính chi phí và reward. Sau 4 bước này em build state vector mới 92 chiều cho timestep kế tiếp. Đây là MDP không tất định vì cả workload, giá và interrupt đều ngẫu nhiên — agent phải học chính sách robust trước biến động.

---

## Slide 8 — THAM SỐ MÔI TRƯỜNG

**Hiển thị:**

**Thông số Episode:**

- **Quy mô:** 672 bước / episode, mỗi bước = 15 phút thực tế (1 tuần)
- **Kết thúc:** dừng khi đạt 672 bước (không có điều kiện thắng/thua)

**Quy trình 1 bước (Step):**

1. Agent quan sát state → chọn action (qua action mask)
2. Environment thực thi action: provision / release / convert / hold
3. Workload generator sinh job mới theo `hourly_profile`
4. Market simulator update giá Spot + interrupt event
5. Tính reward: savings, penalty, bonus
6. Update state cho bước tiếp theo

**Mục tiêu huấn luyện:**

> Tìm policy `π(a|s)` tối đa hóa **tổng phần thưởng tích lũy** sau 672 bước trên 4 scenario:
> `stable`, `volatile`, `spike`, `az_divergence`.

**Speaker notes:**
> Một episode kéo dài 672 bước, tương đương 1 tuần thực tế. Mỗi step thực hiện theo 6 giai đoạn: quan sát, chọn action có mask, env thực thi, sinh workload mới, update market, tính reward. Agent train trên 4 scenario khác nhau để học tổng quát: stable cho học cơ bản, volatile cho biến động, spike cho tải đột biến, az_divergence cho khi 1 AZ đắt lên.

---

## Slide 9 — THAM SỐ CHÍNH — Môi trường

**Hiển thị:**

| Tham số | Giá trị | Ý nghĩa |
|---|:---:|---|
| `N_TYPES` | 5 | m5.large, c5.xlarge, r5.large, m5.xlarge, c5.2xlarge |
| `N_AZS` | 3 | ap-southeast-1a / 1b / 1c |
| `N_POOLS` | 15 | `N_TYPES × N_AZS` |
| `STATE_DIM` | 92 | Vector trạng thái chuẩn hóa `[0,1]` |
| `N_ACTIONS` | 121 | `8 × 15 + 1` |
| `MAX_STEPS` | 672 | 1 tuần × 24h × 4 step/h |
| `MAX_INSTANCES` | 20 | Giới hạn cluster |
| `MAX_PER_AZ` | 10 | Giới hạn mỗi AZ |
| `MAX_VCPU` | 160 | Giới hạn tài nguyên |
| `STEP_INTERVAL` | 15 min | Chu kỳ quyết định |

**Scenarios huấn luyện:**

`stable` · `volatile` · `spike` · `az_divergence`

**Speaker notes:**
> Các tham số cố định mô tả kích thước bài toán. 5 type × 3 AZ là 15 pool. Cluster giới hạn 20 instance để mô phỏng quota AWS. Step interval 15 phút khớp với chu kỳ scaling thực tế của Karpenter. 4 scenario được sample đều với trọng số 1.0 mỗi cái.

---

## Slide 10 — THAM SỐ CHÍNH — Agent DQN

**Hiển thị:**

| Nhóm | Tham số | Giá trị |
|---|---|:---:|
| **Thuật toán** | Double DQN + Dueling + PER | ✓ |
| **Network** | Factored Q-network | 2 × 512 hidden |
| **Learning rate** | `lr` | `2e-5` |
| **Discount** | `γ` | `0.99` |
| **Batch size** | — | `512` |
| **Replay buffer** | — | `500,000` |
| **Exploration** | ε-greedy, decay | `1.0 → 0.08 / 450k steps` |
| **Target net** | sync | mỗi `8,000` grad steps |
| **PER** | `α / β_start` | `0.5 / 0.5` |
| **Episodes** | total | `6,000` |
| **Hardware** | GPU | RTX 2000 Ada / Tesla T4 |
| **Training time** | — | ~12-15 giờ |

**Action selection:**
> `action = argmax_a Q(s, a)`  với `a` chỉ trong `valid_actions` (sau action mask)

**Speaker notes:**
> Em dùng kết hợp 3 cải tiến DQN: Double DQN giảm overestimation, Dueling tách giá trị state và advantage, PER ưu tiên transition khó học. Network factored để học action 121 chiều hiệu quả hơn. Learning rate thấp 2e-5 chống Q-value bùng nổ. Buffer lớn 500k để tận dụng off-policy. Train 6000 episode khoảng 12-15 giờ trên GPU consumer.

---

## Slide 11 — TÓM TẮT MÔ HÌNH HỆ THỐNG

**Hiển thị:**

**Bài toán Spot orchestration → MDP có cấu trúc:**

| Thành phần | Mô tả |
|---|---|
| **State (92)** | Thị trường + workload + health + utilization |
| **Action (121)** | 8 op × 15 pool + HOLD, có action mask |
| **Reward** | Savings − SLA/interrupt/migration penalty + bonus |
| **Episode** | 672 bước × 15 phút = 1 tuần |
| **Environment** | 15 pool (5 type × 3 AZ), 4 scenario |
| **Agent** | Factored Double Dueling DQN + PER |

**Thông điệp:**

> Thiết kế **state-action-reward đầy đủ** + **action mask chặt chẽ** + **agent có cấu trúc phù hợp** cho phép học chính sách cân bằng cost ↔ SLA trên môi trường mô phỏng.

**Tiếp theo:** Kết quả huấn luyện và đánh giá trên 4 scenario.

**Speaker notes:**
> Tóm lại phần thiết kế hệ thống. Bốn thành phần: state 92 chiều, action 121, reward đa mục tiêu, episode 1 tuần. Môi trường có 15 pool và 4 scenario. Agent kết hợp 3 cải tiến DQN. Đây là nền tảng cho phần tiếp theo — kết quả huấn luyện và so sánh với baseline.

---

# IV. KẾT QUẢ HUẤN LUYỆN VÀ ĐÁNH GIÁ

---

## Slide 12 — KẾT QUẢ HUẤN LUYỆN — Đường cong training

**Hiển thị:** 4 chart song song theo dõi training v41 (RunPod RTX 2000 Ada, ~1,800 episodes)

```
┌─────────────────────────┬─────────────────────────┐
│      📉 COST             │      📊 TRAINING LOSS    │
│   1700 → 1100$/ep       │   Bướu → hội tụ ~1.0    │
│   (tiết kiệm dần)       │   (training stable)     │
├─────────────────────────┼─────────────────────────┤
│      ✅ SLA              │      💰 REWARD           │
│   91% → 97%             │   -2000 → +1200         │
│   (vượt mục tiêu 95%)   │   (có lợi nhuận)        │
└─────────────────────────┴─────────────────────────┘
```

**Diễn giải từng chart:**

| Chart | Xu hướng | Ý nghĩa |
|---|---|---|
| **Cost** | 1,700$ → 1,100$ | Agent học cách vận hành **rẻ hơn dần** |
| **Loss** | Tăng → hội tụ ~1.0 | Pattern bình thường của DQN + PER, **không divergence** |
| **SLA** | 91% → 97% | Vượt **mục tiêu 95%** sau ~ep 400 |
| **Reward** | -2000 → +1200 | Vượt mốc **break-even** sau ~ep 500 |

**Eval-gated checkpointing:**
- Mỗi 100 episode → chạy greedy eval trên 4 scenario × 3 episode
- Chỉ save `best_eval.pth` khi **SLA ≥ 94.5%** VÀ score cải thiện
- → Tránh save model "lệch": cost thấp nhưng SLA tệ

→ **Cả 4 chỉ số đều hội tụ tốt** — không có dấu hiệu collapse, model đáng tin để eval.

**Speaker notes (~90s):**
> Slide này trình bày diễn biến huấn luyện DQN agent qua khoảng 1,800 episode trên RunPod RTX 2000 Ada. Em quan sát 4 chỉ số song song.
>
> **Chart Cost** ở góc trái trên — chi phí trung bình mỗi episode giảm đều từ khoảng **1,700 đô xuống còn 1,100**. Đường cong không dốc đứng mà giảm có nhịp, cho thấy agent học có chiến lược chứ không phải random.
>
> **Chart Training loss** ở phải trên — loss có hình **dạng bướu**: tăng nhanh ban đầu rồi giảm dần. Đây là pattern bình thường của DQN dùng prioritized experience replay. Loss tăng lúc đầu vì agent đang khám phá nhiều state mới, network chưa kịp adapt. Sau khoảng episode 200, loss giảm dần và hội tụ về ~1.0, ổn định, không có dấu hiệu divergence.
>
> **Chart SLA** dưới trái — tỉ lệ job hoàn thành đúng hạn tăng từ **91% lên 97%**. Đường ngang 95% là mục tiêu em đặt ra. Agent vượt mục tiêu sau khoảng episode 400 và duy trì ổn định cho tới cuối.
>
> **Chart Reward** dưới phải — reward tổng tăng từ **-2,000 lên +1,200**. Đường ngang 0 là mốc 'break-even'. Sau khoảng episode 500, reward vượt 0 và tiếp tục tăng, cho thấy agent thực sự học được cách tối ưu chứ không chỉ tránh phạt.
>
> Tổng kết: cả 4 chỉ số đều cho thấy training hội tụ tốt. Em dùng kỹ thuật eval-gated checkpointing nên best model em chọn là tại điểm SLA cao và cost thấp đồng thời, không phải điểm cuối training.

**Q&A có thể bị hỏi:**

| Câu hỏi | Trả lời sẵn |
|---|---|
| *Loss tăng đầu có bất thường?* | Không — pattern bình thường của DQN do epsilon-greedy exploration. State distribution thay đổi → loss tăng tạm thời. |
| *Sao chỉ ~1,800 episode, không phải 6,000?* | Em dùng early stopping (patience 12 × 100 ep). Khi eval không cải thiện qua 1,200 episode → dừng. |
| *SLA 97% có đủ tốt?* | Mục tiêu 95% (chuẩn SLA cloud). 97% vượt mục tiêu. Cao hơn 99% tốn chi phí không tương xứng. |
| *Reward dao động mạnh cuối — có ổn không?* | Variance episode-level là tự nhiên do workload + interrupt có yếu tố ngẫu nhiên. Xu hướng quan trọng hơn từng điểm. |

---

## Slide 13a — ĐIỀU KIỆN ĐÁNH GIÁ

**Hiển thị:**

> **Script:** `experiments/evaluate_kaggle.py`
> **Model:** `best_mixed.pth` của v41 (checkpoint eval-gated tốt nhất)
> **Output:** `results/kaggle/eval_v41/{eval_summary.json, *.png, run_log.txt}`

**Setup chạy đánh giá:**

| Tham số | Giá trị |
|---|---|
| **Số scenario** | 4 (stable, volatile, spike, az_divergence) |
| **Số episode / scenario** | **30** |
| **Tổng số episode** | 4 × 30 = **120 episode** |
| **Số bước / episode** | 168 |
| **Workload** | `base_arrival=15/h, peak=2.0×, avg_duration=3h` |
| **Seeding** | `seed = ep × 7 + 42` (deterministic) |
| **ε (exploration)** | **0.0** (greedy — không random) |
| **Action mask** | Có — chỉ chọn action hợp lệ |

**7 phương pháp được so sánh:**

| Nhóm | Phương pháp |
|---|---|
| **Mô hình đề xuất** | `DQN v41` (best_mixed.pth) |
| **Baseline ngây thơ** | `OnDemand`, `Spot`, `Random` |
| **Baseline rule-based** | `Threshold`, `CheapestAZ`, `CheapestType` |

**Đảm bảo tính công bằng:**

- ✅ Cùng env config, workload, seed → mọi agent thấy **cùng 1 tình huống**
- ✅ Greedy `ε=0` → loại trừ may rủi exploration
- ✅ 30 episode/scenario → đủ trung bình hóa biến động ngẫu nhiên
- ✅ Cùng action mask → so sánh apple-to-apple

**Speaker notes:**
> Trước khi xem kết quả, em nói rõ điều kiện đánh giá để đảm bảo công bằng. Em chạy script evaluate_kaggle.py với model best_mixed.pth — đây là checkpoint v41 được chọn qua eval-gated trong training. Mỗi scenario chạy 30 episode, tổng cộng 120 episode. Mỗi episode có 168 bước. Workload generator dùng cấu hình arrival rate 15 job/giờ, peak 2x, mỗi job trung bình 3 giờ. Seeding deterministic theo công thức ep × 7 + 42 — cùng seed thì kết quả giống nhau, đảm bảo reproducibility. Quan trọng nhất là em dùng epsilon = 0 — agent chạy greedy 100%, không random, để loại trừ may rủi. Bảy phương pháp được so sánh gồm DQN của em, 3 baseline ngây thơ và 3 baseline rule-based. Tất cả cùng env, cùng seed, cùng action mask — đảm bảo so sánh công bằng.

---

## Slide 13 — KẾT QUẢ ĐÁNH GIÁ — So sánh baseline

**Hiển thị:**

> **Nguồn dữ liệu:** `results/kaggle/eval_v41/eval_summary.json` — đánh giá DQN v41 với **6 baseline** trên **4 scenario**.

**Tổng hợp trên 4 scenario (giá trị trung bình):**

| Phương pháp | Cost ($) | SLA | Savings vs OD |
|---|:---:|:---:|:---:|
| `Spot-only` (toàn Spot) | ~97 | 39% | 85% (nhưng SLA tệ) |
| `Threshold` | ~187 | 67% | 71% |
| `CheapestAZ` | ~202 | 71% | 68% |
| `CheapestType` | ~208 | 72% | 67% |
| `Random` | ~496 | 75% | 22% |
| **OnDemand-only** (baseline) | 636 | 89% | 0% |
| 🟢 **DQN v41 (đề xuất)** | **400** | **95%** | **37%** |

→ DQN là phương pháp **duy nhất đạt cả 2 mục tiêu**: tiết kiệm ≥ 30% **và** SLA ≥ 95%.

**Kết quả chi tiết DQN v41 trên 4 scenario:**

| Scenario | Cost ($) | SLA | Savings vs OD |
|---|:---:|:---:|:---:|
| `stable` | 388.3 | 94.7% | **38.97%** |
| `volatile` | 398.6 | 95.0% | **37.36%** |
| `spike` | 417.1 | 95.0% | **34.45%** |
| `az_divergence` | 398.0 | 95.1% | **37.45%** |

**Thông điệp:**

> **DQN v41 đạt mục tiêu đề ra: trung bình ~37% savings, SLA ~95% trên cả 4 scenario.**

**Speaker notes:**
> Đây là kết quả eval lấy trực tiếp từ file eval_summary.json của v41. Em so sánh với 6 baseline khác nhau. Spot-only tiết kiệm tới 85% nhưng SLA chỉ 39% — không khả thi. Các rule-based như Threshold, CheapestAZ, CheapestType tiết kiệm 67-71% nhưng SLA chỉ 67-72%, vi phạm mục tiêu. Random làm sanity check. OD-only an toàn nhưng không tiết kiệm. DQN của em là phương pháp duy nhất đạt được cả 2 mục tiêu: tiết kiệm 37% và SLA 95%. Chi tiết từng scenario rất ổn định: SLA dao động 94.7-95.1%, savings dao động 34.4-38.97%. Scenario spike khó nhất vì workload đột biến nên savings hơi thấp hơn, nhưng SLA vẫn giữ 95%.

---

## Slide 14 — KẾT QUẢ ĐÁNH GIÁ — Phân tích hành vi agent

**Hiển thị:**

> **Nguồn:** `results/kaggle/eval_v41/dqn_action_dist.png` — phân bố action của DQN v41 trên 4 scenario kết hợp.

**Phân bố action (tỉ lệ %):**

| Action | Tỉ lệ | Diễn giải |
|---|:---:|---|
| `PROVISION_SPOT` | **31.5%** | Thao tác phổ biến nhất → agent ưu tiên Spot |
| `RELEASE_ONDEMAND` | **15.4%** | Giải phóng OD đắt khi không cần |
| `PROVISION_ONDEMAND` | **14.3%** | Provision OD khi Spot không khả thi |
| `RESERVE_CAPACITY` | **8.9%** | Dự phòng OD khi rủi ro interrupt cao |
| `REBALANCE_SPOT` | **8.7%** | Di chuyển Spot giữa pool |
| `CONVERT_TO_SPOT` | **7.5%** | Chuyển OD → Spot để tiết kiệm |
| `RELEASE_SPOT` | **7.1%** | Giải phóng Spot dư |
| `HOLD` | **4.1%** | Hiếm khi giữ nguyên |
| `CONVERT_TO_ONDEMAND` | **2.6%** | Ít convert sang OD — tránh drain cost |

*[Hình: bar chart action distribution]*

**Quan sát chính:**

- 🟢 **PROVISION_SPOT là action thường xuyên nhất (31.5%)** → agent thực sự ưu tiên Spot
- 🟢 **CONVERT_TO_OD thấp nhất trong nhóm có ý nghĩa (2.6%)** → agent học được rằng convert tốn drain, hiếm khi dùng
- 🟢 **PROVISION_OD + RELEASE_OD ≈ 30%** → agent có dùng OD nhưng vòng đời ngắn (provision → release sớm)
- 🔵 **HOLD chỉ 4.1%** → agent **rất active**, can thiệp gần như mỗi step

**Thông điệp:**

> Phân bố action **không đồng đều** (không random) → agent học được **chính sách có cấu trúc**: ưu tiên Spot, tránh convert tốn kém, can thiệp chủ động.

**Speaker notes:**
> Em phân tích phân bố action thực tế của DQN v41 trên 4 scenario. Action thường xuyên nhất là PROVISION_SPOT chiếm 31.5%, chứng tỏ agent thực sự ưu tiên Spot rẻ. Đáng chú ý là CONVERT_TO_ONDEMAND chỉ 2.6% — agent học được rằng convert phải drain instance nên tốn kém, hiếm khi dùng. RELEASE_ONDEMAND 15.4% và PROVISION_ONDEMAND 14.3% gần bằng nhau, cho thấy agent dùng OD chỉ khi cần thiết rồi giải phóng sớm chứ không giữ lâu. HOLD chỉ 4.1% — agent rất chủ động, gần như mỗi step đều can thiệp gì đó. Phân bố này lệch rõ ràng, không đều, chứng tỏ agent không random mà học được chính sách có cấu trúc rõ ràng.

---

## Slide 15 — ABLATION — Áp dụng cho Workload CI/CD ngắn

**Hiển thị:**

> **Câu hỏi:** Agent được train với job dài 3 giờ. Liệu có **generalize** sang CI/CD job ngắn (~15 phút) không?

**Setup ablation:**

| Tham số | Eval gốc | **Eval CI/CD** |
|---|:---:|:---:|
| `avg_job_duration` | 12 step (3 giờ) | **1 step (15 phút)** |
| `base_arrival_rate` | 3.75 / step (15/h) | **15 / step (60/h)** |
| Khác | giữ nguyên | giữ nguyên |

→ Model **không retrain** — chỉ thay workload lúc eval.

**Kết quả tổng hợp:**

| Phương pháp | Cost ($) | SLA | Savings vs OD |
|---|---:|---:|---:|
| OnDemand | 2,567 | 99.3% | 0% |
| Threshold | 668 | 93.0% | 74.0% |
| CheapestAZ | 671 | 92.9% | 73.9% |
| CheapestType | 666 | 92.8% | 74.1% |
| Random | 2,122 | 96.5% | 17.4% |
| Spot-only | 378 | 75.2% | 85.3% (SLA tệ) |
| 🟢 **DQN v41** | **645** | **97.4%** | **74.9%** |

**Per-scenario DQN v41 (CI/CD workload):**

| Scenario | Cost | SLA | Savings |
|---|---:|---:|---:|
| stable | 637.6 | 97.30% | **75.16%** |
| volatile | 645.1 | 97.46% | **74.87%** |
| spike | 659.1 | 97.35% | **74.33%** |
| az_divergence | 640.1 | 97.35% | **75.07%** |

**So sánh với eval gốc (in-distribution):**

| Metric | Eval gốc (3h) | **Eval CI/CD (15p)** | Δ |
|---|:---:|:---:|:---:|
| Savings | 37.1% | **74.9%** | **×2** |
| SLA | 95.0% | **97.4%** | +2.4pp |

**Thông điệp:**

> Agent **generalize tốt** sang workload CI/CD ngắn dù chưa thấy bao giờ — savings tăng gấp đôi, SLA vẫn ≥ 97%. Lý do: job ngắn → ít rủi ro interrupt giữa job → agent tận dụng Spot triệt để hơn.

**Speaker notes:**
> Sau khi có kết quả eval chính, em làm thêm 1 thí nghiệm ablation: liệu agent train với job dài 3 giờ có dùng được cho CI/CD thực tế với job 15 phút không. Em giữ nguyên model v41, chỉ thay workload config: avg_job_duration từ 12 step xuống 1 step, arrival rate tăng 4 lần để giữ áp lực. Kết quả vượt mong đợi: savings tăng từ 37% lên 75%, SLA tăng từ 95% lên 97.4%. Lý do là với job ngắn, rủi ro spot interrupt giữa job giảm gần như bằng 0 — agent có thể tận dụng spot triệt để hơn. Đây là tín hiệu tốt cho việc triển khai thực tế trong hệ thống CI/CD: agent học chính sách robust, không phụ thuộc cứng vào độ dài job cụ thể.
