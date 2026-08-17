# Đề xuất Pipeline nghiên cứu: Sinh dataset ảnh mô phỏng cứu hộ từ ảnh thiên tai

## 1. Bối cảnh và bài toán nghiên cứu

### 1.1. Bài toán
Pipeline này nhằm giải quyết bài toán thiếu dữ liệu có nhãn cho mô hình cứu hộ thông minh. Các bộ dataset ảnh thiên tai hiện có (ví dụ incidents-1M) cung cấp ảnh thực tế, nhưng thiếu các mẫu ảnh mô phỏng cảnh cứu hộ đang diễn ra — loại dữ liệu quan trọng để huấn luyện mô hình nhận diện tình huống, lập kế hoạch cứu hộ, hoặc mô phỏng thảo luận.

### 1.2. Giả định nghiên cứu
Nếu tạo được một bộ dataset có cấu trúc gồm cặp (ảnh thiên tai gốc, ảnh mô phỏng cứu hộ, mô tả ngữ cảnh, hướng dẫn chỉnh sửa, metadata), bộ dataset này có thể:
- Phục vụ huấn luyện mô hình đa phương thức hiểu tình huống cứu hộ.
- Dùng làm benchmark cho các phương pháp sinh/chỉnh sửa ảnh rescue simulation.
- Công bố cho cộng đồng nghiên cứu.

### 1.3. Đầu vào
- Bộ ảnh thiên tai thực tế: incidents-1M (hoặc tập con có sẵn).
- Mô hình ngôn ngữ đa phương thức: Gemma 4 (zero-shot, không fine-tune).
- Mô hình sinh/chỉnh sửa ảnh: FLUX-2 klein 4B + ControlNet / LoRA.

### 1.4. Mức độ can thiệp
- Pipeline tập trung vào mức can thiệp **trung bình**: giữ nguyên bối cảnh và cấu trúc ảnh gốc, chỉ thêm các đối tượng cứu hộ phù hợp.
- Không thay đổi bố cục lớn, không xóa đối tượng nền quan trọng.
- Ngưỡng preservation score cố định >= 0.70 áp dụng cho tất cả các mức can thiệp.

### 1.5. Đầu ra mong muốn
- Một bộ dataset gồm N cặp ảnh gốc – ảnh biến thể cứu hộ, kèm metadata và chỉ số đánh giá.
- Báo cáo phân tích chất lượng dataset.
- Quy trình tái sử dụng được cho các phiên bản mô hình khác nhau.

---

## 2. Kiến trúc pipeline đề xuất

```
Disaster Image (incidents-1M)
        │
        ▼
┌─────────────────────────┐
│ Bước 1: Tiền xử lý &    │
│        Lọc dữ liệu       │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Bước 2: Hiểu cảnh        │
│        (Scene            │
│        Understanding)    │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Bước 3: Sinh Rescue      │
│        Instructions      │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Bước 4: Sinh k ảnh       │
│        (Multi-prompt     │
│        Generation)       │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Bước 5: Chọn n ảnh       │
│        tốt nhất          │
│        (Quality-based     │
│        Selection)        │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Bước 6: Đánh giá         │
│        chất lượng        │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Bước 7: Lặp lại nếu      │
│        chưa đạt chuẩn   │
└─────────────────────────┘
        │
        ▼
   Dataset + Metadata
```

---

## 3. Chi tiết từng bước

### Bước 1: Tiền xử lý và lọc dữ liệu

**Mục tiêu:** Từ incidents-1M, lọc ra các ảnh có chất lượng đủ tốt để làm đầu vào cho pipeline.

**Công cụ:**
- Python script + Pillow (PIL) để kiểm tra kích thước, định dạng.
- pandas để quản lý metadata.
- (Tùy chọn) Mô hình phát hiện mờ/noise: Laplacian variance, BRISQUE.

**Chỉ số:**
- `valid_image_ratio`: tỷ lệ ảnh vượt lọc / tổng ảnh.
- `resolution_pass_rate`: tỷ lệ ảnh có resolution >= ngưỡng (ví dụ 512x512).
- `noise_score`: điểm mờ/nhiễu (dùng Laplacian hoặc BRISQUE).
- `scene_type_coverage`: bao phủ các loại cảnh (flood, fire, landslide, v.v.).

**Đầu vào:** incidents-1M raw images.
**Đầu ra:** danh sách `filtered_images.jsonl` gồm `image_path`, `width`, `height`, `noise_score`, `scene_type_hint`.

---

### Bước 2: Hiểu cảnh (Scene Understanding)

**Mục tiêu:** Từ mỗi ảnh gốc, trích xuất mô tả văn bản chi tiết về bối cảnh thiên tai, các đối tượng có mặt, và các yếu tố liên quan đến cứu hộ.

**Công cụ:**
- Gemma 4 (zero-shot, không fine-tune) mô hình đa phương thức, multimodal để phân tích ảnh và sinh caption.
- (Tùy chọn) CLIP ViT-L/14 để kiểm tra độ liên quan giữa caption và ảnh (CLIP score).
- (Tùy chọn) Mô hình nhận diện đối tượng: YOLO, DETR để trích xuất bounding box làm kiểm chứng.

**Chỉ số:**
- `scene_cls_accuracy`: độ chính xác phân loại cảnh (nếu có ground-truth hoặc dùng Gemma 4 tự đánh giá).
- `object_detection_mAP`: độ chính xác phát hiện đối tượng (nếu có nhãn).
- `clip_score`: độ tương đồng giữa caption sinh ra và ảnh đầu vào.
- `caption_length`: kiểm soát độ dài mô tả (tránh quá ngắn hoặc quá dài).

**Đầu vào:** ảnh đã lọc từ bước 1.
**Đầu ra:** `scene_descriptions.jsonl` gồm `image_id`, `scene_summary`, `objects_detected`, `rescue_relevant_elements`, `clip_score`.

---

### Bước 3: Sinh Rescue Instructions

**Mục tiêu:** Từ mô tả cảnh, sinh ra các hướng dẫn chỉnh sửa ảnh (instructions) để biến ảnh thiên tai thành ảnh mô phỏng cứu hộ. Mỗi instruction cần nêu rõ đối tượng thêm vào, đối tượng giữ nguyên, và góc nhìn.

**Công cụ:**
- Gemma 4 (zero-shot, không fine-tune) được prompt với task cụ thể để sinh rescue instructions.
- Prompt template dựa trên few-shot examples (lấy ví dụ từ bộ dữ liệu nhỏ thủ công ban đầu).
- Template instruction:
  - "Thêm [đối tượng cứu hộ] vào [vị trí], giữ nguyên [các đối tượng/cảnh nền cần giữ]."
  - "Chèn [đối tượng] ở [góc nhìn], mức độ can thiệp: [nhẹ/trung bình/mạnh]."

**Chỉ số:**
- `instruction_diversity_score`: sử dụng embedding (ví dụ Gemma 4 embeddings) để đo độ đa dạng của các instruction sinh ra.
- `action_coverage`: tỷ lệ các loại hành động cứu hộ được bao phủ (rescue_boat, helicopter, firefighter, v.v.).
- `plausibility_rate`: tỷ lệ instruction hợp lý với cảnh (đánh giá bởi Gemma 4 judge hoặc human review).
- `avg_instruction_length`: kiểm soát độ dài instruction.

**Đầu vào:** `scene_descriptions.jsonl`.
**Đầu ra:** `rescue_instructions.jsonl` gồm `image_id`, `scene_summary`, `instruction`, `rescue_actions`, `preserve_elements`, `viewpoint`.

---

### Bước 4: Sinh k ảnh (Multi-prompt Generation)

**Mục tiêu:** Để chống lỗi sinh ảnh (artifact, sai object, sai bố cục), thay vì sinh một ảnh cho mỗi instruction, pipeline sinh ra k ảnh bằng cách đổi prompt/instruction cho cùng một ảnh gốc. Cách này tăng khả năng tìm ra instruction phù hợp nhất với cảnh, từ đó sinh ra ảnh đạt chất lượng cao.

**Cách hoạt động:**
- Với mỗi ảnh gốc, sinh ra k phiên bản rescue instruction khác nhau (khác cách diễn đạt, khác đối tượng cứu hộ đề xuất, khác mức độ chi tiết).
- Mỗi instruction biến thể được đưa vào FLUX-2 klein 4B để sinh một ảnh.
- Các ảnh sinh ra có cùng ảnh gốc nhưng khác prompt, khác seed.
- Sử dụng FLUX-2 klein 4B để sinh ảnh cho mỗi biến thể prompt.

**Công cụ:**
- **FLUX-2 klein 4B** (mô hình sinh ảnh chính, thay thế SDXL).
- **ControlNet** (Canny, Depth, OpenPose) để giữ cấu trúc ảnh gốc.
- **LoRA** fine-tune trên domain rescue để cải thiện chất lượng sinh đối tượng cứu hộ.
- **Prompt-to-Prompt** hoặc **Attention Sharing** để duy trì tính nhất quán giữa ảnh gốc và ảnh biến thể.

**Tham số đề xuất:**
- k = 3 đến 5 ảnh mỗi instruction (có thể điều chỉnh theo thực nghiệm).
- Mỗi ảnh có `prompt_variant_id` riêng, `seed` riêng, `cfg_scale` riêng.

**Chỉ số:**
- `generation_success_rate`: tỷ lệ sinh ảnh thành công (không lỗi, không artifact nặng).
- `prompt_variant_diversity`: độ đa dạng giữa các prompt biến thể (đo bằng embedding distance).
- `per_variant_clip_iqa`: điểm chất lượng của từng biến thể.

**Đầu vào:** `rescue_instructions.jsonl` + ảnh gốc.
**Đầu ra:** `candidate_images/` + `generation_log.jsonl` gồm `image_id`, `variant_id`, `instruction`, `model_config`, `clip_iqa`, `ssim`.

---

### Bước 5: Chọn n ảnh tốt nhất (Quality-based Selection)

**Mục tiêu:** Từ k ảnh sinh ra ở bước 4, chọn ra n ảnh tốt nhất dựa trên chỉ số đánh giá, để đưa vào dataset cuối cùng. Cơ chế này giúp giảm thiểu ảnh bị lỗi, tăng chất lượng trung bình của dataset.

**Cách hoạt động:**
- Với mỗi ảnh gốc, thu thập k ảnh sinh kèm chỉ số.
- Sắp xếp theo điểm tổng hợp (composite score).
- Chọn top n ảnh (ví dụ n = 1 hoặc n = 2).
- Các ảnh không được chọn được lưu vào `rejected/` để phân tích lỗi.

**Công cụ:**
- **CLIP-IQA** đánh giá từng ảnh.
- **SSIM** so với ảnh gốc.
- **CMMD** kiểm tra lệch phân phối.
- **Gemma 4 judge** đánh giá semantic consistency.
- Hàm composite score O, lấy giá trị nhỏ nhất giữa semantic consistency (SC) và preservation quality (PQ):
  ```
  O = min(SC, PQ)
  ```
  Hàm này đảm bảo ảnh phải đạt đồng thời cả hai tiêu chí: giữ ngữ cảnh ban đầu và giữ cấu trúc ảnh gốc. Nếu một trong hai chỉ số thấp, ảnh sẽ bị loại bỏ.

**Chỉ số:**
- `O_score`: điểm tổng hợp dùng để xếp hạng, được tính bằng `min(semantic_consistency, preservation_score)`.
- `selection_pass_rate`: tỷ lệ ảnh được chọn vào top-n.
- `top_n_quality_gap`: chênh lệch điểm giữa ảnh đứng đầu và ảnh đứng cuối trong top-n.

**Tham số đề xuất:**
- k = 3–5 (số ảnh sinh ra mỗi instruction).
- n = 1–2 (số ảnh được chọn).
- Ngưỡng O_score để chọn top-n: >= 0.70.

**Đầu vào:** `candidate_images/` + `generation_log.jsonl`.
**Đầu ra:** `selected_images/` + `selection_log.jsonl` gồm `image_id`, `variant_id`, `composite_score`, `rank`, `selection_status` (selected / rejected).

---

### Bước 6: Đánh giá chất lượng tổng hợp

**Mục tiêu:** Đánh giá từng mẫu ảnh đã được chọn (top-n) để quyết định có đưa vào dataset hay không.

**Công cụ:**
- **CLIP-IQA** (mô hình đánh giá chất lượng ảnh dựa trên CLIP).
- **CMMD** (Characteristic Multimodal Distance) để so sánh phân phối ảnh sinh với ảnh gốc.
- **Gemma 4** đóng vai trò judge: nhận ảnh gốc, ảnh sinh, và instruction, đưa ra đánh giá về tính hợp lệ, sự nhất quán ngữ nghĩa, và mức độ thực tế.
- **Human review** trên tập validation (ít nhất 10% tổng mẫu) để tạo ground-truth phụ trợ.

**Chỉ số:**
- `clip_iqa`: điểm chất lượng ảnh.
- `cmmd`: khoảng cách phân phối giữa ảnh sinh và ảnh gốc (càng thấp càng tốt).
- `semantic_consistency`: điểm nhất quán ngữ nghĩa (Gemma 4 judge scale 1-5 hoặc binary).
- `preservation_score`: điểm giữ nguyên cấu trúc gốc.
- `human_pass_rate`: tỷ lệ qua kiểm tra người (trên tập validation).

**Ngưỡng chấp nhận đề xuất (có thể điều chỉnh sau pilot):**
- CLIP-IQA >= 0.70
- Semantic consistency >= 0.75 (Gemma 4 judge hoặc human)
- Preservation score >= 0.70
- CMMD <= 0.20 (hoặc ổn định so với phân phối ảnh gốc)
- Human pass rate >= 0.80 (trên tập validation)

**Đầu vào:** ảnh đã chọn từ bước 5.
**Đầu ra:** `evaluation_results.jsonl` gồm `image_id`, `variant_id`, `clip_iqa`, `cmmd`, `semantic_consistency`, `preservation_score`, `quality_status` (accepted / rejected).

---

### Bước 7: Vòng lặp cải thiện (Iterative Refinement)

**Mục tiêu:** Đảm bảo pipeline hội tụ về chất lượng đủ tốt cho public dataset, không chỉ sinh một lần rồi dừng.

**Công cụ:**
- **Automated prompt tuner**: điều chỉnh instruction nếu ảnh sinh bị rejected.
- **Gemma 4 feedback loop**: dùng Gemma 4 phân tích lỗi ảnh sinh (ví dụ: "đối tượng cứu hộ bị biến dạng", "bố cục bị thay đổi quá nhiều") và đề xuất prompt/instruction mới.
- **Hyperparameter search**: grid search hoặc Bayesian optimization trên các tham số (CFG scale, ControlNet condition scale, LoRA weight, số bước denoising, k, n).

**Chỉ số:**
- `regeneration_rate`: tỷ lệ mẫu cần sinh lại so với tổng mẫu.
- `acceptance_rate`: tỷ lệ mẫu được chấp nhận sau tối đa K lần sinh lại (ví dụ K=3).
- `quality_improvement_delta`: chênh lệch điểm đánh giá trước và sau khi tối ưu.
- `convergence_rate`: tốc độ pipeline đạt ngưỡng chấp nhận.

**Điều kiện dừng:**
- Acceptance rate >= ngưỡng đề ra (ví dụ >= 70%).
- Hoặc đạt số mẫu mục tiêu.

**Đầu vào:** `evaluation_results.jsonl` (các mẫu rejected).
**Đầu ra:** `final_dataset/` gồm các mẫu accepted, kèm metadata đầy đủ.

---

## 4. Phân công công việc

### 4.1. Thành phần nhóm và trách nhiệm

| Thành phần | Trách nhiệm chính | Công cụ cần chuẩn bị |
|---|---|---|
| **Nhóm Xử lý ảnh** | Tiền xử lý, lọc dữ liệu, tách scene, quản lý metadata | Python, PIL, pandas |
| **Nhóm Vision-Language** | Phân tích cảnh bằng Gemma 4, sinh scene description, đánh giá semantic consistency | Gemma 4 API/local, CLIP, YOLO (tùy chọn) |
| **Nhóm Prompt & Instruction** | Thiết kế prompt template, sinh rescue instructions, tối ưu prompt cho FLUX-2 | Gemma 4, few-shot dataset |
| **Nhóm Sinh ảnh** | Fine-tune LoRA (nếu có), chạy FLUX-2 klein 4B, điều khiển ControlNet, quản lý seed/variant | FLUX-2 klein 4B, ControlNet, LoRA, diffusers library |
| **Nhóm Đánh giá & Lọc** | Tính metrics, chọn top-n ảnh, quản lý vòng lặp cải thiện | CLIP-IQA, CMMD, SSIM, Gemma 4 judge |
| **Nhóm Tổng hợp Dataset** | Đóng gói dataset, viết dataset card, quản lý split, chuẩn bị release | Python, Hugging Face datasets, pandas |

### 4.2. Lộ trình triển khai

**Giai đoạn 1: Pilot (1–2 tuần)**
- Chọn subset nhỏ từ incidents-1M (ví dụ 500–1,000 ảnh).
- Chạy pipeline từ bước 1 đến bước 6.
- Thu thập human feedback trên tập validation (100–200 mẫu).
- Xác định ngưỡng metric phù hợp, xác định k và n tối ưu.

*Checkpoint:* Có bộ pilot dataset (~500 mẫu accepted), báo cáo phân tích lỗi chính, điều chỉnh prompt template và model config.

**Giai đoạn 2: Tối ưu và mở rộng (2–4 tuần)**
- Tối ưu prompt template cho Gemma 4 (zero-shot) dựa trên kết quả pilot, không cần fine-tune.
- Fine-tune LoRA cho FLUX-2 trên domain rescue (nếu có dữ liệu phù hợp).
- Chạy full pipeline trên tập lớn hơn (ví dụ 10,000–50,000 ảnh).
- Tự động hóa vòng lặp bước 7.

*Checkpoint:* Dataset 10k–50k mẫu, acceptance rate >= ngưỡng, FID ổn định.

**Giai đoạn 3: Chuẩn hóa và công bố (1–2 tuần)**
- Hoàn thiện metadata, split train/val/test.
- Viết báo cáo kỹ thuật (dataset card).
- Chuẩn bị license và ghi nguồn.
- Đăng tải lên Hugging Face hoặc nền tảng công bố dataset.

---

## 5. Cấu trúc dataset đầu ra

### 5.1. Metadata cho mỗi mẫu

```json
{
  "sample_id": "rescue_000123_variant_02",
  "source_image": "images/source/incidents_000123.jpg",
  "generated_image": "images/generated/rescue_000123_variant_02.jpg",
  "source_id": "incidents_000123",
  "scene_summary": "Flooded residential area with collapsed house structure and trapped vehicle.",
  "rescue_actions": ["rescue_boat", "firefighter", "emergency_light"],
  "rescue_instruction": "Add a rescue boat to the flooded street in front of the house, keep the damaged house structure and street layout unchanged.",
  "viewpoint": "front_view",
  "generation_mode": "edit",
  "model_version": "flux2_klein_4b_controlnet_canny_lora_rescue_v1",
  "seed": 42,
  "variant_id": "variant_02",
  "generation_config": {
    "controlnet_type": "canny",
    "lora_weight": 0.7,
    "cfg_scale": 7.5,
    "steps": 30,
    "k_candidates": 5,
    "n_selected": 1,
    "candidate_source": "prompt_variation"
  },
  "metrics": {
    "clip_iqa": 0.81,
    "ssim_vs_original": 0.72,
    "semantic_consistency": 0.80,
    "preservation_score": 0.76,
    "cmmd": 0.15,
    "O_score": 0.76
  },
  "quality_status": "accepted",
  "split": "train",
  "scene_type": "flood",
  "source_dataset": "incidents-1M",
  "generation_timestamp": "2025-08-03T08:00:00+07:00"
}
```

### 5.2. Định dạng lưu trữ đề xuất
- `dataset/`
  - `images/source/` — ảnh gốc.
  - `images/generated/` — ảnh sinh đã được chọn (top-n).
  - `images/candidates/` — tất cả k ảnh sinh ra trước khi chọn (để phân tích).
  - `images/rejected/` — các mẫu rejected kèm lý do.
  - `metadata/train.jsonl`, `val.jsonl`, `test.jsonl` — metadata từng split.
  - `reports/` — báo cáo đánh giá chất lượng tập dữ liệu.

---

## 6. Tiêu chuẩn đánh giá dataset

### 6.1. Ngưỡng mẫu đơn (per-sample)
| Chỉ số | Công cụ | Ngưỡng đề xuất | Mô tả |
|---|---|---|---|
| CLIP-IQA | CLIP-IQA model | >= 0.70 | Độ tự nhiên và chất lượng ảnh |
| Semantic consistency | Gemma 4 judge | >= 0.75 | Mức độ giữ ngữ cảnh ban đầu + hành động cứu hộ hợp lý |
| Preservation score | SSIM + Depth/seg check | >= 0.70 | Mức độ giữ nguyên cấu trúc, bố cục, đối tượng nền |
| O_score | min(SC, PQ) | >= 0.70 | Điểm tổng hợp: ảnh phải đạt đồng thời SC và PQ |
| CMMD | CMMD metric | <= 0.20 | Phân phối ảnh sinh không lệch quá xa so với ảnh gốc |
| Human pass rate | Human review | >= 80% | Tỷ lệ qua kiểm tra thủ công (trên val/test) |

### 6.2. Chỉ số pipeline (pipeline-level)
| Chỉ số | Công cụ | Mục tiêu |
|---|---|---|
| CMMD | CMMD metric | Thấp, ổn định so với phân phối ảnh gốc |
| Diversity score | Embedding clustering | Cao (các biến thể khác nhau của cùng ảnh gốc) |
| Coverage score | Action label histogram | Bao phủ nhiều loại hành động cứu hộ |
| Acceptance rate | Pipeline metric | >= 70% |
| Regeneration rate | Pipeline metric | Thấp (hiệu quả prompt/model) |
| Top-n selection efficiency | Pipeline metric | Cao (nhiều ảnh trong top-n đạt ngưỡng) |

---

## 7. Rủi ro và giảm thiểu

| Rủi ro | Mô tả | Giải pháp giảm thiểu |
|---|---|---|
| Ảnh sinh bị biến dạng đối tượng cứu hộ | Gemma 4 sinh instruction mơ hồ, FLUX-2 sinh sai object | Thêm constraint vào prompt, dùng Gemma 4 judge kiểm tra instruction trước khi sinh ảnh, tăng k (nhiều candidates) |
| Cấu trúc ảnh bị thay đổi quá nhiều | ControlNet không đủ mạnh, LoRA weight quá cao | Điều chỉnh ControlNet scale, thêm SSIM threshold trong bước chọn top-n |
| Mode collapse / lệch phân phối | Dùng FLUX-2 mặc định, không phù hợp domain rescue | Fine-tune LoRA trên ảnh rescue thực tế nếu có, dùng CMMD theo dõi |
| Chi phí inference cao | Chạy FLUX-2 + Gemma 4 với k lớn | Dùng quantization (GPTQ/AWQ) cho Gemma 4, LCM LoRA cho FLUX-2 để giảm bước denoising, điều chỉnh k phù hợp |
| Metadata không nhất quán | Prompt template thay đổi giữa các bước | Khóa prompt template version, log đầy đủ generation_config |
| Chọn sai ảnh trong top-n | Composite score chưa phù hợp | Điều chỉnh ngưỡng O_score, bổ sung human review trên tập validation |
| Chất lượng rescue instruction chưa đủ | Zero-shot Gemma 4 chưa hiểu đúng domain rescue | Tối ưu prompt template, bổ sung few-shot examples, đánh giá lại plausibility_rate |

---

## 8. Kết luận

Pipeline này khác với pipeline trong pipeline-1-prd.md ở chỗ:
- pipeline-1-prd.md tập trung vào việc sinh dữ liệu huấn luyện cho InstructPix2Pix (image-to-image instruction following) bằng cách dùng LLM sinh instruction + Stable Diffusion sinh ảnh cặp.
- Pipeline này mở rộng ý tưởng đó sang bài toán rescue simulation: dùng Gemma 4 (zero-shot) để hiểu cảnh thiên tai, sinh rescue instructions, sau đó dùng FLUX-2 klein 4B + ControlNet/LoRA để sinh ảnh mô phỏng cứu hộ, có cơ chế multi-prompt generation (k ảnh từ các instruction khác nhau) và quality-based selection với hàm O = min(SC, PQ), có vòng lặp đánh giá và tối ưu.

Điểm khác biệt cốt lõi:
- Đầu vào là ảnh thiên tai thực tế, không phải ảnh tổng hợp.
- Dùng FLUX-2 klein 4B thay vì SDXL làm mô hình sinh ảnh chính.
- Có cơ chế sinh k ảnh từ các rescue instruction biến thể khác nhau và chọn n ảnh tốt nhất bằng hàm O = min(SC, PQ) để chống lỗi sinh ảnh.
- Dùng CMMD thay vì FID để đánh giá lệch phân phối.
- Mức can thiệp giữ ở mức trung bình, ngưỡng preservation score cố định >= 0.70 (không phân chia theo mức độ can thiệp).
- Sử dụng Gemma 4 zero-shot (không fine-tune) cho cả scene understanding và instruction generation.
- Có cơ chế đánh giá và lọc tự động, có human review trên tập validation.
- Có mục tiêu public dataset có cấu trúc và metadata chuẩn hóa.
