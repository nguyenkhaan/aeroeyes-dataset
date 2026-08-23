# Kế hoạch tích hợp SDQM vào AeroEyes Dataset Pipeline

> **Trạng thái:** Phase 3 đã triển khai (V-Info + regression)  
> **Tham chiếu:** [SDQM paper & repo](https://github.com/ayushzenith/SDQM) · [arXiv:2510.06596](https://arxiv.org/abs/2510.06596)  
> **Pipeline hiện tại:** `docs/pipeline/pipeline-2-prd.md`

---

## 1. Mục tiêu

Tích hợp **Synthetic Dataset Quality Metric (SDQM)** vào pipeline sinh ảnh cứu hộ để:

1. Đánh giá chất lượng **cấp dataset** (real vs synthetic) bổ sung cho CMMD.
2. Cung cấp submetrics có ý nghĩa nghiên cứu (MAUVE, α-Precision, Separability, …).
3. Chuẩn bị nền tảng cho Phase 2 (full SDQM với YOLO labels) phục vụ huấn luyện object detection.

SDQM **không thay thế** quality gate per-image (`O_score`, `SSIM`) — nó chạy **sau batch** trên tập ảnh đã accepted, tương tự CMMD.

---

## 2. Hiện trạng pipeline

### 2.1. Luồng đánh giá hiện có

```
main.py
  ├── Per-sample: evaluate_quality() → sc_score, pq_score, o_score, ssim
  ├── Quality gate: passes_quality_gate()
  ├── Lưu REAL_IMAGES_DIR / GEN_IMAGES_DIR
  └── Cuối batch: run_cmmd_report() → CMMD score
```

### 2.2. Module liên quan

| File | Vai trò |
|---|---|
| `src/evaluation/quality.py` | CLIP-IQA, O_score, SSIM |
| `src/evaluation/cmmd.py` | Wrapper CMMD (dynamic import) |
| `src/core/config.py` | Hằng số cấu hình |
| `main.py` | Entrypoint VPS |

### 2.3. Gap so với SDQM

| Yêu cầu SDQM | Pipeline hiện tại | Phase giải quyết |
|---|---|---|
| Embeddings (.pkl + .csv) | Chưa có | **Phase 1** |
| Cặp real/synthetic dirs | Có (`REAL_IMAGES_DIR`, `GEN_IMAGES_DIR`) | Phase 1 |
| YOLO labels (`.txt` bbox) | Có (auto-label Phase 2) | **Phase 2** |
| `data.yaml` class map | Có (`config/sdqm/data.yaml`) | **Phase 2** |
| V-Info (ultralytics tùy biến) | Chưa có | Phase 3 |

---

## 3. Phạm vi triển khai theo phase

### Phase 1 — Partial SDQM (triển khai ngay)

**Mục tiêu:** Chạy được SDQM embedding metrics trên thư mục ảnh hiện có, không cần YOLO.

**Submetrics:**

| Type | Metric | Mô tả ngắn |
|---|---|---|
| `similarity` | MAUVE, Frontier Integral | Độ tương đồng phân phối embedding |
| `fdg` | α-Precision, β-Recall, Authenticity | Fidelity, Diversity, Generality |
| `lcm` | Cluster Metric | Phân cụm embedding |
| `separability` | Separability accuracy | Khả năng phân tách real/synthetic |

**Deliverables Phase 1:**

- `docs/pipeline/sdqm-integration-plan.md` (file này)
- `src/evaluation/sdqm_embedding.py` — tạo embedding DINOv2
- `src/evaluation/sdqm.py` — wrapper gọi `calculate_sdqm()`
- Cập nhật `src/core/config.py`, `src/evaluation/__init__.py`, `main.py`
- `requirements-sdqm.txt` — deps riêng, tránh xung đột
- `.gitignore` — ignore `third_party/SDQM/`, `data/sdqm/`

**Luồng Phase 1:**

```
REAL_IMAGES_DIR + GEN_IMAGES_DIR
    → sdqm_embedding.embed_image_directory() × 2
    → sdqm.compute_dataset_sdqm(metric_types=partial)
    → sdqm_report.json + in console
```

### Phase 2 — YOLO auto-labeling (đã triển khai)

**Mục tiêu:** Full annotation metrics (Label Overlap, Spatial Distribution, Bounding Box Match, Pixel Intensity).

**Đã làm:**

1. Class map rescue domain: `config/sdqm/data.yaml`
2. Module `src/evaluation/yolo_export.py`:
   - Grounding DINO zero-shot trên ảnh real + gen
   - Export YOLO layout: `images/train/`, `labels/train/`, `data.yaml`
3. `compute_dataset_sdqm()` mở rộng metric types Phase 2 (trừ V-Info)

**Luồng Phase 2:**

```
REAL_IMAGES_DIR + GEN_IMAGES_DIR
    → yolo_export.export_yolo_pair()
    → sdqm_embedding (từ yolo/.../images/train, paths chứa /images/)
    → sdqm.calculate_sdqm(metric_types=Phase2)
    → sdqm_report.json (kèm yolo_layout)
```

**Output YOLO:**

```
data/output/sdqm/yolo/
  real/images/train/ + labels/train/ + data.yaml
  synthetic/images/train/ + labels/train/ + data.yaml
```

**Biến môi trường Phase 2:**

| Biến | Mặc định | Mô tả |
|---|---|---|
| `SDQM_YOLO_EXPORT` | `true` | Bật auto-label trước SDQM |
| `SDQM_YOLO_DATA_YAML` | `config/sdqm/data.yaml` | Class map |
| `SDQM_GROUNDING_DINO_MODEL` | `IDEA-Research/grounding-dino-tiny` | Detector |
| `SDQM_BOX_THRESHOLD` | `0.25` | Ngưỡng bbox |
| `SDQM_TEXT_THRESHOLD` | `0.25` | Ngưỡng text match |

**Tiêu chí hoàn thành Phase 2:**

- [x] `config/sdqm/data.yaml` với 5 class rescue
- [x] `yolo_export.py` export YOLO + Grounding DINO
- [x] `compute_dataset_sdqm()` dùng YOLO layout + annotation metrics
- [ ] Verify end-to-end trên VPS với SDQM repo + deps

### Phase 3 — Full SDQM + V-Info (đã triển khai)

**Mục tiêu:** Super metric đầy đủ như paper (bao gồm V-Info).

**Đã làm:**

1. `src/evaluation/sdqm_vinfo.py` — layout YOLO cho V-Info (copy/symlink fallback)
2. Gọi `get_v_info()` từ SDQM với dataset `rescue` (standard YOLO val)
3. `src/evaluation/sdqm_regression.py` — Pearson/Spearman vs mAP từ history CSV
4. `scripts/setup_sdqm_vinfo.sh` — cài custom ultralytics trên VPS
5. Tích hợp vào `compute_dataset_sdqm()` + `main.py`

**Luồng Phase 3:**

```
YOLO export (Phase 2)
    → prepare_vinfo_yolo_layout()   # synthetic=train, real=val
    → get_v_info()                  # YOLO11n train + val
    → merge v_info_* vào sdqm_report.json
    → append sdqm_history.csv
    → regression_report.json (nếu >= 3 rows có map)
```

**Cài đặt V-Info:**

```bash
bash scripts/setup_sdqm_vinfo.sh
# hoặc thủ công:
pip install -e third_party/SDQM/dataset_interpretability/v_info/ultralytics
```

**Biến môi trường Phase 3:**

| Biến | Mặc định | Mô tả |
|---|---|---|
| `SDQM_VINFO_ENABLED` | `true` | Bật V-Info |
| `SDQM_VINFO_DATASET` | `rescue` | Loại validator (dùng standard YOLO val) |
| `SDQM_APPEND_HISTORY` | `true` | Ghi mỗi run vào history CSV |
| `SDQM_HISTORY_CSV` | `{SDQM_OUTPUT_DIR}/sdqm_history.csv` | Lịch sử metrics |
| `SDQM_MAP_CSV` | `""` | CSV thay thế history (cột `map`) |
| `SDQM_MAP_COLUMN` | `map` | Tên cột mAP |
| `SDQM_MAP_VALUE` | `""` | mAP của run hiện tại, được ghi cùng SDQM history |
| `SDQM_RUN_REGRESSION` | `true` | Chạy correlation khi đủ rows |
| `SDQM_MIN_REGRESSION_ROWS` | `3` | Tối thiểu rows cho regression |
| `SDQM_SUMMARY_PATH` | `reports/sdqm_summary.md` | Báo cáo Markdown dataset-level |

**V-Info metrics output:**

- `conditional_iou`, `predictive_iou`, `v_info_iou`
- `conditional_conf`, `predictive_conf`, `v_info_conf`
- `conditional_fusion`, `predictive_fusion`, `v_info_fusion`

**Tiêu chí hoàn thành Phase 3:**

- [x] Module V-Info với layout cross-platform
- [x] Kiểm tra custom ultralytics + skip graceful
- [x] Regression SDQM vs mAP (history CSV)
- [ ] Verify end-to-end trên Linux VPS với GPU

### Phase 4 — Gắn vào PRD Bước 5–6 (triển khai một phần)

- [ ] Dùng SDQM submetrics làm tie-breaker khi `O_score` gần nhau. Cần thiết kế selection theo batch vì SDQM hiện là dataset-level.
- [x] Thêm `sdqm` block có scope `dataset` vào `data/output/evaluation_metadata.jsonl`.
- [x] Báo cáo dataset-level trong `reports/sdqm_summary.md`.

---

## 4. Kiến trúc code (Phase 1 + 2)

```
aeroeyes-dataset/
├── config/sdqm/
│   └── data.yaml                # class map rescue (committed)
├── third_party/
│   └── SDQM/                    # git clone (gitignored)
├── data/
│   └── output/sdqm/             # embeddings + yolo + reports (gitignored)
│       ├── yolo/real/...
│       ├── yolo/synthetic/...
│       ├── real_embeddings.pkl
│       └── sdqm_report.json
├── docs/pipeline/
│   └── sdqm-integration-plan.md
├── requirements-sdqm.txt
└── src/
    ├── core/config.py
    └── evaluation/
        ├── cmmd.py
        ├── sdqm_embedding.py
        ├── sdqm.py
        ├── sdqm_vinfo.py          # Phase 3
        ├── sdqm_regression.py     # Phase 3
        └── yolo_export.py         # Phase 2
```

### 4.1. Pattern wrapper (giống CMMD)

`src/evaluation/sdqm.py` sẽ:

1. Kiểm tra `SDQM_REPO_DIR/sdqm.py` tồn tại.
2. Thêm repo vào `sys.path` (SDQM import nội bộ).
3. Gọi `calculate_sdqm()` với `metric_type` partial.
4. Trả về `dict[str, float]` đã flatten.

### 4.2. Embedding tự chủ

Không dùng `embedding.py` gốc của SDQM (có lỗi `Image.Open`, phụ thuộc CLIP cũ).  
Dùng **DINOv2** (`facebook/dinov2-small`) qua `transformers` — đã có trong project.

Output format tương thích SDQM:

- `{prefix}.pkl` — pickle của `np.ndarray` shape `(N, D)`
- `{prefix}.csv` — cột `file_path` (đường dẫn tuyệt đối tới ảnh)

---

## 5. Cấu hình môi trường

### 5.1. Clone SDQM repo

```bash
git clone https://github.com/ayushzenith/SDQM.git third_party/SDQM
```

Hoặc set biến môi trường:

```bash
export SDQM_REPO_DIR=/path/to/SDQM
```

### 5.2. Cài dependencies SDQM (Phase 1)

```bash
pip install -r requirements-sdqm.txt
```

### 5.3. Biến môi trường mới

| Biến | Mặc định | Mô tả |
|---|---|---|
| `SDQM_REPO_DIR` | `{PROJECT_ROOT}/third_party/SDQM` | Đường dẫn repo SDQM |
| `SDQM_OUTPUT_DIR` | `{OUTPUT_DIR}/sdqm` | Thư mục embedding + report |
| `SDQM_ENABLED` | `true` | Bật/tắt chạy SDQM cuối pipeline |
| `SDQM_EMBEDDING_MODEL` | `facebook/dinov2-small` | Model embedding |

### 5.4. Hằng số trong `config.py`

```python
SDQM_METRIC_TYPES = [
    "similarity", "fdg", "lcm", "separability",
    "distribution", "bounding_box", "label_overlap", "spatial",
]
SDQM_YOLO_EXPORT = True
SDQM_MIN_IMAGES = 2
```

---

## 6. Tích hợp vào `main.py`

Thêm hàm `run_sdqm_report()` sau `run_cmmd_report()`:

```python
def run_sdqm_report() -> dict[str, float] | None:
    if not SDQM_ENABLED:
        return None
    cleanup()  # giải phóng VRAM Gemma/FLUX trước
    try:
        metrics = compute_dataset_sdqm(
            ref_dir=REAL_IMAGES_DIR,
            eval_dir=GEN_IMAGES_DIR,
        )
        # in summary + lưu JSON
        return metrics
    except Exception as exc:
        print(f"SDQM calculation failed: {exc}")
        return None
```

Gọi ở cuối `main.py`:

```python
run_cmmd_report()
run_sdqm_report()
```

---

## 7. Metadata output (Phase 1)

File `data/output/sdqm/sdqm_report.json`:

```json
{
  "ref_dir": "data/real_reference",
  "eval_dir": "data/gen_reference",
  "embedding_model": "facebook/dinov2-small",
  "metric_types": ["similarity", "fdg", "lcm", "separability"],
  "metrics": {
    "Dataset Similarity_mauve": 0.68,
    "Dataset Similarity_frontier_integral": 0.71,
    "FDG_fidelity_alpha_precision": 0.75,
    "FDG_diversity_beta_recall": 0.62,
    "FDG_generality_authenticity": 0.70,
    "Cluster Metric": 0.55,
    "log_Cluster Metric": 0.48,
    "accuracy_Separability": 0.82
  }
}
```

---

## 8. Rủi ro và giảm thiểu

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| SDQM repo chưa clone | Cao | Error message rõ + hướng dẫn clone |
| Xung đột deps (mauve-text, deap) | Trung bình | `requirements-sdqm.txt` riêng |
| VRAM không đủ khi embed | Trung bình | `cleanup()` trước SDQM; DINOv2-small |
| Mẫu quá ít (`LIMIT_IMAGES=5`) | Thấp | `SDQM_MIN_IMAGES=2`; cảnh báo nếu N < 10 |
| Windows symlink (V-Info Phase 3) | Trung bình | Chỉ chạy V-Info trên Linux VPS |

---

## 9. Tiêu chí hoàn thành Phase 1

- [x] Plan document tồn tại và được team review
- [x] `python -m py_compile` pass cho các file mới
- [ ] Khi có SDQM repo + deps: `run_sdqm_report()` tạo `sdqm_report.json`
- [x] Khi thiếu repo: pipeline vẫn chạy xong, in warning không crash
- [x] CMMD và quality gate per-image không bị ảnh hưởng

---

## 10. Verify trên VPS

```bash
# 1. Clone SDQM
git clone https://github.com/ayushzenith/SDQM.git third_party/SDQM

# 2. Cài deps SDQM
pip install -r requirements-sdqm.txt

# 3. Chạy pipeline (sau khi có ảnh accepted)
python main.py

# 4. Kiểm tra output
cat data/output/sdqm/sdqm_report.json
```

---

## 11. Timeline đề xuất

| Phase | Thời gian ước lượng | Phụ thuộc |
|---|---|---|
| Phase 1 (partial SDQM) | 1–2 ngày | Clone SDQM, deps |
| Phase 2 (YOLO export) | 3–5 ngày | Class map, detector |
| Phase 3 (V-Info full) | 2–3 ngày | Linux VPS, ultralytics |
| Phase 4 (PRD integration) | 1–2 ngày | Phase 1 stable |

---

## 12. Tài liệu tham khảo

- [SDQM GitHub](https://github.com/ayushzenith/SDQM)
- [SDQM DeepWiki](https://deepwiki.com/ayushzenith/SDQM)
- [arXiv:2510.06596](https://arxiv.org/abs/2510.06596)
- Pipeline PRD: `docs/pipeline/pipeline-2-prd.md`
- CMMD wrapper (pattern): `src/evaluation/cmmd.py`
