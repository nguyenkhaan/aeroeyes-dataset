# Plan 001: Hoàn thiện benchmark và pipeline auto-annotation bounding box

> **Hướng dẫn thực thi**: Làm lần lượt từng bước và chạy lệnh kiểm tra ở cuối
> mỗi bước. Nếu gặp một điều kiện trong mục “Điều kiện dừng”, dừng và báo cáo,
> không tự thay đổi protocol. Khi hoàn tất, cập nhật trạng thái trong
> `plans/README.md`.
>
> **Kiểm tra drift trước khi làm**:
> `git diff --stat 2fe21a4..HEAD -- main_bounding_boxes.py src/evaluation src/core/config.py scripts tests config docs`
> Nếu các file liên quan đã thay đổi, đối chiếu lại các bằng chứng trong mục
> “Hiện trạng” trước khi triển khai.

## Trạng thái

- **Ưu tiên**: P0
- **Công sức**: L, dự kiến 5–8 ngày làm việc tùy thời gian annotation
- **Rủi ro**: MED — thay đổi logic matching và định dạng artifact đánh giá
- **Phụ thuộc**: không
- **Loại**: research, correctness, tests, reproducibility
- **Planned at**: commit `2fe21a4`, 2026-10-20

## Vì sao cần thực hiện

Pipeline hiện tại báo cáo `bbox_passed` khi vùng thay đổi hợp lệ và có ít nhất
một đối tượng cứu hộ được xác nhận. Chỉ số này không đo precision, recall hoặc
localization: một ảnh có năm đối tượng thật nhưng chỉ phát hiện một đối tượng
vẫn có thể pass. Nghiên cứu cần một ground truth độc lập theo cặp Before/After,
một evaluation harness chuẩn, tập validation để tuning, test độc lập và ablation
cho từng cải tiến.

Mục tiêu cuối cùng là trả lời riêng ba câu hỏi:

1. **Detection**: detector tìm được bao nhiêu đối tượng cứu hộ thật trong ảnh After?
2. **Localization**: box dự đoán bám sát đối tượng đến mức nào?
3. **Change verification**: pipeline có phân biệt đúng đối tượng mới với đối
   tượng đã tồn tại trong ảnh Before không?

## Hiện trạng đã kiểm chứng

- `src/evaluation/bounding_box_pipeline.py:199` đặt `bbox_passed` bằng
  `valid_change AND confirmed_rescue_count > 0`.
- `src/evaluation/object_matching.py:52-68` ghép đối tượng cùng lớp theo greedy,
  phụ thuộc thứ tự detection.
- `src/evaluation/object_matching.py:92-127` refine box bằng bounding rectangle
  của mọi pixel delta nằm trong detection box; nhiễu rời rạc có thể làm box quá
  rộng hoặc delta thiếu có thể làm box quá nhỏ.
- `src/evaluation/yolo_export.py:52-62` bỏ nhãn không khớp class map; mở rộng
  prompt mà không có ontology sẽ làm mất detection hoặc tạo ánh xạ sai nghĩa.
- `src/evaluation/bounding_box_pipeline.py:158-182` resize ảnh After khi kích
  thước khác ảnh Before nhưng vẫn giữ đường dẫn ảnh After gốc.
- `src/evaluation/yolo_export.py:229-235` ghi `train` và `val` cùng trỏ đến
  `images/train`; cấu trúc này chỉ phù hợp artifact tạm cho SDQM, không phải
  đánh giá supervised độc lập.
- Các test hiện có là unit test với dữ liệu tổng hợp. Chưa có benchmark ảnh thật,
  ground truth hay evaluation end-to-end.

## Quy ước phải khóa trước khi code

### Ontology năm lớp loại trừ nhau

| Lớp | Bao gồm | Không tự động ánh xạ |
|---|---|---|
| `firefighter` | người có dấu hiệu rõ là lính cứu hỏa | `rescue worker`, cảnh sát, nhân viên y tế nếu không đủ bằng chứng |
| `rescue_boat` | thuyền/raft đang phục vụ cứu hộ | thuyền dân dụng không có dấu hiệu cứu hộ |
| `helicopter` | trực thăng cứu hộ hoặc trực thăng được bổ sung theo chỉ dẫn | máy bay cánh cố định |
| `ambulance` | xe cứu thương nhận diện được | mọi xe khẩn cấp nói chung |
| `emergency_vehicle` | xe cứu hỏa, rescue truck và xe khẩn cấp không thuộc `ambulance` | `ambulance` |

Các từ đồng nghĩa rõ ràng như `fire fighter` có thể ánh xạ tự động. Nhãn khái
quát hoặc mơ hồ phải chuyển sang human review, không ép về một lớp để tăng recall.

### Đơn vị ground truth

Mỗi cặp phải có:

- `pair_id`, đường dẫn Before và After, source/event group;
- box và class trên ảnh Before;
- box và class trên ảnh After;
- với mỗi object After: `is_newly_added`, `is_valid_rescue`, `occluded`,
  `ambiguous`, `annotation_notes`;
- `generation_status`: `valid`, `missing_requested_object`, `invalid_object`,
  `severe_scene_corruption`, hoặc `ambiguous`.

Không tính lỗi sinh ảnh thành false negative của detector. Báo cáo riêng
generation failure, detection failure, matching failure và localization failure.

## Phạm vi

### Trong phạm vi

- `docs/bounding_box/annotation-guideline.md` (tạo mới)
- `config/bounding_box/ontology.yaml` (tạo mới)
- `config/bounding_box/benchmark.yaml` (tạo mới)
- `scripts/build_bbox_benchmark.py` (tạo mới)
- `scripts/evaluate_bounding_boxes.py` (tạo mới)
- `src/evaluation/bbox_metrics.py` (tạo mới)
- `src/evaluation/object_matching.py`
- `src/evaluation/bounding_box_pipeline.py`
- `src/evaluation/yolo_export.py`
- `src/core/config.py`
- `main_bounding_boxes.py`
- `scripts/detect_bounding_boxes.slurm`
- các test tương ứng dưới `tests/`

Ground-truth image/annotation artifact có thể nằm ngoài Git do dung lượng, nhưng
manifest, checksum, phiên bản và hướng dẫn tái tạo phải được lưu trong repo.

### Ngoài phạm vi

- `main.py`, FLUX/Gemma generation logic;
- SDQM scoring;
- `src/vision/disaster_filter.py` và CLIP filtering;
- mọi file dưới `third_party/`;
- huấn luyện hoặc fine-tune detector mới.

## Git workflow

- Tạo branch `feat/bbox-research-benchmark`.
- Commit theo từng giai đoạn với conventional commits, ví dụ
  `feat: add paired bounding box benchmark`.
- Không push hoặc mở PR nếu chưa được yêu cầu.

## Giai đoạn 1 — Khóa protocol và tạo benchmark

### Bước 1.1: Viết annotation guideline

Tạo `docs/bounding_box/annotation-guideline.md` gồm:

- định nghĩa năm lớp loại trừ nhau;
- quy tắc box cho vật thể bị che khuất, bị cắt biên và vật thể rất nhỏ;
- quy tắc xác định mới/cũ khi so Before với After;
- quy tắc xử lý nhãn mơ hồ;
- taxonomy lỗi generation/detection/matching/localization;
- ít nhất hai ví dụ dương và hai phản ví dụ cho mỗi lớp.

**Kiểm tra**:

```bash
test -s docs/bounding_box/annotation-guideline.md
grep -q "is_newly_added" docs/bounding_box/annotation-guideline.md
grep -q "ambiguous" docs/bounding_box/annotation-guideline.md
```

Kỳ vọng: exit code 0 cho cả ba lệnh.

### Bước 1.2: Chọn mẫu và tách validation/test

Tạo manifest từ 500 cặp bằng seed cố định. Tập chính phải được lấy mẫu đại diện;
challenge subset gồm các trường hợp khó được lấy có chủ đích và báo cáo riêng.

- Khởi đầu với 100 cặp: 40 validation, 60 test.
- Split theo source/event group, không chỉ theo stem.
- Không xem số 100 là cố định. Sau annotation, mỗi lớp trong test phải có tối
  thiểu 20 đối tượng newly-added hợp lệ để báo cáo per-class metric. Nếu thiếu,
  bổ sung cặp trước khi khóa test hoặc ghi rõ lớp đó không đủ dữ liệu.
- Challenge subset chỉ lấy từ validation hoặc tách hẳn; không dùng nó để tuyên
  bố hiệu năng đại diện trên test.

Tạo `scripts/build_bbox_benchmark.py` nhận `--seed`, `--validation-size`,
`--test-size`, `--metadata-dir`, `--output-manifest`. Script phải lưu source
group, checksum file, split và sampling reason.

**Kiểm tra**:

```bash
venv/bin/python scripts/build_bbox_benchmark.py --help
venv/bin/python scripts/build_bbox_benchmark.py \
  --metadata-dir /datastore/khanhnd/models/aeroeyes_output/output/_metadata \
  --output-manifest /tmp/bbox_benchmark.json
venv/bin/python -m unittest tests.test_bbox_benchmark
```

Kỳ vọng: manifest có 100 `pair_id` duy nhất, không có source/event group xuất
hiện ở cả validation và test, test pass.

### Bước 1.3: Gán nhãn và kiểm tra chất lượng

Import cả Before và After vào CVAT theo guideline. Double-annotate tối thiểu 20%
số cặp, sau đó adjudicate bất đồng về class, new/existing và box. CVAT hỗ trợ
consensus annotation; nếu không dùng chức năng này, lưu bảng bất đồng và kết quả
adjudication tương đương.

Khóa artifact thành `bbox-gt-v1`, lưu checksum và số instance theo split/class.
Không chỉnh `bbox-gt-v1` sau khi xem kết quả test; sửa lỗi annotation phải tạo
phiên bản mới và ghi changelog.

**Nghiệm thu thủ công bắt buộc**:

- 100% cặp có Before/After và trạng thái generation;
- 100% object After có `is_newly_added` và `is_valid_rescue`;
- tất cả bất đồng trong phần double annotation đã adjudicate;
- có bảng đếm instance theo split/class.

## Giai đoạn 2 — Evaluation harness và baseline

### Bước 2.1: Viết evaluation harness

Tạo `src/evaluation/bbox_metrics.py` và CLI
`scripts/evaluate_bounding_boxes.py`. Đầu vào là ground truth manifest và reports
JSON của pipeline. Matching prediction–GT phải một-một, cùng lớp, theo confidence
và IoU. Một GT không được tính TP cho nhiều prediction.

Báo cáo riêng:

1. detector trên tất cả rescue object trong After;
2. phân loại new/existing của các object đã detect;
3. pipeline cuối trên newly-added valid rescue object;
4. generation success rate;
5. pair pass rate, chỉ như chỉ số bổ trợ.

Metrics bắt buộc:

- Precision, Recall, F1 tại IoU 0.50;
- AP50 và mAP50:95;
- per-class AP/Recall;
- mean matched IoU, kèm số FN để tránh diễn giải riêng lẻ;
- FP/FN và failure taxonomy;
- inference time và peak VRAM khi có trong manifest.

Recall@IoU 0.30 chỉ dùng để phân tích localization thô. Với lớp không có GT,
trả về `insufficient_data`, không tự gán 0 hoặc 1.

**Kiểm tra**:

```bash
venv/bin/python -m unittest tests.test_bbox_metrics
```

Test phải bao phủ: duplicate predictions, wrong class, no prediction, no GT,
IoU ngay dưới/trên ngưỡng, một prediction cạnh tranh hai GT và lớp thiếu dữ liệu.

### Bước 2.2: Chạy baseline đã đóng băng

Chạy code hiện tại trên validation mà chưa tune. Lưu manifest gồm Git commit,
model ID/revision, prompt, ontology version, threshold, seed, dataset checksum,
Python/Torch/CUDA version, GPU và runtime.

Phân loại lỗi thành:

- generation failure;
- detector miss;
- wrong class;
- before/after matching error;
- delta rejection;
- localization failure;
- false positive.

**Kiểm tra**:

```bash
venv/bin/python scripts/evaluate_bounding_boxes.py \
  --ground-truth bbox-gt-v1.json \
  --predictions <baseline-validation-run>/reports \
  --output <baseline-validation-run>/metrics.json
```

Kỳ vọng: metrics JSON có đủ ba tầng đánh giá, per-class counts và failure counts;
tổng failure breakdown khớp số object/pair đã xử lý.

## Giai đoạn 3 — Khảo sát và ablation trên validation

### Bước 3.1: Benchmark bốn pipeline theo cùng protocol

So sánh:

1. Grounding DINO — baseline;
2. Grounding DINO + SAM2 — đo riêng tác dụng localization;
3. YOLO-World + SAM2 — detector thay thế;
4. Florence-2 + SAM2 qua Autodistill Grounded SAM 2 — phương án tham khảo.

Không gọi Autodistill là detector độc lập; ghi rõ nó là framework và cấu hình
Grounded SAM 2 của nó dùng Florence-2 để grounding. Dùng cùng ontology, ảnh,
hardware và protocol đo. Chạy smoke test 5–20 ảnh trước để ước lượng runtime và
VRAM, sau đó benchmark toàn validation.

Ghi: AP/Recall, mean IoU, runtime/ảnh, peak VRAM, số lỗi cài đặt và thời gian
human correction ước tính trên một subset cố định.

### Bước 3.2: Tuning ontology, prompt và threshold

Thêm alias rõ nghĩa; nhãn mơ hồ phải được ghi vào review queue. Sweep threshold
theo lưới định trước trên validation. Không chọn ngưỡng bằng test.

Lưu toàn bộ run, kể cả run kém; bảng kết quả phải có ID cấu hình. Chọn operating
point theo mục tiêu recall nhưng đặt precision floor trước khi xem kết quả.

### Bước 3.3: Matching ablation

So sánh tối thiểu:

- greedy IoU hiện tại;
- global/Hungarian matching với class + IoU;
- global matching với class + IoU + normalized center distance;
- appearance feature chỉ thêm nếu hai cấu hình trên vẫn lỗi ở ca dịch chuyển lớn.

Không dùng heuristic `preservation_score` làm bằng chứng bảo toàn bối cảnh.

### Bước 3.4: Delta và SAM2 ablation

Chạy cùng detector/config với các biến thể:

| ID | Matching | Delta validation | Box đầu ra |
|---|---|---|---|
| A | hiện tại | có | Grounding DINO |
| B | cải tiến | có | Grounding DINO |
| C | cải tiến | không | Grounding DINO |
| D | cải tiến | có | SAM2-refined |
| E | cải tiến | không | SAM2-refined |

Nếu dùng padding hoặc SAM2 fallback, sweep padding trên validation và ghi cờ
`refinement_method`, `sam2_quality`, `fallback_used` cho từng prediction.

SAM2 chỉ được giữ nếu cải thiện localization hoặc metrics cuối với chi phí được
chấp nhận. Delta mask nên là tín hiệu xác minh; không mặc định dùng nó làm biên
cuối của object box.

**Nghiệm thu giai đoạn 3**:

- bảng tất cả cấu hình trên cùng validation;
- một cấu hình được chọn theo tiêu chí đã công bố trước;
- ablation cho thấy đóng góp của prompt, matching, delta và SAM2;
- file cấu hình cuối được khóa và có checksum.

## Giai đoạn 4 — Hardening, test cuối và chạy 500 cặp

### Bước 4.1: Sửa tính đúng đắn và khả năng chạy lại

- Từ chối cặp Before/After khác kích thước và ghi failure reason; không resize
  âm thầm trong pipeline hiện tại.
- Mỗi run dùng thư mục riêng, không trộn report cũ.
- Checkpoint từng cặp bằng ghi file nguyên tử.
- Resume phải parse report, kiểm tra schema, stem, checksum đầu vào, config hash
  và output liên quan; không chỉ kiểm tra file tồn tại.
- Manifest phải lưu Git commit, model revision, ontology/GT version, prompt,
  thresholds, seed, data checksum, môi trường và tài nguyên.
- Mọi threshold matching/delta/refinement phải đến từ config và được ghi vào run.

**Kiểm tra**:

```bash
venv/bin/python -m unittest discover -s tests -p 'test_bbox*.py'
venv/bin/python -m unittest discover -s tests -p 'test_bounding_box_pipeline.py'
venv/bin/python -m unittest discover -s tests -p 'test_change_detection.py'
```

Thêm test cho mismatch kích thước, resume report hỏng, config hash thay đổi,
stale output và interrupted run.

### Bước 4.2: Khóa cấu hình và chạy test đúng một lần

Đặt config ở trạng thái immutable. Chạy trên 60 test pair hoặc số mở rộng đã
được khóa. Không tune lại sau khi xem test. Nếu phát hiện bug implementation,
sửa bug, tăng version protocol và báo cáo minh bạch việc chạy lại.

Báo cáo metric chính trên test, bootstrap confidence interval khi đủ dữ liệu,
per-class counts và `insufficient_data` cho lớp thiếu mẫu.

### Bước 4.3: Chạy toàn bộ 500 cặp

Trước tiên benchmark 20 cặp để dự trù thời gian và VRAM. Sau đó submit toàn bộ
500 bằng config hash đã khóa. Chạy human review trên mẫu ngẫu nhiên định trước;
không chỉ chọn ảnh đẹp.

Đầu ra cuối:

- YOLO labels và paired reports;
- run manifest và checksum;
- validation/test metrics;
- bảng tool benchmark và ablation;
- failure analysis;
- 6–10 visualization gồm TP, FP, FN, matching error và refinement trước/sau;
- phần hạn chế nghiên cứu.

## Điều kiện hoàn thành

- [ ] Annotation guideline và ontology v1 đã khóa.
- [ ] Ground truth có Before/After, thuộc tính new/existing và generation status.
- [ ] Validation/test độc lập theo source/event group.
- [ ] Ít nhất 20% cặp được double-annotate và adjudicate.
- [ ] Evaluation harness vượt qua toàn bộ edge-case tests.
- [ ] Baseline chạy trên validation với manifest đầy đủ.
- [ ] Bốn pipeline được benchmark theo cùng protocol hoặc có lý do kỹ thuật rõ
      ràng cho phương án không chạy được.
- [ ] Có ablation prompt, matching, delta và SAM2.
- [ ] Config cuối được khóa trước khi chạy test.
- [ ] Test chỉ được dùng cho báo cáo cuối, không dùng tuning.
- [ ] Pipeline từ chối mismatch kích thước, hỗ trợ checkpoint/resume có kiểm tra
      tính toàn vẹn và cô lập output theo run.
- [ ] Chạy đủ 500 cặp bằng config cuối và human review mẫu ngẫu nhiên.
- [ ] Báo cáo tách generation, detection, matching và localization failures.

## Điều kiện dừng

Dừng và báo cáo nếu:

- Không thể xác định source/event group để tách dữ liệu mà không leakage.
- Một lớp không đủ object cho test sau vòng lấy mẫu đầu; không tự thay đổi tỷ lệ
  split hoặc gộp lớp nếu chưa cập nhật protocol.
- Ground-truth schema không biểu diễn được new/existing hoặc generation failure.
- In-scope code đã drift khiến bằng chứng hiện trạng không còn đúng.
- Công cụ yêu cầu model/API không thể chạy hoặc license không phù hợp; ghi nhận
  là kết quả khảo sát, không thay bằng công cụ khác mà không sửa protocol.
- Sửa implementation sau khi đã xem test; cần version và khóa lại protocol trước
  khi chạy lại.

## Ghi chú bảo trì

- Nếu sau này huấn luyện detector, tạo plan riêng cho train/val/test split theo
  source/event group. Không tái sử dụng cấu trúc `val: images/train` của SDQM.
- Khi thay ontology, bump version và không so trực tiếp metric giữa hai version
  nếu ground truth chưa được migrate, adjudicate và ghi changelog.
- Reviewer cần kiểm tra kỹ one-to-one matching, việc loại generation failures
  khỏi detector recall và tính toàn vẹn của resume/checkpoint.
