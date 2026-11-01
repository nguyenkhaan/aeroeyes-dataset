# Bounding-box research benchmark

Thư mục này chứa protocol cho benchmark phát hiện đối tượng cứu hộ mới trong
cặp ảnh Before/After.

## Thành phần giai đoạn 1

- `annotation-guideline.md`: quy tắc ontology, vẽ box, new/existing và kiểm soát chất lượng.
- `../../config/bounding_box/ontology.yaml`: định nghĩa lớp và alias.
- `../../config/bounding_box/benchmark.yaml`: protocol validation/test và metric.
- `../../scripts/build_bbox_benchmark.py`: tạo manifest tái lập từ metadata sinh ảnh.

## Tạo manifest trên UIT HPC

Chạy từ project root sau khi đã có đủ 500 cặp Before/After:

```bash
venv/bin/python scripts/build_bbox_benchmark.py \
  --metadata-dir /datastore/khanhnd/models/aeroeyes_output/output/_metadata \
  --output-manifest /datastore/khanhnd/models/aeroeyes_output/bbox_benchmark/bbox-gt-v1-manifest.json
```

Kích thước split và seed mặc định được đọc từ
`config/bounding_box/benchmark.yaml`. Chỉ dùng các cờ `--validation-size`,
`--test-size` và `--seed` khi chủ động tạo một protocol khác.

Manifest lưu split, nhãn sự kiện, đường dẫn ảnh, source group và SHA-256 của
từng ảnh. Chạy lại với cùng dữ liệu và seed phải tạo cùng nội dung.

Nếu metadata không có `source_group`, `source_event_id`, `event_id` hoặc
`incident_id`, script dùng `pair_id` làm fallback và in cảnh báo. Cách fallback
này giữ Before/After của cùng cặp ở chung một split nhưng không chứng minh các
ảnh gần trùng hoặc cùng sự kiện nguồn đã được tách nhóm. Phải ghi hạn chế này
trong báo cáo; trước khi huấn luyện detector cần bổ sung source group đáng tin cậy.

## Kiểm tra manifest

```bash
venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path(
    "/datastore/khanhnd/models/aeroeyes_output/"
    "bbox_benchmark/bbox-gt-v1-manifest.json"
)
manifest = json.loads(path.read_text(encoding="utf-8"))

print("COUNTS:", manifest["counts"])
print("SOURCE GROUP METHODS:", Counter(
    item["source_group_method"] for item in manifest["pairs"]
))
print("LABEL COUNTS:", manifest["label_counts"])
PY
```

Không bắt đầu annotation nếu:

- manifest không đủ 40 validation và 60 test;
- cùng một source group xuất hiện ở cả hai split;
- có ảnh bị thiếu hoặc checksum không đọc được;
- ontology hoặc guideline vẫn còn điểm chưa thống nhất.

Nếu mọi `source_group_method` đều là `pair_id_fallback`, validation và test chỉ
được chứng minh là không trùng `pair_id`. Chưa thể khẳng định hai ảnh gần trùng
hoặc cùng sự kiện nguồn không nằm ở hai split. Phải ghi hạn chế này trong báo
cáo và không dùng benchmark này để tuyên bố khả năng tổng quát hóa theo sự kiện.

## Chuẩn bị bundle cho CVAT

Tạo bundle sau khi manifest đã vượt qua các kiểm tra trên:

```bash
venv/bin/python scripts/prepare_bbox_annotation.py \
  --manifest /datastore/khanhnd/models/aeroeyes_output/bbox_benchmark/bbox-gt-v1-manifest.json \
  --output-dir /datastore/khanhnd/models/aeroeyes_output/bbox_benchmark/annotation-bundle
```

Script kiểm tra lại SHA-256 của 200 ảnh trước khi đóng gói và tạo:

- `validation-images.zip` và `test-images.zip` cho annotation chính;
- hai archive `*-double-annotation-images.zip` chứa đúng 20% cặp được chọn tái
  lập để người thứ hai gán độc lập;
- `annotation-mapping.json` ánh xạ tên frame CVAT về `pair_id` và Before/After;
- `cvat-labels.json` chứa năm rectangle label cùng `pair_status` tag;
- `bbox-gt-v1.template.json` làm khung cho artifact sau khi convert export CVAT.
- `bundle-summary.json` lưu count và SHA-256 của toàn bộ artifact trong bundle.

Tên frame có dạng `0000__00_before__<pair_id>` và
`0000__01_after__<pair_id>`, do đó Before và After của mỗi cặp nằm liền nhau khi
CVAT dùng thứ tự lexicographical. Không đổi tên frame sau khi tạo bundle.

Khi tạo task CVAT:

1. Tạo hai task chính riêng cho validation và test.
2. Dán nội dung `cvat-labels.json` vào tab **Raw** của phần Labels hoặc truyền
   file này qua `cvat-cli`.
3. Upload archive tương ứng, chọn thứ tự **Lexicographical**, và đặt segment
   size là số chẵn để một job không cắt đôi cặp Before/After.
4. Chỉ vẽ rectangle cho năm lớp ontology.
5. Trên frame After, đặt `is_newly_added`, `is_valid_rescue`,
   `occlusion_level`; thêm đúng một tag `pair_status` và chọn
   `generation_status`.
6. Trên frame Before, vẽ mọi object thuộc ontology; các thuộc tính After được
   bỏ qua khi convert.
7. Tạo hai task độc lập cho archive double annotation và giao cho annotator thứ
   hai. Không cho annotator thứ hai xem nhãn của task chính.
8. Export bằng **CVAT for images 1.1** để giữ rectangle, tag và custom
   attributes.

CVAT không tự tạo đầy đủ custom attributes khi chỉ import dataset, vì vậy label
schema phải được cấu hình trước khi annotation.

## Kiểm tra pilot và chuyển export CVAT

Sau 10 cặp calibration, export task bằng **CVAT for images 1.1** rồi tạo artifact
pilot. Cờ `--allow-incomplete` chỉ dùng trong giai đoạn này:

```bash
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
BUNDLE_DIR="$BENCH_DIR/annotation-bundle"

venv/bin/python scripts/convert_cvat_bbox_annotations.py \
  --template "$BUNDLE_DIR/bbox-gt-v1.template.json" \
  --mapping "$BUNDLE_DIR/annotation-mapping.json" \
  --cvat-export "$BENCH_DIR/cvat/validation-pilot.zip" \
  --output "$BENCH_DIR/bbox-gt-v1.validation-pilot.json" \
  --split validation \
  --allow-incomplete
```

File `bbox-gt-v1.validation-pilot.summary.json` liệt kê cặp chưa làm và cặp còn
`ambiguous`. Một cặp chỉ được tính là hoàn tất khi ảnh After có đúng một tag
`pair_status`. Vì vậy cặp không có object vẫn phải có tag này. Trước khi khóa
artifact, xử lý mọi mục trong `unresolved` và không để thuộc tính object After ở
trạng thái `ambiguous`.

Khi đã gán đủ một split, chạy lại không có `--allow-incomplete`. Converter sẽ
từ chối khóa artifact nếu thiếu tag, còn nhãn mơ hồ, trùng frame, sai tên frame,
box ngoài biên hoặc dùng label ngoài ontology:

```bash
venv/bin/python scripts/convert_cvat_bbox_annotations.py \
  --template "$BUNDLE_DIR/bbox-gt-v1.template.json" \
  --mapping "$BUNDLE_DIR/annotation-mapping.json" \
  --cvat-export "$BENCH_DIR/cvat/validation-main.zip" \
  --output "$BENCH_DIR/bbox-gt-v1.validation.json" \
  --split validation
```

Mỗi nguồn export được lưu kèm tên task, task ID và SHA-256 trong artifact để có
thể truy vết. Không dùng export từ task double annotation làm nhãn chính trước
khi hoàn tất bước so sánh bất đồng và adjudication.

## Quy trình annotation

1. Annotator đọc guideline và gán thử 10 cặp calibration.
2. Chốt bất đồng trước khi gán tập chính.
3. Gán cả Before và After, bao gồm trạng thái generation.
4. Double-annotate tối thiểu 20% số cặp.
5. Adjudicate nhãn mơ hồ và mọi bất đồng.
6. Xuất artifact `bbox-gt-v1`, lưu checksum và không sửa sau khi mở kết quả test.

Tập validation được phép dùng để tuning. Tập test chỉ được mở sau khi cấu hình
pipeline cuối đã được khóa.

## Đánh giá bounding box

Artifact ground truth dùng cho evaluation phải giữ cấu trúc `pairs` của
manifest và bổ sung `generation_status`, `before_objects`, `after_objects` theo
schema trong guideline. Mỗi object After bắt buộc có `object_id`, `class_name`,
`bbox_xyxy`, `is_newly_added` và `is_valid_rescue`.

Chạy baseline trên validation bằng lệnh:

```bash
venv/bin/python scripts/evaluate_bounding_boxes.py \
  --ground-truth /datastore/khanhnd/models/aeroeyes_output/bbox_benchmark/bbox-gt-v1.json \
  --predictions /datastore/khanhnd/models/aeroeyes_output/output/bounding_boxes/reports \
  --split validation \
  --run-manifest /datastore/khanhnd/models/aeroeyes_output/bbox_benchmark/baseline-run.yaml \
  --output /datastore/khanhnd/models/aeroeyes_output/bbox_benchmark/baseline-validation-metrics.json
```

Sao chép `config/bounding_box/run-manifest.example.yaml` để tạo run manifest,
sau đó thay toàn bộ giá trị `REPLACE_WITH_*` bằng thông tin thực tế. Lưu Git
commit, model ID/revision, prompt, threshold, checksum ground truth, phiên bản
Python/Torch/CUDA, GPU, thời gian chạy và peak VRAM trước khi so sánh các cấu
hình.

CLI mặc định chỉ đánh giá `validation`. Chỉ dùng `--split test` sau khi đã khóa
pipeline cuối. `--split all` dành cho kiểm tra artifact, không dùng để tuning
hoặc công bố kết quả test.

Metrics JSON tách riêng:

- `generation`: tỷ lệ cặp sinh ảnh hợp lệ;
- `detector`: detection trên mọi rescue object hợp lệ trong ảnh After;
- `change_verification`: phân loại new/existing trên các object detector đã bắt
  được tại IoU 0.50, kèm coverage để không che detector miss;
- `final_pipeline`: kết quả cuối trên newly-added valid rescue objects;
- `pair_pass_rate`: chỉ số vận hành bổ trợ;
- `failure_taxonomy`: lỗi sinh ảnh theo pair; các nguyên nhân bỏ sót loại trừ
  nhau theo object; và false positive prediction được báo ở trường riêng;
- `performance`: runtime và peak VRAM nếu report có cung cấp.

AP dùng nội suy 101 điểm. `AP50` và `mAP50:95` tổng hợp macro trên các lớp có
ground truth; Precision/Recall/F1 tổng hợp micro. Lớp không có ground truth trả
về `insufficient_data`. Recall tại IoU 0.30 chỉ dùng để chẩn đoán box định vị
thô, không dùng làm kết quả chính.

### Đánh giá có nhãn mơ hồ — protocol v2

`schema_version=2` và `evaluation_protocol.version=bbox-metrics-v2-ambiguity-aware`
thay đổi cách tính prediction trên vùng mơ hồ. Lưu kết quả vào file mới và chạy
lại baseline bằng cùng protocol trước khi so sánh cấu hình; không so trực tiếp
precision/AP với file metrics v1.

- Chỉ cặp `generation_status=valid` tham gia detector/final metrics. Cặp sinh
  lỗi và cặp generation mơ hồ được thống kê riêng trong `annotation_coverage`.
- Detector dùng object có `is_valid_rescue=true` làm GT; object có validity mơ
  hồ là ứng viên ignore. Newness mơ hồ không loại object khỏi detector GT.
- Pipeline cuối dùng object có cả validity và newness là `true`. Object có thể
  vừa hợp lệ vừa mới nhưng còn thuộc tính mơ hồ là ứng viên ignore. Object đã
  xác định `is_valid_rescue=false` hoặc `is_newly_added=false` không che FP cuối.
- Prediction phải cùng cặp, cùng lớp và đạt IoU của metric đang tính mới được
  ghép với ignore. Mỗi ứng viên ignore chỉ nhận một prediction. Prediction được
  ghép với GT chắc chắn trước; prediction trùng một GT chắc chắn đã ghép vẫn là
  FP, kể cả khi cũng overlap vùng ignore.
- Ignore được tính lại tại từng ngưỡng IoU cho AP50, mAP50:95 và recall@0.30.
  Các count trong `overall`/`per_class` báo tại IoU 0.50:
  `raw_prediction_count = prediction_count + ignored_prediction_count`.
  `ignored_gt_count` là số ứng viên ignore, không cộng vào mẫu số recall.
- `failure_taxonomy.false_positive_predictions` dùng cùng chính sách với
  `final_pipeline`, tránh hai phần báo FP khác nhau.

Đây là chính sách đánh giá có điều kiện trên nhãn xác định được, không tự xác
nhận một nhãn đã review hay chốt ground truth. Có note và được ignore không
thay thế adjudication. Khi review còn dang dở, kết quả vẫn là thăm dò; báo cáo
coverage kèm metric để không diễn giải thành hiệu năng trên toàn bộ 40 cặp.

Chạy lại trên server bằng reports hiện có, không cần GPU:

```bash
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
RUN_DIR="$BENCH_DIR/runs/baseline-ambiguity-v2-$(date +%Y%m%d-%H%M%S)"

venv/bin/python scripts/evaluate_bounding_boxes.py \
  --ground-truth "$BENCH_DIR/bbox-gt-v1.validation-review.json" \
  --predictions /datastore/khanhnd/models/aeroeyes_output/output/bounding_boxes/reports \
  --split validation \
  --output "$RUN_DIR/metrics.json"
```

Kiểm tra implementation:

```bash
venv/bin/python -m unittest tests.test_bbox_metrics
venv/bin/python -m unittest tests.test_bbox_ambiguity
venv/bin/python scripts/evaluate_bounding_boxes.py --help
```

## Thí nghiệm Grounding DINO trên validation

Checkpoint này dùng các report tách riêng cho từng job. Nó chọn ảnh theo
`bbox-gt-v1-manifest.json`, chỉ lấy split `validation`, kiểm tra SHA-256 ảnh và
kiểm tra ground truth cùng nguồn trước khi nạp model. Ảnh Before/After khác kích
thước bị từ chối. Job không ghi vào `output/bounding_boxes` cũ.

Hai cấu hình đầu tiên chỉ khác ngưỡng box:

| Config | Box threshold | Text threshold | Rescue boat threshold | NMS IoU |
|---|---:|---:|---:|---:|
| `dino-baseline.json` | 0.25 | 0.25 | 0.33 | 0.50 |
| `dino-low-threshold.json` | 0.18 | 0.25 | 0.33 | 0.50 |

Cả hai dùng cùng model `IDEA-Research/grounding-dino-tiny`, cùng prompt gốc và
seed 20261020. Job ghi lại trạng thái deterministic của Torch; seed một mình
không bảo đảm các phép GPU luôn xác định hoàn toàn.
`0.18` là điểm thử, chưa phải ngưỡng được chọn. Ngưỡng riêng của `rescue_boat`
vẫn là 0.33 vì pipeline lấy giá trị lớn hơn giữa ngưỡng box và ngưỡng riêng.
Như vậy thay đổi 0.25 xuống 0.18 không hạ ngưỡng riêng của lớp thuyền.

Transformer phiên bản cũ dùng tên `box_threshold`, phiên bản mới dùng `threshold`;
cả hai có thể có `text_threshold`. Job kiểm tra API và dừng nếu không thể áp
dụng ngưỡng text, để cấu hình được ghi khớp với suy luận. Xem [Hugging Face
Grounding DINO documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino).

Trước tiên chạy smoke test **5 cặp cho mỗi cấu hình**:

```bash
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-baseline.json 5
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-low-threshold.json 5
```

Sau khi cả hai hoàn tất và đã kiểm tra log/metrics, chạy cùng 40 cặp:

```bash
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-baseline.json 40
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-low-threshold.json 40
```

Theo dõi bằng `squeue -u "$USER"` và `sacct -j <JOB_ID>`; log nằm tại
`logs/bbox_experiment_<JOB_ID>.out` và `.err`. Mỗi job tạo thư mục mới
`$BENCH_DIR/runs/bbox-experiment-<JOB_ID>/` gồm reports, visualization, nhãn,
`ground-truth.selected.json`, `run-manifest.json` và `metrics.json`. Script từ
chối thư mục đã tồn tại. Run manifest ghi cấu hình, checksum, pair ID, git,
model revision nếu Hugging Face cung cấp, Python/Torch/Transformers/CUDA, GPU,
thời gian và peak VRAM.

Subset 5 cặp được lấy cách đều theo `pair_id` đã sắp xếp và là smoke test, không
đại diện toàn tập. Evaluator chỉ dùng `ground-truth.selected.json` tương ứng nên
không tính các cặp chưa chạy là FN. Có thể nhận `insufficient_data` khi subset
không có GT đủ điều kiện. Chỉ so sánh cấu hình trên **cùng pair ID, cùng nhãn,
cùng protocol evaluator v2**. Ground truth chưa adjudicate và các cặp generation
mơ hồ vẫn bị loại khỏi detection metrics; ghi rõ coverage khi báo kết quả.

So sánh hai job cùng quy mô bằng ID thực tế của chúng:

```bash
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
BASE_JOB=<JOB_ID_BASELINE>
LOW_JOB=<JOB_ID_LOW_THRESHOLD>

venv/bin/python scripts/compare_bbox_experiments.py \
  --baseline "$BENCH_DIR/runs/bbox-experiment-$BASE_JOB" \
  --candidate "$BENCH_DIR/runs/bbox-experiment-$LOW_JOB" \
  --vary box_threshold
```

Thay hai placeholder `<...>` bằng số job trước khi chạy; dấu `<`/`>` là ký hiệu
giữ chỗ, không gõ vào shell. Script từ chối so sánh nếu khác pair, GT checksum,
evaluator version, git commit, hoặc có thay đổi cấu hình ngoài trường được chọn.
Nó kiểm tra metrics gắn đúng run manifest và không có report bị thiếu. Nếu model
revision hoặc Git không xác nhận được, hoặc GT chưa khóa, kết quả có warning.
Script không xác minh được adjudication, kể cả khi converter ghi trạng thái
`validation_locked`; cần người review xác nhận trước khi công bố metric.
Kết quả trả chênh lệch precision, recall, AP, FP/FN, thời gian và VRAM. Với thử
prompt, dùng `--vary prompt` và giữ các trường cấu hình khác giống nhau.

Muốn thử prompt/ngưỡng khác, sao chép một JSON config và đổi `name`, `prompt`
hoặc ngưỡng; không sửa trực tiếp kết quả đã chạy. Các nhãn detector không khớp
5 lớp hoặc alias cho phép trong `ontology.yaml` sẽ bị bỏ qua. Trong các job thí
nghiệm, chỉ `automatic_aliases` được chuyển tự động về lớp canonical; các
`review_terms` không được tự động chuyển. NMS và ngưỡng riêng áp dụng trên lớp
canonical; ví dụ `emergency boat` vẫn phải qua ngưỡng `rescue_boat=0.33`.

### Thử mở rộng prompt bằng alias ontology

Sau thử nghiệm `text_threshold=0.18`, `firefighter` tăng recall trên validation
nhưng `emergency_vehicle` vẫn còn nhiều FN. Hai config sau có cùng model, seed,
ngưỡng box 0.18, ngưỡng text 0.18 và mapping alias; chỉ khác prompt:

- `dino-prompt-control.json`: prompt 5 lớp gốc;
- `dino-prompt-aliases.json`: thêm `fire fighter`, `fire truck`, `fire engine`,
  `rescue truck`, đều thuộc `automatic_aliases` của ontology hiện tại.

Chạy lại **cả hai** sau khi commit checkpoint này vì code mapping đã thay đổi.
Không dùng job 8090 làm control trực tiếp cho phép so sánh prompt: nó dùng commit
trước khi có mapping alias. Prompt mở rộng có thể tăng FP hoặc sinh nhãn không
khớp ontology, không mặc định cải thiện recall.

```bash
# Smoke test; sau khi kiểm tra thành công, thay 5 bằng 40 cho cả hai job.
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-prompt-control.json 5
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-prompt-aliases.json 5
```

Khi hai job hoàn tất, so sánh bằng `scripts/compare_bbox_experiments.py` với
`--vary prompt`. Thay hai số ID bằng ID thực tế:

```bash
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
BASE_JOB=1234  # ID của dino-prompt-control
ALIAS_JOB=5678 # ID của dino-prompt-aliases

venv/bin/python scripts/compare_bbox_experiments.py \
  --baseline "$BENCH_DIR/runs/bbox-experiment-$BASE_JOB" \
  --candidate "$BENCH_DIR/runs/bbox-experiment-$ALIAS_JOB" \
  --vary prompt
```

Run manifest ghi SHA-256 của ontology và class map; comparator từ chối khi các
hash này khác nhau và cảnh báo nếu run cũ không có hash. `label_mapping_counts`
cho biết số label sau postprocess, số lần alias tự động được dùng và số label
không map được. Đây là số detection trước lọc bổ sung/NMS trên cả Before và
After, không phải số object thật hay số object duy nhất.

### Chuyển manifest Ground Truth validation v2

`scripts/adapt_bbox_strict_v2.py` đọc manifest trong ZIP v2, kiểm tra ID với
benchmark validation và tách `after_objects` khỏi `ignore_regions`. Output lưu
checksum nguồn và giữ nguyên `generation_status`. Evaluator cũ chủ động từ chối
output này; sử dụng `scripts/evaluate_bbox_strict_v2.py` để tính metric.

```bash
venv/bin/python scripts/adapt_bbox_strict_v2.py \
  --bundle /duong-dan/AeroEyes_GT_VALIDATION_V2_FINAL.zip \
  --benchmark-manifest /duong-dan/bbox-gt-v1-manifest.json \
  --output /duong-dan/validation-v2-adapted.json
```

Evaluator v2 dùng toàn bộ cặp validation, kể cả cặp ambiguous hoặc sinh lỗi,
và ghi riêng thống kê trạng thái sinh. Detector dùng tất cả confirmed After;
pipeline dùng confirmed new. Ở cả hai tầng, positive được ghép trước với IoU
0.5; vùng ignore cùng lớp/cặp được ghép một-một với dự đoán còn lại tại IoU
0.5. Dự đoán trùng positive đã ghép vẫn được tính là FP. Kết quả gồm micro F1,
macro F1 trên lớp có GT, IoU trung bình của các box đã ghép, thống kê theo lớp,
và phép thử độ nhạy khi coi các vùng ignore là không có vùng bỏ qua. Chính sách
v2 khác evaluator cũ nên không so trực tiếp metric giữa hai phiên bản GT.
Sau khi tạo các file metric từ cùng GT, `scripts/compare_bbox_strict_v2.py`
kiểm tra checksum ground truth, evaluator, phạm vi và ontology trước khi in
kết quả và từng cấu hình. Cấu hình khác nhiều tham số chỉ cho phép so sánh
tổng thể, không tách được tác động riêng của từng tham số.

Có thể đánh giá lại report inference đã lưu, không cần chạy lại GPU. Lệnh từ chối
report thiếu/thừa, checksum benchmark khác, job chưa hoàn tất và không ghi đè
output. Hash của từng report được ghi tại thời điểm đánh giá; các job inference
cũ chưa lưu hash từng report nên không thể xác minh chúng chưa thay đổi sau chạy.

```bash
venv/bin/python scripts/evaluate_bbox_strict_v2.py \
  --ground-truth /duong-dan/validation-v2-adapted.json \
  --run-dir /duong-dan/bbox-experiment-8105 \
  --output /duong-dan/metrics-strict-v2-8105.json
```

### Thử prompt bổ sung cho xe cứu hộ

`dino-supplemental-vehicles.json` giữ nguyên prompt, ngưỡng và mapping của
`dino-prompt-control.json`, sau đó chạy thêm ba prompt độc lập `ambulance`,
`fire truck`, `fire engine` trên cả Before và After. Chỉ box thuộc lớp mục tiêu
được giữ từ mỗi lượt bổ sung. Tất cả box gốc và bổ sung được NMS cùng lớp trước
khi vào pipeline; box bổ sung có thể thay box gốc nếu có confidence cao hơn.
Các prompt phải khớp alias được phép trong ontology. Confidence giữa các prompt
không nhất thiết được hiệu chuẩn như nhau; đây là thử nghiệm, không đảm bảo tăng
recall hoặc precision.

Sau khi commit và pull cùng phiên bản code, chạy hai job 40 cặp:

```bash
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-prompt-control.json 40
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-supplemental-vehicles.json 40
```

Script SLURM vẫn ghi metric v1 trong mỗi run; dùng
`scripts/evaluate_bbox_strict_v2.py` với `strict-v2-results/validation-v2-adapted.json`
để đánh giá cả hai report theo GT v2, rồi dùng `compare_bbox_strict_v2.py`.
Run manifest ghi `supplemental_prompt_counts` cho từng prompt trên cả Before và
After trước NMS tổng hợp; đó không phải số đối tượng duy nhất hoặc số TP.
Lượt inference bổ sung làm tăng thời gian chạy. Kết quả mới vẫn mang giới hạn
GT một người rà soát và không thay thế đánh giá trên tập test.

### Thử ánh xạ nhãn ghép firefighter

Job 8095 có nhiều label thô `firefighter_fire_fighter`. Phép thử tiếp theo
chỉ ánh xạ chính xác label này thành `firefighter`; các label ghép nhiều lớp
vẫn không được tự động ánh xạ. Đây là giả thuyết cần đo, không đảm bảo đúng
về mặt thị giác hoặc cải thiện recall. Chế độ mặc định `exact` không đổi.

Sau khi commit, chạy lại hai cấu hình có cùng prompt mở rộng, chỉ khác
`label_mapping_mode`, trên cùng bản ground truth và cùng commit:

```bash
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-prompt-aliases.json 40
sbatch scripts/bbox_experiment.slurm config/bounding_box/dino-prompt-compound.json 40
```

So sánh hai job mới bằng `--vary label_mapping_mode`. Để so với prompt gốc,
chạy thêm `dino-prompt-control.json` ở cùng commit và dùng `--vary prompt`
để so với job `dino-prompt-aliases.json`. Không dùng trực tiếp các job 8094/8095
cho comparator do khác commit. Giữ ground truth cố định giữa các job.

Để kiểm tra selection và checksum không dùng GPU, chọn một thư mục **chưa tồn tại**:

```bash
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
venv/bin/python scripts/run_bbox_experiment.py \
  --manifest "$BENCH_DIR/bbox-gt-v1-manifest.json" \
  --ground-truth "$BENCH_DIR/bbox-gt-v1.validation-review.json" \
  --config config/bounding_box/dino-baseline.json \
  --output-dir "$BENCH_DIR/runs/prepare-check-$(date +%Y%m%d-%H%M%S)" \
  --limit 5 --prepare-only
```
