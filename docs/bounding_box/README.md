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

Kiểm tra implementation:

```bash
venv/bin/python -m unittest tests.test_bbox_metrics
venv/bin/python scripts/evaluate_bounding_boxes.py --help
```
