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
