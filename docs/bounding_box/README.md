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
