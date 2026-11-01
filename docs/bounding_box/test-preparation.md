# Chuẩn bị tập test bounding box

## Trạng thái

- Nhóm xác nhận 60 cặp test chưa được gán nhãn.
- File `test-images.zip` trên máy làm việc đã được kiểm tra tên: 120 ảnh PNG,
  60 cặp Before/After đầy đủ, không trùng pair hash. SHA-256 của file đã kiểm tra:
  `7a204ef70003a27864368e65ace4da6ce0049553ff613e82a07db9a6208f29cf`.
  Kiểm tra này không chứng minh file khớp manifest trên server hoặc các nhóm sự
  kiện độc lập.
- Ứng viên hiện tại là `config/bounding_box/dino-ambulance-clip.json` tại commit
  `7c2a4309c7fbadb06734aa2fcf589ca277bedd38`, kèm model DINO revision
  `a2bb814dd30d776dcf7e30523b00659f4f141c71` và CLIP revision
  `57c216476eefef5ab752ec549e440a49ae4ae5f3`. Cần đối chiếu commit và trạng
  thái worktree trong run manifest job 8282 trên server trước khi coi đây là
  checkpoint có thể tái lập.
- Validation strict-v2 của job 8282: 40 cặp, pipeline TP=97, FP=144, FN=32,
  P=0.40249, R=0.75194, F1=0.52432. Đây là kết quả thăm dò; GT do một người rà
  soát, quy tắc CLIP được chọn sau khi xem validation.
- Manifest được báo trước đó dùng `pair_id_fallback` cho cả 100 cặp, nên chưa
  chứng minh được chia tách theo source/event group. Plan 001 yêu cầu dừng trước
  kết luận test độc lập khi chưa xác định được nhóm sự kiện. Cần kiểm tra khả
  năng khôi phục nguồn hoặc thống nhất và ghi rõ giới hạn protocol trước khi
  gán nhãn toàn bộ / công bố kết quả. Không tự đổi split đã có annotation.
- Runner và strict-v2 adapter/evaluator hiện dành cho validation; chưa dùng
  chúng để chạy hoặc chấm 60 cặp test. Cần triển khai và kiểm tra riêng sau khi
  protocol cùng artifact GT test được xác nhận.

## Kiểm tra nguồn và run trước khi gán nhãn

Trên server, từ project root, chạy kiểm tra read-only sau. Các trường trùng
giữa hai split là dấu hiệu cần xem lại; không có trùng chính xác cũng không
chứng minh không có ảnh gần trùng hoặc cùng sự kiện.

```bash
cd /datastore/khanhnd/aeroeyes_cloudian/aeroeyes-dataset
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark

venv/bin/python - "$BENCH_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "bbox-gt-v1-manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
pairs = manifest["pairs"]
print("COUNTS:", Counter(pair["split"] for pair in pairs))
print("GROUP METHODS:", Counter(pair.get("source_group_method") for pair in pairs))
for field in ("source_group", "image_key", "before_sha256", "url"):
    values = {}
    for split in ("validation", "test"):
        values[split] = {str(pair[field]).strip() for pair in pairs
                         if pair.get("split") == split and pair.get(field)}
    overlap = values["validation"] & values["test"]
    print(f"{field}: validation={len(values['validation'])} "
          f"test={len(values['test'])} exact_overlap={len(overlap)}")

run_path = root / "runs/bbox-experiment-8282/run-manifest.json"
run = json.loads(run_path.read_text(encoding="utf-8"))
print("RUN STATUS:", run.get("status"))
print("RUN GIT:", run.get("git"))
print("RUN CONFIG:", json.dumps(run.get("config"), ensure_ascii=False))
print("DINO REVISION:", run.get("result", {}).get("model_revision"))
print("CLIP DETAILS:", json.dumps(run.get("result", {}).get("vehicle_clip_filter"), ensure_ascii=False))
PY

sha256sum "$BENCH_DIR/bbox-gt-v1-manifest.json" \
  "$BENCH_DIR/annotation-bundle/test-images.zip" \
  "$BENCH_DIR/strict-v2-results/validation-v2-adapted.json"
```

Đối chiếu manifest với SHA-256 đã ghi khi tạo benchmark
`5ce034905ca113b0cfd3f67fa39e1906ca9a43397d44e7d0057c328e48b1faed`, ZIP với
giá trị ở trên và GT validation với SHA-256 trong metrics job 8282. Nếu nguồn
không khôi phục được, ghi rõ test chỉ tách theo cặp; không mô tả là độc lập
theo sự kiện.

## Chuẩn bị CVAT sau khi thống nhất protocol

1. Dùng task test riêng, cùng project/ontology với validation; không import
   prediction hay nhãn validation. Chọn **Lexicographical** và segment size số
   chẵn để giữ hai ảnh của một cặp cùng job. Đối chiếu đủ 120 frame.
2. Gán Before/After theo `annotation-guideline.md`; ảnh After cần box và các
   thuộc tính new/existing, valid/ambiguous, occlusion và notes khi cần. Mỗi
   After cần đúng một tag `pair_status` với `generation_status`, kể cả khi
   không có đối tượng. Có thể giữ `ambiguous` kèm lý do thay vì tự đoán.
3. Cần prompt gốc theo pair để xác định tình huống `missing_requested_object`.
   Nếu chưa có prompt, giữ trạng thái phù hợp là `ambiguous` và ghi lý do;
   không khẳng định thất bại sinh ảnh chỉ từ ảnh.
4. Nếu thực hiện theo protocol double annotation, dùng task độc lập có sẵn từ
   `test-double-annotation-images.zip` (20% = 12 cặp), annotator thứ hai không
   xem nhãn chính. Ghi rõ phạm vi review và việc đã adjudicate hay chưa.
5. Export **CVAT for images 1.1**, giữ file gốc và checksum. Không dùng
   converter v1 để ép trạng thái `ambiguous` thành nhãn chắc chắn: cần adapter
   và evaluator test tương thích ignore regions như strict-v2 validation.

Sau khi chốt protocol, GT và cấu hình, đánh giá test một lần. Báo cáo số lượng
GT mỗi lớp; nếu lớp không đạt tối thiểu 20 đối tượng mới theo plan, ghi không đủ
dữ liệu cho kết luận theo lớp. Nếu thay đổi code/quy tắc sau khi xem kết quả
test, ghi rõ phiên bản và việc tập test đã bị lộ cho lần thử tiếp theo.
