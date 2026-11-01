# Kế hoạch hoàn thiện nghiên cứu bounding box

Kế hoạch được tạo ngày 2026-10-20 tại commit `2fe21a4`. Thực hiện theo đúng
thứ tự bên dưới vì mỗi giai đoạn sử dụng artifact đã được khóa từ giai đoạn
trước. Không đánh giá trên test trước khi khóa cấu hình cuối.

## Thứ tự và trạng thái

| Plan | Nội dung | Ưu tiên | Công sức | Phụ thuộc | Trạng thái |
|---|---|---|---|---|---|
| 001 | Hoàn thiện benchmark và pipeline auto-annotation bounding box | P0 | L | — | IN PROGRESS |

Trạng thái hợp lệ: `TODO`, `IN PROGRESS`, `DONE`, `BLOCKED`, `REJECTED`.

## Nguyên tắc thực nghiệm

- `bbox_passed` là chỉ số vận hành, không phải recall.
- Validation được dùng để phân tích lỗi và chọn cấu hình; test chỉ được chạy
  sau khi cấu hình đã khóa.
- Tập challenge dùng để phân tích lỗi, không được gộp vào kết quả đại diện nếu
  được lấy mẫu có chủ đích.
- SAM2 và delta mask chỉ được giữ khi ablation trên validation chứng minh lợi ích.
- Ground truth phải mô tả quan hệ Before/After và phân biệt lỗi sinh ảnh với lỗi
  detection, matching và localization.

## Hạng mục chưa làm trong plan này

- Huấn luyện detector mới từ đầu.
- Chạy lại CLIP filtering trên server mới; đây là benchmark lọc dữ liệu riêng.
- Audit sâu pipeline sinh ảnh, SDQM và mã trong `third_party/`.
