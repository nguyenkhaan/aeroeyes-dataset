# T1 — Rà soát và thống kê validation bounding box

**Ngày rà soát:** 03/11/2026. **Trạng thái:** Đã hoàn thành đối chiếu và thống kê theo ba artifact được cung cấp; các giới hạn về nguồn sự kiện và tình trạng adjudication vẫn cần theo dõi.

## 1. Artifact và đối chiếu

Ba file do nhóm cung cấp được sao chép vào `data/output/t1-input/` trên máy làm việc. Thư mục này không được Git theo dõi; cần lưu trữ artifact gốc ở nơi dùng chung để nhóm khác tái lập.

| Artifact | SHA-256 |
|---|---|
| `bbox-gt-v1-manifest.json` | `5ce034905ca113b0cfd3f67fa39e1906ca9a43397d44e7d0057c328e48b1faed` |
| `validation-v2-adapted.json` | `ec43309668199aa7368956e39716d2b57f0f25c24ee5c60e78fd73c4f98aa8ee` |
| `metrics-8282.json` | `33381cc3781ffec04227300c0132d645f31625afb8a6b4b6629a31572451b9c4` |

- Manifest có 100 cặp: 40 validation và 60 test; GT v2 có 40 pair ID duy nhất và trùng đúng tập validation của manifest.
- SHA-256 manifest khớp giá trị nguồn được ghi trong GT v2 và metrics. SHA-256 GT v2 khớp `ground_truth_sha256` trong metrics. Đã tính hash trực tiếp trên ba file; chưa đối chiếu trực tiếp ảnh, file strict manifest nguồn, run manifest hay report prediction.
- GT dùng schema `strict-validation-v2-adapter-1`, dataset ID `AeroEyes_GT_validation_v2_reviewed_single_annotator`, trạng thái `single_annotator_reviewed`.
- Metrics ghi protocol `bbox-strict-validation-v2-iou50-1`, job/config `dino-ambulance-clip`, commit `7c2a4309c7fbadb06734aa2fcf589ca277bedd38`. Số liệu coverage và số GT theo lớp khớp phép đếm từ GT. Chưa chạy lại inference/evaluator hoặc xác minh các report prediction gốc.

## 2. Thống kê Ground Truth v2

Số liệu dưới đây được tính trực tiếp từ 40 cặp trong GT adapted. **Một object** là một instance trong danh sách tương ứng; **một cặp có lớp** là cặp có ít nhất một object mới xác nhận thuộc lớp đó. Một cặp có thể có nhiều lớp, nên số cặp theo lớp không cộng thành 40.

| Lớp | Before: object | After: xác nhận mới | After: xác nhận đã tồn tại | Tổng After xác nhận | Số cặp có object mới | Ignore regions |
|---|---:|---:|---:|---:|---:|---:|
| `firefighter` | 5 | 61 | 5 | 66 | 11 | 108 |
| `rescue_boat` | 0 | 15 | 0 | 15 | 6 | 2 |
| `helicopter` | 0 | 1 | 0 | 1 | 1 | 0 |
| `ambulance` | 0 | 8 | 0 | 8 | 6 | 0 |
| `emergency_vehicle` | 1 | 44 | 1 | 45 | 27 | 9 |
| **Tổng object** | **6** | **129** | **6** | **135** | — | **119** |

`After: xác nhận mới` là mẫu số GT cho metric pipeline cuối; detector dùng cả 135 object After đã xác nhận, bao gồm 6 object cũ. Vùng ignore không được mặc định là object cứu hộ dương tính và không cộng vào 135. Trong GT này 119 vùng ignore nằm trên 23 cặp: tất cả đều có `is_valid_rescue=ambiguous`; 112 vùng có `is_newly_added=true`, 7 vùng còn lại có `is_newly_added=ambiguous`. Các thuộc tính này chồng lấn, không cộng hai nhóm để ra số vùng.

Đối tượng `helicopter` mới chỉ có **1 instance trong 1 cặp**, `ambulance` có **8 instance trong 6 cặp**. Cả hai cần được ưu tiên xem xét khi chọn mẫu validation bổ sung. `firefighter` có 108/119 vùng ignore, cho thấy cần rà soát các trường hợp khó nhận diện vai trò; không thể kết luận toàn bộ vùng này là firefighter thực sự.

### Trạng thái sinh ảnh

| `generation_status` | Số cặp |
|---|---:|
| `valid` | 27 |
| `ambiguous` | 10 |
| `missing_requested_object` | 1 |
| `severe_scene_corruption` | 2 |
| **Tổng** | **40** |

`ambiguous` ở bảng này là trạng thái **cấp cặp**, không phải số object hoặc số ignore region. Theo protocol, các cặp này không bị loại toàn bộ: các object xác nhận vẫn được đánh giá và các vùng ignore được xử lý riêng.

GT adapted không chứa trường `occlusion`/`occlusion_level` trong các object. Vì vậy, ba artifact này chưa đủ để thống kê mức che khuất; cần strict manifest nguồn hoặc export annotation phù hợp và kiểm tra schema trước khi tính.

## 3. Đối chiếu metrics job 8282

Metrics ghi kết quả pipeline trên 40 cặp: **TP 97, FP 144, FN 32; Precision 40,25%, Recall 75,19%, F1 52,43%**. Mẫu số 129 object mới và số lượng từng lớp khớp GT đã kiểm đếm; 119 ignore regions cũng khớp coverage. 322 prediction đầu vào gồm 81 prediction được ignore, còn 241 prediction được chấm là 97 TP và 144 FP. Đây là số liệu từ metrics đã lưu, không phải lần đánh giá lại từ prediction gốc.

GT và cấu hình được dùng để phát triển trên validation; kết quả không phải đánh giá độc lập trên test. Mean matched IoU chỉ xét các box đã ghép đúng và không phản ánh các object bị bỏ sót.

## 4. Nguồn dữ liệu và tách split

Manifest có đủ `source_group`, `image_key`, URL, checksum Before và checksum After ở cả 100 cặp, nhưng **100/100 source group được tạo bằng `pair_id_fallback`**. So sánh các giá trị đã ghi trong manifest giữa 40 validation và 60 test cho thấy 0 giá trị trùng chính xác ở `pair_id`, `source_group`, `image_key`, URL, checksum Before và checksum After. Đây là đối chiếu metadata trong manifest, chưa tính lại checksum từ ảnh và chưa kiểm tra ảnh gần trùng.

Không trùng chính xác không chứng minh hai split độc lập theo sự kiện: fallback chỉ gắn nhóm theo cặp. Báo cáo trước ghi `image_summary.json` khớp 100/100 cặp và không có bốn trường mã sự kiện đã kiểm tra; file summary không nằm trong ba artifact lần này nên nhận định đó chưa được kiểm tra lại. Khi chọn 60 cặp validation mới, cần đối chiếu nguồn, ID và checksum với cả hai split, đồng thời ghi rõ các nguồn chưa xác minh và các mẫu tăng cường lớp hiếm.

## 5. Kiểm soát chất lượng và giới hạn

- Các tài liệu review cục bộ cho thấy hai export từng được so sánh trên 8 cặp; file đối chiếu còn các ô quyết định để trống. Bằng chứng này không cho biết có quyết định sau đó ở nơi khác hay không, và export riêng không xác nhận tính độc lập của quá trình gán nhãn.
- Artifact GT hiện tại tự ghi `single_annotator_reviewed`; chưa có bằng chứng trong ba file cho thấy toàn bộ GT v2 đã được hai người rà soát độc lập và adjudicate. Cần xác nhận tình trạng thực tế với nhóm và tiếp tục review theo protocol.
- Bản trích xuất cục bộ của export ban đầu chứa 40 frame Before với 6 box và 40 frame After với 268 box. Đây là annotation **trước vòng rà soát GT v2**, không phải GT dương tính và không được so trực tiếp như kết quả đánh giá v2.
- Chưa có artifact để xác nhận nguồn sự kiện, tính checksum ảnh thực, xác minh report prediction và thống kê occlusion. Các giới hạn này không làm thay đổi phép đếm trên ba file đã cung cấp, nhưng cần ghi rõ khi mở rộng dataset và báo cáo kết quả.

## 6. Đề xuất cho bước tiếp theo

1. Dùng bảng mục 2 làm baseline về độ bao phủ lớp và số cặp khi chuẩn bị T5; tìm thêm helicopter và ambulance, ghi rõ phương pháp chọn có chủ đích.
2. Dùng 119 vùng ignore, đặc biệt 108 vùng mang nhãn `firefighter`, làm đầu vào cho T2 để làm rõ dấu hiệu nhận diện và quy tắc xử lý mơ hồ; không tự đổi các vùng này thành nhãn dương tính.
3. Xác nhận với nhóm trạng thái review 8 cặp và khả năng khôi phục metadata sự kiện; lưu lại bằng chứng và không dùng test để lựa chọn cấu hình.
