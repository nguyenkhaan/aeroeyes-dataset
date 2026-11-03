# Kế hoạch tuần: chuẩn hóa và thử nghiệm quy trình auto-annotation

**Thời gian:** 03–08/11/2026. Lịch công việc chính dự kiến từ thứ Ba đến thứ Sáu; cuối tuần là thời gian dự phòng, không mặc định nhóm phải làm việc. Phân công theo vai trò và quy mô pilot là đề xuất, cần điều chỉnh theo nhân lực, dữ liệu và quyền truy cập thực tế.

**Mục tiêu tuần:** Có quy trình annotation bốn round đủ rõ để thử, kiểm tra được khả năng đưa prediction vào CVAT, hoàn thành pilot đo thời gian và chất lượng, đồng thời chuẩn bị nguồn dữ liệu để mở rộng validation từ 40 lên 100 cặp. Mức 1–2 phút/cặp là mục tiêu cần kiểm chứng, không phải kết quả đã đạt.

Kế hoạch này là bước ngắn hạn tiếp nối [Plan 001](001-complete-bbox-research-benchmark.md); các quy tắc về tách test và giới hạn nguồn dữ liệu vẫn được giữ. Danh sách theo dõi cá nhân ở [tasks/todo.md](../tasks/todo.md); thư mục `tasks` không được theo dõi trong Git.

## Kết quả cần có cuối tuần

1. Bản cập nhật guideline và checklist bốn round, kèm các điểm còn cần giảng viên hoặc nhóm thống nhất.
2. Bảng thống kê GT validation v2 hiện có và danh sách ứng viên cho 60 cặp bổ sung, kèm phương pháp chọn và tình trạng kiểm tra trùng nguồn. Nếu chưa đủ ảnh, ghi rõ số có sẵn và phần thiếu.
3. Kết quả thử prediction → CVAT → export trên một số cặp, ghi rõ dữ liệu giữ được và các thao tác còn thủ công.
4. Bảng pilot so sánh ba phương pháp, có thời gian, thao tác sửa và kết quả kiểm tra chất lượng độc lập; nếu chưa đủ điều kiện so sánh, báo rõ phần còn thiếu.
5. Quyết định bước tiếp theo: dùng hay tiếp tục sửa luồng AI hỗ trợ, khối lượng annotation tuần sau và những vấn đề cần trao đổi với thầy.

## Đã có và chưa có

- Đã có guideline và ontology năm lớp, công cụ chuẩn bị bundle CVAT, chuyển đổi export, evaluator strict-v2 và các cấu hình Grounding DINO/CLIP. Xem [hướng dẫn bounding box](../docs/bounding_box/README.md) và [báo cáo tuần trước](../docs/reports/bao-cao-cai-tien-auto-annotation.md).
- Báo cáo ghi nhận 40 cặp validation GT v2 và 60 cặp test đã chuẩn bị nhưng chưa gán nhãn. GT validation chưa được xác minh độc lập toàn bộ; đối chiếu artifact thực tế trước khi thống kê.
- Công cụ chuẩn bị bundle và đọc export không chứng minh luồng import prediction vào CVAT đã hoàn chỉnh. Cần thử trực tiếp, bao gồm dữ liệu detection cho cả Before và After.
- Chưa chứng minh được các split độc lập theo sự kiện; `pair_id_fallback` chỉ bảo đảm danh tính cặp, không loại trừ ảnh gần trùng hoặc cùng sự kiện.

## Công việc trong tuần

### T1 — Rà soát dữ liệu và thống kê 40 cặp

**Mục tiêu:** Biết chính xác dữ liệu, phân bố lớp và các giới hạn hiện có trước khi chọn thêm mẫu.

- Đối chiếu manifest, GT strict-v2, checksum, đường dẫn artifact và trạng thái double annotation/adjudication thực tế; không giả định artifact trên server đã có ở máy hiện tại.
- Thống kê số cặp; số đối tượng theo lớp, Before/After, mới/cũ và ambiguous theo trường có sẵn; báo rõ định nghĩa mẫu số và những trường không thể thống kê. Đặc biệt ghi số `helicopter` và `ambulance` mới.
- Kiểm tra trường nguồn, mức độ đầy đủ metadata và bằng chứng đã kiểm tra trùng validation/test. Không diễn giải không trùng chính xác thành độc lập theo sự kiện.

**Đầu ra:** Bảng thống kê và danh sách dữ liệu/giới hạn cần xử lý, lưu tại `docs/bounding_box/` hoặc thư mục báo cáo nhóm thống nhất; ghi vị trí artifact lớn bên ngoài Git.

**Nghiệm thu:** Tổng theo lớp khớp GT đang dùng, nêu phiên bản/checksum và phạm vi ambiguous; thông tin chưa truy cập được được đánh dấu chưa xác minh. **Phụ thuộc:** quyền truy cập GT/manifest; có thể làm song song T2.

### T2 — Chốt guideline và thiết kế bốn round

**Mục tiêu:** Người gán có cùng cách hiểu trước khi thử quy trình.

- Rà soát [annotation-guideline.md](../docs/bounding_box/annotation-guideline.md) và [ontology.yaml](../config/bounding_box/ontology.yaml), bổ sung ví dụ và quy tắc cho lớp dễ nhầm, che khuất, box sát phần nhìn thấy, Before/After, đối tượng mới/cũ và tình huống mơ hồ. Phân biệt tên lớp chuẩn với các cụm prompt; không tự gộp cụm gần nghĩa.
- Mô tả đầu vào, thao tác, đầu ra và checklist cho bốn lượt: tạo đề xuất; tìm đối tượng thiếu; chỉnh box và thuộc tính; kiểm tra chất lượng. Có thể làm các lượt liên tiếp theo cặp hoặc theo batch, ghi cách thực tế đã dùng khi pilot.
- Một đối tượng phân biệt được có một box; vùng gộp chỉ được dùng làm ghi chú review tạm, không xuất thành object GT. Ghi nhận các điểm chưa thống nhất thay vì ép nhãn.
- Thống nhất cách xem prompt sinh ảnh để xác định `generation_status` và xử lý trường hợp thiếu prompt; giữ `ambiguous` khi chưa đủ bằng chứng.

**Đầu ra:** Guideline cập nhật, checklist bốn round và danh sách câu hỏi còn mở; chỉ thay ontology/schema nếu nhóm đã thống nhất và có kế hoạch phiên bản tương ứng.

**Nghiệm thu:** Ít nhất hai người cùng áp dụng trên một số cặp calibration, ghi bất đồng và quyết định xử lý; mỗi round có điều kiện hoàn thành rõ. **Phụ thuộc:** tài liệu hiện có; calibration cần ảnh và người review.

### T3 — Kiểm tra luồng AI hỗ trợ trên CVAT

**Mục tiêu:** Xác định có thể tái sử dụng prediction hiện tại để giảm thao tác đến đâu.

- Khảo sát report/detection trung gian của pipeline, bao gồm Before và After, lớp, tọa độ và khả năng tạo file import tương thích CVAT đang dùng. Không giả định chỉ output box đối tượng mới là đủ cho GT.
- Thử import trên 2–3 cặp phát triển, kiểm tra đúng ảnh, box, lớp và chỉnh sửa/xóa; export lại và kiểm tra khả năng giữ thuộc tính, tag, frame mapping. Ghi rõ thuộc tính cần người gán bổ sung và mọi giới hạn của định dạng.
- Nếu thiếu converter, ghi thiết kế và ước lượng nhỏ nhất để thực hiện. Chỉ phát triển khi thời gian và điều kiện cho phép; vẫn tiến hành pilot thủ công nếu chưa có luồng AI.

**Đầu ra:** Ghi chú kiểm thử, ví dụ artifact và danh sách khoảng trống; module dự kiến khảo sát gồm `src/evaluation/bbox_annotation.py`, `src/evaluation/cvat_bbox.py`, `scripts/prepare_bbox_annotation.py` và report pipeline.

**Nghiệm thu:** Mỗi box thử nghiệm đối chiếu được ảnh và lớp; có kết luận rõ về tọa độ, thuộc tính và import/export. Không khai báo thành công nếu mới chỉ xuất nhãn YOLO hoặc đọc export CVAT. **Phụ thuộc:** CVAT, ảnh/prediction và T2 bản nháp.

### T4 — Pilot đo thời gian và chất lượng

**Mục tiêu:** Kiểm tra bốn round và AI có giảm tổng công người mà vẫn giữ chất lượng hay không.

- Ba phương án: A — thủ công hiện tại; B — thủ công bốn round; C — prediction hỗ trợ và bốn round. Dùng cùng guideline và tiêu chí chất lượng cuối cho cả ba.
- Đề xuất tối thiểu 15 cặp phát triển chưa quen thuộc, chia 5 cặp/phương án, phân bổ độ khó và số đối tượng tương đối tương đồng; nếu có nhiều người, luân phiên người/phương án. Các cặp này tách khỏi test; ưu tiên tách khỏi 60 cặp dự kiến làm validation và ghi danh sách/chính sách nếu không thể. Mẫu nhỏ chỉ giúp phát hiện vấn đề, chưa đủ để kết luận chắc chắn.
- Không cho cùng một người gán lặp lại cùng ảnh ở các phương án. Ghi thời gian thao tác theo round, chờ và xử lý AI, review/chốt nhãn; ghi số box tạo, chỉnh, xóa và lỗi được phát hiện. Nêu rõ thời gian tổng đã bao gồm hay chưa bao gồm review độc lập.
- Người review độc lập kiểm tra box thiếu/thừa, nhãn, box và thuộc tính so với ảnh và guideline; ghi bất đồng, adjudication và phạm vi đã review. So sánh trung bình/trung vị theo cặp, thời gian tổng và lỗi còn lại; không xem một phương án nhanh hơn là tốt hơn nếu chất lượng giảm.
- Chốt trước pilot ngưỡng chất lượng nhóm chấp nhận và mức giảm thời gian mong muốn. Nếu chưa thống nhất được ngưỡng, báo số liệu mô tả và để mở quyết định; không tự tuyên bố đạt.

**Đầu ra:** Danh sách mẫu và phân công, bảng ghi thời gian/thao tác, kết quả review và nhận xét A/B/C.

**Nghiệm thu:** Mỗi cặp có phương án, người, thời gian và trạng thái review; kết quả nêu quy mô mẫu, giới hạn và kết luận phù hợp. Nếu T3 chưa sẵn sàng, hoàn thành A/B và ghi C chưa kiểm chứng, không thay bằng số ước tính. **Phụ thuộc:** T2; phương án C thêm T3.

### T5 — Chuẩn bị 60 cặp validation bổ sung

**Mục tiêu:** Sẵn sàng mở rộng validation, ưu tiên độ bao phủ và giữ nguyên test.

- Từ dữ liệu mới ngoài validation/test/pilot, lập danh sách ứng viên và lý do chọn, ưu tiên lớp hiếm, đối tượng nhỏ, che khuất, cảnh khó và tình huống nhiều đối tượng khi có thể xác minh. Dự kiến 60 cặp mới; ghi số đủ điều kiện thực tế.
- Đối chiếu pair ID, đường dẫn, checksum và metadata nguồn có sẵn; rà soát ảnh gần trùng hoặc nguồn liên quan trong khả năng. Gắn cờ nguồn chưa xác minh; không khẳng định độc lập theo sự kiện khi chưa có bằng chứng.
- Giữ nguyên 40 validation và 60 test đã có; không chạy lại cơ chế chọn split một cách làm thay đổi tập cũ. Chuẩn bị manifest bổ sung có thể truy vết; nếu cần sửa script, lập task riêng sau khi xác định yêu cầu.
- Phân biệt mẫu lấy theo phân bố tự nhiên với mẫu tăng cường có chủ đích; ghi phương án double annotation và adjudication. Kết quả trên tập được tăng cường phải được mô tả đúng cách chọn.

**Đầu ra:** Manifest/danh sách ứng viên, phương pháp chọn, trạng thái kiểm tra trùng và ước lượng nhân lực gán nhãn dựa trên pilot.

**Nghiệm thu:** Mỗi cặp có ID, ảnh Before/After, tình trạng nguồn, lý do chọn và bằng chứng kiểm tra; không trùng cặp test đã biết. Nếu chưa đủ 60 cặp hoặc thiếu dữ liệu nguồn, báo rõ thay vì tự lấy từ test. **Phụ thuộc:** T1, quyền truy cập nguồn ảnh; ước lượng phụ thuộc T4.

### T6 — Tổng hợp và ra quyết định

**Mục tiêu:** Có nội dung cụ thể để báo cáo thầy và triển khai tuần sau.

- Tổng hợp tiến độ T1–T5, bằng chứng, khó khăn và số liệu pilot; phân biệt đã hoàn thành, đang làm và đề xuất.
- Quyết định bước tiếp theo cho workflow, quy mô annotation validation, việc cần sửa ở CVAT/pipeline, và các câu hỏi cần giảng viên xác nhận.
- Nếu T1–T5 hoàn tất sớm và đủ nhân lực, bắt đầu gán thử một batch nhỏ trong 60 cặp theo quy trình đã thống nhất; lưu version, kết quả review và số cặp thực tế. Không coi 100 cặp đã hoàn thành khi chưa gán và review đầy đủ.

**Đầu ra:** Báo cáo ngắn cuối tuần và đề xuất tuần tiếp theo. **Nghiệm thu:** Mỗi kết luận có nguồn hoặc số liệu, nêu rõ điều chưa chứng minh. **Phụ thuộc:** kết quả thực tế T1–T5.

## Lịch gợi ý và vai trò

| Thời gian | Công việc chính | Vai trò đề xuất |
|---|---|---|
| Thứ Ba 03/11 | Bắt đầu T1; rà soát guideline và chốt bản nháp T2; kiểm tra quyền truy cập CVAT/GT. | Người phụ trách dữ liệu; người phụ trách annotation. |
| Thứ Tư 04/11 | Calibration T2; thử T3 trên 2–3 cặp; thiết kế và chuẩn bị mẫu pilot T4. | Annotation/reviewer; người phụ trách pipeline/CVAT. |
| Thứ Năm 05/11 | Chạy T4; song song chọn ứng viên và kiểm tra trùng T5. | Người gán và reviewer độc lập; người phụ trách dữ liệu. |
| Thứ Sáu 06/11 | Review pilot; hoàn thiện T5, tổng hợp T6 và thống nhất việc tuần sau. | Người tổng hợp cùng các vai trò trên. |
| 07–08/11 | Dự phòng nếu nhóm có thời gian; chỉ bắt đầu batch validation sau khi quy trình đủ rõ. | Theo nguồn lực thực tế. |

Vai trò có thể do cùng một người đảm nhận khi nguồn lực hạn chế, nhưng phần review độc lập cần người khác với người gán trên mẫu được kiểm tra. Nếu không có người thứ hai, báo rõ chưa kiểm chứng độc lập.

## Không đặt làm kết quả bắt buộc trong tuần

- Hoàn thiện toàn bộ 100 cặp validation hoặc 5.500 cặp dataset.
- Benchmark nhiều detector, triển khai SAM2 đầy đủ hoặc chạy thí nghiệm refinement diện rộng; chỉ khảo sát khi không ảnh hưởng T1–T6.
- Dùng 60 cặp test để tune hoặc lựa chọn phương pháp. Test chỉ đánh giá sau khi khóa cấu hình và làm rõ protocol; không khẳng định độc lập theo sự kiện khi chưa xác minh.
