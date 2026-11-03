# Quy trình annotation bốn round và calibration

**Trạng thái:** Dự thảo T2, chờ nhóm calibration trên ảnh thật trước khi khóa. Áp dụng cùng [guideline](annotation-guideline.md), [ontology hiện hành](../../config/bounding_box/ontology.yaml) và protocol strict-v2. Quy trình này chưa được đo để chứng minh đạt 1–2 phút/cặp.

## Chuẩn bị

- Chuẩn bị cặp Before/After, `pair_id`, ảnh và prompt gốc; xem prompt như thông tin đối chiếu, không xem là bằng chứng vật thể tồn tại. Chọn task CVAT, ontology, mapping và format theo [hướng dẫn hiện có](README.md#chuẩn-bị-bundle-cho-cvat).
- Kiểm tra ảnh đúng cặp và đúng thứ tự, nhãn CVAT khớp năm lớp, frame After có thể đặt tag `pair_status`. Không thay đổi ontology hoặc GT đã khóa để thử quy trình.
- Nếu dùng AI, xem prediction là đề xuất chưa được xác nhận; kiểm tra có proposal cho cả Before và After hay không. Luồng import và khả năng giữ thuộc tính phải được T3 kiểm chứng, không mặc định đã hoạt động.
- Chuẩn bị log thời gian và danh sách review có `pair_id`, frame, vị trí/box, lý do, người ghi, trạng thái và quyết định. Nếu trường hợp không thể chọn lớp có căn cứ, lưu review ngoài annotation theo guideline.
- Chọn làm bốn round liên tiếp trên từng cặp hoặc theo batch nhỏ, tùy cách thực tế; ghi lại cách đã dùng để so sánh ở T4. Không bỏ lượt kiểm tra ảnh toàn cảnh chỉ vì đã có prediction.

## Round 1 — Tạo annotation ban đầu

**Đầu vào:** Cặp ảnh và nhãn ontology; có thể thêm proposal AI đã xác định đúng ảnh.  
**Mục tiêu:** Tạo danh sách ứng viên cho cả hai ảnh, ưu tiên tìm đối tượng.

1. Xem Before và After; vẽ box cho đối tượng nhìn thấy hoặc kiểm tra proposal AI và thêm box hiển nhiên còn thiếu.
2. Mỗi đối tượng phân biệt được có box riêng. Cho phép box thô rộng hơn đối tượng trong lúc làm, nhưng vẫn phải gắn với một ứng viên cụ thể, nằm trong ảnh và không dùng box nhóm làm object.
3. Đánh dấu trường hợp chưa chắc và lý do vào note/review queue. Vùng bao nhiều người có thể ghi như khu vực cần xem trong log, **không** lưu/xuất như object GT.

**AI có thể hỗ trợ:** phát hiện box và gợi ý lớp; người gán chịu trách nhiệm xét từng box.  
**Đầu ra/điều kiện chuyển round:** danh sách ứng viên ban đầu trên cả hai ảnh; mọi vùng không gán được lớp có căn cứ được ghi review. Box còn thô chưa được coi là GT hoàn chỉnh.  
**Lỗi chú ý:** sai frame, dự đoán trùng, box nhóm, tin nhãn từ prompt hoặc thiếu đối tượng Before.

## Round 2 — Tìm đối tượng bị bỏ sót

**Đầu vào:** Ảnh và ứng viên từ round 1.  
**Mục tiêu:** Giảm bỏ sót bằng cách rà soát ảnh, không chỉ nhìn những nơi đã có box.

1. Quét có hệ thống toàn ảnh Before và After, kể cả biên, nền xa, vùng bị che và nơi có nhiều đối tượng; tạm ẩn box khi cần để thấy phần bị che bởi overlay.
2. Bổ sung từng đối tượng phân biệt được, tách box nhóm nếu gặp; ghi đối tượng không đủ căn cứ vào review queue. Đối chiếu Before/After và đánh dấu tạm các trường hợp mới/cũ chưa chắc.
3. Không tạo đối tượng chỉ vì nó xuất hiện trong prompt hoặc vì ảnh After có vùng thay đổi.

**AI có thể hỗ trợ:** đề xuất vùng nghi bỏ sót; người gán vẫn quét toàn ảnh.  
**Đầu ra/điều kiện chuyển round:** hai frame đã được rà soát toàn cảnh, các đối tượng có thể phân biệt đã có box ứng viên, vùng chưa kết luận đã được ghi lại.  
**Lỗi chú ý:** người nhỏ ở nền xa, vật thể sát nhau, vật thể cắt biên, nhầm hiện vật trong Before là mới.

## Round 3 — Chỉnh box, lớp và thuộc tính

**Đầu vào:** danh sách sau round 2, Before/After và prompt.  
**Mục tiêu:** Đưa box và thuộc tính về đúng guideline.

1. Chỉnh box sát phần nhìn thấy của từng đối tượng; không bao bóng, khói, vật cầm tay không thuộc cấu trúc phương tiện. Box dừng ở biên ảnh; note `truncated` khi bị cắt.
2. Chốt một lớp có căn cứ; phân biệt `ambulance` với `emergency_vehicle`. Với ứng viên chưa đủ căn cứ, ghi ambiguity hoặc review theo guideline, không ép nhãn.
3. Trên After, đặt `is_newly_added`, `is_valid_rescue`, `occlusion_level`, `annotation_notes` theo bằng chứng. Dùng `ambiguous` có lý do nếu chưa thể xác nhận. Phân biệt vật thể bị che với bị cắt biên.
4. Trên After đặt đúng một tag `pair_status`, gồm `generation_status` và `pair_notes` khi cần. Xem prompt gốc để đánh giá yêu cầu; khi thiếu thông tin không tự kết luận vật thể được yêu cầu bị thiếu.

**AI có thể hỗ trợ:** đề xuất chỉnh biên hoặc gợi ý thuộc tính; chỉ chấp nhận sau khi người gán đối chiếu.  
**Đầu ra/điều kiện chuyển round:** mọi box có lớp và biên được rà soát, thuộc tính After và tag cặp được điền; mục chưa rõ có note/review.  
**Lỗi chú ý:** box quá rộng/cắt đối tượng; thuộc tính mặc định chưa được xem; nhầm `occlusion_level` và `truncated`; thiếu tag khi không có object.

## Round 4 — Review chất lượng và bàn giao

**Đầu vào:** annotation sau round 3, review queue và prompt.  
**Mục tiêu:** Phát hiện lỗi còn lại và ghi nhận rõ trạng thái sẵn sàng.

1. Xem lại ảnh toàn cảnh và từng box; kiểm tra box thiếu/thừa/trùng, lớp, biên, thuộc tính, new/existing, generation status và đúng frame.
2. Giải quyết hoặc chuyển reviewer các mục không chắc. So sánh ảnh, bối cảnh và prompt; ghi người review, quyết định, bằng chứng/lý do. `ambiguous` vẫn được giữ nếu đã xem mà bằng chứng chưa đủ; phân biệt với mục chưa được xem.
3. Trên mẫu double annotation, người thứ hai gán độc lập trước khi xem nhãn người đầu; so sánh, adjudicate bất đồng và giữ hai export gốc. Một người gán rồi tự kiểm tra không thay thế double annotation.
4. Export CVAT for images 1.1, lưu mapping, phiên bản và checksum; kiểm tra format và đường xử lý ambiguous theo protocol. Không ép ambiguous thành nhãn chắc chắn hoặc dùng converter v1 như thể đã xử lý strict-v2.

**AI có thể hỗ trợ:** đánh dấu box nghi trùng/sai; không quyết định GT cuối.  
**Đầu ra/điều kiện hoàn thành:** checklist cuối, danh sách review đã adjudicate hoặc chuyển xử lý có lý do, export truy vết được; các trường hợp còn ambiguity được biểu diễn và báo cáo rõ. Nếu còn mục chưa review hoặc đường export không giữ được trạng thái, cặp chưa sẵn sàng khóa GT.  
**Lỗi chú ý:** xóa proposal sai nhưng không rà FN; gán nhãn hợp lý sai lớp; coi nhãn chưa review là ambiguous đã adjudicate; mất tag hoặc note khi export.

## Checklist bàn giao cho mỗi cặp

- [ ] Ảnh Before/After và `pair_id` khớp mapping; đã quét cả hai ảnh.
- [ ] Mỗi đối tượng phân biệt được có box riêng và đúng lớp; không còn box nhóm/trùng.
- [ ] Box bám phần nhìn thấy, nằm trong ảnh; che khuất và `truncated` được ghi đúng.
- [ ] Mỗi object After có `is_newly_added`, `is_valid_rescue`, `occlusion_level`; note có lý do khi mơ hồ.
- [ ] After có đúng một `pair_status`; `generation_status` được đối chiếu prompt hoặc ghi thiếu bằng chứng. Before không gắn tag này.
- [ ] Review queue được xử lý hoặc bàn giao có người phụ trách; mục chưa gán được lớp không bị mất.
- [ ] Export, mapping, người gán/review, version và checksum được lưu; đã kiểm tra trạng thái ambiguity còn nguyên.

Trong CVAT tên thuộc tính là `occlusion_level`; schema annotation cũ dùng tên `occlusion` sau chuyển đổi. Adapter strict-v2 hiện có không mang trường occlusion trong artifact adapted, nên không dùng artifact đó để tuyên bố đã giữ hoặc thống kê được occlusion.

## Calibration trước annotation chính

**Đề xuất theo protocol hiện hành:** hai người cùng gán độc lập 10 cặp phát triển, sau đó cùng reviewer đối chiếu. Chọn cặp chưa quen thuộc, tách 60 test và tránh trùng pilot/validation bổ sung; ghi rõ nếu không bảo đảm được. Có thể ưu tiên các ca có khả năng gồm firefighter khó nhận diện, xe dễ nhầm, che khuất/cắt biên, nhiều người và Before/After khó ghép. Không khẳng định có đủ tất cả loại ca nếu dữ liệu chưa được xem.

1. Chọn và ghi 10 `pair_id`, đường dẫn ảnh/prompt, lý do chọn, nguồn, split và version guideline. Kiểm tra không thuộc test; nếu chưa đủ 10 cặp hợp lệ thì ghi số có được, chưa tuyên bố calibration hoàn tất.
2. Tạo hai task độc lập A/B với cùng ảnh và prompt; người B không xem box hoặc kết luận của A trước khi nộp. Ghi người, thời gian, phiên bản guideline, hình thức làm theo cặp/batch và checksum export; không yêu cầu giống nhau về tốc độ.
3. So sánh object thiếu/thừa, lớp, biên, `is_newly_added`, `is_valid_rescue`, occlusion và generation status. Ghép box theo hình học chỉ để tìm bất đồng; không coi IoU đơn lẻ là bằng chứng đúng. Với 8 cặp đã có review cũ, chỉ dùng như ví dụ thảo luận có nguồn gốc, không tự coi là bộ gán độc lập mới.
4. Reviewer đối chiếu ảnh, Before/After và prompt, ghi từng quyết định, lý do và bất đồng chưa giải quyết. Nếu không đủ căn cứ, giữ ambiguous và nêu giới hạn; không ép quyết định chắc chắn. Cập nhật guideline nếu thấy quy tắc còn gây hiểu khác nhau.
5. Nhóm xem lại guideline sau thay đổi; nếu thay đổi ảnh hưởng quy tắc trọng yếu, gán thử lại ca liên quan trước khi khóa. Lưu version cuối và quyết định khóa; bộ calibration không tự trở thành GT validation.

**Điều kiện đạt calibration:** có hai export độc lập cho đủ 10 cặp hợp lệ; các bất đồng và tình huống không đủ bằng chứng đã được ghi quyết định hoặc trạng thái còn mở; guideline cuối và quy tắc ambiguity đã được nhóm thống nhất. Nếu chưa có người review hoặc dữ liệu, báo chưa hoàn tất; không tự đánh dấu đạt.

## Mẫu log calibration

Sao chép bảng dưới đây vào tài liệu làm việc hoặc spreadsheet và bổ sung một dòng cho mỗi bất đồng; không đưa thông tin phỏng đoán thay cho kết quả quan sát.

| Pair ID / frame | Vị trí / đối tượng | Chủ đề | Nhãn A | Nhãn B | Bằng chứng | Quyết định / lý do | Người review / ngày | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| Chưa thực hiện | — | — | — | — | — | — | — | Chờ chọn mẫu |

Thông tin chung cần lưu: danh sách 10 cặp và split/nguồn, người A/B/reviewer, version guideline và ontology, thời gian, đường dẫn hai export và checksum, quyết định cập nhật/khóa guideline, số mục đã xử lý và còn mở. Không ghi log này là kết quả đã thực hiện trước khi nhóm hoàn thành.
