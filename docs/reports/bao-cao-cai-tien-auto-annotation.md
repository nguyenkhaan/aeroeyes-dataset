# Báo cáo tiến độ cải thiện tự động gán nhãn bounding box

Báo cáo ghi lại quá trình thực hiện từ nhận xét của thầy về chất lượng lọc đối tượng, yêu cầu tăng recall và refine bounding box đến cấu hình Grounding DINO kết hợp CLIP đã chạy trên validation. Mỗi giai đoạn trình bày vấn đề cần giải quyết, lý do lựa chọn cách làm, thay đổi đã triển khai, kết quả và quyết định tiếp theo. Các kết quả được phân biệt theo phiên bản ground truth để tránh nhầm thay đổi cách đánh giá với cải thiện thuật toán.

## 1. Quá trình triển khai phương pháp và lý do điều chỉnh

### 1.1. Xác định lại mục tiêu sau nhận xét của thầy

Thầy đề nghị nhóm khảo sát công cụ hỗ trợ annotation tự động vì kết quả lọc còn chưa ổn định, đồng thời tập trung tăng recall và refine bounding box. Từ nhận xét này, nhóm cần trả lời ba câu hỏi riêng:

- Detector có tìm thấy các đối tượng cứu hộ thực sự xuất hiện trong ảnh After hay không?
- Pipeline có xác định đúng đối tượng mới được thêm vào so với ảnh Before hay không?
- Bounding box đầu ra có bao phủ đúng đối tượng hay bị rộng, lệch hoặc cắt mất một phần đối tượng?

Điểm xuất phát trước đó là kết quả 401/500 cặp đạt `bbox_passed`, tương ứng 80,2%. Tuy nhiên, điều kiện này chỉ yêu cầu vùng thay đổi hợp lệ và có ít nhất một đối tượng cứu hộ mới được hệ thống xác nhận. Nếu ảnh có năm đối tượng thật nhưng hệ thống chỉ tìm được một đối tượng, cặp ảnh vẫn có thể pass. Vì vậy, 80,2% là tỷ lệ vượt điều kiện vận hành; chưa đo được object recall hoặc chất lượng định vị box.

Nhóm chuyển từ việc chỉ đếm số cặp pass sang đo TP, FP, FN và IoU dựa trên nhãn thủ công. Quyết định này là lý do phải xây dựng benchmark trước khi tiếp tục chỉnh ngưỡng hoặc thêm mô hình.

Phạm vi thực hiện của báo cáo là tự động gán bounding box cho đối tượng cứu hộ trong cặp Before/After. Quá trình sinh ảnh, các điểm chất lượng ảnh và phần lọc ảnh thảm họa trước đây chỉ được dùng làm đầu vào hoặc bối cảnh, không được trình bày như đóng góp mới trong giai đoạn này.

### 1.2. Phân định vai trò công cụ và chọn hướng triển khai trước

Nhóm tiếp tục sử dụng Grounding DINO vì pipeline đã có khả năng phát hiện theo prompt, xuất box và xử lý cặp Before/After. Việc giữ detector hiện có cho phép khảo sát nguyên nhân bỏ sót và nhầm lớp trước khi thay toàn bộ hệ thống.

| Công cụ/phương án | Vai trò trong công việc | Trạng thái đến hiện tại |
|---|---|---|
| CVAT | Gán box, thuộc tính đối tượng và trạng thái cặp để xây dựng ground truth | Đã dùng cho 40 cặp validation |
| Grounding DINO | Đề xuất box theo prompt; thử ngưỡng, prompt và ánh xạ nhãn | Đã triển khai và chạy thực nghiệm |
| CLIP | So sánh nội dung vùng xe với mô tả ambulance/fire truck khi hai lớp dự đoán xung đột | Đã thử trên kết quả lưu, replay và tích hợp inference mới |
| SAM2 hoặc phương pháp refine khác | Khảo sát khả năng cải thiện biên box sau detection | Chưa có thực nghiệm so sánh trong giai đoạn này |
| Các detector/framework auto-annotation khác | Cung cấp phương án so sánh ngoài Grounding DINO | Chưa hoàn thành benchmark đa công cụ |

CVAT là công cụ gán nhãn; CLIP trong triển khai này là bước phân biệt một số dự đoán xe; hai công cụ không được xem như detector thay thế độc lập đã được benchmark. Việc khảo sát nhiều công cụ theo yêu cầu của thầy mới hoàn thành một phần, trong khi phần thực nghiệm đã đi sâu vào cải thiện pipeline đang có.

### 1.3. Xây dựng benchmark theo cặp và tách validation/test

**Vấn đề.** Chỉ gán box trên ảnh After không đủ để xác định đối tượng mới được sinh thêm. Đồng thời, nếu dùng cùng dữ liệu để chỉnh ngưỡng và báo cáo kết quả cuối, kết quả có thể phụ thuộc quá nhiều vào chính các ví dụ đã xem.

**Cách triển khai.** Nhóm tạo benchmark 100 cặp từ tập dữ liệu đang có, gồm 40 cặp validation để phân tích và thử phương án, cùng 60 cặp test dành cho giai đoạn đánh giá sau khi chốt cấu hình. Mỗi cặp luôn giữ hai ảnh Before/After trong cùng một split. Quá trình chọn mẫu được lưu trong manifest với seed cố định `20261020`.

Manifest ghi `pair_id`, split, khóa ảnh nguồn, nhãn bối cảnh, URL, đường dẫn hai ảnh, checksum và cách xác định nhóm nguồn. Việc lưu các trường này giúp nhận diện đúng dữ liệu đã dùng khi chạy lại hoặc khi thay đổi phiên bản annotation.

Nhóm xác nhận manifest có đủ 100 `pair_id` duy nhất. Tuy nhiên, metadata không cung cấp các trường nhóm sự kiện cho quá trình tạo manifest; cả 100 cặp dùng `pair_id_fallback`. Vì vậy, phép tách hiện tại được xác định ở cấp cặp; chưa chứng minh được độc lập theo sự kiện. Đây là giới hạn đã được ghi lại, không được bỏ qua chỉ vì số lượng validation/test đúng thiết kế.

**Kết quả và quyết định.** Nhóm dùng 40 cặp validation cho toàn bộ các vòng cải tiến được báo cáo dưới đây. Chưa gán nhãn hoặc đánh giá hiệu năng trên 60 cặp test.

### 1.4. Thiết kế nhãn thủ công đúng với bài toán đối tượng cứu hộ mới

**Vấn đề.** Một đối tượng không được phát hiện có thể do detector bỏ sót, do đối tượng vốn không được sinh ra, hoặc do bản thân hình sinh quá mơ hồ để xác định. Nếu chỉ có tên lớp và box, các tình huống này dễ bị trộn lẫn.

**Thay đổi đã thực hiện.** Nhóm xây dựng ontology năm lớp và thống nhất cách phân biệt:

| Lớp | Quy tắc chính |
|---|---|
| `firefighter` | Cần dấu hiệu phù hợp với lính cứu hỏa; không tự chuyển mọi người mặc áo phản quang thành firefighter |
| `rescue_boat` | Thuyền hoặc bè có bằng chứng phục vụ cứu hộ |
| `helicopter` | Trực thăng thuộc phạm vi cứu hộ hoặc được bổ sung theo yêu cầu; không gồm máy bay cánh cố định |
| `ambulance` | Xe cứu thương có dấu hiệu nhận diện phù hợp |
| `emergency_vehicle` | Xe chữa cháy, rescue truck và phương tiện khẩn cấp đường bộ không thuộc ambulance |

Các lớp được thiết kế loại trừ nhau ở cấp nhãn của một đối tượng. Đặc biệt, không dùng `emergency_vehicle` như lớp bao trùm để đồng thời gán cho cùng một ambulance trong ground truth.

Ngoài box và lớp, annotation trên ảnh After có `is_newly_added`, `is_valid_rescue`, `occlusion_level` và ghi chú khi cần. Mỗi ảnh After có tag `pair_status` chứa `generation_status`, kể cả khi không có đối tượng. Nhãn Before dùng để nhận diện các đối tượng đã tồn tại.

Phân biệt này dẫn đến hai cách xử lý quan trọng:

- Nếu prompt yêu cầu ambulance nhưng ảnh không sinh được ambulance nhận diện được, cần ghi nhận tình trạng sinh ảnh; không tạo một box ground truth tưởng tượng để tính detector bỏ sót.
- Nếu một phần cặp ảnh mơ hồ nhưng một số đối tượng vẫn nhận diện rõ, các đối tượng rõ vẫn có thể được dùng làm ground truth theo protocol phù hợp.

### 1.5. Đóng gói CVAT và hoàn thiện annotation validation

Nhóm chuẩn bị archive ảnh và mapping để tên frame chứa số thứ tự cặp, vai trò Before/After và `pair_id`. Cách đặt tên giữ hai ảnh của một cặp cạnh nhau theo thứ tự lexicographical, đồng thời giúp chuyển export CVAT về đúng cặp trong manifest.

Các thành phần triển khai gồm `prepare_bbox_annotation.py` để tạo bundle, `annotation-mapping.json` để liên kết frame và `convert_cvat_bbox_annotations.py` để đọc export **CVAT for images 1.1**. Định dạng này được dùng để giữ box, tag và custom attributes.

Trong lần chuyển pilot ban đầu, artifact mới thể hiện 1/40 cặp hoàn tất và còn mục chưa giải quyết. Đây là trạng thái annotation đang làm, chưa phải ground truth hoàn chỉnh. Sau đó nhóm hoàn thiện các job CVAT cho đủ 40 cặp và bổ sung `pair_status` còn thiếu.

Nhóm từng thực hiện annotation độc lập lần hai trên 8 cặp validation, tương ứng 20% tập validation, và có bước chuẩn bị đối chiếu bất đồng. Theo tiến độ thực tế, việc adjudicate toàn bộ được tạm hoãn để chạy baseline thăm dò. Do đó, không mô tả dữ liệu đã được hai người thống nhất toàn bộ.

**Điều chỉnh đối với ambiguous.** Trong quá trình gán nhãn, nhóm gặp các đối tượng khó phân biệt về lớp, tính hợp lệ hoặc tính mới. Hướng xử lý được điều chỉnh từ việc cố hoàn tất nhãn chắc chắn sang lưu trạng thái `ambiguous` có lý do. Cần phân biệt trường hợp thiếu tag/chưa làm với trường hợp người gán đã xem nhưng thực sự không đủ bằng chứng. Hai trường hợp không có cùng ý nghĩa.

### 1.6. Bổ sung prompt gốc để rà soát generation status

**Lý do.** Quan sát ảnh Before/After có thể xác định một số thay đổi, nhưng chưa đủ để kết luận mô hình có đáp ứng yêu cầu sinh ảnh hay không. Ví dụ, việc không nhìn thấy xe cứu thương chỉ là `missing_requested_object` nếu yêu cầu gốc có đề nghị bổ sung xe cứu thương.

Nhóm lấy lại `editing_instruction` và `flux_prompt` từ metadata theo `pair_id`, kết hợp với ảnh Before, ảnh After và nhãn CVAT để rà soát. Việc này hỗ trợ phân biệt lỗi sinh ảnh với lỗi phát hiện và làm rõ lý do của các trạng thái mơ hồ.

Sau vòng rà soát, nhóm đóng gói **ground truth validation v2** với nhãn chắc chắn và các vùng chưa chắc chắn tách riêng. Bản này được một người rà soát; có hỗ trợ trong quá trình kiểm tra và xử lý nhãn, nhưng không được xem là annotation độc lập kép hoặc bộ nhãn đã được chứng minh không còn bỏ sót.

### 1.7. Điều chỉnh cách đánh giá khi có nhãn mơ hồ

**Vấn đề.** Coi toàn bộ vùng mơ hồ là âm tính có thể làm một prediction hợp lý bị tính FP. Ngược lại, tự coi đối tượng mơ hồ là ground truth dương tính cũng có thể làm sai số TP/FN. Ngoài ra, loại toàn bộ một cặp chỉ vì trạng thái generation mơ hồ sẽ làm mất các đối tượng rõ ràng trong cặp đó.

Nhóm đã đi qua các bước đánh giá có phạm vi khác nhau:

| Giai đoạn đánh giá | Phạm vi và cách xử lý | Cách sử dụng kết quả |
|---|---|---|
| Các kết quả ban đầu | Dùng annotation ban đầu; cách xử lý ambiguous còn được điều chỉnh | Chẩn đoán, không làm mốc so xuyên phiên bản |
| Evaluator ambiguity-aware trước GT v2 | Trong 40 cặp đã xử lý, chỉ 17 cặp đủ điều kiện vào các chỉ số detector/pipeline; 21 cặp ambiguous và 2 cặp generation failure bị loại khỏi phạm vi này | So sánh các thí nghiệm sớm có cùng protocol |
| Strict validation v2 | Đánh giá cả 40 cặp dựa trên đối tượng xác nhận; dùng ignore regions riêng | Mốc đánh giá cho các thí nghiệm phương tiện và CLIP sau này |

Ở strict-v2, prediction được ghép một-một với ground truth cùng lớp tại IoU 0,5. Các đối tượng dương tính chắc chắn được ưu tiên. Prediction trùng lặp một đối tượng dương tính không được tự chuyển thành ignore để tránh FP. Sau đó, một prediction còn lại có thể được ignore nếu khớp vùng mơ hồ cùng lớp theo quy tắc một-một. Cách này tránh việc một vùng ignore che toàn bộ các prediction trùng lặp.

Hai mức kết quả được báo cáo riêng:

1. **Detector:** box dự đoán trên ảnh After so với mọi đối tượng cứu hộ xác nhận trong After, kể cả đối tượng đã có từ Before.
2. **Toàn pipeline:** box cuối sau đối sánh và xử lý delta so với các đối tượng xác nhận là mới.

Các chỉ số chính là:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
```

Macro F1 được tính trên các lớp có ground truth để nhìn thêm ảnh hưởng theo lớp. Điều này cần thiết vì F1 tổng thể có thể tăng trong khi một lớp ít mẫu mất toàn bộ prediction đúng.

Các kết quả AP50/mAP của evaluator trước được lưu trong các run tương ứng. Evaluator strict-v2 hiện dùng IoU 0,5 và báo cáo precision, recall, F1, số lượng theo lớp cùng mean matched IoU; không gán các số AP/mAP cũ cho bảng strict-v2.

**Quyết định.** Sau khi đổi ground truth hoặc protocol, nhóm đánh giá lại các report đã lưu. Không diễn giải việc metric thay đổi do sửa cách đánh giá như bằng chứng detector tự cải thiện.

### 1.8. Tạo runner thí nghiệm riêng và thử ngưỡng Grounding DINO

Để mỗi thay đổi có thể so sánh được, nhóm triển khai `run_bbox_experiment.py`, cấu hình JSON riêng cho từng phương án và job Slurm. Mỗi run có thư mục riêng, danh sách cặp đã chọn, config, phiên bản model, Git commit, môi trường và thông tin tài nguyên.

Runner chỉ chọn validation cho các vòng tuning, xác minh checksum ảnh và từ chối Before/After khác kích thước trong thí nghiệm. Khi so sánh một yếu tố, các yếu tố còn lại cần giữ cố định. Các job chạy 5 cặp đầu tiên dùng để quan sát khả năng chạy, chưa đủ để chọn cấu hình.

Cụ thể, hai job 8027/8028 chỉ có 2/5 cặp được evaluator khi đó đưa vào đánh giá; ba cặp còn lại bị loại do trạng thái ambiguous. Trên phần được chấm chỉ có ba GT đủ điều kiện và cả hai cấu hình đều có TP bằng 0. Nhóm không dùng kết quả nhỏ này để kết luận phương án không hiệu quả, mà chuyển sang chạy đủ 40 cặp ở hai job 8082/8083. Bước này giúp tránh chọn cấu hình dựa trên một subset rất ít đối tượng thật.

Nhóm cũng ghi nhận sự khác nhau giữa các phiên bản API hậu xử lý của Transformers. Runner thí nghiệm yêu cầu hậu xử lý thực sự hỗ trợ cả box threshold và text threshold; nếu chỉ truyền được một ngưỡng thì phép so sánh hai ngưỡng không còn đúng ý nghĩa.

**Thử nghiệm thứ nhất: giảm riêng box threshold.** Hai job 8082 và 8083 xử lý 40 cặp với box threshold lần lượt 0,25 và 0,18, trong khi text threshold giữ 0,25. Các chỉ số trên 17 cặp đủ điều kiện không đổi. Kết quả cho thấy trong cấu hình và dữ liệu này, hạ riêng box threshold chưa mang lại thêm đối tượng khớp ground truth. Chưa thể từ đó kết luận box threshold không có tác dụng trong mọi trường hợp.

**Thử nghiệm thứ hai: giảm text threshold.** Job 8090 giữ box threshold 0,18 và giảm text threshold từ 0,25 xuống 0,18 so với job 8083. Detector tăng từ 28 lên 37 TP; pipeline tăng từ 27 lên 35 TP. Phần lớn lợi ích detector đến từ firefighter: TP tăng từ 17 lên 25. Emergency vehicle tăng từ 5 lên 6 TP.

Kết quả này gợi ý việc tạo/giữ nhãn sau hậu xử lý là một nút thắt đáng quan tâm. Nhóm tiếp tục dùng mức 0,18 cho các thử nghiệm prompt tiếp theo, đồng thời theo dõi FP vì hạ ngưỡng có thể giữ thêm cả dự đoán nhầm.

### 1.9. Thử prompt dài và xử lý nhãn ghép

**Giả thuyết ban đầu.** Thêm từ đồng nghĩa như `fire fighter`, `fire truck`, `fire engine` và `rescue truck` vào cùng prompt có thể giúp detector tìm được nhiều cách biểu diễn đối tượng hơn.

Nhóm thử prompt gốc:

```text
firefighter . rescue boat . helicopter . ambulance . emergency vehicle .
```

và prompt mở rộng:

```text
firefighter . fire fighter . rescue boat . helicopter .
ambulance . emergency vehicle . fire truck . fire engine . rescue truck .
```

**Kết quả quan sát.** Prompt dài tạo nhiều nhãn ghép như `firefighter_fire_fighter` hoặc các chuỗi chứa từ của nhiều lớp. Một phần nhãn không khớp ontology nên bị bỏ trước khi vào đánh giá. Vì vậy, thêm nhiều từ vào prompt không đồng nghĩa với tăng số box hợp lệ đầu ra.

So sánh 8094/8095 cho thấy detector recall giảm từ 50,68% xuống 26,03%; pipeline recall giảm từ 48,61% xuống 23,61% theo protocol cũ. Nhóm sau đó chạy thêm đối chứng 8105 để so với 8103 ở cùng commit. Các số liệu ở lần so sánh tương thích này tái hiện xu hướng giảm.

**Điều chỉnh ánh xạ nhãn.** Nhóm thử thêm đúng một ánh xạ `firefighter_fire_fighter → firefighter`, so sánh job 8103 và 8104. Đây là nhãn ghép từ hai cách viết của cùng một lớp. Những nhãn ghép nhiều lớp hoặc khái niệm rộng như `rescue worker` không được tự ép về firefighter vì có thể làm sai nghĩa.

Ánh xạ này khôi phục nhiều detection bị bỏ: theo protocol cũ, detector recall tăng từ 26,03% lên 53,42%; pipeline recall tăng từ 23,61% lên 47,22%. Tuy nhiên, khi đánh giá lại trên GT strict-v2, prompt mở rộng kèm ánh xạ vẫn có pipeline recall 47,29%, F1 40,53%, thấp hơn prompt gốc có recall 49,61%, F1 46,38%. Cả hai phương án prompt mở rộng khi đó không có ambulance TP trên GT v2.

**Quyết định.** Nhóm giữ lại việc chuẩn hóa và alias có nghĩa rõ ràng, nhưng không chọn prompt dài cộng ánh xạ nhãn ghép làm cấu hình ứng viên chính. Hướng tiếp theo là tách prompt bổ sung theo từng lớp thay vì tiếp tục kéo dài một prompt chung.

### 1.10. Rà soát lại ground truth trước khi tiếp tục chọn cấu hình

Sau các thử nghiệm prompt, nhóm nhận thấy số lượng cặp ambiguous và đối tượng chưa chắc chắn ảnh hưởng lớn đến cách đọc metric. Nhóm quyết định hoàn thiện nhãn trước khi tiếp tục kết luận phương án nào tốt hơn.

Trong vòng này, prompt gốc được bổ sung để xem lại yêu cầu sinh ảnh; những vùng có nhãn hoặc thuộc tính chưa đủ chắc chắn được giữ riêng dưới dạng ignore regions. Ground truth v2 gồm:

| Thành phần | Số lượng |
|---|---:|
| Cặp validation | 40 |
| Đối tượng xác nhận trong After | 135 |
| Trong đó: đối tượng mới xác nhận | 129 |
| Trong đó: đối tượng đã tồn tại | 6 |
| Vùng mơ hồ lưu dưới dạng ignore | 119 |

Trạng thái generation của 40 cặp gồm 27 valid, 10 ambiguous, một missing requested object và hai severe scene corruption. Cặp ambiguous không bị loại toàn bộ trong strict-v2: chỉ đối tượng đủ điều kiện được đưa vào tập dương tính, còn các vùng mơ hồ được xử lý theo quy tắc ignore.

Việc số cặp valid/ambiguous thay đổi so với annotation cũ là kết quả của rà soát nhãn trên dữ liệu đã có; không phải kết quả sinh lại ảnh. Vì vậy, không dùng sự thay đổi số cặp valid giữa hai phiên bản GT để khẳng định mô hình sinh ảnh được cải thiện.

Nhóm triển khai adapter và evaluator riêng để đọc đúng bản GT này, rồi đánh giá lại các report 8105, 8103 và 8104. Việc đánh giá lại cho phép so các phương án trên cùng GT mà không cần sinh lại ảnh. Đây là mốc chuyển protocol; các phần thực nghiệm bên dưới đều dùng strict-v2 trên cả 40 cặp.

### 1.11. Tách prompt bổ sung cho ambulance và emergency vehicle

**Vấn đề còn lại.** Với prompt chung, lớp ambulance và emergency vehicle còn bỏ sót nhiều. Phương án prompt dài lại tạo nhãn ghép khó xử lý.

**Thay đổi triển khai.** Nhóm giữ prompt gốc, sau đó chạy thêm ba lượt độc lập cho từng ảnh:

- `ambulance .` với lớp mục tiêu ambulance;
- `fire truck .` với lớp mục tiêu emergency vehicle;
- `fire engine .` với lớp mục tiêu emergency vehicle.

Mỗi lượt bổ sung chỉ giữ detection thuộc lớp mục tiêu. Nhãn prompt phải khớp alias được ontology cho phép. Các box gốc và box bổ sung được gộp, rồi NMS cùng lớp trước khi vào đối sánh Before/After. Phương pháp áp dụng cho cả hai ảnh để tránh thay đổi cách phát hiện chỉ ở một phía của cặp.

**Kết quả và đánh đổi.** Trên strict-v2, job 8238 tăng pipeline TP từ 64 lên 99 so với đối chứng 8237; FN giảm từ 65 xuống 30. Recall tăng 49,61% lên 76,74%. Tuy nhiên, FP tăng từ 83 lên 188, làm precision giảm 43,54% xuống 34,49%.

Lợi ích theo lớp rất rõ ở detector: ambulance TP tăng từ 1 lên 7 trên 8 GT; emergency vehicle TP tăng từ 12 lên 44 trên 45 GT. Nhưng ambulance cũng tạo nhiều FP, đặc biệt trên các phương tiện khẩn cấp khác.

**Quyết định.** Prompt bổ sung là hướng có hiệu quả tăng recall trong validation này, nhưng chưa đủ để tạo nhãn đáng tin cậy mà không cần thêm xử lý nhầm lớp. Nhóm giữ phương án này làm nền cho vòng điều chỉnh tiếp theo.

### 1.12. Điều chỉnh confidence riêng của prompt ambulance

Nhóm thử ba mức `min_confidence` là 0,30, 0,45 và 0,60 cho riêng prompt bổ sung ambulance, đối chiếu với cấu hình chưa thêm ngưỡng riêng. Các lượt `fire truck` và `fire engine` được giữ như cũ.

Ngưỡng được áp dụng sau hậu xử lý của lượt prompt bổ sung và trước NMS tổng hợp, trên cả Before và After. Nó không trực tiếp đặt ngưỡng mới cho detection từ prompt gốc; dù vậy, box gốc vẫn có thể bị ảnh hưởng gián tiếp qua NMS và đối sánh.

Lý do không nâng ngưỡng chung cho tất cả lớp là lỗi nổi bật nằm ở ambulance. Một thay đổi toàn cục sẽ tác động thêm firefighter, boat hoặc helicopter mà chưa có bằng chứng cần thiết.

Các mức 0,45 và 0,60 loại được nhiều FP ambulance, nhưng chỉ giữ được 1/8 ambulance GT. Mức 0,30 còn giữ 5/8 và được dùng làm điểm làm việc để phân tích tiếp. Đây là lựa chọn thăm dò nhằm giữ nhiều ambulance hơn các mức ngưỡng cao, không phải kết quả tối ưu hóa đã chứng minh. So với cấu hình chưa thêm ngưỡng riêng, mức 0,30 vẫn mất hai ambulance TP.

Nhóm cũng không xem confidence giữa các prompt là các xác suất đã hiệu chỉnh tương đương. Vì vậy, confidence riêng giúp giảm số lượng box, nhưng chưa giải quyết được bản chất nhầm lớp.

### 1.13. Phân tích vị trí FP và thử ưu tiên lớp theo IoU

**Bằng chứng dẫn đến giả thuyết mới.** Sau khi chọn mức 0,30, nhóm xuất phân tích FP/FN theo cặp, lớp, tọa độ và quan hệ với GT. Trong 34 FP ambulance ở job 8240, có 27 box chồng lên GT emergency vehicle theo ngưỡng IoU của phân tích; bảy box còn lại thuộc nhóm unmatched.

Nhóm dùng kết quả này như dấu hiệu hình học để xem lại ảnh. Một FP chồng lên GT khác lớp chưa tự chứng minh hoàn toàn nguyên nhân sai; đồng thời, việc chồng lên GT emergency vehicle không bảo đảm có một prediction emergency vehicle tương ứng.

**Phương án thử.** Với job 8253, sau NMS cùng lớp, loại ambulance nếu trong cùng ảnh có prediction emergency vehicle có IoU từ 0,75. Prediction emergency vehicle được giữ bất kể confidence tương đối. Job 8252 là đối chứng cùng cấu hình nhưng chưa bật quy tắc này.

**Kết quả.** Pipeline FP giảm từ 162 xuống 129 và F1 tăng từ 50,00% lên 52,57%. Tuy nhiên, TP giảm 97 xuống 92; riêng ambulance giảm từ 5 TP xuống 0 TP. Macro F1 giảm từ 49,11% xuống 44,85%.

**Quyết định.** Không chọn quy tắc ưu tiên cứng dù F1 tổng thể tăng. Kết quả cho thấy cần xem đồng thời recall theo lớp và macro F1; chỉ tối ưu một chỉ số tổng hợp có thể che việc một lớp mất toàn bộ detection đúng.

### 1.14. Xem hình đối chiếu để hiểu nhầm lẫn giữa hai lớp xe

Nhóm tạo bộ hình gồm bốn ô: Before có GT, After có GT, dự đoán After của đối chứng và dự đoán After của phương án thử. Bộ hình chọn tất cả ambulance GT mới được ghép đúng ở đối chứng nhưng mất ghép ở ứng viên, rồi thêm tối đa năm cặp có nhiều FP ambulance chồng lên GT emergency vehicle.

Bộ review có chín hình. Năm ambulance GT mất ghép nằm trong bốn cặp 0005, 0021, 0028 và 0039; cặp 0005 có hai đối tượng. Các hình còn lại tập trung vào các ca xe chữa cháy bị dự đoán thêm nhãn ambulance.

Quan sát cho thấy cùng một vùng xe có thể nhận cả hai nhãn. Trường hợp GT xác nhận ambulance vẫn có thể được detector chấm emergency vehicle cao hơn. Do đó, độ chồng lấn hoặc ưu tiên confidence đơn thuần chưa đủ để quyết định lớp.

Bộ hình được chọn theo lỗi cần phân tích, không phải mẫu ngẫu nhiên đại diện cho toàn bộ validation. Việc mất ghép trong kết quả cũng cần được đọc cùng dấu vết xử lý Before/After, không chỉ suy nguyên nhân từ một hình After.

**Quyết định.** Chuyển sang thử tín hiệu thị giác của vùng xe bằng CLIP thay vì tiếp tục loại ambulance chỉ dựa trên hình học.

### 1.15. Chấm điểm CLIP trước khi thay đổi prediction

Nhóm triển khai một bước chẩn đoán riêng, đọc các report cố định của job 8252. Chỉ ambulance đang xung đột với prediction emergency vehicle tại IoU ít nhất 0,75 mới được đưa vào chấm điểm.

Với mỗi box, vùng ảnh được mở thêm 10% chiều rộng ở mỗi phía trái/phải và 10% chiều cao ở mỗi phía trên/dưới, rồi cắt trong biên ảnh. CLIP so sánh vùng này với hai mô tả:

```text
a photo of an ambulance
a photo of a fire truck
```

Đặt chênh lệch điểm:

```text
margin = cosine(crop, ambulance) - cosine(crop, fire truck)
```

Dấu của margin thể hiện CLIP nghiêng về mô tả nào trong cặp mô tả đã chọn. Nó không có nghĩa là xác suất đối tượng chắc chắn thuộc lớp đó; hai mô tả cũng không bao quát mọi loại phương tiện khẩn cấp.

Job 8276 chấm 45 box xung đột trong 24 cặp: bảy box trên Before và 38 box trên After. Trong bước đối chiếu hình học trên After với GT tại IoU từ 0,5:

| Nhóm đối chiếu | Số box | Margin > 0 | Margin ≤ 0 |
|---|---:|---:|---:|
| Trùng GT ambulance | 5 | 5 | 0 |
| Trùng GT emergency vehicle | 27 | 13 | 14 |
| Không trùng GT hai lớp trên | 6 | 2 | 4 |

Đây là phân tích quan hệ hình học, chưa phải TP/FP chính thức của toàn pipeline. Tuy nhiên, kết quả gợi ý dấu margin có thể giúp loại một phần prediction ambulance nhầm mà vẫn giữ các box ambulance đúng đã quan sát.

**Quyết định.** Thử quy tắc zero-margin: giữ ambulance xung đột nếu margin > 0; loại nếu margin ≤ 0. Ngưỡng này được chọn sau khi xem validation nên phải được mô tả là thăm dò.

### 1.16. Replay toàn bộ bước đối sánh bằng điểm CLIP đã lưu

Nhóm chưa đưa ngay giả thuyết từ bảng điểm thành kết luận về pipeline. Bước tiếp theo dùng lại detection của job 8252 và điểm CLIP 8276, áp dụng quy tắc cho cả Before lẫn After, rồi tính lại delta, đối sánh và kết quả cuối.

Lý do cần làm lại đối sánh là việc loại box trên Before có thể thay đổi cách một box trên After được phân loại là đối tượng đã tồn tại hay mới thêm. Chỉ xóa một số dòng khỏi report After hoặc chỉ đếm margin âm sẽ chưa đánh giá đúng tác động toàn pipeline.

Replay yêu cầu tái tạo được kết quả delta, matching và metrics gốc trước khi đánh giá ứng viên. Job 8280 loại bốn box ở Before và 18 box ở After. Kết quả cuối giữ 97 TP và 32 FN, trong khi FP giảm 162 xuống 144. Ambulance giữ 5 TP và giảm FP 34 xuống 16.

**Quyết định.** Vì replay giữ được recall của điểm làm việc đang dùng và giảm FP, nhóm tiếp tục tích hợp CLIP vào inference thực tế. Kết quả replay vẫn chưa cho biết chạy model mới có tái tạo cùng kết quả hay không.

### 1.17. Tích hợp CLIP tùy chọn và chạy inference mới

Nhóm thêm tùy chọn `vehicle_clip_filter` vào cấu hình thí nghiệm. Khi bật, CLIP xử lý box xe xung đột sau NMS cùng lớp và trước đối sánh. Khi không bật, cấu hình đối chứng tiếp tục hành vi cũ. Quy tắc CLIP không được bật đồng thời với quy tắc ưu tiên cứng theo lớp.

Để khớp với phép thử trước, nhóm giữ nguyên prompt, cách cắt vùng xe, IoU xung đột 0,75 và margin bằng 0. Model CLIP được cố định ở revision đã dùng khi chấm điểm. Report từng cặp ghi box, điểm hai mô tả, margin và quyết định giữ/loại; run manifest ghi model, chính sách và bộ đếm.

Hai job mới được chạy trên cùng commit:

- **8281:** cấu hình đối chứng `dino-ambulance-min030.json`.
- **8282:** cấu hình `dino-ambulance-clip.json`, khác đối chứng ở việc bật CLIP.

Kết quả strict-v2 của 8282 khớp các chỉ số đã in ra của replay 8280. So sánh hai job cũng xác nhận cấu hình chỉ khác tùy chọn CLIP. Vì vậy, hiệu quả giảm FP quan sát được không chỉ xuất hiện trong bước xử lý report đã lưu; nó được tái hiện khi thực hiện inference mới trên cùng validation.

### 1.18. Chuẩn bị test và ghi nhận giới hạn nguồn dữ liệu

Nhóm đã chuẩn bị `test-images.zip` gồm 120 ảnh, tương ứng 60 cặp đầy đủ. Checksum ZIP trên máy làm việc và server khớp nhau. Công cụ xuất prompt theo split đã tạo `test-generation-prompts.json`; lần đối chiếu cuối xác nhận đủ 60 cặp, tên frame khớp mapping và không thiếu editing instruction.

Nhóm đồng thời kiểm tra nguy cơ trùng nguồn. Không phát hiện trùng chính xác giữa validation và test ở `source_group`, `image_key`, checksum ảnh Before và URL. Tuy nhiên, source group hiện chỉ là fallback theo pair ID, nên kết quả không trùng ở trường này không chứng minh độc lập sự kiện.

Đối chiếu với `image_summary.json` khớp 100/100 cặp. Summary có các trường `downloaded_file`, `incidents`, `places`, `url`; không có bốn trường mã sự kiện đã kiểm tra. Đây là thông tin về summary hiện có, chưa chứng minh mọi nguồn metadata khác đều không có thông tin sự kiện.

**Trạng thái hiện tại.** Chưa gán nhãn 60 cặp test; người thực hiện dự kiến tiếp tục vào tuần sau. Chưa chạy đánh giá hiệu năng trên tập này. Các kết luận của báo cáo đến hiện tại đều dựa trên validation, với giới hạn phép tách theo cặp chưa được xác minh độc lập theo sự kiện.

## 2. Tổng hợp kết quả thực nghiệm và cách diễn giải

### 2.1. Kết quả giai đoạn ngưỡng và prompt theo protocol cũ

Bảng dưới thuộc cùng giai đoạn annotation trước strict-v2: mỗi run xử lý 40 cặp, nhưng các chỉ số detector/pipeline chỉ sử dụng 17 cặp đủ điều kiện. Không nối các số này thành đường cải thiện trực tiếp với bảng GT v2.

| Phép thử | Job | Detector recall | Pipeline precision | Pipeline recall | Pipeline F1 | Quyết định |
|---|---|---:|---:|---:|---:|---|
| Box 0,25; text 0,25 | 8082 | 38,36% | 71,05% | 37,50% | 49,09% | Mốc ngưỡng ban đầu |
| Box 0,18; text 0,25 | 8083 | 38,36% | 71,05% | 37,50% | 49,09% | Hạ riêng box chưa có lợi ích đo được |
| Box 0,18; text 0,18 | 8090 | 50,68% | 70,00% | 48,61% | 57,38% | Giữ để thử prompt tiếp |
| Prompt mở rộng, mapping exact | 8103 | 26,03% | 51,52% | 23,61% | 32,38% | Không chọn; nhiều nhãn ghép bị bỏ |
| Prompt mở rộng, mapping firefighter compound | 8104 | 53,42% | 57,63% | 47,22% | 51,91% | Khôi phục một phần nhãn nhưng chưa là lựa chọn cuối |

Job 8105 là đối chứng prompt gốc chạy lại cùng commit với 8103/8104, có pipeline precision 70,00%, recall 48,61%, F1 57,38%. Điều này giúp đọc kết quả prompt mở rộng trong một phép so sánh tương thích, thay vì gộp các job khác phiên bản code.

### 2.2. Đánh giá lại prompt trên ground truth strict-v2

| Cấu hình | Job report | TP / FP / FN toàn pipeline | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Prompt gốc | 8105 | 64 / 83 / 65 | 43,54% | 49,61% | 46,38% |
| Prompt mở rộng, mapping exact | 8103 | 30 / 69 / 99 | 30,30% | 23,26% | 26,32% |
| Prompt mở rộng, mapping compound | 8104 | 61 / 111 / 68 | 35,47% | 47,29% | 40,53% |

Các phương án trên đã được chấm lại trên cùng 40 cặp và 129 đối tượng mới của GT v2. Bảng này cho thấy khôi phục nhãn ghép có lợi so với chính prompt mở rộng chưa xử lý, nhưng chưa vượt prompt gốc ở F1 pipeline. Nhóm vì vậy chuyển sang prompt bổ sung độc lập.

### 2.3. Hiệu quả và chi phí FP khi bổ sung prompt

| Chỉ số toàn pipeline, strict-v2 | Prompt đối chứng 8237 | Prompt bổ sung 8238 |
|---|---:|---:|
| TP | 64 | 99 |
| FP | 83 | 188 |
| FN | 65 | 30 |
| Precision | 43,54% | 34,49% |
| Recall | 49,61% | 76,74% |
| F1 | 46,38% | 47,60% |
| Ambulance TP / FP / FN | 1 / 1 / 7 | 7 / 60 / 1 |

Recall pipeline tăng **27,13 điểm phần trăm**, tương ứng thêm 35 TP trên tập này. Đồng thời FP tăng 105 và precision giảm **9,04 điểm phần trăm**. Kết quả đáp ứng một phần mục tiêu tăng recall nhưng cho thấy còn nhiều nhãn cần chỉnh trước khi sử dụng cho annotation tự động.

### 2.4. Vì sao giữ ngưỡng ambulance 0,30 thay vì chọn F1 cao nhất

Bốn job 8239–8242 được dùng cho vòng so sánh ngưỡng riêng của prompt bổ sung ambulance.

| Ngưỡng riêng | Job | TP / FP / FN pipeline | Precision pipeline | Recall pipeline | F1 pipeline | Ambulance TP / FP / FN | Recall ambulance |
|---|---|---:|---:|---:|---:|---:|---:|
| Chưa thêm ngưỡng riêng | 8239 | 99 / 188 / 30 | 34,49% | 76,74% | 47,60% | 7 / 60 / 1 | 87,50% |
| 0,30 | 8240 | 97 / 162 / 32 | 37,45% | 75,19% | 50,00% | 5 / 34 / 3 | 62,50% |
| 0,45 | 8241 | 93 / 130 / 36 | 41,70% | 72,09% | 52,84% | 1 / 2 / 7 | 12,50% |
| 0,60 | 8242 | 93 / 129 / 36 | 41,89% | 72,09% | 52,99% | 1 / 1 / 7 | 12,50% |

Nhóm giữ 0,30 để tiếp tục tìm cách phân biệt lớp khi vẫn còn nhiều ambulance đúng. Chọn 0,45 hoặc 0,60 chỉ vì F1 tổng thể cao hơn sẽ không phù hợp với ưu tiên giữ recall ambulance trong vòng thử này.

Tuy vậy, cấu hình 0,30 cũng không có recall cao nhất: nó bỏ thêm hai ambulance đúng so với cấu hình chưa thêm ngưỡng riêng. Lựa chọn hiện tại cần được hiểu là điểm làm việc thăm dò, chưa phải operating point tối ưu đã được xác nhận trên dữ liệu độc lập.

### 2.5. So sánh loại cứng theo IoU và phương án CLIP

| Chỉ số toàn pipeline, strict-v2 | Đối chứng 8252 | Loại cứng theo IoU 8253 | CLIP inference mới 8282 |
|---|---:|---:|---:|
| TP | 97 | 92 | 97 |
| FP | 162 | 129 | 144 |
| FN | 32 | 37 | 32 |
| Precision | 37,45% | 41,63% | 40,25% |
| Recall | 75,19% | 71,32% | 75,19% |
| F1 | 50,00% | 52,57% | 52,43% |
| Macro F1 | 49,11% | 44,85% | 51,75% |
| Ambulance TP / FP / FN | 5 / 34 / 3 | 0 / 1 / 8 | 5 / 16 / 3 |

Đối chứng của phép inference CLIP là job 8281; các số tương ứng trong bảng được tái tạo từ đối chứng này. Job 8252 là mốc của phép thử loại cứng và replay; bảng không khẳng định các run này có cùng commit. Hai so sánh có đối chứng cùng phiên bản là 8252/8253 và 8281/8282.

Loại cứng có F1 tổng thể nhỉnh hơn CLIP trong bảng, nhưng làm recall ambulance bằng 0. CLIP giữ các TP ambulance đã có của đối chứng và giảm 18 FP ambulance, tương ứng giảm khoảng 52,94% FP của lớp này. Vì vậy, nhóm chọn tiếp tục đánh giá cấu hình CLIP thay vì ưu tiên cứng theo lớp.

### 2.6. Kết quả cuối của hai job inference mới 8281 và 8282

| Chỉ số | Đối chứng 8281 | CLIP 8282 | Chênh lệch |
|---|---:|---:|---:|
| Detector TP / FP / FN | 105 / 186 / 30 | 105 / 168 / 30 | FP giảm 18 |
| Detector precision | 36,08% | 38,46% | +2,38 điểm % |
| Detector recall | 77,78% | 77,78% | Không đổi |
| Detector F1 | 49,30% | 51,47% | +2,17 điểm % |
| Pipeline TP / FP / FN | 97 / 162 / 32 | 97 / 144 / 32 | FP giảm 18 |
| Pipeline precision | 37,45% | 40,25% | +2,80 điểm % |
| Pipeline recall | 75,19% | 75,19% | Không đổi |
| Pipeline F1 | 50,00% | 52,43% | +2,43 điểm % |
| Pipeline macro F1 | 49,11% | 51,75% | +2,64 điểm % |

Chi tiết theo lớp của pipeline CLIP 8282:

| Lớp | GT mới | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Firefighter | 61 | 40 | 57 | 21 | 41,24% | 65,57% |
| Rescue boat | 15 | 10 | 4 | 5 | 71,43% | 66,67% |
| Helicopter | 1 | 1 | 2 | 0 | 33,33% | 100,00% |
| Ambulance | 8 | 5 | 16 | 3 | 23,81% | 62,50% |
| Emergency vehicle | 44 | 41 | 65 | 3 | 38,68% | 93,18% |
| Tổng | 129 | 97 | 144 | 32 | 40,25% | 75,19% |

Recall 100% của helicopter chỉ dựa trên một đối tượng GT mới, nên chưa có tính ổn định để kết luận cho lớp. Ambulance cũng chỉ có tám GT mới. Những lớp ít mẫu cần được báo cáo cùng số lượng, không chỉ tỷ lệ.

Mean matched IoU pipeline của hai job đều khoảng **0,7882**. Chỉ số này được tính trên các cặp đã ghép đúng; nó không bao gồm các đối tượng bỏ sót. Kết quả hiện tại không chứng minh CLIP cải thiện biên box: đóng góp đo được là giảm dự đoán nhầm lớp trong khi giữ các TP của đối chứng.

### 2.7. Cấu hình và tài nguyên của thí nghiệm cuối

| Thành phần | Thiết lập đã sử dụng |
|---|---|
| Detector | `IDEA-Research/grounding-dino-tiny` |
| Revision DINO | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| CLIP | `openai/clip-vit-base-patch16` |
| Revision CLIP | `57c216476eefef5ab752ec549e440a49ae4ae5f3` |
| Box threshold / text threshold | 0,18 / 0,18 |
| Ngưỡng riêng rescue boat | 0,33 |
| NMS cùng lớp | IoU 0,5 |
| Ngưỡng prompt bổ sung ambulance | 0,30 |
| IoU xung đột ambulance–emergency vehicle | 0,75 |
| Quy tắc CLIP | Loại ambulance xung đột khi margin ≤ 0 |
| Seed | `20261020` |
| Commit của hai job 8281/8282 | `7c2a4309c7fbadb06734aa2fcf589ca277bedd38` |
| GPU | NVIDIA H200 MIG 2g.35gb |
| Dữ liệu đánh giá | 40 cặp validation, GT strict-v2 |

Thời gian được run manifest ghi nhận tăng từ khoảng **77,30 giây** ở đối chứng lên **81,89 giây** khi bật CLIP. Đây là thời gian của lượt chạy được đo, không bao gồm thời gian chờ hàng đợi Slurm và không phải benchmark tốc độ qua nhiều lần lặp.

Peak VRAM do PyTorch ghi nhận tăng từ khoảng **1,48 GiB** lên **2,03 GiB**. Chi phí tăng này đi kèm bước phân biệt vùng xe. Chưa dùng các số của một cặp job để khẳng định hiệu năng thời gian ổn định trên mọi phần cứng hoặc quy mô dữ liệu.

### 2.8. Mức độ hoàn thành so với yêu cầu và việc còn tiếp tục

| Nội dung từ nhận xét của thầy | Điều đã thực hiện | Điều còn thiếu |
|---|---|---|
| Tăng recall | Điều chỉnh text threshold, tách prompt bổ sung, đo TP/FN theo lớp | Tiếp tục đánh giá điểm đánh đổi recall/precision và dữ liệu độc lập |
| Giảm nhãn xe nhầm | Phân tích FP, thử ưu tiên cứng, CLIP theo vùng xung đột, replay và inference mới | Ambulance vẫn có 16 FP, ba FN trong validation; chưa giải quyết mọi loại nhầm xe |
| Refine bounding box | Có cơ chế refine bằng delta trong pipeline và đã đo mean matched IoU | Chưa có ablation riêng chứng minh cải thiện localization, chưa thử SAM2 |
| Survey công cụ auto-annotation | Đã sử dụng và phân định vai trò CVAT, Grounding DINO, CLIP | Chưa có so sánh nhiều detector/framework theo cùng protocol |
| Cơ sở đánh giá | Có benchmark theo cặp, GT v2, ignore policy và metrics theo lớp | GT chưa được xác minh độc lập toàn bộ; số mẫu lớp hiếm còn ít |
| Đánh giá test | Đã chuẩn bị 60 cặp và đủ 60 prompt đối chiếu | Chưa gán nhãn, chưa chạy đánh giá test; dự kiến tiếp tục tuần sau |

Cấu hình hiện tại là ứng viên dựa trên các vòng thăm dò validation. CLIP giúp cải thiện precision và F1 so với điểm làm việc 0,30 mà giữ recall của điểm đó; không thể diễn đạt là CLIP làm tăng recall so với chính đối chứng này hoặc cấu hình đã tối ưu toàn bộ bài toán.

Các lựa chọn ngưỡng, prompt và chính sách được đưa ra sau khi xem validation. Kết quả vì thế chưa đo được khả năng tổng quát hóa trên dữ liệu chưa dùng để lựa chọn phương án. Thêm vào đó, chưa xác minh được nhóm sự kiện giữa hai split, nên chưa có cơ sở công bố tính độc lập theo sự kiện.

Trong giai đoạn tiếp theo, nhóm dự kiến hoàn thiện nhãn test theo tiến độ tuần sau, tiếp tục các nội dung refine/survey còn thiếu trên validation và chỉ đánh giá test sau khi xác định rõ cấu hình cùng phạm vi kết luận. Nếu tiếp tục điều chỉnh sau khi đã xem test, cần ghi rõ lần đánh giá đó không còn là test độc lập cho cấu hình mới.
