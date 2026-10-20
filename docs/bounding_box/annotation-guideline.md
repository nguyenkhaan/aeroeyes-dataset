# Hướng dẫn gán nhãn benchmark bounding box AeroEyes

## 1. Mục tiêu

Benchmark đánh giá khả năng phát hiện, định vị và xác nhận các đối tượng cứu hộ
mới được bổ sung trong cặp ảnh thiên tai Before/After. Ground truth phải cho
phép tách bốn loại lỗi:

1. **Generation failure**: ảnh After không chứa đối tượng cứu hộ hợp lệ như yêu cầu.
2. **Detection failure**: đối tượng tồn tại nhưng detector không tìm thấy hoặc gán sai lớp.
3. **Matching failure**: pipeline phân loại sai đối tượng mới và đối tượng đã tồn tại.
4. **Localization failure**: detector tìm đúng lớp nhưng bounding box không đủ chính xác.

`bbox_passed` chỉ là chỉ số vận hành của pipeline, không phải precision hoặc recall.

## 2. Đơn vị annotation

Mỗi mẫu là một cặp ảnh có chung `pair_id`:

- **Before**: ảnh thiên tai trước khi chỉnh sửa;
- **After**: ảnh sau khi mô hình sinh bổ sung hoạt động cứu hộ;
- **Before objects**: mọi đối tượng thuộc ontology nhìn thấy trong ảnh Before;
- **After objects**: mọi đối tượng thuộc ontology nhìn thấy trong ảnh After;
- **Pair attributes**: trạng thái sinh ảnh và ghi chú tình huống đặc biệt.

Mỗi object trong ảnh After phải có:

- `class_name`;
- bounding box;
- `is_newly_added`: `true`, `false` hoặc `ambiguous`;
- `is_valid_rescue`: `true`, `false` hoặc `ambiguous`;
- `occlusion`: `none`, `partial` hoặc `heavy`;
- `annotation_notes` khi có điểm không rõ ràng.

## 3. Ontology

Năm lớp phải loại trừ nhau ở cấp annotation. Danh sách alias chính thức nằm tại
`config/bounding_box/ontology.yaml`.

### 3.1. `firefighter`

Gán nhãn khi người trong ảnh có đủ dấu hiệu nhận diện là lính cứu hỏa, chẳng hạn
đồ bảo hộ, mũ, bình dưỡng khí, vòi chữa cháy hoặc hoạt động chữa cháy rõ ràng.

Ví dụ dương:

- người mặc đồ bảo hộ chữa cháy và đang sử dụng vòi nước;
- người có mũ và đồng phục cứu hỏa đứng cạnh xe chữa cháy trong hiện trường.

Không gán tự động:

- người mặc áo phản quang nhưng không có dấu hiệu là lính cứu hỏa;
- `rescue worker`, nhân viên y tế hoặc cảnh sát không xác định được vai trò.

### 3.2. `rescue_boat`

Gán cho thuyền hoặc bè được sử dụng rõ ràng để cứu hộ, sơ tán hoặc vận chuyển
người bị ảnh hưởng trong bối cảnh thảm họa.

Ví dụ dương:

- xuồng bơm hơi chở đội cứu hộ trong vùng ngập;
- thuyền có ký hiệu cứu hộ đang tiếp cận nạn nhân.

Không gán tự động:

- thuyền dân dụng không thể hiện hoạt động cứu hộ;
- vật thể hoặc đống đổ nát có hình dạng giống thuyền.

### 3.3. `helicopter`

Gán cho trực thăng tham gia hoặc được bổ sung trong bối cảnh cứu hộ. Không gán
máy bay cánh cố định vào lớp này.

Ví dụ dương:

- trực thăng cứu hộ bay gần hiện trường thiên tai;
- trực thăng được bổ sung theo editing instruction.

Phản ví dụ:

- máy bay chở khách;
- hình dạng mơ hồ trên trời không đủ nhận diện là trực thăng.

### 3.4. `ambulance`

Gán khi phương tiện được nhận diện rõ là xe cứu thương qua hình dạng, ký hiệu,
đèn hoặc chữ trên xe. `ambulance` loại trừ với `emergency_vehicle`.

Ví dụ dương:

- xe cứu thương có biểu tượng y tế hoặc chữ ambulance;
- xe cứu thương đang tham gia sơ tán tại hiện trường.

Phản ví dụ:

- xe van trắng thông thường;
- xe chữa cháy hoặc rescue truck.

### 3.5. `emergency_vehicle`

Gán cho xe chữa cháy, rescue truck hoặc xe phản ứng khẩn cấp đường bộ không phải
xe cứu thương.

Ví dụ dương:

- fire truck có thang, vòi hoặc ký hiệu cứu hỏa;
- rescue truck có đèn và nhận diện cứu hộ rõ ràng.

Phản ví dụ:

- xe cứu thương;
- xe địa hình hoặc xe tải thông thường không có dấu hiệu ứng cứu.

## 4. Quy tắc vẽ bounding box

1. Vẽ box nhỏ nhất bao quanh toàn bộ phần nhìn thấy của một đối tượng.
2. Không mở rộng box để chứa bóng, khói, vòi nước hoặc vật thể người đó đang cầm,
   trừ khi phần đó thuộc cấu trúc của phương tiện.
3. Với đối tượng bị che một phần, box bao quanh phần nhìn thấy; đặt
   `occlusion=partial` hoặc `heavy`.
4. Với đối tượng bị cắt bởi biên ảnh, box dừng ở biên ảnh và ghi chú `truncated`.
5. Hai người đứng sát nhau vẫn phải có hai box nếu phân biệt được từng người.
6. Không vẽ một box chung cho một nhóm người hoặc nhiều phương tiện.
7. Đối tượng quá mơ hồ để xác định lớp được đánh dấu review, không ép nhãn.

## 5. Xác định đối tượng mới

Annotator phải xem đồng thời Before và After:

- `true`: đối tượng After không có đối tượng cùng danh tính trong Before;
- `false`: cùng đối tượng đã tồn tại trong Before dù box, màu hoặc hình dạng có
  thay đổi nhẹ;
- `ambiguous`: thay đổi phối cảnh hoặc mức sinh lại ảnh khiến không thể kết luận.

Không dùng chỉ riêng IoU để tạo ground truth new/existing. Annotator cần xem vị
trí tương đối, hình dạng, màu sắc, bối cảnh và quan hệ với các vật thể xung quanh.

## 6. Trạng thái sinh ảnh

- `valid`: ảnh After chứa ít nhất một đối tượng cứu hộ hợp lệ theo yêu cầu và
  không bị hỏng cảnh nghiêm trọng.
- `missing_requested_object`: không thấy đối tượng cứu hộ hợp lệ được yêu cầu.
- `invalid_object`: có vật thể được sinh nhưng sai nghĩa hoặc biến dạng đến mức
  không thể gán nhãn.
- `severe_scene_corruption`: bố cục hoặc phần lớn nội dung ảnh bị thay đổi/hỏng.
- `ambiguous`: chưa đủ bằng chứng để kết luận; phải adjudicate.

Generation failure không được tính là detector false negative.

## 7. Quy trình kiểm soát chất lượng

1. Annotator đọc guideline và gán thử cùng 10 cặp calibration.
2. Thảo luận bất đồng và khóa guideline trước khi gán benchmark chính.
3. Ít nhất 20% cặp được hai người gán độc lập.
4. So sánh class, new/existing và IoU giữa hai bộ annotation.
5. Người review adjudicate mọi bất đồng và mọi nhãn `ambiguous`.
6. Khóa artifact với version, checksum, ngày và người review.
7. Mọi sửa đổi sau khi khóa phải tạo version mới và changelog.

## 8. Tách validation và test

- Validation dùng để phân tích lỗi, chọn prompt, threshold, matching và SAM2.
- Test chỉ được chạy sau khi đã khóa cấu hình cuối.
- Không tiếp tục tune trên test rồi báo cáo lại trên cùng tập đó.
- Challenge subset chứa trường hợp khó phải nằm trong validation hoặc được báo
  cáo riêng; không gộp vào metric đại diện nếu được lấy mẫu có chủ đích.
- Nếu một lớp trong test có quá ít object newly-added hợp lệ, báo cáo
  `insufficient_data` hoặc bổ sung mẫu trước khi mở kết quả test.

## 9. Schema annotation đề xuất

```json
{
  "pair_id": "sha256-stem",
  "split": "validation",
  "generation_status": "valid",
  "before_objects": [
    {
      "object_id": "before-1",
      "class_name": "firefighter",
      "bbox_xyxy": [10, 20, 80, 190],
      "occlusion": "none"
    }
  ],
  "after_objects": [
    {
      "object_id": "after-1",
      "class_name": "firefighter",
      "bbox_xyxy": [200, 120, 280, 350],
      "is_newly_added": "true",
      "is_valid_rescue": "true",
      "occlusion": "partial",
      "annotation_notes": ""
    }
  ],
  "pair_notes": ""
}
```

Các thuộc tính `is_newly_added` và `is_valid_rescue` dùng chuỗi enum
`"true"`, `"false"`, `"ambiguous"` để tương thích với thuộc tính dạng danh mục
của công cụ annotation. Các tọa độ dùng quy ước
`x_min, y_min, x_max, y_max` theo pixel của đúng ảnh được gán nhãn.
