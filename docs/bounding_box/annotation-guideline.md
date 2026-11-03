# Hướng dẫn gán nhãn benchmark bounding box AeroEyes

> Bản bổ sung T2 ngày 03/11/2026: **dự thảo để calibration**, chưa khóa cho
> đợt annotation mới. Ontology và schema hiện hành chưa thay đổi. Quy trình thao
> tác nằm tại [annotation-four-rounds.md](annotation-four-rounds.md).

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
- `occlusion_level` trong CVAT: `none`, `partial` hoặc `heavy`;
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

### 3.6. Tình huống khó và cách ghi nhận

- **Người mặc áo phản quang:** áo hoặc mũ bảo hộ thông thường, đứng gần xe cứu
  hỏa, hay prompt yêu cầu firefighter không tự xác nhận người đó là firefighter.
  Tìm thêm đặc điểm trang bị hoặc hoạt động chữa cháy. Nếu có ứng viên hợp lý
  nhưng chưa đủ bằng chứng, đưa vào review; không đổi thành nhãn chắc chắn để
  tăng số lượng.
- **Xe cứu thương và xe khẩn cấp:** nhãn, chữ, biểu tượng và cấu trúc phải được
  xem cùng ảnh. Xe van trắng hay đèn nhấp nháy đơn lẻ không đủ xác nhận xe cứu
  thương. Nếu là xe cứu thương rõ ràng, chỉ dùng `ambulance`; xe chữa cháy rõ
  ràng dùng `emergency_vehicle`. Với xe bị che, mờ hoặc biến dạng khiến không
  phân biệt được, đưa review; không gán đồng thời cả hai lớp cho một xe.
- **Thuyền và trực thăng:** thuyền dân dụng chưa thể hiện cứu hộ và dấu vết
  trên trời không đủ nhận diện trực thăng không tự thành nhãn xác nhận. Xem
  bối cảnh, ảnh Before/After và prompt; prompt không thay thế bằng chứng ảnh.
- **Đối tượng nhỏ, chồng lấn hoặc bị che:** nếu phân biệt được từng đối tượng,
  mỗi đối tượng có một box bao phần nhìn thấy. Không đặt ngưỡng kích thước
  không có trong protocol. Nếu không đủ bằng chứng để tách hoặc xác định đối
  tượng, ghi review và lý do; không đoán số lượng.
- **Biến dạng do sinh ảnh:** nếu không đủ nhận diện lớp hoặc tính hợp lệ, không
  tạo box chắc chắn chỉ vì prompt yêu cầu. Đánh giá `generation_status` riêng
  với nhãn của những đối tượng khác vẫn nhận diện được.

Với một ứng viên có lớp hợp lý nhưng tính cứu hộ chưa chắc, có thể lưu box ở
lớp ứng viên cùng `is_valid_rescue=ambiguous` và giải thích trong
`annotation_notes`; nếu tính mới không rõ thì ghi `is_newly_added=ambiguous`.
Đây là ứng viên cần review, không phải GT xác nhận. Nếu không thể chọn lớp ứng
viên có căn cứ, ghi vào **danh sách review ngoài annotation** với `pair_id`,
Before/After, vị trí và lý do, thay vì ép một trong năm lớp. Không tự tạo nhãn
`unknown` hoặc coi một vùng ignore theo lớp là có thể che mọi lớp. Khi adjudicate,
giữ lại `ambiguous` nếu vẫn thiếu bằng chứng và nêu rõ giới hạn của cách biểu
diễn trong artifact; không xóa âm thầm các trường hợp chưa xử lý.

### 3.7. Tên lớp và thuật ngữ trong prompt

Tên lớp chuẩn là năm mã ở mục 3 và không đổi theo tiếng Anh Anh/Mỹ. Bảng dưới
giải thích cách dùng, **không tự bổ sung alias hoặc thay cấu hình**. Alias tự
động hiện hành và từ cần review được định nghĩa trong ontology.

| Thuật ngữ | Sử dụng trong annotation và prompt |
|---|---|
| `firefighter`, `fire fighter` | Alias đã có cho `firefighter`; vẫn cần xem bằng chứng trong ảnh. |
| `fire engine`, `fire truck` | Cách gọi xe chữa cháy; cả hai đã là alias `emergency_vehicle`. Có thể thử làm prompt riêng, nhưng prompt không chứng minh đối tượng tồn tại. |
| `ambulance` | Alias đã có cho `ambulance`; không đồng nhất với mọi `emergency vehicle`. |
| `rescue boat`, `inflatable rescue boat` | Alias đã có cho `rescue_boat`; `boat` chung cần review bối cảnh. |
| `helicopter`, `rescue helicopter` | Alias đã có cho `helicopter`; `aircraft` chung cần review. |
| `rescue worker`, `emergency responder`, `paramedic`, `rescue vehicle` | Thuật ngữ rộng hoặc vai trò khác; không tự ánh xạ, cần review theo ontology. |

## 4. Quy tắc vẽ bounding box

1. Vẽ box nhỏ nhất bao quanh toàn bộ phần nhìn thấy của một đối tượng.
2. Không mở rộng box để chứa bóng, khói, vòi nước hoặc vật thể người đó đang cầm,
   trừ khi phần đó thuộc cấu trúc của phương tiện.
3. Với đối tượng bị che một phần, box bao quanh phần nhìn thấy; đặt
   `occlusion_level=partial` hoặc `heavy` trong CVAT.
4. Với đối tượng bị cắt bởi biên ảnh, box dừng ở biên ảnh và ghi chú `truncated`.
5. Hai người đứng sát nhau vẫn phải có hai box nếu phân biệt được từng người.
6. Không vẽ một box chung cho một nhóm người hoặc nhiều phương tiện.
7. Đối tượng quá mơ hồ để xác định lớp được đánh dấu review, không ép nhãn.

`occlusion_level` mô tả mức bị **vật khác che**, không tự suy từ việc vật thể
bị cắt ở biên. `none`: không thấy bị che; `partial`: còn nhận diện được phần
đáng kể nhưng có phần bị che; `heavy`: bị che phần lớn, chỉ khi vẫn có đủ bằng
chứng nhận diện. Chưa đặt ngưỡng phần trăm cứng; ghi lý do các ca ranh giới để
calibration. Với trường hợp vừa che vừa cắt biên, ghi cả mức occlusion và note
`truncated`.

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

Phải xem `editing_instruction`/prompt gốc cùng Before và After khi kết luận yêu
cầu sinh ảnh được đáp ứng. Nếu prompt thiếu hoặc không đủ bằng chứng, ghi
`ambiguous` và lý do; không suy ra `missing_requested_object` chỉ từ ảnh.
Mỗi frame After cần đúng một tag `pair_status` chứa `generation_status`, kể cả
khi không có box. Không dùng tag này trên Before. Sau review, vẫn có thể giữ
`ambiguous` nếu bằng chứng không đủ; ghi rõ ai đã xem và vì sao.

Generation failure không được tính là detector false negative.

## 7. Quy trình kiểm soát chất lượng

1. Annotator đọc guideline và gán thử cùng 10 cặp calibration.
2. Thảo luận bất đồng và khóa guideline trước khi gán benchmark chính.
3. Ít nhất 20% cặp được hai người gán độc lập.
4. So sánh class, new/existing và IoU giữa hai bộ annotation.
5. Người review adjudicate mọi bất đồng và mọi nhãn `ambiguous`.
6. Khóa artifact với version, checksum, ngày và người review.
7. Mọi sửa đổi sau khi khóa phải tạo version mới và changelog.

Gán thử 10 cặp là mục tiêu protocol, không coi là đã hoàn tất khi chưa có hai
bộ nhãn và log thảo luận. Calibration dùng ảnh phát triển tách test; không dùng
GT 40 cặp đang phân tích để giả định hai người gán độc lập nếu họ đã xem nhãn.
Chi tiết chọn mẫu, biểu mẫu và điều kiện review ở
[annotation-four-rounds.md](annotation-four-rounds.md).

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
