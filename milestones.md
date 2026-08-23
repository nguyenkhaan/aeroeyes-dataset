# Milestones: Rescue Image Generation Pipeline on VPS

## Mục tiêu
Chuyển pipeline trong repo từ trạng thái prototype/Kaggle sang trạng thái có thể chạy ổn định trên VPS bằng `sbatch.sh`, đồng thời giữ đúng hướng của `docs/pipeline/pipeline-2-prd.md`.

Mục tiêu ngắn hạn là tạo được một luồng sinh ảnh cứu hộ đơn giản, chạy được end-to-end trên một tập mẫu nhỏ, không cần mở rộng sang phần đánh giá phức tạp của PRD ở giai đoạn đầu.

## Nguyên tắc làm việc
- Không chạy pipeline thật trên máy local.
- Chỉ kiểm tra bằng đọc source, phân tích logic, kiểm tra cú pháp, và xác nhận cấu hình.
- Notebook Kaggle là tài liệu tham chiếu, không phải entrypoint triển khai trên VPS.
- Ưu tiên sửa các lỗi làm job fail ngay trước, sau đó mới dọn cấu trúc và cải thiện chất lượng.

## Hiện trạng đã kiểm tra
- `main.py` là entrypoint hiện tại của code VPS.
- `sbatch.sh` đang gọi `train.py`, nhưng repo không có file này.
- `main.py` đang dùng biến chưa khởi tạo như `count` và `skipped`.
- `main.py` chưa tạo `metadata_dir` trước khi ghi file metadata.
- Repo đang có song song 2 lớp file:
  - `src/core/config.py`, `src/helper/*`, `src/generation/*`, `src/vision/*`
  - `src/config.py`, `src/utility.py`
- `data/input/eccv_train.json` là file rất lớn, chỉ nên xem vài dòng đầu để hiểu schema, không quét toàn bộ.

## Milestone 1: Khóa lại contract chạy tối thiểu
Mục tiêu: làm rõ entrypoint, cấu hình, và các biến nền để project không chết ngay khi khởi động.

### Việc cần làm
- Chuẩn hóa `main.py` làm entrypoint chính.
- Sửa `sbatch.sh` để gọi đúng entrypoint.
- Đưa các đường dẫn quan trọng về `src/core/config.py` hoặc biến môi trường.
- Khởi tạo các biến runtime cơ bản trong `main.py`.

### Files liên quan
- `main.py`
- `sbatch.sh`
- `src/core/config.py`
- `src/helper/loading_dataset.py`
- `README.md`

### Tiêu chí hoàn thành
- `sbatch.sh` không còn gọi file không tồn tại.
- `main.py` có thể parse được ở mức cú pháp và không còn biến runtime chưa khởi tạo ở phần khởi đầu.
- Cấu hình đường dẫn không phụ thuộc cứng vào Kaggle.

### Verify
- `python -m py_compile main.py src/**/*.py`
- `bash -n sbatch.sh`

## Milestone 2: Làm sạch luồng sinh ảnh đơn giản
Mục tiêu: giữ nguyên pipeline tối thiểu của Kaggle nhưng tách rõ các bước để có thể chạy trên VPS.

### Việc cần làm
- Rà lại luồng:
  - load dataset
  - lọc sample có label dương
  - tải ảnh
  - resize/crop
  - sinh scene description
  - sinh rescue instruction
  - build prompt
  - sinh ảnh
  - lưu ảnh và metadata
- Sửa các lỗi logic trong vòng lặp:
  - đếm sample đã sinh
  - đếm sample bỏ qua
  - xử lý resume
  - xử lý exception theo từng bước
- Đảm bảo metadata được ghi ra đúng thư mục.

### Files liên quan
- `main.py`
- `src/helper/image.py`
- `src/helper/memory.py`
- `src/helper/loading_dataset.py`
- `src/vision/scene_description.py`
- `src/vision/rescue_instruction.py`
- `src/vision/__init__.py`

### Tiêu chí hoàn thành
- Luồng chính không còn lỗi logic hiển nhiên khi đọc source.
- Các bước xử lý có đường thoát lỗi rõ ràng.
- Metadata và ảnh output được ghi vào thư mục dự kiến.

### Verify
- Soát logic thủ công theo từng bước
- `python -m py_compile main.py src/**/*.py`

## Milestone 3: Đồng bộ model loading và API inference
Mục tiêu: làm cho phần load Gemma/FLUX nhất quán giữa notebook cũ và code VPS mới.

### Việc cần làm
- Soát lại `src/generation/gemma.py` và `src/generation/flux.py`.
- Kiểm tra model ID, dtype, device placement, và API tương thích.
- Thống nhất cách gọi `parse_response` hay fallback sang decode string nếu API không hỗ trợ.
- Tách cấu hình model ra khỏi logic chạy.

### Files liên quan
- `src/generation/gemma.py`
- `src/generation/flux.py`
- `src/core/config.py`
- `src/vision/scene_description.py`
- `src/vision/rescue_instruction.py`

### Tiêu chí hoàn thành
- Model loader có giao diện ổn định.
- Không còn phụ thuộc mơ hồ giữa Gemma 3/Gemma 4 trong code.
- Không có chỗ nào hardcode model khác với cấu hình trung tâm mà không được giải thích.

### Verify
- Kiểm tra import tĩnh
- `python -m py_compile main.py src/**/*.py`

## Milestone 4: Chuẩn hóa sbatch cho VPS
Mục tiêu: có một script submit job chạy được trên VPS và không phụ thuộc vào chi tiết Kaggle.

### Việc cần làm
- Sửa `sbatch.sh` để:
  - gọi đúng file chạy
  - `cd` vào project root trước khi chạy
  - activate đúng môi trường
  - ghi log vào thư mục tồn tại
  - không dùng tham chiếu mơ hồ như `train.py`
- Kiểm tra các biến môi trường cần thiết cho Hugging Face token.
- Giảm rủi ro từ cấu hình GPU/MPS nếu VPS không hỗ trợ đúng cách.

### Files liên quan
- `sbatch.sh`
- `README.md`
- `.env.example`

### Tiêu chí hoàn thành
- Script batch nhìn vào là biết chạy file nào, ở đâu, với môi trường nào.
- Không còn đường dẫn shell placeholder chưa thay thế.
- Log/error file có chỗ ghi hợp lệ.

### Verify
- `bash -n sbatch.sh`
- Review thủ công toàn bộ script trước khi submit job thật

## Milestone 5: Dọn dự án và giảm nhiễu
Mục tiêu: giảm nhầm lẫn giữa các file cũ và file mới, để bảo trì dễ hơn.

### Việc cần làm
- Xác định file nào là legacy:
  - `src/config.py`
  - `src/utility.py`
- Quyết định giữ, xóa, hay chuyển thành wrapper tương thích.
- Dọn README để phản ánh đúng entrypoint hiện tại.
- Nếu cần, tách một launcher rõ ràng hơn cho VPS.

### Files liên quan
- `src/config.py`
- `src/utility.py`
- `README.md`
- `CHANGELOG.md`

### Tiêu chí hoàn thành
- Một đường đi chính duy nhất cho cấu hình và utility.
- Tài liệu không gây hiểu nhầm giữa notebook và VPS code.

### Verify
- Rà soát import toàn repo
- Đảm bảo không còn import trùng không cần thiết

## Checkpoint: Sau Milestone 1-2
- [ ] `sbatch.sh` trỏ đúng entrypoint
- [ ] `main.py` không còn lỗi biến chưa khởi tạo
- [ ] Luồng sinh ảnh tối thiểu được tổ chức rõ ràng
- [ ] Không có bước nào yêu cầu chạy model để xác nhận cú pháp

## Checkpoint: Sau Milestone 3-4
- [ ] Loader model và cấu hình nhất quán
- [ ] Batch script có thể submit được trên VPS
- [ ] Các giả định Kaggle đã được tách ra khỏi runtime VPS

## Checkpoint: Hoàn tất
- [ ] Có thể submit job trên VPS mà không sửa tay trực tiếp trong code chạy chính
- [ ] Pipeline tạo được ảnh đầu ra và metadata cho một tập mẫu nhỏ
- [ ] Tài liệu milestone phản ánh đúng trạng thái code

## Rủi ro và giảm thiểu
| Rủi ro | Mức ảnh hưởng | Giảm thiểu |
|---|---|---|
| `train.py` không tồn tại nhưng `sbatch.sh` đang gọi | Cao | Chuyển entrypoint sang `main.py` hoặc tạo launcher mới rõ ràng |
| Biến runtime chưa khởi tạo trong `main.py` | Cao | Khởi tạo đầy đủ và kiểm tra bằng compile tĩnh |
| Khác biệt model/API giữa notebook và source mới | Cao | Chuẩn hóa loader và tách config |
| Đường dẫn Kaggle hardcode | Trung bình | Đưa về config/env trên VPS |
| Job VPS fail vì script batch thiếu `cd`/activate môi trường | Cao | Sửa script batch trước khi submit |

## Open Questions
- VPS của bạn dùng scheduler nào chính xác: Slurm chuẩn hay một lớp wrapper riêng?
- `main.py` sẽ là entrypoint cuối cùng hay bạn muốn tạo thêm một file launcher riêng cho job submit?
- Bạn muốn giữ lại `src/config.py` và `src/utility.py` như wrapper tương thích hay hợp nhất chúng vào `src/core` và `src/helper`?

