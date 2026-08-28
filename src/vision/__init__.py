def build_flux_prompt(
    scene_description: str,
    rescue_instruction: str,
) -> str:
    """
    Xây dựng prompt cuối cùng cho FLUX để chỉnh sửa ảnh.
    """
    prompt = f"""
Bạn đang chỉnh sửa một bức ảnh thiên tai đã có sẵn.

1. Cảnh gốc
----------------
{scene_description}

2. Hướng dẫn chỉnh sửa
------------------------
{rescue_instruction}

3. Yêu cầu bắt buộc
--------------------
- Giữ nguyên cảnh thiên tai gốc.
- Giữ nguyên tất cả tòa nhà, phương tiện, đường giao thông và các vật thể môi trường hiện có.
- Không thay đổi loại thiên tai.
- Chỉ thêm các hoạt động cứu hộ có tính thực tế.
- Trộn các nhân sự cứu hộ, phương tiện và thiết bị mới thêm vào một cách tự nhiên.
- Giữ ánh sáng, bóng đổ và góc nhìn hợp lý với thực tế.
- Giữ tỷ lệ vật thể chính xác.
- Tạo hình người có cấu trúc cơ thể hợp lý và thực tế.
- Không để xuất hiện artefact, vết cắt, thay đổi quá mức hoặc kết cấu không tự nhiên.

4. Phong cách hình ảnh
----------------------
- Chụp ảnh tài liệu thiên tai
- Nhiếp ảnh báo chí
- Phản ánh ứng phó khẩn cấp trong thực tế
- Chất lượng màu tự nhiên
- Kết cấu thật
- Độ chân thực cao
- Không mang phong cách điện ảnh quá mức
"""
    return prompt.strip()