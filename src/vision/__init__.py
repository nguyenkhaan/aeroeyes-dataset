def build_flux_prompt(
    scene_description: str,
    rescue_instruction: str,
) -> str:
    """
    Build the final prompt for FLUX image editing.
    """
    prompt = f"""
Bạn đang chỉnh sửa một bức ảnh thiên tai hiện có.

### Cảnh ban đầu
--------------
{scene_description}
### Hướng dẫn chỉnh sửa
--------------------
{rescue_instruction}
### Yêu cầu
    - Giữ nguyên cảnh thiên tai ban đầu.
    - Giữ nguyên tất cả tòa nhà, phương tiện, con đường và đối tượng môi trường hiện có.
    - Không thay đổi loại thiên tai.
    - Chỉ thêm các hoạt động cứu hộ thực tế.
    - Hòa trộn tự nhiên nhân viên cứu hộ, phương tiện và thiết bị mới thêm.
    - Duy trì ánh sáng, bóng đổ và phối cảnh thực tế.
    - Duy trì tỷ lệ chính xác giữa các đối tượng.
    - Tạo con người đúng về mặt giải phẫu.
    - Chỉnh sửa ảnh liền mạch, không có lỗi nhìn thấy được.
### Phong cách
    - Nhiếp ảnh thiên tai mang tính tài liệu
    - Ảnh báo chí
    - Hoạt động ứng phó khẩn cấp ngoài đời thực
    - Chỉnh màu tự nhiên
    - Kết cấu chân thực
    - Độ chân thực cao
    - Không mang phong cách điện ảnh
"""
    return prompt.strip()