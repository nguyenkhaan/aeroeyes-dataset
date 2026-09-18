#!/usr/bin/env bash
#SBATCH --job-name=aeroeyes_run2
#SBATCH --output=/datastore/cndt_khanhnd/models/aeroeyes_output_run2/logs/slurm_%j.log
#SBATCH --error=/datastore/cndt_khanhnd/models/aeroeyes_output_run2/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=defq

# ════════════════════════════════════════════════════════════
# AeroeEyes Pipeline — Lần chạy 2
#   BASE_SEED=100, GUIDANCE_SCALE=3.0
#   Output: aeroeyes_output_run2/
# Mục đích: Thu thập điểm dữ liệu thứ 2 cho SDQM–mAP regression
# ════════════════════════════════════════════════════════════
set -euo pipefail

# ── Đường dẫn ─────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/venv/bin/python"

# ── Kiểm tra venv ─────────────────────────────────────────
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: venv không tồn tại. Chạy setup_vps.sh trước." >&2
  exit 1
fi

# ════════════════════════════════════════════════════════════
# THAM SỐ LẦN CHẠY 2 — Thay đổi so với lần 1
# ════════════════════════════════════════════════════════════

# ── Đường dẫn lưu trữ ─────────────────────────────────────
export AEROEYES_MODEL_DIR="/datastore/cndt_khanhnd/models/aeroeyes_model"
export AEROEYES_OUTPUT_DIR="/datastore/cndt_khanhnd/models/aeroeyes_output_run2"

# ── HuggingFace token ──────────────────────────────────────
# Đọc từ .env nếu có, hoặc set trực tiếp ở đây
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi
# Hoặc uncomment dòng dưới và điền token:
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# ── Tham số sinh ảnh (KHÁC lần 1) ─────────────────────────
export IMAGE_SIZE=1024
export LIMIT_IMAGES=500
export BASE_SEED=100          # Lần 1: 50  →  Lần 2: 100
export GUIDANCE_SCALE=3.0     # Lần 1: 3.5 →  Lần 2: 3.0
export NUM_INFERENCE_STEPS=20
export MAX_CONSECUTIVE_ERRORS=10

# ── Timeout ────────────────────────────────────────────────
export GENERATION_TIMEOUT_SECONDS=244800   # 68 giờ
export MODEL_LOAD_TIMEOUT_SECONDS=1800
export STAGE_TIMEOUT_SECONDS=900
export EVALUATION_TIMEOUT_SECONDS=3600

# ════════════════════════════════════════════════════════════
# ⚠️  SDQM MAP VALUE — Điền mAP50 từ YOLOv8 lần 1 vào đây
# ────────────────────────────────────────────────────────────
# Cách lấy mAP50 từ YOLOv8 lần 1:
#   1. Xem file results.csv trong thư mục yolo run của lần 1
#   2. Lệnh: awk -F',' 'END{print $NF}' <yolo_run1>/results.csv
#   3. Hoặc: cat <yolo_run1>/results.csv | python3 -c "
#        import sys,csv; rows=list(csv.DictReader(sys.stdin))
#        print('mAP50:', rows[-1]['metrics/mAP50(B)'].strip())"
# Ví dụ: export SDQM_MAP_VALUE=0.734
export SDQM_MAP_VALUE=""       # ← ĐIỀN VÀO ĐÂY trước khi submit
# ════════════════════════════════════════════════════════════

export SDQM_MAP_COLUMN=map
export SDQM_APPEND_HISTORY=true
export SDQM_RUN_REGRESSION=true

# ── In cấu hình ───────────────────────────────────────────
mkdir -p "$AEROEYES_OUTPUT_DIR/logs"

echo "======================================================"
echo " AeroeEyes Run 2  |  Job: $SLURM_JOB_ID  |  $(date)"
echo "======================================================"
echo "  BASE_SEED          = $BASE_SEED"
echo "  GUIDANCE_SCALE     = $GUIDANCE_SCALE"
echo "  NUM_INFERENCE_STEPS= $NUM_INFERENCE_STEPS"
echo "  OUTPUT_DIR         = $AEROEYES_OUTPUT_DIR"
echo "  SDQM_MAP_VALUE     = ${SDQM_MAP_VALUE:-'(chưa điền)'}"
echo "  Python             = $VENV_PYTHON"
echo "======================================================"
echo ""

# ── Kiểm tra SDQM_MAP_VALUE ───────────────────────────────
if [[ -z "$SDQM_MAP_VALUE" ]]; then
  echo "WARNING: SDQM_MAP_VALUE chưa được điền." >&2
  echo "  SDQM history sẽ không ghi mAP và regression sẽ bị bỏ qua." >&2
  echo "  Điền SDQM_MAP_VALUE vào script rồi submit lại để có đủ dữ liệu." >&2
  echo ""
fi

cd "$ROOT_DIR"

# ── Giai đoạn 1: Sinh ảnh ─────────────────────────────────
echo "[$(date +%H:%M:%S)] === Bắt đầu sinh ảnh (main.py) ==="
"$VENV_PYTHON" main.py
echo "[$(date +%H:%M:%S)] === Sinh ảnh hoàn tất ==="
echo ""

# ── Giai đoạn 2: Đánh giá + SDQM ─────────────────────────
echo "[$(date +%H:%M:%S)] === Bắt đầu đánh giá (main_evaluation.py) ==="
"$VENV_PYTHON" main_evaluation.py
echo "[$(date +%H:%M:%S)] === Đánh giá hoàn tất ==="
echo ""

# ── Tổng kết và hướng dẫn download ───────────────────────
echo "======================================================"
echo " HOÀN TẤT! Download các file sau về máy:"
echo "======================================================"
OUTPUT="$AEROEYES_OUTPUT_DIR/output"
echo "  scp $USER@slurm.uit.edu.vn:$OUTPUT/evaluation_report.csv ."
echo "  scp $USER@slurm.uit.edu.vn:$OUTPUT/evaluation_summary.json ."
echo "  scp $USER@slurm.uit.edu.vn:$OUTPUT/evaluation_metadata.jsonl ."
echo "  scp -r $USER@slurm.uit.edu.vn:$OUTPUT/sdqm/ ./sdqm_run2/"
echo "======================================================"
