#!/usr/bin/env bash
# ============================================================
# run_age.sh  —  训练 / 测试 / Grad-CAM 一键脚本
# 用法:
#   bash run_age.sh train          # 仅训练
#   bash run_age.sh test           # 仅测试
#   bash run_age.sh gradcam        # 仅 Grad-CAM
#   bash run_age.sh train test     # 训练完接着测试
#   bash run_age.sh train test gradcam  # 全流程
# ============================================================

set -euo pipefail

# ---- 路径配置 -----------------------------------------------
PYTHON="${PYTHON:-python}"                    # 可用 PYTHON=/path/to/python bash run_age.sh 覆盖
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/ubuntu/桌面/SAM2PATH/pelvis/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
TEST_LOG="${LOG_DIR}/test_${TIMESTAMP}.log"
GRADCAM_LOG="${LOG_DIR}/gradcam_${TIMESTAMP}.log"

# ---- GPU 设置 -----------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING=1

# ---- 环境检查 -----------------------------------------------
echo "=================================================="
echo "  Python  : $("${PYTHON}" --version 2>&1)"
echo "  PyTorch : $("${PYTHON}" -c 'import torch; print(torch.__version__)')"
echo "  CUDA    : $("${PYTHON}" -c 'import torch; print(torch.version.cuda)')"
echo "  GPU     : $("${PYTHON}" -c \
    'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo "  CWD     : ${SCRIPT_DIR}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=================================================="

cd "${SCRIPT_DIR}"

# ---- 工具函数 -----------------------------------------------
run_train() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ▶ 开始训练 (log → ${TRAIN_LOG})"
    "${PYTHON}" -u main_age.py 2>&1 | tee "${TRAIN_LOG}"
    echo "[$(date '+%H:%M:%S')] ✔ 训练完成"
}

run_test() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ▶ 开始测试 (log → ${TEST_LOG})"
    "${PYTHON}" -u test.py 2>&1 | tee "${TEST_LOG}"
    echo "[$(date '+%H:%M:%S')] ✔ 测试完成"
}

run_gradcam() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ▶ 开始生成 Grad-CAM (log → ${GRADCAM_LOG})"
    "${PYTHON}" -u gradcam.py 2>&1 | tee "${GRADCAM_LOG}"
    echo "[$(date '+%H:%M:%S')] ✔ Grad-CAM 完成"
}

# ---- 参数解析 -----------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "用法: bash run_age.sh [train] [test] [gradcam]"
    echo "  train    — 5-fold 交叉验证训练"
    echo "  test     — 测试集评估"
    echo "  gradcam  — 生成 Grad-CAM 热力图"
    exit 0
fi

for STAGE in "$@"; do
    case "${STAGE}" in
        train)   run_train   ;;
        test)    run_test    ;;
        gradcam) run_gradcam ;;
        *)
            echo "未知参数: ${STAGE}  (可选: train | test | gradcam)"
            exit 1
            ;;
    esac
done

echo ""
echo "=================================================="
echo "  全部任务完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
