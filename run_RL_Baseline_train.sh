#!/bin/bash
# filepath: /home/wen/tr/LLMTSCS/run_RL_Baseline_train.sh
# 
# RL Baseline 并行训练脚本（支持 Hangzhou 4x4 与 Jinan 3x4）
# 
# 用法：
#   bash run_RL_Baseline_train.sh hangzhou 100          # Hangzhou 100轮
#   bash run_RL_Baseline_train.sh jinan 100              # Jinan 100轮
#   bash run_RL_Baseline_train.sh hangzhou              # 默认100轮
#

set -e

# ==================== 配置参数 ====================
DATASET=${1:-"hangzhou"}
NUM_ROUNDS=${2:-100}
PROJ_NAME="RL_Baseline_Train"
DURATION=15
MAX_PARALLEL_JOBS=8              # 最多并行训练8个方法

# ==================== 按数据集设置路网与流量文件 ====================
if [ "${DATASET}" == "hangzhou" ]; then
  ROAD_NET="4_4"
  TRAFFIC_FILE="synthetic_8000_1h.json"
  DATA_DIR="./data/Hangzhou/${ROAD_NET}"
elif [ "${DATASET}" == "jinan" ]; then
  ROAD_NET="3_4"
  TRAFFIC_FILE="synthetic_8000_1h.json"
  DATA_DIR="./data/Jinan/${ROAD_NET}"
else
  echo "❌ 不支持的数据集: ${DATASET}（仅支持 hangzhou / jinan）"
  echo "用法: bash $0 [hangzhou|jinan] [num_rounds]"
  exit 1
fi

# 检查训练流量文件
if [ ! -f "${DATA_DIR}/${TRAFFIC_FILE}" ]; then
  echo "❌ 训练流量文件不存在: ${DATA_DIR}/${TRAFFIC_FILE}"
  exit 1
fi

# ==================== 创建日志目录 ====================
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="./logs/train_${DATASET}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# 统计信息
SUMMARY_FILE="${LOG_DIR}/training_summary.txt"
PID_FILE="${LOG_DIR}/pids.txt"

# ==================== 打印开始信息 ====================
cat > "${SUMMARY_FILE}" << EOF
═══════════════════════════════════════════════════════════
  RL Baseline 并行训练开始
═══════════════════════════════════════════════════════════

配置信息:
  数据集: ${DATASET}
  路网规模: ${ROAD_NET}
  训练流量: ${TRAFFIC_FILE}
  训练轮数: ${NUM_ROUNDS}
  最大并行数: ${MAX_PARALLEL_JOBS}
  日志目录: ${LOG_DIR}
  开始时间: $(date '+%Y-%m-%d %H:%M:%S')

═══════════════════════════════════════════════════════════
EOF

cat "${SUMMARY_FILE}"

# ==================== 定义8个方法的训练函数 ====================

train_mplight() {
  local METHOD="MPLight"
  local SCRIPT="run_mplight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "MPLight" \
    --mod "EfficientMPLight" \
    --model "MPLight" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_colight() {
  local METHOD="Colight"
  local SCRIPT="run_colight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "Colight" \
    --mod "EfficientColight" \
    --model "Colight" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_attendlight() {
  local METHOD="Attend"
  local SCRIPT="run_attendlight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "AttendLight" \
    --mod "Attend" \
    --model "AttendLight" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_efficient_presslight() {
  local METHOD="EfficientPressLight"
  local SCRIPT="run_efficient_presslight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "${METHOD}" \
    --mod "${METHOD}" \
    --model "${METHOD}" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_efficient_mplight() {
  local METHOD="EfficientMPLight"
  local SCRIPT="run_efficient_mplight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "${METHOD}" \
    --mod "${METHOD}" \
    --model "${METHOD}" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_efficient_colight() {
  local METHOD="EfficientColight"
  local SCRIPT="run_efficient_colight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "${METHOD}" \
    --mod "${METHOD}" \
    --model "${METHOD}" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_advanced_colight() {
  local METHOD="AdvancedColight"
  local SCRIPT="run_advanced_colight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "${METHOD}" \
    --mod "${METHOD}" \
    --model "${METHOD}" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

train_advanced_mplight() {
  local METHOD="AdvancedMPLight"
  local SCRIPT="run_advanced_mplight.py"
  local LOG_FILE="${LOG_DIR}/${METHOD}.log"
  
  echo "[$(date '+%H:%M:%S')] 【${METHOD}】训练开始（PID: $$）" | tee -a "${LOG_FILE}"
  
  python "${SCRIPT}" \
    --memo "AdvancedMPLight" \
    --mod "AdvancedMPLight" \
    --model "AdvancedMplight" \
    --proj_name "${PROJ_NAME}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --duration "${DURATION}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】训练完成" >> "${LOG_FILE}"
  else
    echo "[$(date '+%H:%M:%S')] ✗ 【${METHOD}】训练失败 (EXIT CODE: ${EXIT_CODE})" >> "${LOG_FILE}"
  fi
  return ${EXIT_CODE}
}

# ==================== 并行训练管理器 ====================

# 启动所有训练任务
echo ""
echo "════════════════════════════════════════════════════"
echo "【开始并行训练】"
echo "════════════════════════════════════════════════════"
echo ""

rm -f "${PID_FILE}"

# 后台启动每个训练
train_mplight &
echo $! >> "${PID_FILE}"
sleep 2  # 稍微错开启动时间，避免瞬间 I/O 拥堵

train_colight &
echo $! >> "${PID_FILE}"
sleep 2

train_attendlight &
echo $! >> "${PID_FILE}"
sleep 2

train_efficient_presslight &
echo $! >> "${PID_FILE}"
sleep 2

train_efficient_mplight &
echo $! >> "${PID_FILE}"
sleep 2

train_efficient_colight &
echo $! >> "${PID_FILE}"
sleep 2

train_advanced_colight &
echo $! >> "${PID_FILE}"
sleep 2

train_advanced_mplight &
echo $! >> "${PID_FILE}"

# ==================== 等待所有任务完成 ====================

echo "【监控训练进度】"
echo ""

# 读取所有 PID
declare -a PIDS
mapfile -t PIDS < "${PID_FILE}"

# 定期监控
FAILED_JOBS=()
COMPLETED_JOBS=0
TOTAL_JOBS=${#PIDS[@]}

while [ ${COMPLETED_JOBS} -lt ${TOTAL_JOBS} ]; do
  COMPLETED_JOBS=0
  
  for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    
    if ! kill -0 ${PID} 2>/dev/null; then
      COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
      
      # 检查退出状态
      wait ${PID} 2>/dev/null
      EXIT_CODE=$?
      
      if [ ${EXIT_CODE} -ne 0 ]; then
        FAILED_JOBS+=(${PID})
      fi
    fi
  done
  
  # 打印进度
  RUNNING=$((TOTAL_JOBS - COMPLETED_JOBS))
  echo "[$(date '+%H:%M:%S')] 训练中... 运行中: ${RUNNING}/${TOTAL_JOBS} | 已完成: ${COMPLETED_JOBS}/${TOTAL_JOBS}"
  
  sleep 30
done

# ==================== 最终统计 ====================

echo ""
echo "════════════════════════════════════════════════════"
echo "【所有训练完成】"
echo "════════════════════════════════════════════════════"
echo ""

# 收集结果
SUCCESS_COUNT=0
FAILED_COUNT=0

for LOG_FILE in ${LOG_DIR}/*.log; do
  if [ -f "${LOG_FILE}" ]; then
    if grep -q "✓ 【.*】训练完成" "${LOG_FILE}"; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    elif grep -q "✗ 【.*】训练失败" "${LOG_FILE}"; then
      FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
  fi
done

# 生成最终报告
cat >> "${SUMMARY_FILE}" << EOF

完成时间: $(date '+%Y-%m-%d %H:%M:%S')

训练统计:
  总方法数: 8
  成功: ${SUCCESS_COUNT}
  失败: ${FAILED_COUNT}

详细日志:
EOF

for LOG_FILE in ${LOG_DIR}/*.log; do
  if [ -f "${LOG_FILE}" ]; then
    METHOD_NAME=$(basename "${LOG_FILE}" .log)
    if grep -q "✓ 【.*】训练完成" "${LOG_FILE}"; then
      echo "  ✓ ${METHOD_NAME}" >> "${SUMMARY_FILE}"
    else
      echo "  ✗ ${METHOD_NAME}" >> "${SUMMARY_FILE}"
    fi
  fi
done

echo ""
cat "${SUMMARY_FILE}"
echo ""
echo "═════════════════════════════════════════════════════════════"
echo "✅ 训练完成！详细日志请查看: ${LOG_DIR}/"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "📊 汇总信息:"
echo "  - 成功: ${SUCCESS_COUNT}/8"
echo "  - 失败: ${FAILED_COUNT}/8"
echo "  - 日志目录: ${LOG_DIR}"
echo ""
