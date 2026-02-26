#!/bin/bash

# RL Baseline 评测脚本（支持 Hangzhou 和 Jinan）
# 
# 用法：
#   bash run_RL_Baseline_test.sh hangzhou
#   bash run_RL_Baseline_test.sh jinan
#
# 该脚本对齐了 run_RL_Baseline_train.sh 的并行结构，并支持以下模型：
# AdvancedColight, Colight, EfficientPressLight, AdvancedMPLight, 
# EfficientColight, MPLight, AttendLight, EfficientMPLight, PressLight

set -e

# ==================== 配置参数 ====================
DATASET=${1:-"hangzhou"}
PROJ_NAME="RL_Baseline_Test"
DURATION=15 # 绿灯相位的持续时长

# 9 个 RL 方法
# declare -a METHODS=(
#     "MPLight"
#     "Colight"
#     "AttendLight"
#     "EfficientPressLight"
#     "EfficientMPLight"
#     "EfficientColight"
#     "AdvancedColight"
#     "AdvancedMPLight"
# )

declare -a METHODS=(
    "AttendLight"
)

# ==================== 根据数据集配置流量文件 ====================
if [ "${DATASET}" == "hangzhou" ]; then
    SOURCE_TRAFFIC="hangzhou_synthetic_8000_1h.json"
    TEST_TRAFFIC_FILES=(
        "anon_4_4_hangzhou_real.json"
        "anon_4_4_hangzhou_real_5816.json"
        "anon_4_4_hangzhou_synthetic_24000_60min.json"
        "synthetic_8000_1h.json"
    )
    ROAD_NET="4_4"
    DATA_DIR="./data/Hangzhou/4_4"
elif [ "${DATASET}" == "jinan" ]; then
    SOURCE_TRAFFIC="jinan_synthetic_8000_1h.json"
    TEST_TRAFFIC_FILES=(
        "anon_3_4_jinan_real.json"
        "anon_3_4_jinan_real_2000.json"
        "anon_3_4_jinan_real_2500.json"
        "anon_3_4_jinan_synthetic_24000_60min.json"
        "synthetic_8000_1h.json"
    )
    ROAD_NET="3_4"
    DATA_DIR="./data/Jinan/3_4"
else
    echo "❌ 错误: 不支持的数据集 '${DATASET}'"
    echo "用法: bash $0 [hangzhou|jinan]"
    exit 1
fi

# ==================== 打印配置信息 ====================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         ${DATASET^^} 模型评测 (并行模式)                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "数据集: ${DATASET}"
echo "路网规模: ${ROAD_NET}"
echo "训练源: ${SOURCE_TRAFFIC}"
echo "评测目标: ${#TEST_TRAFFIC_FILES[@]} 个流量文件"
echo "方法数: ${#METHODS[@]} (${METHODS[*]})"
echo ""

# ==================== 检查数据文件 ====================
if [ ! -f "${DATA_DIR}/${SOURCE_TRAFFIC}" ]; then
    echo "⚠️  警告: 训练源文件不存在: ${DATA_DIR}/${SOURCE_TRAFFIC}"
fi

MISSING_FILES=0
for TARGET in "${TEST_TRAFFIC_FILES[@]}"; do
    if [ ! -f "${DATA_DIR}/${TARGET}" ]; then
        echo "⚠️  警告: 测试文件不存在: ${DATA_DIR}/${TARGET}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ ${MISSING_FILES} -gt 0 ]; then
    read -p "⚠️  发现 ${MISSING_FILES} 个缺失的数据文件，是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ==================== 创建日志目录 ====================
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="./logs/test_${DATASET}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
RESULTS_FILE="${LOG_DIR}/results.csv"
PID_FILE="${LOG_DIR}/pids.txt"

# CSV Header
echo "Method,Dataset,Road_Net,Source,Target,ATT,AWT,Status,Timestamp" > "${RESULTS_FILE}"

# ==================== 定义评测函数 ====================
test_method() {
    local METHOD=$1
    local METHOD_LOG="${LOG_DIR}/${METHOD}.log"
    
    # 默认参数
    local PARAM_MEMO="${METHOD}"
    local PARAM_MOD="${METHOD}"
    local PARAM_MODEL="${METHOD}"

    # 根据方法名调整参数，保持与 train 脚本一致
    case "${METHOD}" in
        "MPLight")
            PARAM_MEMO="MPLight"
            PARAM_MOD="EfficientMPLight"
            PARAM_MODEL="MPLight"
            ;;
        "Colight")
            PARAM_MEMO="Colight"
            PARAM_MOD="EfficientColight"
            PARAM_MODEL="Colight"
            ;;
        "AttendLight")
            PARAM_MEMO="AttendLight"
            PARAM_MOD="Attend"
            PARAM_MODEL="AttendLight"
            ;;
    esac
    
    echo "[$(date '+%H:%M:%S')] 【${METHOD}】评测任务启动 (PID: $$)" | tee -a "${METHOD_LOG}"

    # 检查模型是否存在
    MODEL_DIR="./model_weights/${PARAM_MEMO}"
    if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A ${MODEL_DIR} 2>/dev/null)" ]; then
        echo "  ⚠️  警告: 未找到 ${METHOD} 的模型权重 (${MODEL_DIR})，跳过该方法" | tee -a "${METHOD_LOG}"
        for TARGET in "${TEST_TRAFFIC_FILES[@]}"; do
            echo "${METHOD},${DATASET},${ROAD_NET},${SOURCE_TRAFFIC},${TARGET},N/A,N/A,No Model,$(date '+%Y-%m-%d %H:%M:%S')" >> "${RESULTS_FILE}"
        done
        return 0
    fi

    # 遍历每个测试流量文件
    for TARGET in "${TEST_TRAFFIC_FILES[@]}"; do
        if [ ! -f "${DATA_DIR}/${TARGET}" ]; then
            continue
        fi

        echo "  -> 评测: ${SOURCE_TRAFFIC} -> ${TARGET}" | tee -a "${METHOD_LOG}"
        
        # 独立的日志文件用于 python 输出
        CASE_LOG="${LOG_DIR}/${METHOD}_${TARGET%.*}.log"
        
        # 调用 run_RL_transfer.py
        # 注意：这里假设 run_RL_transfer.py 已经适配了所有的方法名
        python run_RL_transfer.py \
            -memo "${PARAM_MEMO}" \
            -mod "${PARAM_MOD}" \
            -model "${PARAM_MODEL}" \
            -dataset "${DATASET}" \
            -traffic_file_source "${SOURCE_TRAFFIC}" \
            -traffic_file "${TARGET}" \
            -proj_name "${PROJ_NAME}" \
            -duration "${DURATION}" \
            2>&1 | tee "${CASE_LOG}" >> "${METHOD_LOG}"
        
    done
    
    echo "[$(date '+%H:%M:%S')] ✓ 【${METHOD}】所有评测完成" | tee -a "${METHOD_LOG}"
}

# ==================== 启动并行任务 ====================
echo ""
echo "════════════════════════════════════════════════════"
echo "【开始并行评测】"
echo "════════════════════════════════════════════════════"
echo ""

rm -f "${PID_FILE}"

for METHOD in "${METHODS[@]}"; do
    test_method "${METHOD}" &
    echo $! >> "${PID_FILE}"
    # 稍微错开启动时间，避免瞬间 I/O 拥堵
    sleep 1
done

# ==================== 监控进度 ====================
echo ""
echo "【监控评测进度】"

declare -a PIDS
if [ -f "${PID_FILE}" ]; then
    mapfile -t PIDS < "${PID_FILE}"
else
    PIDS=()
fi

TOTAL_JOBS=${#PIDS[@]}
COMPLETED_JOBS=0
declare -a FAILED_JOBS

while [ ${COMPLETED_JOBS} -lt ${TOTAL_JOBS} ]; do
    COMPLETED_JOBS=0
    for PID in "${PIDS[@]}"; do
        if ! kill -0 ${PID} 2>/dev/null; then
            COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
        fi
    done
    
    RUNNING=$((TOTAL_JOBS - COMPLETED_JOBS))
    echo -ne "\r[$(date '+%H:%M:%S')] 评测中... 运行中: ${RUNNING}/${TOTAL_JOBS} | 已完成: ${COMPLETED_JOBS}/${TOTAL_JOBS}"
    sleep 5
done

echo ""
echo ""
echo "════════════════════════════════════════════════════"
echo "【所有评测完成】"
echo "════════════════════════════════════════════════════"

# ==================== 生成报告 ====================
SUMMARY_FILE="${LOG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
═══════════════════════════════════════════════════════════
  ${DATASET^^} ${ROAD_NET} 评测总结
═══════════════════════════════════════════════════════════

评测时间: $(date '+%Y-%m-%d %H:%M:%S')
数据集: ${DATASET}
路网规模: ${ROAD_NET}
训练源: ${SOURCE_TRAFFIC}

方法列表:
$(printf "  - %s\n" "${METHODS[@]}")

详细结果:
