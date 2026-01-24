#!/bin/bash

# 运行基于RL方法的BaseLine实验，在hangzhou路网，synthetic车流数据集上进行强化学习模型的训练
# 包括主流的RL方法，mplight,attendlight,presslight,colight,efficient-colight,advanced-colight
# 保留训练结果，记录各个训练记录的日志和模型参数

set -e

# ==================== 配置参数 ====================
export CUDA_VISIBLE_DEVICES=""
DATASET="hangzhou"
TRAFFIC_FILE="synthetic_8000_1h.json"  # synthetic数据集
PROJ_NAME="RL_Baseline_Hangzhou_Synthetic"
DURATION=15  # 每个phase的执行时间(秒)
NUM_GENERATORS=1  # 生成器数量
PARALLEL_JOBS=6  # 并行运行的方法数（根据GPU/CPU资源调整）
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="./logs/RL_Baseline_${TIMESTAMP}"
TEMP_DIR="/tmp/rl_baseline_${TIMESTAMP}"
mkdir -p "${TEMP_DIR}"

# ==================== 创建日志目录 ====================
mkdir -p "${LOG_DIR}"
echo "Baseline RL Methods Experiment" > "${LOG_DIR}/experiment_summary.log"
echo "Started at: $(date)" >> "${LOG_DIR}/experiment_summary.log"
echo "Dataset: ${DATASET}" >> "${LOG_DIR}/experiment_summary.log"
echo "Traffic File: ${TRAFFIC_FILE}" >> "${LOG_DIR}/experiment_summary.log"
echo "===============================================" >> "${LOG_DIR}/experiment_summary.log"

# ==================== 定义运行的RL方法 ====================
declare -a RL_METHODS=(
    "MPLight|run_mplight.py"
    "AttendLight|run_attendlight.py"
    "PressLight|run_presslight.py"
    "CoLight|run_colight.py"
    "EfficientColight|run_efficient_colight.py"
    "AdvancedColight|run_advanced_colight.py"
)

# ==================== 运行每个RL方法（并行版本）====================
TOTAL_METHODS=${#RL_METHODS[@]}
SUCCESS_COUNT=0
FAILED_METHODS=()
RUNNING_PIDS=()
RUNNING_METHODS=()
RESULTS_FILE="${TEMP_DIR}/results_tracking.txt"

echo "开始并行运行 RL 方法 (最多 ${PARALLEL_JOBS} 个并行任务)"
echo ""

# 创建一个函数来运行单个方法
run_method() {
    local METHOD_NAME=$1
    local SCRIPT_NAME=$2
    local METHOD_LOG=$3
    
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${START_TIME}] 开始训练 ${METHOD_NAME}" | tee -a "${LOG_DIR}/experiment_summary.log"
    
    # 运行对应的脚本，捕获返回值
    if python "${SCRIPT_NAME}" \
        --dataset "${DATASET}" \
        --traffic_file "${TRAFFIC_FILE}" \
        --proj_name "${PROJ_NAME}" \
        --duration "${DURATION}" \
        --gen "${NUM_GENERATORS}" \
        --memo "${METHOD_NAME}" \
        2>&1 | tee "${METHOD_LOG}"; then
        
        END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[${END_TIME}] ✓ ${METHOD_NAME} 训练完成" | tee -a "${LOG_DIR}/experiment_summary.log"
        
        # 从日志中提取最后的指标
        if grep -q "test_avg_travel_time_over" "${METHOD_LOG}"; then
            ATT=$(grep "test_avg_travel_time_over" "${METHOD_LOG}" | tail -1 | grep -oP ":\s*\K[0-9.]+")
            AWT=$(grep "test_avg_waiting_time_over" "${METHOD_LOG}" | tail -1 | grep -oP ":\s*\K[0-9.]+")
            echo "  ATT: ${ATT}, AWT: ${AWT}" >> "${LOG_DIR}/experiment_summary.log"
        fi
        
        echo "SUCCESS:${METHOD_NAME}" >> "${RESULTS_FILE}"
    else
        END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[${END_TIME}] ✗ ${METHOD_NAME} 训练失败" | tee -a "${LOG_DIR}/experiment_summary.log"
        echo "FAILED:${METHOD_NAME}" >> "${RESULTS_FILE}"
    fi
}

# 导出函数和变量供后台进程使用
export -f run_method
export DATASET TRAFFIC_FILE PROJ_NAME DURATION NUM_GENERATORS LOG_DIR TEMP_DIR RESULTS_FILE

# 遍历所有方法，并行运行
for ((i=0; i<${TOTAL_METHODS}; i++)); do
    IFS='|' read -r METHOD_NAME SCRIPT_NAME <<< "${RL_METHODS[$i]}"
    CURRENT=$((i+1))
    
    METHOD_LOG="${LOG_DIR}/${METHOD_NAME}_${TIMESTAMP}.log"
    
    echo "═══════════════════════════════════════════════════════════"
    echo "  ⏳ [${CURRENT}/${TOTAL_METHODS}] 准备启动 ${METHOD_NAME}"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # 在后台运行方法，但先检查是否已达到并行限制
    while [ $(jobs -r | wc -l) -ge ${PARALLEL_JOBS} ]; do
        sleep 10
        # 清理已完成的后台任务
        wait -n 2>/dev/null || true
    done
    
    # 启动新任务
    run_method "${METHOD_NAME}" "${SCRIPT_NAME}" "${METHOD_LOG}" &
    RUNNING_PIDS+=($!)
    RUNNING_METHODS+=("${METHOD_NAME}")
done

# 等待所有后台任务完成
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "所有任务已启动，正在等待完成..."
echo "═══════════════════════════════════════════════════════════"
echo ""

for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# 统计结果
if [ -f "${RESULTS_FILE}" ]; then
    SUCCESS_COUNT=$(grep -c "SUCCESS:" "${RESULTS_FILE}" 2>/dev/null || echo 0)
    FAILED_COUNT=$(grep -c "FAILED:" "${RESULTS_FILE}" 2>/dev/null || echo 0)
    
    if [ ${FAILED_COUNT} -gt 0 ]; then
        FAILED_METHODS=($(grep "FAILED:" "${RESULTS_FILE}" | cut -d: -f2))
    fi
fi

# ==================== 生成总结报告 ====================
echo "" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "═══════════════════════════════════════════════════════════" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "实验总结" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "═══════════════════════════════════════════════════════════" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "成功完成: ${SUCCESS_COUNT}/${TOTAL_METHODS}" | tee -a "${LOG_DIR}/experiment_summary.log"

if [ ${#FAILED_METHODS[@]} -gt 0 ]; then
    echo "失败的方法:" | tee -a "${LOG_DIR}/experiment_summary.log"
    for method in "${FAILED_METHODS[@]}"; do
        echo "  - ${method}" | tee -a "${LOG_DIR}/experiment_summary.log"
    done
fi

echo "结束时间: $(date)" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "日志目录: ${LOG_DIR}" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "模型保存目录: ./model/" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "训练记录目录: ./records/" | tee -a "${LOG_DIR}/experiment_summary.log"
echo "" | tee -a "${LOG_DIR}/experiment_summary.log"

# ==================== 清理临时文件 ====================
rm -rf "${TEMP_DIR}"


