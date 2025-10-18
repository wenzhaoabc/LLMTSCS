#!/bin/bash

# 定义数据集和对应的交通文件
declare -A runs
runs["jinan"]="anon_3_4_jinan_real.json anon_3_4_jinan_real_2000.json anon_3_4_jinan_real_2500.json"
runs["hangzhou"]="anon_4_4_hangzhou_real.json anon_4_4_hangzhou_real_5816.json"

# 创建一个目录来存放日志文件
LOG_DIR="fixedtime_logs"
mkdir -p $LOG_DIR

# 遍历所有组合并执行
for dataset in "${!runs[@]}"; do
  for traffic_file in ${runs[$dataset]}; do
    # 为日志文件生成一个描述性的名字，包含数据集、文件名和时间戳
    log_file="${LOG_DIR}/${dataset}_${traffic_file%.json}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Running: dataset=${dataset}, traffic_file=${traffic_file}"
    echo "Log will be saved to: ${log_file}"
    
    # 执行命令，并将标准输出和错误输出都重定向到日志文件
    python run_fixedtime.py \
      --dataset "$dataset" \
      --traffic_file "$traffic_file" \
      --proj_name "TSCS_FIXED" > "$log_file" 2>&1
      
    echo "Finished. See log for details."
    echo "---------------------------------"
  done
done

echo "All fixed time runs completed."


