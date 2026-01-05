#!/bin/bash

# ================= 配置区域 =================
MODELS=(
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_1"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_2"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_3"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_4"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_5"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_1"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_2"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_3"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_4"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_5"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_1"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_2"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_3"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_4"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_5"
)

INPUT_JSON="/root/codespace/gaozhitao/PSP_bmk/phybench/dataset/PHYBench-questions_v1.json"
GPU=4
PORT=8004
# ============================================

# 创建汇总日志
SUMMARY_LOG="phybench_final_summary.log"
echo "PhyBench Evaluation Summary - $(date)" > $SUMMARY_LOG

for MODEL_PATH in "${MODELS[@]}"; do
    # 解析名称用于定位输出文件
    ROUND_NAME=$(basename "$MODEL_PATH")
    EXP_NAME=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
    RESULT_FILE="phybench_results_${EXP_NAME}_${ROUND_NAME}.jsonl"

    echo "---------------------------------------------------------"
    echo "当前测评模型: $EXP_NAME | $ROUND_NAME"
    
    # 1. 运行推理 (Python 内部会自动起停 vLLM)
    python infer_with_server.py \
        --model_path "$MODEL_PATH" \
        --gpu "$GPU" \
        --port "$PORT" \
        --input_file "$INPUT_JSON"

    # 2. 运行评估并存入汇总日志
    if [ -f "$RESULT_FILE" ]; then
        echo "推理完成，开始计算 EED 指标..."
        python eval.py --input "$RESULT_FILE" >> $SUMMARY_LOG
    else
        echo "错误: 未能生成结果文件 $RESULT_FILE" >> $SUMMARY_LOG
    fi

    # 3. 预防性清理端口
    pkill -f "vllm.*--port $PORT" || true
    sleep 10
done

echo "所有任务已完成。请查看 $SUMMARY_LOG"


# nohup bash auto_phybench.sh > phybench_run.log 2>&1 &