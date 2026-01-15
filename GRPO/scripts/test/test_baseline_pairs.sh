#!/bin/bash

# =============================================================================
# test_baseline_pairs.sh - 测试所有recallers两两组合的baseline性能
# =============================================================================
# 
# 用法: ./test_baseline_pairs.sh <dataset> <gpu_id>
#
# 示例:
#   ./test_baseline_pairs.sh ml-1m 0
#   ./test_baseline_pairs.sh Amazon_All_Beauty 1
#
# =============================================================================

export PYTHONPATH=/data/sjc4fq/ColdRec/AmazonReviews2023
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <gpu_id>"
    echo ""
    echo "Examples:"
    echo "  $0 ml-1m 0"
    echo "  $0 Amazon_All_Beauty 1"
    exit 1
fi

DATASET=$1
GPU_ID=$2

cd /data/sjc4fq/ColdRec/AmazonReviews2023
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 基础参数
profile_cutoff=20
final_k=50
seed=42

# 定义所有可用的recallers
ALL_RECALLERS=("BPR" "SASRec" "ItemKNN" "LightGCN" "Pop" "SimpleX")

# 生成所有两两组合
echo "================================================"
echo "Testing All Recaller Pairs Baseline Performance"
echo "================================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Available recallers: ${ALL_RECALLERS[@]}"
echo "================================================"

# 创建结果目录
mkdir -p results

# 生成所有两两组合
pairs=()
for i in "${!ALL_RECALLERS[@]}"; do
    for j in "${!ALL_RECALLERS[@]}"; do
        if [ $i -lt $j ]; then
            pairs+=("${ALL_RECALLERS[$i]} ${ALL_RECALLERS[$j]}")
        fi
    done
done

echo "Total pairs to test: ${#pairs[@]}"
echo ""

# 汇总文件
SUMMARY_FILE="results/baseline_pairs_summary_${DATASET}_$(date +%Y%m%d_%H%M%S).txt"
echo "Baseline Performance for Recaller Pairs - $DATASET" > $SUMMARY_FILE
echo "Started at: $(date)" >> $SUMMARY_FILE
echo "Total pairs: ${#pairs[@]}" >> $SUMMARY_FILE
echo "================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# 测试每个组合
pair_num=0
for combo in "${pairs[@]}"; do
    pair_num=$((pair_num + 1))
    combo_name=$(echo "$combo" | tr ' ' '_')
    
    echo ""
    echo "================================================"
    echo "[$pair_num/${#pairs[@]}] Testing: $combo"
    echo "================================================"
    
    # 运行baseline测试
    python GRPO/models/main_pure.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --recbole_models $combo \
        --test_baseline \
        --final_k $final_k \
        --seed $seed \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff
    
    # 检查结果文件是否存在
    result_file="results/baseline_results_${DATASET}_${combo_name}.json"
    if [ -f "$result_file" ]; then
        echo "✅ Results saved to: $result_file"
        
        # 提取关键指标并写入汇总文件
        if command -v jq &> /dev/null; then
            # 如果有jq，提取NDCG@50和Recall@50
            ndcg50=$(jq -r '.avg_score_weight."ndcg@50" // "N/A"' $result_file 2>/dev/null)
            recall50=$(jq -r '.avg_score_weight."recall@50" // "N/A"' $result_file 2>/dev/null)
            echo "$combo_name: NDCG@50=$ndcg50, Recall@50=$recall50" >> $SUMMARY_FILE
        else
            echo "$combo_name: Results saved (use jq to extract metrics)" >> $SUMMARY_FILE
        fi
    else
        echo "❌ Warning: Result file not found: $result_file"
        echo "$combo_name: FAILED" >> $SUMMARY_FILE
    fi
done

echo ""
echo "================================================"
echo "All pairs tested!"
echo "================================================"
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "Individual results saved to:"
echo "  results/baseline_results_${DATASET}_*.json"
echo ""
