#!/bin/bash
# =============================================================================
# Policy Gradient Baseline Experiments
# Run PG-based personalized fusion optimization across datasets and recaller combos
# =============================================================================

set -e  # Exit on error

# Configuration
DATA_PATH="./dataset"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="results/pg"
SEED=42
FINAL_K=50

# PG parameters (from paper Section C.3)
NUM_EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=1e-4
REG_WEIGHT=0.01  # Small value to not dominate PG loss
EMBEDDING_DIM=64
DELTA_MAX=10.0
EPSILON=1e-6
NUM_SAMPLES=1
TOP_K_CHANNEL_ITEMS=10

# User limits (set to empty string for all users)
NUM_TRAIN_USERS=5000
NUM_TEST_USERS=1000

# Device (cuda or cpu)
DEVICE="cuda"

# Datasets to run
DATASETS=("ml-1m" "steam")

# Recaller combinations to test
# Each combination is a space-separated list of models
declare -a RECALLER_COMBOS=(
    "BPR SimpleX LightGCN"
    "BPR SimpleX LightGCN SASRec"
    "BPR LightGCN SASRec"
    "SimpleX LightGCN SASRec"
    "BPR SimpleX SASRec"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/pg_experiments_$(date +%Y%m%d_%H%M%S).log"
echo "Policy Gradient Baseline Experiments" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Counter for experiments
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#RECALLER_COMBOS[@]}))
CURRENT=0

# Run experiments
for DATASET in "${DATASETS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    
    for COMBO in "${RECALLER_COMBOS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # Convert combo string to array
        read -ra MODELS <<< "$COMBO"
        COMBO_NAME=$(IFS="_"; echo "${MODELS[*]}")
        
        echo "" | tee -a "$LOG_FILE"
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: $DATASET with ${MODELS[*]}" | tee -a "$LOG_FILE"
        echo "Time: $(date)" | tee -a "$LOG_FILE"
        echo "-------------------------------------------" | tee -a "$LOG_FILE"
        
        # Build command
        CMD="python GRPO/baselines/baseline_pg.py \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --checkpoint_dir $CHECKPOINT_DIR \
            --output_dir $OUTPUT_DIR \
            --recbole_models ${MODELS[*]} \
            --final_k $FINAL_K \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --reg_weight $REG_WEIGHT \
            --embedding_dim $EMBEDDING_DIM \
            --delta_max $DELTA_MAX \
            --epsilon $EPSILON \
            --num_samples $NUM_SAMPLES \
            --top_k_channel_items $TOP_K_CHANNEL_ITEMS \
            --device $DEVICE \
            --seed $SEED"
        
        # Add user limits if specified
        if [ -n "$NUM_TRAIN_USERS" ]; then
            CMD="$CMD --num_train_users $NUM_TRAIN_USERS"
        fi
        if [ -n "$NUM_TEST_USERS" ]; then
            CMD="$CMD --num_test_users $NUM_TEST_USERS"
        fi
        
        echo "Command: $CMD" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # Run the experiment
        if $CMD 2>&1 | tee -a "$LOG_FILE"; then
            echo "✅ Success: $DATASET - $COMBO_NAME" | tee -a "$LOG_FILE"
        else
            echo "❌ Failed: $DATASET - $COMBO_NAME" | tee -a "$LOG_FILE"
        fi
        
        echo "" | tee -a "$LOG_FILE"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All experiments completed at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Summary of results
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "RESULTS SUMMARY" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

for result_file in "$OUTPUT_DIR"/pg_results_*.json; do
    if [ -f "$result_file" ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "File: $(basename $result_file)" | tee -a "$LOG_FILE"
        # Extract key metrics using python
        python -c "
import json
with open('$result_file') as f:
    data = json.load(f)
    config = data.get('config', {})
    pg = data.get('pg_fusion', {}).get('pg_optimized', {})
    weights = data.get('pg_fusion', {}).get('avg_weights', {})
    print(f\"  Dataset: {config.get('dataset', 'N/A')}\")
    print(f\"  Recallers: {config.get('recaller_combo', 'N/A')}\")
    print(f\"  NDCG@50: {pg.get('ndcg@50', 0):.4f}\")
    print(f\"  Recall@50: {pg.get('recall@50', 0):.4f}\")
    print(f\"  Avg Weights: {weights}\")
" 2>/dev/null || echo "  (Could not parse results)" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Done!" | tee -a "$LOG_FILE"

