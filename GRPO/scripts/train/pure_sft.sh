#!/bin/bash

export PYTHONPATH=/data/sjc4fq/ColdRec/AmazonReviews2023

# DATASET=Musical_Instruments
# export MASTER_PORT=12346
# export CUDA_VISIBLE_DEVICES=2

# DATASET=Sports_and_Outdoors
# export MASTER_PORT=12344
# export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=12366
# export CUDA_VISIBLE_DEVICES=4

# DATASET=Gift_Cards
# export MASTER_PORT=12347
# export CUDA_VISIBLE_DEVICES=3

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# Usage: ./pure_sft.sh <dataset_name> <gpu_id>
# Example: ./pure_sft.sh Amazon_All_Beauty 0

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset_name> <gpu_id>"
    echo "Example: $0 Amazon_All_Beauty 0"
    exit 1
fi

# 运行命令
cd /data/sjc4fq/ColdRec/AmazonReviews2023
export CUDA_VISIBLE_DEVICES=$2
profile_cutoff=20
model_name=meta-llama/Llama-3.2-1B-Instruct
final_k=5
# model_name=microsoft/deberta-v3-base
# model_name=mistralai/Ministral-3-3B-Base-2512

models="Pop SimpleX LightGCN"

# echo "================================================"
# echo "Dataset: $1"
# echo "GPU: $2"
# echo "Model: $model_name"
# echo "================================================"

echo "================================================"
echo "Generating pure SFT data..."
echo "================================================"
python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --gen_sft_test \
    --final_k $final_k \
    --seed 42 \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \
    --gen_sft_train \
    --gen_sft_eval \

echo "================================================"
echo "Training pure SFT model..."
echo "================================================"
python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --do_sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 500 \
    --eval_steps 500 \
    --max_length 1536 \
    --final_k $final_k \
    --seed 42 \
    --gradient_checkpointing \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \

echo "================================================" 
echo "Testing pure SFT model..."
echo "================================================"
python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --do_test_sft \
    --final_k $final_k \
    --seed 42 \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff

echo "================================================"
echo "Pure SFT training completed!"
echo "================================================"
