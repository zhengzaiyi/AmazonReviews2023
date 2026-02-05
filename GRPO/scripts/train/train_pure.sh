#!/bin/bash

export PYTHONPATH=/data/sjc4fq/ColdRec/AmazonReviews2023

# DATASET=Musical_Instruments
# export MASTER_PORT=12346
# export CUDA_VISIBLE_DEVICES=2

# DATASET=Sports_and_Outdoors
# export MASTER_PORT=12344
# export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=12368
# export CUDA_VISIBLE_DEVICES=4

# DATASET=Gift_Cards
# export MASTER_PORT=12347
# export CUDA_VISIBLE_DEVICES=3

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT="pure-grpo"

# 运行命令
cd /data/sjc4fq/ColdRec/AmazonReviews2023

PARALLEL_SIZE=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
models="LightGCN SASRec"
final_k=50
model_name=meta-llama/Llama-3.2-1B-Instruct
profile_cutoff=20

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
    --gen_sft_test \

# echo "================================================"
# echo "Running Pure Classification Training: SFT"
# echo "================================================"
# accelerate launch --config_file GRPO/configs/soft_acc.yaml \
#     GRPO/models/main_pure.py \
#     --do_sft \
#     --dataset $1 \
#     --data_path dataset \
#     --model_name $model_name \
#     --output_dir GRPO/data/pure_models \
#     --recbole_models $models \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --warmup_steps 100 \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --eval_steps 500 \
#     --max_length 1536 \
#     --final_k $final_k \
# #     --padding_side left \
#     --seed 42

# echo "================================================"
# echo "Running Pure Classification Training: SFT + GRPO"
# echo "================================================"
# accelerate launch --config_file GRPO/configs/soft_acc.yaml \
#     GRPO/models/main_pure.py \
#     --do_sft \
#     --do_grpo \
#     --dataset $1 \
#     --data_path dataset \
#     --model_name $model_name \
#     --output_dir GRPO/data/pure_models \
#     --recbole_models $models \
#     --final_k $final_k \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --tau_gumbel 1.0 \
#     --top_p 0.9 \
#     --epsilon 0.2 \
#     --beta 0.01 \
#     --num_generations 4 \
#     --grpo_lr 1e-6 \
#     --grpo_epochs 1 \
# #     --seed 42

echo "================================================"
echo "Running Pure Classification Training: GRPO Only"
echo "================================================"
accelerate launch --config_file GRPO/configs/soft_acc.yaml \
    GRPO/models/main_pure.py \
    --do_grpo \
    --dataset $1 \
    --data_path dataset \
    --model_name $model_name \
    --output_dir GRPO/data/pure_models \
    --recbole_models $models \
    --final_k $final_k \
    --logging_steps 10 \
    --save_steps 500 \
    --tau_gumbel 1.0 \
    --top_p 0.9 \
    --noise_scale 0.1 \
    --epsilon 0.2 \
    --beta 0.1 \
    --sync_ref_model \
    --merge_method top_k \
    --ref_model_sync_steps 500 \
    --num_generations 8 \
    --grpo_lr 1e-5 \
    --grpo_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --seed 42

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
    --do_test_grpo \
    --final_k $final_k \
    --seed 42 \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \
    --merge_method top_k    


