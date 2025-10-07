
export PYTHONPATH=/home/zzheng3/AmazonReviews2023

# DATASET=Musical_Instruments
# export MASTER_PORT=12346
# export CUDA_VISIBLE_DEVICES=2

# DATASET=Sports_and_Outdoors
# export MASTER_PORT=12344
# export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=12366
export CUDA_VISIBLE_DEVICES=$2

# DATASET=Gift_Cards
# export MASTER_PORT=12347
# export CUDA_VISIBLE_DEVICES=3

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled


# 运行命令
cd /home/zzheng3/AmazonReviews2023
python GRPO/main_trl.py \
    --use_hf_local \
    --dataset $1 \
    --data_path dataset \
    --do_train \
    --use_vllm \
    --hf_model meta-llama/Llama-3.2-1B-Instruct

# PARALLEL_SIZE=4
# accelerate launch --config_file ref/grpo.yaml\
#     GRPO/main_trl.py \
#     --use_hf_local \
#     --dataset $DATASET \
#     --data_path dataset \
#     --parallel_size $PARALLEL_SIZE \
#     --do_train \
#     --use_vllm \
#     --hf_model Qwen/Qwen2.5-0.5B