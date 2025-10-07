export PYTHONPATH=/home/zzheng3/AmazonReviews2023
export CUDA_VISIBLE_DEVICES=$2
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# 运行测试命令
cd /home/zzheng3/AmazonReviews2023
python GRPO/main_trl.py \
    --use_hf_local \
    --dataset $1 \
    --data_path dataset \
    --recbole_models BPR ItemKNN FPMC Pop SASRec\
    --do_test_recaller \
    --use_vllm