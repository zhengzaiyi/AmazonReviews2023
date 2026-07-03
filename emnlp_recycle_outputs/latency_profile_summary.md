# Latency Profile Summary

- Date: 2026-05-24T22:25:27
- Host: uvavast
- Command: `GRPO/scripts/rebuttal/latency_profile.py --datasets ml-1m steam Food --model_names meta-llama/Llama-3.2-1B-Instruct --recbole_models ItemKNN LightGCN Pop --profile_cutoff 500000 --manual_history_cutoff 20 --batch_size 1 --sample_users 500 --device cuda:1`
- Output CSV: `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/latency_profile.csv`
- Routing: classification forward pass only; no token generation.
- Recall: online recaller calls are timed separately; recaller/model initialization is excluded.
- Prompts: loaded from the saved test dataset; purchase history is manually truncated before tokenization.
- Prompt construction time is not included.

## LaTeX Table

| Model | Dataset | Hist max | Prompt tok. | Recall p50 | Recall p95 | Route p50 | Route p95 | E2E p50 | E2E p95 | E2E Users/s | Peak Mem. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | ml-1m | 20 | 1433.2 | 4.2 | 4.5 | 230.0 | 240.0 | 234.2 | 244.3 | 4.24 | 5.55 |
| Llama-3.2-1B-Instruct | steam | 20 | 964.9 | 5.0 | 5.3 | 120.0 | 239.4 | 125.0 | 244.4 | 6.41 | 5.58 |
| Llama-3.2-1B-Instruct | Food | 20 | 3794.8 | 38.8 | 40.8 | 994.4 | 1052.5 | 1033.5 | 1091.6 | 1.14 | 12.21 |

## Provenance

### Llama-3.2-1B-Instruct / ml-1m

- Checkpoint: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`
- Test dataset: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`
- Hardware: NVIDIA A100 80GB PCIe
- Requested device: cuda:1
- Resolved device: cuda
- Batch size: 1
- Source artifact profile cutoff: 500000
- Manual history cutoff: 20
- Source history length mean/p50/p95/max: 37.0/43.0/46.0/46
- Profiled history length mean/p50/p95/max: 19.7/20.0/20.0/20
- Sample users: 500
- Sample seed: 42
- Recall included: True
- Recall users: 500
- Recall top-k: 50
- Recall channels: itemknn lightgcn pop
- Max length: 2777
- Device note: Mapped requested physical cuda:1 to local cuda via CUDA_VISIBLE_DEVICES=1.

### Llama-3.2-1B-Instruct / steam

- Checkpoint: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`
- Test dataset: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`
- Hardware: NVIDIA A100 80GB PCIe
- Requested device: cuda:1
- Resolved device: cuda
- Batch size: 1
- Source artifact profile cutoff: 500000
- Manual history cutoff: 20
- Source history length mean/p50/p95/max: 16.6/9.0/46.0/46
- Profiled history length mean/p50/p95/max: 11.4/9.0/20.0/20
- Sample users: 500
- Sample seed: 42
- Recall included: True
- Recall users: 413
- Recall top-k: 50
- Recall channels: itemknn lightgcn pop
- Max length: 3228
- Device note: Mapped requested physical cuda:1 to local cuda via CUDA_VISIBLE_DEVICES=1.

### Llama-3.2-1B-Instruct / Food

- Checkpoint: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`
- Test dataset: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`
- Hardware: NVIDIA A100 80GB PCIe
- Requested device: cuda:1
- Resolved device: cuda
- Batch size: 1
- Source artifact profile cutoff: 500000
- Manual history cutoff: 20
- Source history length mean/p50/p95/max: 27.1/27.5/46.0/46
- Profiled history length mean/p50/p95/max: 15.6/20.0/20.0/20
- Sample users: 500
- Sample seed: 42
- Recall included: True
- Recall users: 455
- Recall top-k: 50
- Recall channels: itemknn lightgcn pop
- Max length: 9137
- Device note: Mapped requested physical cuda:1 to local cuda via CUDA_VISIBLE_DEVICES=1.

