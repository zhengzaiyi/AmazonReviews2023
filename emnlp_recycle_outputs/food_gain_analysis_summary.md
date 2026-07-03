# Food Gain Analysis Summary

- Date: 2026-05-24T23:43:10
- Host: uvavast
- Command: `GRPO/scripts/rebuttal/food_gain_analysis.py --datasets ml-1m steam Food --model_names meta-llama/Llama-3.2-1B-Instruct --recbole_models ItemKNN LightGCN Pop --output_dir emnlp_recycle_outputs --channels pop itemknn lightgcn --metric ndcg@50 --device cuda`
- Output CSV: `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/food_gain_analysis.csv`
- Method: manual RoutePO evaluation with cached per-user predictions; no new RoutePO training.
- Oracle distribution uses fractional credit for ties.

## LaTeX Table

| Dataset | Oracle channel entropy | Routing weight variance | RoutePO N@50 gain |
|---|---:|---:|---:|
| ml-1m | 1.0511 | 0.024836 | 0.021101 |
| steam | 0.6970 | 0.093468 | 0.002817 |
| Food | 0.7150 | 0.146924 | 0.047529 |

## Result Interpretation

Overall, Food shows the clearest RoutePO benefit, ml-1m shows a moderate benefit, and steam is almost saturated by its best single-channel baseline.

| Dataset | Baseline N@50 | RoutePO N@50 | Gain | Relative lift |
|---|---:|---:|---:|---:|
| ml-1m | 0.279250 | 0.300351 | 0.021101 | 7.6% |
| steam | 0.484243 | 0.487061 | 0.002817 | 0.6% |
| Food | 0.411495 | 0.459025 | 0.047529 | 11.6% |

### Why Food Gains the Most

Food has the strongest and most interpretable routing pattern. The oracle distribution is dominated by ItemKNN but still leaves a meaningful minority of users for LightGCN:

- Oracle distribution: pop=6.2%, itemknn=74.0%, lightgcn=19.8%
- RoutePO mean weights: pop=2.4%, itemknn=72.9%, lightgcn=24.7%

This means RoutePO largely learns the right high-level behavior: route most users toward ItemKNN, reserve a smaller but non-trivial portion for LightGCN, and suppress Pop. This alignment explains why Food obtains the largest absolute gain, improving NDCG@50 from 0.411495 to 0.459025.

### Why Steam Gains Little

Steam is dominated by LightGCN. The oracle assigns 77.0% of users to LightGCN, and the best single-channel baseline is already LightGCN with NDCG@50=0.484243. The other two channels are much weaker on average:

- itemknn: 0.231150
- lightgcn: 0.484243
- pop: 0.274135

Because the best single channel is already very strong, personalized routing has limited room to improve. Mixing in weaker channels can help a few users, but it also risks hurting many others. This is why RoutePO only improves steam by 0.002817.

### Why ml-1m Is in the Middle

ml-1m has high oracle diversity:

- Oracle distribution: pop=19.6%, itemknn=38.0%, lightgcn=42.4%
- Normalized oracle entropy: 0.9567

This shows that different users genuinely prefer different channels. However, RoutePO's routing weights are also relatively smooth, with normalized mean weight entropy 0.8918. In other words, the model does not commit as sharply as it does on Food. As a result, ml-1m gets a useful but smaller gain of 0.021101.

### Oracle Headroom

An oracle upper bound was computed by selecting, for each user, the single channel with the highest NDCG@50. This measures how much per-user channel selection could help if routing were perfect.

| Dataset | Best-single baseline | RoutePO | Oracle | Oracle headroom | Headroom captured |
|---|---:|---:|---:|---:|---:|
| ml-1m | 0.279250 | 0.300351 | 0.391264 | 0.112013 | 18.8% |
| steam | 0.484243 | 0.487061 | 0.550141 | 0.065898 | 4.3% |
| Food | 0.411495 | 0.459025 | 0.524019 | 0.112524 | 42.2% |

Food captures the largest fraction of available oracle headroom, which further supports the conclusion that its routing weights are better aligned with per-user channel complementarity. Steam has some oracle headroom in theory, but RoutePO captures little of it because the LightGCN baseline is already dominant and the alternative channels are often risky to mix in.

### Suggested Takeaway

These results suggest that RoutePO's gains are not determined by oracle entropy alone. The gain depends on two conditions:

1. The dataset must have meaningful per-user complementarity across channels.
2. The learned routing weights must align with that complementarity.

Food satisfies both conditions, so it gains the most. ml-1m has strong complementarity but less decisive learned routing, so it gains moderately. steam is mostly controlled by one strong channel, so the routing opportunity is limited.

### Caveats

- This is a manual RoutePO evaluation using cached per-user predictions; no new RoutePO training was run for this analysis.
- The oracle distribution uses fractional credit for ties.
- For stronger paper claims, it would be useful to add paired bootstrap significance tests and report agreement between the RoutePO top-weight channel and the oracle-winning channel.

## Details

### ml-1m

- Users: 1004
- Oracle distribution: pop=19.6%, itemknn=38.0%, lightgcn=42.4%
- Oracle entropy normalized: 0.9567
- RoutePO weight entropy normalized mean: 0.8918
- RoutePO ndcg@50: 0.300351
- Baseline ndcg@50: 0.279250
- Gain: 0.021101

- Prediction cache: `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_ml-1m_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`
- Checkpoint: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`
- Test dataset: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`
- Best single channel: `lightgcn`

### steam

- Users: 2398
- Oracle distribution: pop=10.3%, itemknn=12.6%, lightgcn=77.0%
- Oracle entropy normalized: 0.6344
- RoutePO weight entropy normalized mean: 0.6215
- RoutePO ndcg@50: 0.487061
- Baseline ndcg@50: 0.484243
- Gain: 0.002817

- Prediction cache: `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_steam_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`
- Checkpoint: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`
- Test dataset: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`
- Best single channel: `lightgcn`

### Food

- Users: 1922
- Oracle distribution: pop=6.2%, itemknn=74.0%, lightgcn=19.8%
- Oracle entropy normalized: 0.6508
- RoutePO weight entropy normalized mean: 0.3798
- RoutePO ndcg@50: 0.459025
- Baseline ndcg@50: 0.411495
- Gain: 0.047529

- Prediction cache: `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_Food_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`
- Checkpoint: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`
- Test dataset: `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`
- Best single channel: `itemknn`
