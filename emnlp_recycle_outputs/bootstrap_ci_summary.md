# Paired Bootstrap CI Summary

- Date: 2026-05-24T20:58:35
- Host: uvavast
- Command: `GRPO/scripts/rebuttal/paired_bootstrap_ci.py --datasets ml-1m steam Food --model_names meta-llama/Llama-3.2-1B-Instruct --recbole_models ItemKNN LightGCN Pop --output_dir emnlp_recycle_outputs --bootstrap_samples 10000 --seed 42`
- Output CSV: `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/bootstrap_ci.csv`
- Method: paired user-level bootstrap over per-user metric deltas.
- Caveat: this measures robustness across users, not multi-seed training variance.

## LaTeX Table

| Dataset | Metric | Compared to | Delta RoutePO | 95% CI | p-value |
|---|---|---|---:|---:|---:|
| ml-1m | ndcg@20 | best_single:lightgcn | 0.024326 | [0.014708, 0.033815] | 0.0000 |
| ml-1m | ndcg@50 | best_single:lightgcn | 0.021101 | [0.012195, 0.029943] | 0.0000 |
| ml-1m | recall@20 | best_single:lightgcn | 0.067995 | [0.054199, 0.082205] | 0.0000 |
| ml-1m | recall@50 | best_single:lightgcn | 0.054714 | [0.041450, 0.068526] | 0.0000 |
| steam | ndcg@20 | best_single:lightgcn | 0.001484 | [-0.002631, 0.005823] | 0.4988 |
| steam | ndcg@50 | best_single:lightgcn | 0.002817 | [-0.001288, 0.007118] | 0.1870 |
| steam | recall@20 | best_single:lightgcn | 0.033723 | [0.026042, 0.041528] | 0.0000 |
| steam | recall@50 | best_single:lightgcn | 0.039116 | [0.031227, 0.047199] | 0.0000 |
| Food | ndcg@20 | best_single:itemknn | 0.048638 | [0.036654, 0.061102] | 0.0000 |
| Food | ndcg@50 | best_single:itemknn | 0.047529 | [0.035992, 0.059491] | 0.0000 |
| Food | recall@20 | best_single:itemknn | 0.121618 | [0.106183, 0.137401] | 0.0000 |
| Food | recall@50 | best_single:itemknn | 0.114299 | [0.100294, 0.128608] | 0.0000 |

## Result Interpretation

Overall, Food and ml-1m show statistically robust RoutePO improvements across all four metrics. Steam shows robust Recall gains, but its NDCG gains are not statistically significant because the 95% confidence intervals cross zero.

| Dataset | NDCG@50 delta | 95% CI | Significant? | Recall@50 delta | 95% CI | Significant? |
|---|---:|---:|---|---:|---:|---|
| ml-1m | 0.021101 | [0.012195, 0.029943] | yes | 0.054714 | [0.041450, 0.068526] | yes |
| steam | 0.002817 | [-0.001288, 0.007118] | no | 0.039116 | [0.031227, 0.047199] | yes |
| Food | 0.047529 | [0.035992, 0.059491] | yes | 0.114299 | [0.100294, 0.128608] | yes |

### Absolute Metrics and Relative Lift

The bootstrap table reports only RoutePO deltas. The corresponding absolute metrics and relative lifts are:

| Dataset | Metric | Baseline | RoutePO | Delta | Relative lift |
|---|---:|---:|---:|---:|---:|
| ml-1m | ndcg@20 | 0.236381 | 0.260707 | 0.024326 | 10.3% |
| ml-1m | ndcg@50 | 0.279250 | 0.300351 | 0.021101 | 7.6% |
| ml-1m | recall@20 | 0.353934 | 0.421929 | 0.067995 | 19.2% |
| ml-1m | recall@50 | 0.496846 | 0.551560 | 0.054714 | 11.0% |
| steam | ndcg@20 | 0.473310 | 0.474794 | 0.001484 | 0.3% |
| steam | ndcg@50 | 0.484243 | 0.487061 | 0.002817 | 0.6% |
| steam | recall@20 | 0.718105 | 0.751828 | 0.033723 | 4.7% |
| steam | recall@50 | 0.758716 | 0.797832 | 0.039116 | 5.2% |
| Food | ndcg@20 | 0.394235 | 0.442873 | 0.048638 | 12.3% |
| Food | ndcg@50 | 0.411495 | 0.459025 | 0.047529 | 11.6% |
| Food | recall@20 | 0.441580 | 0.563198 | 0.121618 | 27.5% |
| Food | recall@50 | 0.503451 | 0.617751 | 0.114299 | 22.7% |

### Food: Strongest Evidence

Food has the strongest and cleanest evidence for RoutePO. All four confidence intervals are strictly positive and far from zero:

- ndcg@20: +0.048638, 95% CI [0.036654, 0.061102]
- ndcg@50: +0.047529, 95% CI [0.035992, 0.059491]
- recall@20: +0.121618, 95% CI [0.106183, 0.137401]
- recall@50: +0.114299, 95% CI [0.100294, 0.128608]

This indicates that RoutePO improves both retrieval coverage and ranking quality on Food. The Recall gains are especially large, while the NDCG gains show that the additional hits are not only appearing at the tail of the top-k list.

### ml-1m: Stable Moderate Gain

ml-1m also shows statistically significant gains on every metric. The NDCG@50 improvement is +0.021101 with 95% CI [0.012195, 0.029943], and Recall@50 improves by +0.054714 with 95% CI [0.041450, 0.068526].

This supports the interpretation that ml-1m has meaningful cross-channel complementarity, but the benefit is smaller than on Food. In relative terms, RoutePO improves NDCG@50 by 7.6% and Recall@50 by 11.0% over the LightGCN best-single baseline.

### Steam: Recall Improves, NDCG Is Not Significant

Steam needs a more careful interpretation. RoutePO significantly improves Recall:

- recall@20: +0.033723, 95% CI [0.026042, 0.041528]
- recall@50: +0.039116, 95% CI [0.031227, 0.047199]

However, the NDCG confidence intervals cross zero:

- ndcg@20: +0.001484, 95% CI [-0.002631, 0.005823], p=0.4988
- ndcg@50: +0.002817, 95% CI [-0.001288, 0.007118], p=0.1870

Thus, for steam, it is safe to claim that RoutePO significantly improves recall, but not that it significantly improves NDCG over the LightGCN best-single baseline. A likely explanation is that RoutePO brings more relevant items into the top-k set, but those added hits are not consistently ranked high enough to produce a robust NDCG improvement.

### Suggested Takeaway

The bootstrap results strengthen the main RoutePO claim for Food and ml-1m: their improvements are not just average gains, but user-level robust gains under paired resampling. For steam, the claim should be metric-specific: RoutePO improves coverage, but its ranking-quality improvement over the already strong LightGCN baseline is not statistically reliable.

### Caveats

- This bootstrap is paired at the user level and measures robustness across users, not multi-seed training variance.
- The p-value is computed from bootstrap mean deltas as `2 * min(P(mean <= 0), P(mean >= 0))`.
- Values printed as p=0.0000 should be interpreted as p < 1e-4 with 10,000 bootstrap samples, not as exactly zero.

## Inputs

- ml-1m ndcg@20: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_ml-1m_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=1004.
- ml-1m ndcg@50: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_ml-1m_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=1004.
- ml-1m recall@20: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_ml-1m_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=1004.
- ml-1m recall@50: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_ml-1m_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/ml-1m/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=1004.
- steam ndcg@20: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_steam_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=2398.
- steam ndcg@50: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_steam_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=2398.
- steam recall@20: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_steam_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=2398.
- steam recall@50: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_steam_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/steam/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `lightgcn`; n=2398.
- Food ndcg@20: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_Food_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `itemknn`; n=1922.
- Food ndcg@50: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_Food_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `itemknn`; n=1922.
- Food recall@20: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_Food_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `itemknn`; n=1922.
- Food recall@50: cache `/data/sjc4fq/ColdRec/AmazonReviews2023/emnlp_recycle_outputs/predictions/manual_predictions_Food_Llama-3.2-1B-Instruct_itemknn_lightgcn_pop_pc500000_ptk3_k50_all.json`; checkpoint `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_ItemKNN_LightGCN_Pop_pc500000`; test data `/data/sjc4fq/ColdRec/AmazonReviews2023/GRPO/data/pure_models/Food/Llama-3.2-1B-Instruct_pure_sft_data_ItemKNN_LightGCN_Pop_500000/test`; best single `itemknn`; n=1922.
