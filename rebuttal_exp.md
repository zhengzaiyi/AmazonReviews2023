# RoutePO EMNLP Experiment Plan

## 0. Mission

This plan is for the experiment/analysis agent. Its job is to produce fast, verified evidence that the writing agent can integrate into the EMNLP/ARR recycle version of RoutePO.

Paper:

- Title: **RoutePO: Personalized Routing Policy Optimization for LLM-based Multi-Channel Recall**
- Original PDF: `C:\Users\donpa\Downloads\1265_RoutePO_Personalized_Rout.pdf`
- Extracted text in current workspace: `C:\Users\donpa\Documents\Codex\2026-05-22\files-mentioned-by-the-user-1265\RoutePO_paper_text.txt`
- OpenReview page: `https://openreview.net/forum?id=SJZu1nN8US#discussion`

Current workspace note:

- This workspace contains only the extracted PDF text.
- First experiment-agent action: locate the actual experiment repo, checkpoints, prediction files, logs, and scripts.

User constraints:

- Do not add new datasets.
- Do not add new recall models.
- Do not narrow the claim.
- Soft-label SFT experiment has already been completed and belongs to the writing/appendix plan; do not rerun soft-label unless result files are unusable.
- All reported values must be traceable to scripts/logs.
- No rushed unstable optional result should be included.

## 1. Reviewer Concerns This Plan Addresses

Primary:

- Latency/cost of LLM routing is not quantified.
- Fixed seed 42 is reported without uncertainty/significance.
- Food gain is much larger than ML-1M/Steam and needs analysis.

Secondary/optional:

- Lightweight router comparison could help answer LLM necessity.
- `tau=1.0` sensitivity could help answer SofT-GRPO hyperparameter concerns.

Do not spend time on:

- New datasets such as Amazon.
- New recall channels such as SASRec/BERT4Rec/BPR/NeuMF/SimpleX.
- Re-running completed soft-label experiments.

## 2. Required Artifact Handoff

For every analysis, produce:

- A machine-readable result file, preferably `.csv` or `.json`.
- A short `.md` summary with exact command, script, date, hardware, input paths, output paths, and interpretation.
- A compact table ready for LaTeX.
- Enough provenance for the writing agent to defend the result.

Recommended output directory in the actual repo:

- `emnlp_recycle_outputs/`

Recommended files:

- `latency_profile.csv`
- `latency_profile_summary.md`
- `bootstrap_ci.csv`
- `bootstrap_ci_summary.md`
- `food_gain_analysis.csv`
- `food_gain_analysis_summary.md`
- `optional_lightweight_router.csv`
- `optional_tau_sensitivity.csv`

## 3. P0: Latency and Cost Profiling

Purpose:

- Answer reviewer concern that LLM routing may be too expensive for recall.

Use existing checkpoints only.

Models:

- Qwen2.5-1.5B-Instruct RoutePO.
- Qwen3-4B-Instruct RoutePO.

Datasets:

- Prefer all three: ML-1M, Steam, Food.
- If time is short, run ML-1M and Food.

Sample size:

- 200 to 500 test users per dataset.
- Use a fixed sample seed and record it.

Measure:

- Average prompt tokens.
- Average generated/output tokens.
- p50 latency.
- p95 latency.
- Mean latency.
- Throughput in users/sec.
- Peak GPU memory.
- Hardware.
- Batch size.
- Decoding settings.
- Whether candidate lists are precomputed.

Recommended schema:

| model | dataset | sample_users | batch_size | prompt_tokens_avg | output_tokens_avg | latency_ms_mean | latency_ms_p50 | latency_ms_p95 | users_per_sec | peak_mem_gb | hardware | decoding | candidate_lists |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|

Minimum LaTeX table columns:

| Model | Dataset | Prompt tok. | Output tok. | p50 Lat. | p95 Lat. | Users/s | Peak Mem. |
|---|---|---:|---:|---:|---:|---:|---:|

Important:

- Time only the LLM routing step unless explicitly measuring end-to-end.
- If end-to-end is measured, separately report retrieval/fusion time and routing time.
- Warm up the model before measuring.
- Record whether measurement uses batch size 1 or batched inference.
- Use the same prompt construction as reported in the paper.

Handoff interpretation:

- State whether 1.5B is close to 4B in quality but cheaper/slower/faster.
- If latency is high, frame it honestly as routing overhead that motivates caching/distillation.

## 4. P0: Paired Bootstrap Confidence Intervals

Purpose:

- Answer fixed-seed and modest-gain concerns without full retraining.

Required input:

- Per-user predictions or per-user metric values for RoutePO and baselines.
- Need at least NDCG@50; include NDCG@20, Recall@20, Recall@50 if easy.

Comparison:

- For each dataset and metric, compare RoutePO against the strongest non-RoutePO baseline in Table 1.
- Use the same model variant as the main claim, preferably RoutePO 4B if Table 1 highlights best results, or RoutePO 1.5B if the paper emphasizes efficiency. Report which one is used.

Known Table 1 strongest non-RoutePO baselines from extracted text:

- ML-1M N@50: Uniform Snake appears strongest at 4.87; Zero-shot LLM is 4.67; PG Fusion is 4.50.
- Steam N@50: LightGCN appears strongest at 5.60; Uniform Snake is 5.29; PG Fusion is 4.75.
- Food N@50: Uniform Snake is 8.94; Zero-shot LLM is 8.86; PG Fusion is 8.07.

Procedure:

1. Compute per-user metric difference: RoutePO minus strongest baseline.
2. Bootstrap users with replacement.
3. Use 10,000 bootstrap samples.
4. Report mean delta, 95% CI, and p-value.
5. Use a fixed bootstrap random seed and record it.

Recommended schema:

| dataset | metric | routepo_variant | baseline | n_users | mean_delta | ci95_low | ci95_high | p_value | bootstrap_samples | seed |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|

Minimum LaTeX table:

| Dataset | Metric | Compared to | Delta RoutePO | 95% CI | p-value |
|---|---|---|---:|---:|---:|

Important wording for handoff:

- This is user-level paired bootstrap, not multi-seed training variance.
- It tests whether observed user-level improvements are robust across users.
- Do not overclaim it as training stochasticity significance.

## 5. P1: Food Gain Analysis

Purpose:

- Explain why Food has much larger NDCG gains.

Use existing predictions and routing weights. No new training.

Compute at least two of:

- Oracle best-channel distribution per dataset.
- Entropy of oracle best-channel distribution.
- RoutePO routing weight variance per dataset.
- RoutePO routing entropy per dataset.
- Gain by user history length bucket.
- Gain by metadata/tag diversity bucket if easy.

Preferred analysis:

1. For each user, compute the best single channel by validation/test NDCG.
2. Compute distribution over POP, ItemKNN, LightGCN.
3. Compute entropy of that distribution.
4. For RoutePO weights, compute per-user entropy and per-channel variance.
5. Compare these statistics across ML-1M, Steam, and Food.

Recommended schema:

| dataset | oracle_pop_pct | oracle_itemknn_pct | oracle_lightgcn_pct | oracle_entropy | routepo_weight_entropy_mean | routepo_weight_variance_mean | routepo_n50_gain |
|---|---:|---:|---:|---:|---:|---:|---:|

Minimum LaTeX table:

| Dataset | Oracle channel entropy | Routing weight variance | RoutePO N@50 gain |
|---|---:|---:|---:|

Interpretation to test:

- Food has more heterogeneous item space and broader user preference patterns.
- Static/global fusion is brittle when channel usefulness varies by user.
- RoutePO benefits more when user-specific routing needs are diverse.

If the statistics do not support this:

- Do not force the explanation.
- Report the most defensible observed pattern.

## 6. P1 Optional: Lightweight Router Sanity Baseline

Timebox:

- Maximum 6 hours.
- Skip if data pipeline is not smooth.
- Do not delay P0 latency or bootstrap.

Purpose:

- Partially answer "why not MLP/GBDT router?"

If feasible:

- Train Logistic Regression or small MLP using existing features only.
- Use the same three recall channels.
- Use the same Snake Fusion.
- Use the same train/validation/test split.

Possible features:

- User history length.
- Channel score summaries.
- Top-K overlap between channels.
- User demographics if already available.
- Simple recaller statistics.

Recommended schema:

| router | uses_language_context | dataset | n50 | r50 | notes |
|---|---|---|---:|---:|---|

Decision rule:

- If the lightweight router is weaker and results are clean, hand it to writing agent for appendix or main text.
- If it is stronger, unstable, or not reproducible, do not include rushed results.
- If skipped, document why.

## 7. P2 Optional: tau Sensitivity

Only do if scripts are already easy to run.

Setup:

- Model: Qwen2.5-1.5B.
- Dataset: one dataset only, preferably Food or ML-1M.
- Values: `tau in {0.5, 1.0, 2.0}`.
- Metrics: NDCG@20/50 and Recall@20/50.

Recommended schema:

| dataset | model | tau | r20 | r50 | n20 | n50 |
|---|---|---:|---:|---:|---:|---:|

If skipped:

- Tell the writing agent to use the default justification sentence:

```tex
We set the Gumbel-Softmax temperature to tau=1.0 as a standard balance between exploration and routing sharpness; a broader sensitivity study is left for future work.
```

<!-- ## 8. Soft-Label Results Handoff

Soft-label SFT experiments are already completed and belong to the writing/appendix plan.

The experiment agent should only help locate existing results if needed.

Do not rerun soft-label unless:

- Existing result files cannot be found, or
- The files are unusable/inconsistent with the current paper split.

If located, provide the writing agent:

- File paths.
- Dataset coverage.
- Exact metrics.
- Whether results are SFT-only, SFT+RL, or both.
- Any caveats. -->

## 9. Experiment Acceptance Criteria

The experiment handoff is ready when:

- Latency/cost profiling has verified numbers or a clear failure note.
- Bootstrap CI has verified numbers or a clear failure note.
- Food gain analysis has verified numbers or a clear failure note.
- Optional lightweight router and tau sensitivity are either cleanly reported or explicitly skipped.
- Every included number has a path to logs/scripts.
- No output table contains `TBD`, placeholders, or guessed values.
- The writing agent receives compact tables plus provenance summaries.

## 10. Final Experiment QA

Before handing results to the writing agent:

- Confirm dataset splits match the paper.
- Confirm method names match Table 1.
- Confirm metric definitions match the paper.
- Confirm whether values are percentages or decimals.
- Confirm random seeds and sample sizes are recorded.
- Confirm hardware and decoding settings are recorded for latency.
- Save all result summaries in a stable output directory.
