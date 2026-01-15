# CEM-Based Multi-Channel Fusion

This directory contains the implementation of Cross-Entropy Method (CEM) optimization for multi-channel recommendation fusion.

## Quick Start

### 1. Test CEM with Synthetic Data

Run the standalone demo to see how CEM works:

```bash
cd /data/sjc4fq/ColdRec/AmazonReviews2023
python GRPO/test_cem_demo.py
```

This will:
- Generate synthetic user data with 3 recommendation channels
- Run CEM optimization to find optimal fusion weights
- Compare with uniform baseline
- Show convergence plot (if matplotlib is available)

### 2. Use CEM in Real Experiments

Integrate CEM evaluation into your baseline experiments:

```bash
python GRPO/baseline_cem.py \
    --dataset Games \
    --data_path ~/data \
    --checkpoint_dir saved \
    --recbole_models BPR LightGCN SASRec \
    --model_name google/flan-t5-base \
    --do_test_sft \
    --final_k 50
```

When testing (`--do_test_sft` or `--do_test_grpo`), the script will automatically:
1. Run standard evaluation (classification accuracy, single-recaller selection)
2. Run multi-channel evaluation (softmax-weighted, uniform-weighted)
3. **Run CEM optimization** and compare results

## Files

### Core Implementation

- **`cem_utils.py`**: CEM optimization algorithms
  - `cem_optimize_fusion_weights()`: Main CEM optimizer
  - `dirichlet_sample()`: Sample weights from Dirichlet distribution
  - `dirichlet_mle_alpha()`: MLE estimation of Dirichlet parameters
  - `fuse_by_quota()`: Quota-based fusion of candidate lists
  - `recall_at_L()`: Compute Recall@L metric

### Integration

- **`baseline_cem.py`**: Main baseline script with CEM integration
  - `evaluate_cem_fusion()`: Wrapper for running CEM on test data
  - Integrated in test section (line ~660)
  - Enhanced summary output with CEM comparison

### Documentation

- **`CEM_INTEGRATION.md`**: Detailed integration guide
- **`test_cem_demo.py`**: Standalone demo with synthetic data

## How CEM Works

```
Initialize: α ~ Uniform(K)  # Dirichlet parameters
For t = 1 to T:
    1. Sample W ~ Dirichlet(α)  # Sample P weight vectors
    2. For each w in W:
         - Fuse candidates using w
         - Compute Recall@L
    3. Select elite E (top 10% by recall)
    4. Fit α_new via MLE on elite samples
    5. Update: α ← 0.7·α + 0.3·α_new  # EMA smoothing
Return: best weight vector w*
```

**Key advantage**: Directly optimizes for Recall@L without requiring gradients!

## Parameters

### CEM-specific
- `cem_iters` (default: 20): Number of CEM iterations
- `cem_population` (default: 256): Population size per iteration
- `cem_elite_frac` (default: 0.1): Elite fraction (top 10%)
- `alpha_smooth` (default: 0.7): EMA smoothing factor for α

### Fusion-specific
- `final_k` (default: 50): Number of items to recommend (L)
- `M` (default: `final_k * 3`): Candidates per channel

## Expected Output

```
==============================================================
CEM-Based Fusion Optimization
==============================================================
Processing 1000 users with 3 recallers
Building candidate lists (M=150 per channel)...
Running CEM optimization (iters=20, pop=256)...

[CEM] iter=00 mean=0.3542 elite_best=0.4123 global_best=0.4123
[CEM] iter=05 mean=0.3912 elite_best=0.4256 global_best=0.4256
[CEM] iter=10 mean=0.4078 elite_best=0.4289 global_best=0.4289
[CEM] iter=15 mean=0.4145 elite_best=0.4298 global_best=0.4298
[CEM] iter=19 mean=0.4156 elite_best=0.4298 global_best=0.4298

==============================================================
CEM Optimization Results
==============================================================
Best Recall@50: 0.4298

Optimized Weights:
  bpr: 0.2341
  lightgcn: 0.4523
  sasrec: 0.3136

Optimized Fusion Performance:
  NDCG@50: 0.3876
  Recall@50: 0.4298

==============================================================
CEM-Optimized Fusion vs. Other Methods
==============================================================
Comparison (NDCG@50):
  Softmax-weighted fusion: 0.3654
  Uniform-weighted fusion: 0.3521
  CEM-optimized fusion: 0.3876

CEM improvement over softmax: +6.08%
```

## Troubleshooting

### Out of Memory
- Reduce `cem_population` (try 128 or 64)
- Reduce `M` (candidates per channel)
- Use CPU instead of CUDA: set `device="cpu"` in `evaluate_cem_fusion()`

### Slow Convergence
- Increase `cem_iters` (try 30 or 50)
- Increase `cem_elite_frac` (try 0.15 or 0.2)
- Decrease `alpha_smooth` for faster updates (try 0.5)

### Poor Results
- Ensure candidate lists have good coverage (increase M)
- Check that ground truth items are present in candidates
- Verify recaller quality (run individual recaller evaluation)

## Citation

If you use this CEM implementation, please cite:

```bibtex
@article{rubinstein1999cross,
  title={The cross-entropy method for combinatorial and continuous optimization},
  author={Rubinstein, Reuven Y},
  journal={Methodology and computing in applied probability},
  volume={1},
  number={2},
  pages={127--190},
  year={1999}
}
```

## Contact

For questions or issues, please refer to the documentation in `CEM_INTEGRATION.md`.




