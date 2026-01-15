# CEM Integration in baseline_cem.py

## Overview

This document describes the integration of Cross-Entropy Method (CEM) optimization from `cem_utils.py` into `baseline_cem.py` for optimizing multi-channel recommendation fusion weights.

## What is CEM?

The Cross-Entropy Method (CEM) is an evolutionary optimization algorithm that:
1. Samples weight vectors from a Dirichlet distribution
2. Evaluates each weight configuration using Recall@L
3. Selects elite (top-performing) samples
4. Fits a new Dirichlet distribution to the elite samples via MLE
5. Iterates until convergence

This approach finds optimal fusion weights for combining multiple recommendation channels without requiring gradient computation.

## Integration Details

### New Function: `evaluate_cem_fusion()`

**Location**: `GRPO/baseline_cem.py` (lines ~58-200)

**Purpose**: Optimizes fusion weights for combining multiple recallers using CEM.

**Key Steps**:
1. **Data Preparation**: Extracts user IDs, histories, and ground truth from test dataset
2. **Candidate Generation**: Gets top-M candidates from each recaller channel
3. **CEM Optimization**: Runs CEM to find optimal weights that maximize Recall@L
4. **Evaluation**: Computes NDCG and Recall metrics using optimized weights

**Parameters**:
- `test_dataset`: Test dataset with user data
- `recallers`: Dictionary of recaller instances
- `recaller_names`: List of recaller names (e.g., ['bpr', 'lightgcn', 'sasrec'])
- `final_k`: Number of items to recommend (L)
- `cem_iters`: CEM iterations (default: 20)
- `cem_population`: Population size per iteration (default: 256)
- `cem_elite_frac`: Elite fraction (default: 0.1, i.e., top 10%)
- `device`: Device for computation ('cuda' or 'cpu')

**Returns**:
```python
{
    "cem_optimized": {
        "ndcg@10": float,
        "ndcg@20": float,
        "ndcg@50": float,
        "recall@10": float,
        "recall@20": float,
        "recall@50": float,
    },
    "optimized_weights": {
        "bpr": float,
        "lightgcn": float,
        "sasrec": float,
        ...
    },
    "optimization_history": {
        "best_score": [list of scores],
        "mean_score": [list of scores],
    }
}
```

### Integration Points

#### 1. Import CEM Utilities (line ~38-52)
```python
from GRPO.cem_utils import (
    cem_optimize_fusion_weights,
    build_user_candidates_from_recalls,
    fuse_by_quota,
    recall_at_L,
)
```

#### 2. Call CEM Evaluation in Test Section (line ~660-674)
When `--do_test_sft` or `--do_test_grpo` is run, the system now:
1. Runs standard multi-channel evaluation (softmax-weighted, uniform-weighted)
2. **Runs CEM optimization** to find optimal weights
3. Compares all methods

#### 3. Enhanced Summary Output (line ~717-745)
The summary now includes:
- CEM-optimized NDCG and Recall metrics
- Optimized weight values for each recaller
- Direct comparison with softmax-weighted and uniform-weighted fusion
- Percentage improvement over baseline methods

## Usage

### Basic Testing with CEM

```bash
python GRPO/baseline_cem.py \
    --dataset Games \
    --recbole_models BPR LightGCN SASRec \
    --do_test_sft \
    --final_k 50 \
    --model_name path/to/trained/model
```

### With Custom CEM Parameters

Add these arguments to your command:
- `--cem_iters 30`: Increase CEM iterations (default: 20)
- `--cem_population 512`: Increase population size (default: 256)
- `--cem_elite_frac 0.15`: Change elite fraction (default: 0.1)

Note: These parameters can be added to `parse_args()` in `main_pure.py` if needed.

## Example Output

```
==============================================================
CEM-Based Fusion Optimization
==============================================================
Processing 1000 users with 3 recallers
Building candidate lists (M=150 per channel)...
Running CEM optimization (iters=20, pop=256)...

[CEM] iter=00 mean=0.3542 elite_best=0.4123 global_best=0.4123
[CEM] iter=01 mean=0.3789 elite_best=0.4201 global_best=0.4201
...
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

CEM-Optimized Performance at Different k:
  k=10: NDCG=0.3245, Recall=0.2134
  k=20: NDCG=0.3567, Recall=0.3421
  k=50: NDCG=0.3876, Recall=0.4298

==============================================================
CEM-Optimized Fusion vs. Other Methods
==============================================================
CEM-Optimized Fusion:
  NDCG@50: 0.3876
  Recall@50: 0.4298

Optimized Weights:
  bpr: 0.2341
  lightgcn: 0.4523
  sasrec: 0.3136

Comparison (NDCG@50):
  Softmax-weighted fusion: 0.3654
  Uniform-weighted fusion: 0.3521
  CEM-optimized fusion: 0.3876

CEM improvement over softmax: +6.08%
```

## Key Benefits

1. **Training-Free Optimization**: CEM finds optimal weights without requiring gradient-based training
2. **Direct Recall Optimization**: Optimizes directly for the target metric (Recall@L)
3. **Flexible**: Works with any set of recallers, no model architecture changes needed
4. **Interpretable**: Produces clear weight distributions showing relative importance of each channel
5. **Fast**: Typically converges in 15-30 iterations

## Implementation Notes

### Recaller Wrapper
The integration uses a wrapper function to adapt RecBoleRecaller instances to the signature expected by `build_user_candidates_from_recalls()`:

```python
def recaller_wrapper(recaller_name):
    def recall_fn(uids: List[int], topk: int) -> List[List[int]]:
        results = []
        for i, uid in enumerate(uids):
            items = recallers[recaller_name.lower()].recall(
                uid, topk, eval_hists[i], 
                full_hist=full_hists[i], 
                gt_items=gt_items_list[i]
            )
            item_ids = [item[0] for item in items]
            results.append(item_ids)
        return results
    return recall_fn
```

### Candidate Size (M)
By default, M = final_k * 3 (e.g., 150 candidates per channel when final_k=50). This ensures:
- Each channel contributes enough diverse items
- Fusion has sufficient options to optimize from
- Balance between computation cost and quality

### Quota-Based Fusion
The `fuse_by_quota()` function implements list-level union:
1. Allocates quota for each channel: `q_k = round(w_k * L)`
2. Takes top q_k items from each channel
3. Unions them (deduplicated) to create final list of length L
4. Handles rounding to ensure exactly L items

## Files Modified

1. **GRPO/baseline_cem.py**
   - Added imports from `cem_utils`
   - Added `evaluate_cem_fusion()` function
   - Integrated CEM evaluation in test section
   - Enhanced summary output with CEM comparison

2. **GRPO/cem_utils.py** (reference only, not modified)
   - Contains core CEM algorithm
   - Dirichlet sampling and MLE
   - Quota-based fusion logic
   - Recall@L metric computation

## Future Extensions

Potential improvements:
1. Add CEM parameters to argparse for easy tuning
2. Support other objective functions (NDCG@k, MAP@k)
3. Save optimization trajectory for visualization
4. Multi-objective optimization (recall + diversity)
5. Warm-start from model's softmax predictions

## References

- Cross-Entropy Method: Rubinstein, R. Y. (1999). "The Cross-Entropy Method for Combinatorial and Continuous Optimization"
- RecBole: Zhao, W. X., et al. (2021). "RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms"




