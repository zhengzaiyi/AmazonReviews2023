import random
import numpy as np
import torch
import math
import json
from typing import List, Dict

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def recall_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    if not gt_items: return 0.0
    k = max(1, k)
    gt = set(gt_items)
    hit = sum(1 for x in rec_list[:k] if x in gt)
    return hit / min(len(gt), k)

def merge_candidates(list_1: List[int], list_2: List[int], w_1: float, final_k: int) -> List[int]:
    w_1 = max(0.0, min(1.0, w_1)); w_2 = 1.0 - w_1
    n_1 = max(1, int(round(w_1 * 10))); n_2 = max(1, int(round(w_2 * 10)))
    merged, seen = [], set(); i = j = 0
    while len(merged) < final_k and (i < len(list_1) or j < len(list_2)):
        for _ in range(n_1):
            if i < len(list_1):
                it = list_1[i]; i += 1
                if it not in seen: merged.append(it); seen.add(it)
                if len(merged) >= final_k: break
        if len(merged) >= final_k: break
        for _ in range(n_2):
            if j < len(list_2):
                it = list_2[j]; j += 1
                if it not in seen: merged.append(it); seen.add(it)
                if len(merged) >= final_k: break
    return merged

def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list: return {}
    keys = [k for k in metrics_list[0].keys()]
    out = {}
    for k in keys:
        out[k] = float(np.mean([m.get(k, 0.0) for m in metrics_list]))
    return out
