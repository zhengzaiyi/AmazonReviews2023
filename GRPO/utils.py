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

def ndcg_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    if not gt_items: return 0.0
    k = max(1, k)
    gt = set(gt_items)
    hit = sum(1 for x in rec_list[:k] if x in gt)
    return hit / min(len(gt), k)

def recall_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    if not gt_items: return 0.0
    k = max(1, k)
    gt = set(gt_items)
    hit = sum(1 for x in rec_list[:k] if x in gt)
    return hit / min(len(gt), k)

def merge_candidates(lists: List[List[int]], ws: List[float], final_k: int) -> List[int]:
    merged = []
    # set scores for each item in each list
    scores = []
    for list_ in lists:
        scores.append({item: i for i, item in enumerate(list_)})
    # weight scores
    for i in range(len(scores)):
        for item in scores[i]:
            scores[i][item] *= ws[i]
    # merge items by scores
    merged = []
    for i in range(final_k):
        max_score = -1
        max_item = None
        for list_ in scores:
            if list_ and list_[0] in list_:
                max_score = list_[0]
                max_item = list_[0]
                break
        if max_item is not None:
            merged.append(max_item)
            scores[scores.index(max_score)] = scores[scores.index(max_score)][1:]
    return merged

def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list: return {}
    keys = [k for k in metrics_list[0].keys()]
    out = {}
    for k in keys:
        out[k] = float(np.mean([m.get(k, 0.0) for m in metrics_list]))
    return out
