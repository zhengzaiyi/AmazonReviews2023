import random
import numpy as np
import torch
import math
import json
from typing import List, Dict
from collections import defaultdict
from GRPO.recallers import RecBoleRecaller

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


TOOLS_DESCRIPTION = {
    "sasrec": {
        "description": "A Transformer-based sequential recommendation model (Self-Attentive Sequential Recommendation). It captures the order and short-term interest patterns from the user's recent interactions.",
        "when_to_use": "Use when the task requires modeling the sequence of recent user interactions or short-term preferences. For example, predicting the next item a user might click or watch.",
        "input": "A chronologically ordered sequence of user-item interactions.",
        "output": "Top-K candidate items predicted as the next likely interactions."
    },
    "bpr": {
        "description": "Bayesian Personalized Ranking, a classic pairwise ranking method based on matrix factorization. It focuses on modeling user preference orderings.",
        "when_to_use": "Use when the task involves general recommendation based on long-term user preferences, without considering sequence order. Suitable for implicit feedback like clicks or likes.",
        "input": "A user-item interaction matrix or embeddings representing user and item factors.",
        "output": "Top-K candidate items ranked by the user's overall preference."
    },
    "lightgcn": {
        "description": "A graph-based recommendation model using Graph Convolutional Networks. It propagates embeddings over a user-item bipartite graph to capture high-order connectivity.",
        "when_to_use": "Use when the task involves graph structures, such as leveraging user-item relations or higher-order neighbor connections. Suitable for social or community-driven recommendations.",
        "input": "A user-item bipartite graph.",
        "output": "Top-K candidate items derived from graph-based user and item embeddings."
    }
}

def build_prompt(profile_json: str, available_models: List[str] = None) -> str:
        if available_models is None:
            available_models = ['sasrec', 'bpr', 'pop']
        models_str = "', '".join(available_models)
        
        # Create an example output format
        import random
        example_output = [{
            "name": available_models[i], 
            "k": random.randint(1, 100), 
            "weight": round(random.uniform(0, 1), 4)
        } for i in range(min(2, len(available_models)))]
        example_output = json.dumps(example_output, indent=2)

        return (
            "You are a multi-channel recall assistant in a recommendation system. Given a user profile JSON, "
            "output ONLY a JSON file describe the usage of different models during the multi-channel recall. Each element must be an object with keys: "
            "\"name\" (string: model name like '"
            + models_str + "'), \"k\" (int 1..500: number of items), \"weight\" (float 0..1: model weight)\n\n"
            f"Available models: \n{[json.dumps(TOOLS_DESCRIPTION[m], indent=2) for m in available_models]}\n"
            f"Profile:\n{profile_json}\n"
            f"Expected output format example:\n{example_output}\n\n"
            "Your JSON response:"
        )


def ndcg_rewards(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    user2items_test: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    final_k: int, 
    **kwargs
):
    """Calculate NDCG rewards"""
    rewards = []
    
    for i, completion_msgs in enumerate(completions):
        try:
            # Extract routing content
            content = completion_msgs[-1]["content"]
            routes = json.loads(content)
            
            # Calculate recommendation results
            candidates = defaultdict(int)
            for route in routes:
                model_name = route.get("name", "")
                k = route.get("k", 10) 
                w = route.get("weight", 1.0)
                
                if model_name in recallers:
                    items = recallers[model_name].recall(uid[i], int(k), histories[i])
                    for item in items:
                        candidates[item[0]] += item[1] * w
            
            # Calculate NDCG
            candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:final_k]
            ndcg = ndcg_at_k(candidates, user2items_test[i], k=final_k)
            rewards.append(ndcg)
        except Exception as e:
            print(f"Error processing completion: {e}")
            rewards.append(0)
    
    return rewards