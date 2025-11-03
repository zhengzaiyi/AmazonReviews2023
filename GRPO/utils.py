import random
import numpy as np
import torch
import math
import json
from typing import List, Dict, Optional
from collections import defaultdict
from GRPO.recallers import RecBoleRecaller

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ndcg_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
    if not gt_items: return 0.0
    k = max(1, k)
    if type(gt_items) == int:
        gt_items = [gt_items]
    gt = set(gt_items)
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item in enumerate(rec_list[:k]):
        if item in gt:
            # relevance = 1 for binary relevance (item is relevant or not)
            # position discount = 1 / log2(i + 2) where i is 0-indexed
            dcg += 1.0 / math.log2(i + 2)
    
    # Calculate IDCG@K (Ideal DCG)
    # In binary relevance case, IDCG is the DCG when all relevant items are at the top
    num_relevant = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    
    # Return NDCG@K
    return dcg / idcg if idcg > 0 else 0.0

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
    },
    "pop": {
        "description": "A simple non-personalized baseline that recommends items purely based on their overall popularity (e.g., number of interactions).",
        "when_to_use": "Use as a baseline for comparison or in cold-start situations where user-specific data is not available.",
        "input": "Global item interaction counts or frequencies.",
        "output": "Top-K items ranked by overall popularity."
    },
    "itemknn": {
        "description": "An item-based collaborative filtering model that recommends items similar to those a user has already interacted with, using item-to-item similarity (e.g., cosine similarity, Jaccard).",
        "when_to_use": "Use when item similarity can effectively capture user preference patterns. Works well in scenarios like e-commerce or content platforms where co-purchase or co-view signals are strong.",
        "input": "Item-item similarity matrix built from historical user-item interactions.",
        "output": "Top-K items most similar to the user’s past interacted items."
    },
    "fpmc": {
        "description": "Factorizing Personalized Markov Chains, a hybrid model combining matrix factorization (long-term user preferences) with first-order Markov chains (short-term sequential patterns). It predicts the next item by considering both user embedding and the transition from the last interacted item.",
        "when_to_use": "Use when the recommendation task involves next-item prediction or session-based recommendation, where both long-term preferences and recent sequential behavior matter.",
        "input": "User embedding (long-term preference) and the last interacted item (short-term context).",
        "output": "Top-K candidate items predicted as the user's next likely interaction."
    }
}

def build_prompt(profile_json: str, available_models: List[str] = None) -> str:
    if available_models is None:
        available_models = ['sasrec', 'bpr', 'pop']
    models_str = "', '".join(available_models)
        
    # Create an example output format
    import random
    example_output = {
        model: {
            "top-k": f'integer between 1 and 500',
            "score-weight": f'float between 0 and 1'
        } for model in available_models
    }
    example_output = json.dumps(example_output, indent=2)

    return (
        "You are a multi-channel recall assistant in a recommendation system. Given a user profile JSON, "
        "output ONLY a JSON file describe the usage of different models during the multi-channel recall. Each element must be an object with keys: "
        "\"name\" (string: model name like '"
        + models_str + "'), \"k\" (integer between 1 and 500: number of items), \"weight\" (float between 0 and 1: model weight)\n\n"
        f"Available models: \n{[json.dumps(TOOLS_DESCRIPTION[m], indent=2) for m in available_models]}\n"
        f"User Profile:\n{profile_json}\n"
        f"Expected output format example:\n{example_output}\n\n"
        f"Please output the JSON file containing the usage of ALL availablemodels."
        "Your JSON response:"
    )


def ndcg_rewards(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    target_items: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    final_k: int, 
    **kwargs
):
    """Abandoned"""
    rewards = []
    
    for i, completion_msgs in enumerate(completions):
        try:
            # Extract routing content
            content = completion_msgs[-1]["content"]
            model_configs = json.loads(content)
            
            # Calculate recommendation results
            candidates = defaultdict(int)
            total_k = 0
            for recaller in recallers.keys():
                total_k += model_configs[recaller]['top-k']
            for recaller in recallers.keys():
                k = model_configs[recaller]['top-k'] * final_k * 1.5 / total_k # TODO: check magic number
                w = model_configs[recaller]['score-weight']
                
                if recaller in recallers:
                    items = recallers[recaller].recall(uid[i], int(k), histories[i])
                    for item in items:
                        candidates[item[0]] += item[1] * w
            
            # Calculate NDCG
            candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:final_k]
            ndcg = ndcg_at_k(candidates, target_items[i], k=final_k)
            rewards.append(ndcg)
        except Exception as e:
            print(f"Error processing completion: {e}")
            rewards.append(0)
    
    return rewards

def multi_channel_recall(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    final_k: int, 
) -> List[List[int]]:
    """Calculate multi-channel recall"""
    recall_results = []
    for i, completion_msgs in enumerate(completions):
        try:
            if isinstance(completion_msgs, str):
                content = completion_msgs
            else:
                content = completion_msgs[-1]["content"]
            model_configs = json.loads(content)
                
            # Calculate recommendation results
            candidates = defaultdict(int)
            total_k = 0
            assert set(model_configs.keys()) <= set(recallers.keys()), f"Model configs keys {model_configs.keys()} are not in recallers keys {recallers.keys()}"
            for recaller in model_configs.keys():
                total_k += model_configs[recaller]['top-k']
            for recaller in model_configs.keys():
                k = model_configs[recaller]['top-k'] * final_k * 1.5 / total_k # TODO: check magic number
                w = model_configs[recaller]['score-weight']
                
                items = recallers[recaller].recall(uid[i], int(k), histories[i])
                for item in items:
                    candidates[item[0]] += item[1] * w
            candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:final_k]
            recall_results.append(candidates)
        except Exception as e:
            print(f"Error processing completion: {e}")
            recall_results.append([])
    return recall_results

def average_recall(
    recallers: Dict[str, RecBoleRecaller],
    uid: List[int],
    histories: List[List[int]],
    final_k: int,
    normalize: Optional[str] = None,
) -> List[List[int]]:
    recall_results = []
    for i, user in enumerate(uid):
        total_scores = 0
        for recaller in recallers.keys():
            scores = recallers[recaller].full_sort_predict(user, histories[i])
            scores = torch.tensor(scores)
            if normalize == 'sigmoid':
                scores = torch.sigmoid(scores)
            elif normalize == 'tanh':
                scores = torch.tanh(scores)
            elif normalize == 'softmax':
                scores = torch.softmax(scores, dim=-1)
            scores = scores.numpy()
            total_scores = scores + total_scores
        topk_indices = np.argsort(total_scores)[::-1][:final_k]
        recall_results.append(topk_indices)
    return recall_results

def evaluate(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    final_k: int, 
    target_items: List[List[int]],
    ks: List[int] = [10, 50],
) -> Dict[str, float]:
    recall_results = multi_channel_recall(
        completions=completions, 
        uid=uid, 
        histories=histories, 
        recallers=recallers, 
        final_k=final_k
    )
    metrics = defaultdict(list)
    for i, recall_result in enumerate(recall_results):
        for k in ks:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(recall_result, target_items[i], k=k))
            metrics[f"recall@{k}"].append(recall_at_k(recall_result, target_items[i], k=k))
    return {
        k: sum(v) / len(v) for k, v in metrics.items()
    }

def jaccard_similarity(list1: List[int], list2: List[int], k: int = None) -> float:
    """
    计算两个推荐列表的 Jaccard 相似度
    
    Args:
        list1
        list2
        k
    
    Returns:
        Jaccard 相似度 (0-1)
    """
    if k is not None:
        list1 = list1[:k]
        list2 = list2[:k]
    
    set1 = set(list1)
    set2 = set(list2)
    
    assert set1 and set2, "set1 and set2 are empty"
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union) if union else 0.0


def rbo_similarity(list1: List[int], list2: List[int], p: float = 0.9) -> float:
    """
    Calculate the RBO (Rank-Biased Overlap) similarity between two sorted lists
    
    Args:
        list1: The first sorted list
        list2: The second sorted list
        p: Continuation probability parameter (0-1),
    
    Returns:
        RBO similarity (0-1)
    """
    def overlap_at_k(list1: List[int], list2: List[int], k: int) -> float:
        return len(set(list1[:k]).intersection(set(list2[:k]))) / k
    
    min_len = min(len(list1), len(list2))
    
    if min_len == 0:
        return 0.0
    
    rbo_sum = 0.0
    for k in range(1, min_len + 1):
        rbo_sum += overlap_at_k(list1, list2, k) * (p ** (k - 1))
    
    # Handle the case of unequal lengths
    if len(list1) != len(list2):
        extrapolated_overlap = overlap_at_k(list1, list2, min_len)
        rbo_sum += extrapolated_overlap * (p ** min_len) * ((1 - p) ** -1)
    else:
        rbo_sum += (p ** min_len) / (1 - p) * overlap_at_k(list1, list2, min_len)
    
    return (1 - p) * rbo_sum

def evaluate_recallers(
    recallers: Dict[str, RecBoleRecaller],
    uid: List[int],
    histories: List[List[int]],
    target_items: List[List[int]],
    final_k: int,
    ks: List[int] = [10, 50],
) -> Dict[str, float]:
    metrics = {recaller: defaultdict(list) for recaller in recallers.keys()}
    metrics['optimal'] = defaultdict(list)
    metrics['average'] = defaultdict(list)
    metrics['static'] = defaultdict(list)
    all_recommendations = {recaller: [] for recaller in recallers.keys()}
    for i in range(len(uid)):
        for recaller_name in recallers.keys():
            items = recallers[recaller_name].recall(uid[i], int(final_k), histories[i])
            item_ids = [item[0] for item in items] if items else []
            all_recommendations[recaller_name].append(item_ids)
            for k in ks:
                ndcg = ndcg_at_k(item_ids, target_items[i], k=k)
                recall = recall_at_k(item_ids, target_items[i], k=k)
                metrics[recaller_name][f"ndcg@{k}"].append(ndcg)
                metrics[recaller_name][f"recall@{k}"].append(recall)
    
    average_recall_results = average_recall(
        recallers=recallers,
        uid=uid,
        histories=histories,
        final_k=final_k,
    )
    static_recall_results = average_recall(
        recallers=recallers,
        uid=uid,
        histories=histories,
        final_k=final_k,
        normalize='softmax'
    )
    
    for i in range(len(uid)):
        for k in ks:
            metrics['average'][f"ndcg@{k}"].append(ndcg_at_k(average_recall_results[i], target_items[i], k=k))
            metrics['average'][f"recall@{k}"].append(recall_at_k(average_recall_results[i], target_items[i], k=k))
            metrics['static'][f"ndcg@{k}"].append(ndcg_at_k(static_recall_results[i], target_items[i], k=k))
            metrics['static'][f"recall@{k}"].append(recall_at_k(static_recall_results[i], target_items[i], k=k))
    for i in range(len(uid)):
        last_k = 10
        best_ndcg = -1
        for recaller_name in recallers.keys():
            if metrics[recaller_name][f"recall@{last_k}"][i] > best_ndcg:
                best_ndcg = metrics[recaller_name][f"ndcg@{last_k}"][i]
                best_recaller = recaller_name
        for k in ks:
            metrics['optimal'][f"ndcg@{k}"].append(metrics[best_recaller][f"ndcg@{k}"][i])
            metrics['optimal'][f"recall@{k}"].append(metrics[best_recaller][f"recall@{k}"][i])


    result = {}
    for recaller_name in metrics.keys():
        result[recaller_name] = {}
        for metric_name, values in metrics[recaller_name].items():
            result[recaller_name][metric_name] = sum(values) / len(values) if values else 0.0

    # compute jaccard similarity and rbo similarity
    recaller_names = list(recallers.keys())
    similarity_metrics = {
        'jaccard': defaultdict(list),
        'rbo': defaultdict(list)
    }
    
    # Compute the similarity between recallers for each user
    from tqdm import tqdm
    # for i in tqdm(range(len(uid)), desc='Computing similarity'):
    #     # Compute the similarity between all recallers
    #     for j, recaller1 in enumerate(recaller_names):
    #         for k, recaller2 in enumerate(recaller_names):
    #             if j < k:  # Avoid duplicate calculations
    #                 list1 = all_recommendations[recaller1][i]
    #                 list2 = all_recommendations[recaller2][i]
                    
    #                 # Compute Jaccard similarity (for the entire list)
    #                 jaccard = jaccard_similarity(list1, list2, k=10)
    #                 similarity_metrics['jaccard'][f"{recaller1}_vs_{recaller2}"].append(jaccard)
                    
    #                 # Compute RBO similarity
    #                 rbo = rbo_similarity(list1, list2, p=0.9)
    #                 similarity_metrics['rbo'][f"{recaller1}_vs_{recaller2}"].append(rbo)
    # result['similarities'] = {
    #     'jaccard': {},
    #     'rbo': {}
    # }
    # for sim_type in ['jaccard', 'rbo']:
    #     for pair_name, values in similarity_metrics[sim_type].items():
    #         avg_similarity = sum(values) / len(values) if values else 0.0
    #         result['similarities'][sim_type][pair_name] = avg_similarity
        
    #     # Compute the overall average similarity for all pairs
    #     all_values = []
    #     for values in similarity_metrics[sim_type].values():
    #         all_values.extend(values)
    #     if all_values:
    #         result['similarities'][sim_type]['average'] = sum(all_values) / len(all_values)
    #     else:
    #         result['similarities'][sim_type]['average'] = 0.0        
    return result