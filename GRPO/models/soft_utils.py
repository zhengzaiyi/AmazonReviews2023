"""
Utilities for soft token generation and beta sampling.
Shared between main_soft.py and trl_trainer.py to avoid code duplication.
Extended with Gumbel-Softmax utilities for SofT-GRPO.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass


# ============== Gumbel-Softmax Utilities for SofT-GRPO ==============

@dataclass
class SoftStepCache:
    """Cache for soft sampling step to enable replay during policy update."""
    support_indices: torch.Tensor  # Indices in vocabulary
    q_prime: torch.Tensor  # log_p + epsilon values
    log_p_old_noise: float  # Log probability of noise
    y_soft: torch.Tensor  # Soft distribution output


def sample_gumbel_noise(shape: Tuple, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    u = torch.rand(shape, device=device).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(u))


def gumbel_softmax_sample(
    logits: torch.Tensor,
    tau: float = 1.0,
    top_p: float = 0.9,
    hard: bool = False,
    noise_scale: float = 1.0
) -> Tuple[torch.Tensor, SoftStepCache]:
    """
    Gumbel-Softmax sampling with top-p filtering for SofT-GRPO.
    
    Args:
        logits: Logits from classification model (num_classes,)
        tau: Temperature for Gumbel-Softmax
        top_p: Nucleus sampling probability threshold
        hard: If True, return hard one-hot (straight-through estimator)
        noise_scale: Scale factor for Gumbel noise (0.0 = no noise, 1.0 = standard)
        
    Returns:
        y_soft: Soft probability distribution
        cache: SoftStepCache for importance ratio computation
    """
    device = logits.device
    
    # Apply top-p filtering
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff
    mask = cumsum <= top_p
    mask[0] = True  # Always include top-1
    
    # Create support set
    support_indices = sorted_indices[mask]
    support_probs = probs[support_indices]
    
    # Renormalize
    support_probs = support_probs / support_probs.sum()
    log_support_probs = torch.log(support_probs + 1e-10)
    
    # Sample Gumbel noise (scaled to control exploration vs exploitation)
    epsilon = sample_gumbel_noise(support_probs.shape, device) * noise_scale
    
    # Compute q' = log(p) + epsilon
    # When noise_scale=0, q_prime = log(p), output = softmax(log(p)/tau) = p^(1/tau) (normalized)
    # When noise_scale=1, standard Gumbel-Softmax
    q_prime = log_support_probs + epsilon
    
    # Gumbel-Softmax
    y_soft_support = F.softmax(q_prime / tau, dim=-1)
    
    # Expand to full vocabulary
    y_soft = torch.zeros_like(logits)
    y_soft[support_indices] = y_soft_support.to(logits.dtype)
    
    # Compute log noise probability
    log_p_noise = (-epsilon - torch.exp(-epsilon)).sum().item()
    
    # Cache for replay
    cache = SoftStepCache(
        support_indices=support_indices,
        q_prime=q_prime,
        log_p_old_noise=log_p_noise,
        y_soft=y_soft
    )
    
    if hard:
        # Straight-through estimator
        idx = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(idx, num_classes=logits.size(-1)).float()
        y_soft = (y_hard - y_soft).detach() + y_soft
    
    return y_soft, cache


def compute_soft_ratio(
    cache: SoftStepCache,
    new_logits: torch.Tensor,
    tau: float = 1.0
) -> float:
    """
    Compute importance ratio for soft step using cached noise.
    
    Args:
        cache: SoftStepCache from old policy sampling
        new_logits: Logits from new policy
        tau: Temperature
        
    Returns:
        Importance ratio r_soft
    """
    device = new_logits.device
    
    # Get new probs for support set
    new_probs = F.softmax(new_logits, dim=-1)
    new_support_probs = new_probs[cache.support_indices]
    
    # Renormalize
    new_support_probs = new_support_probs / (new_support_probs.sum() + 1e-10)
    log_new_support_probs = torch.log(new_support_probs + 1e-10)
    
    # Compute epsilon_new = q' - log(p_new)
    epsilon_new = cache.q_prime - log_new_support_probs
    
    # Compute log probability under new noise
    log_p_new_noise = (-epsilon_new - torch.exp(-epsilon_new)).sum().item()
    
    # Importance ratio
    r_soft = torch.exp(torch.tensor(log_p_new_noise - cache.log_p_old_noise))
    
    return r_soft.item()


def replay_soft_sample(cache: SoftStepCache, tau: float = 1.0) -> torch.Tensor:
    """
    Replay soft sample from cached q' values.
    
    Args:
        cache: SoftStepCache from original sampling
        tau: Temperature
        
    Returns:
        Reconstructed y_soft
    """
    y_soft_support = F.softmax(cache.q_prime / tau, dim=-1)
    y_soft = cache.y_soft.clone()
    y_soft[cache.support_indices] = y_soft_support
    return y_soft


# ============== Multi-Channel Recall ==============

def multi_channel_recall_softmax(
    softmax_weights: torch.Tensor,
    recallers: Dict,
    recaller_names: List[str],
    user_id: int,
    history: List[int],
    total_k: int,
    full_hist: List[int] = None,
    gt_items: List[int] = None,
    normalize_scores: str = 'minmax'
) -> List[Tuple[int, float]]:
    """
    Multi-channel recall using softmax weights from classification.
    
    Args:
        softmax_weights: Probability distribution over recallers
        recallers: Dict of recaller name -> RecBoleRecaller
        recaller_names: Ordered list of recaller names
        user_id: User ID
        history: User history
        total_k: Number of items to return
        full_hist: All interacted items (optional). If provided along with gt_items,
                  items in full_hist but not in gt_items will be excluded
        gt_items: Ground truth items to keep as valid candidates (optional)
        normalize_scores: Normalization method for scores before merging. Options:
            - None: No normalization (default, original behavior)
            - 'minmax': Min-Max normalization to [0, 1]
            - 'zscore': Z-score normalization (mean=0, std=1)
            - 'softmax': Softmax normalization (sum to 1)
        
    Returns:
        List of (item_id, score) sorted by weighted score
    """
    import numpy as np
    
    candidates = defaultdict(float)
    
    for i, name in enumerate(recaller_names):
        weight = softmax_weights[i].item() if torch.is_tensor(softmax_weights[i]) else softmax_weights[i]
        name_lower = name.lower()
        
        if name_lower in recallers:
            items = recallers[name_lower].recall(user_id, total_k, history, full_hist=full_hist, gt_items=gt_items)
            
            if items:
                item_ids = [item[0] for item in items]
                scores = [item[1] for item in items]
                
                # Normalize scores if requested
                if normalize_scores == 'minmax':
                    min_score, max_score = min(scores), max(scores)
                    if max_score > min_score:
                        scores = [(s - min_score) / (max_score - min_score) for s in scores]
                    else:
                        scores = [0.5] * len(scores)  # All same score, set to 0.5
                elif normalize_scores == 'zscore':
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    if std_score > 1e-8:
                        scores = [(s - mean_score) / std_score for s in scores]
                    else:
                        scores = [0.0] * len(scores)  # All same score, set to 0
                elif normalize_scores == 'softmax':
                    # Convert to numpy for softmax
                    scores_tensor = torch.tensor(scores, dtype=torch.float32)
                    scores_softmax = torch.softmax(scores_tensor, dim=0).tolist()
                    scores = scores_softmax
                
                # Merge normalized scores with weights
                for item_id, norm_score in zip(item_ids, scores):
                    candidates[item_id] += norm_score * weight
            else:
                # If no items returned, skip this recaller
                continue
    
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:total_k]


def compute_ndcg_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    """Compute NDCG@k."""
    import math
    if not gt_items:
        return 0.0
    
    gt = set(gt_items) if isinstance(gt_items, list) else {gt_items}
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(rec_list[:k]) if item in gt)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def soft_grpo_reward(
    softmax_outputs: List[torch.Tensor],
    user_ids: List[int],
    histories: List[List[int]],
    ground_truths: List[List[int]],
    recallers: Dict,
    recaller_names: List[str],
    final_k: int = 50
) -> List[float]:
    """
    Compute NDCG rewards using multi-channel recall with softmax weights.
    
    Args:
        softmax_outputs: Softmax vectors from classification
        user_ids: User IDs
        histories: User histories
        ground_truths: Ground truth items
        recallers: Recaller dictionary
        recaller_names: Ordered recaller names
        final_k: Top-k for NDCG
        
    Returns:
        List of NDCG rewards
    """
    rewards = []
    
    for i, weights in enumerate(softmax_outputs):
        try:
            candidates = multi_channel_recall_softmax(
                weights, recallers, recaller_names,
                user_ids[i], histories[i], final_k
            )
            rec_list = [item_id for item_id, _ in candidates]
            reward = compute_ndcg_at_k(rec_list, ground_truths[i], final_k)
            rewards.append(reward)
        except Exception as e:
            print(f"Reward computation error: {e}")
            rewards.append(0.0)
    
    return rewards


# ============== Original Beta Sampling Functions ==============

def build_soft_template(model_names: List[str], use_top_k: bool = False) -> str:
    """Build JSON template with [num][soft_token] placeholders.
    
    Args:
        model_names: List of model names to include in template
        
    Returns:
        JSON string template with placeholders
    """
    lines = ["{"]
    for i, name in enumerate(model_names):
        lines.append(f'  "{name}": {{')
        if use_top_k:
            lines.append('    "top-k": [num][soft_token],')
        lines.append('    "score-weight": [num][soft_token]')
        lines.append("  }" + ("," if i < len(model_names) - 1 else ""))
    lines.append("}")
    return "\n".join(lines)


def sample_from_logits(logit, clamp_range=(1e-6, 1.0 - 1e-6), sample=True):
    """Sample from logits using sigmoid activation.
    
    Args:
        logit: Raw logit value (unbounded real number)
        clamp_range: Range to clamp sampled values
        sample: Whether to sample from Bernoulli or return sigmoid probability
        
    Returns:
        Sampled value in [0, 1] range
    """
    # Convert logit to probability
    prob = torch.sigmoid(logit).clamp(*clamp_range)
    
    if sample:
        # Sample from Bernoulli distribution
        dist = torch.distributions.Bernoulli(prob)
        return dist.sample().item()
    else:
        # Return probability directly
        return prob.item()


def sample_from_beta_params(alpha, beta, clamp_range=(1e-6, 1.0 - 1e-6), sample=True, apply_transform=True):
    """Sample from Beta distribution with given parameters.
    
    Args:
        alpha: Alpha parameter
        beta: Beta parameter  
        clamp_range: Range to clamp sampled values
        sample: Whether to sample or return mean/mode
        apply_transform: Whether to apply softplus + 1.0 transform to alpha/beta
        
    Returns:
        Sampled value from Beta distribution
    """
    # Process parameters if needed
    if apply_transform:
        alpha_processed = torch.nn.functional.softplus(alpha) + 1.0
        beta_processed = torch.nn.functional.softplus(beta) + 1.0
    else:
        alpha_processed = alpha
        beta_processed = beta
    
    # Sample and clamp
    if sample:
        dist = torch.distributions.Beta(alpha_processed, beta_processed)
        return dist.sample().clamp(*clamp_range).item()
    else:
        # return mean value
        # return alpha_processed / (alpha_processed + beta_processed)
        return (alpha_processed - 1) / (alpha_processed + beta_processed - 2)
        # TODO: try (alpha_processed - 1) / (alpha_processed + beta_processed - 2)


def replace_placeholders_with_values(template: str, values: List[float], precision: int = 6) -> str:
    """Replace [num][soft_token] placeholders with numeric values.
    
    Args:
        template: Template string containing placeholders
        values: List of numeric values to substitute
        precision: Number of decimal places for formatting
        
    Returns:
        Template with placeholders replaced by values
    """
    result = template
    for val in values:
        result = result.replace("[num][soft_token]", f"{float(val):.{precision}f}", 1)
    
    # Fill remaining placeholders with default value
    while "[num][soft_token]" in result:
        result = result.replace("[num][soft_token]", f"{0.0:.{precision}f}", 1)
    
    return result


def generate_soft_completions(model, tokenizer, test_dataset, model_names: List[str], max_length: int = 1536) -> List[str]:
    """Generate completions using soft token beta sampling.
    
    Args:
        model: Model with value_head for beta parameter prediction
        tokenizer: Tokenizer with [num] and [soft_token] special tokens
        test_dataset: Dataset containing prompts
        model_names: List of model names for template
        max_length: Maximum sequence length
        
    Returns:
        List of generated completion strings
    """
    completions = []
    device = next(model.parameters()).device
    model.eval()
    model.to(device)
    soft_token_id = tokenizer.convert_tokens_to_ids("[soft_token]")
    num_token_id = tokenizer.convert_tokens_to_ids("[num]")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Generating completions"):
            # Extract prompt
            prompt = test_dataset[i]['prompt']
            prompt_text = prompt[0]['content'] if isinstance(prompt, list) and isinstance(prompt[0], dict) else str(prompt)
            
            # Build full text with template
            template = build_soft_template(model_names)
            full_text = prompt_text + "\n\n" + template
            
            # Tokenize and forward pass
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            # Need to explicitly request hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Sample values from logits (following new BCE logic)
            sampled_values = []
            if hasattr(model, 'value_head') and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                input_ids = inputs['input_ids'][0]
                num_positions = (input_ids == num_token_id).nonzero(as_tuple=True)[0]
                soft_positions = (input_ids == soft_token_id).nonzero(as_tuple=True)[0]
                
                if len(num_positions) > 0 and len(soft_positions) > 0:
                    # Get hidden states at [num] positions
                    hidden_states = outputs.hidden_states[-1][0, num_positions, :]  # (K, H)
                    value_preds = model.value_head(hidden_states)  # (K, 2)
                    
                    # Sample from logits (only use first output as logit)
                    for pred in value_preds:
                        logit = pred[0]  # Only use the first output (mu)
                        sampled_values.append(sample_from_logits(logit))
            
            # Use default values if no sampling occurred
            if not sampled_values:
                num_soft_tokens = template.count("[num][soft_token]")
                sampled_values = [0.5] * num_soft_tokens
            
            # Replace placeholders with sampled values
            completion_text = replace_placeholders_with_values(template, sampled_values)
            completions.append(completion_text)
    
    return completions
