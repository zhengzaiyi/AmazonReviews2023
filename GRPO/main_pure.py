import argparse
import json
import os
from functools import partial
from typing import List, Dict, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datetime import datetime
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    DataCollatorForSeq2Seq,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from GRPO.agents import UserProfileAgent
from GRPO.data import load_dataset
from GRPO.main import initialize_recallers
from GRPO.recallers import RecBoleRecaller
from GRPO.utils import set_seed, build_prompt, ndcg_at_k, recall_at_k
from GRPO.soft_utils import multi_channel_recall_softmax, compute_ndcg_at_k as soft_ndcg
from GRPO.trl_trainer import GRPOTrainer
from GRPO.soft_utils import multi_channel_recall_softmax, compute_ndcg_at_k
from trl import GRPOConfig


# Dataset generation configuration
MIN_HISTORY_FOR_AUGMENTATION = 30
AUGMENTATION_STEP = 10


def create_sft_dataset(
    profile_agent: UserProfileAgent,
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: Dict[str, RecBoleRecaller],
    final_k: int,
    profile_cutoff: int = 20,
    use_augmentation: bool = True,
    use_soft_label: bool = False,
    soft_label_temperature: float = 5.0,
    user_hint_map: Dict[int, str] = None,  # 用户->eval set最佳recaller的映射
    use_self_hint: bool = False,  # 是否使用当前样本的best_recaller作为hint (用于eval set)
    random_history_selection: bool = False,  # 随机选择历史项而非截取最近的
) -> Tuple[Dataset, Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Create dataset where the model predicts the best recaller class"""
    dataset = []
    metrics = {recaller: defaultdict(list) for recaller in recallers.keys()}
    
    # Create label mappings
    recaller_names = sorted(list(recallers.keys()))
    label2id = {name: i for i, name in enumerate(recaller_names)}
    id2label = {i: name for name, i in label2id.items()}
    
    for i, uid in enumerate(user_ids):
        hist = histories[i]
        if 0 in hist:
            hist = hist[:hist.index(0)]
        
        # Combine history with target as full history (target is the last item)
        full_hist = hist + [target_items[i]]
        
        # Fixed history length for eval_hist
        fixed_hist_len = profile_cutoff
        # n_gt: number of ground truth items following eval_hist
        n_gt_ratio = 0.2
        n_gt = max(1, min(5, int(fixed_hist_len * n_gt_ratio)))
        
        # Minimum required length: fixed_hist_len + n_gt
        min_required_len = fixed_hist_len + n_gt
        
        if len(full_hist) < min_required_len:
            continue
        
        # Determine sliding window start positions for augmentation
        # Each window: eval_hist = full_hist[start:start+fixed_hist_len], gt_items = full_hist[start+fixed_hist_len:start+fixed_hist_len+n_gt]
        if use_augmentation and len(full_hist) >= MIN_HISTORY_FOR_AUGMENTATION:
            # Sliding window: different start positions, fixed history length
            max_start = len(full_hist) - min_required_len
            start_positions = list(range(0, max_start + 1, AUGMENTATION_STEP))
            # Always include the last valid position
            if max_start not in start_positions:
                start_positions.append(max_start)
        else:
            # No augmentation: only use the last window (most recent history)
            start_positions = [len(full_hist) - min_required_len]
        
        for start_pos in start_positions:
            gt_items = full_hist[start_pos + fixed_hist_len:start_pos + fixed_hist_len + n_gt]
            if random_history_selection:
                # 随机选择历史项：从 gt_items 之前的所有项中随机选择 profile_cutoff 个，保持时间顺序
                available_hist = full_hist[:start_pos + fixed_hist_len]
                if len(available_hist) >= profile_cutoff:
                    selected_indices = sorted(np.random.choice(len(available_hist), profile_cutoff, replace=False))
                    eval_hist = [available_hist[i] for i in selected_indices]
                else:
                    eval_hist = available_hist
            else:
                eval_hist = full_hist[start_pos:start_pos + fixed_hist_len]
            
            if len(eval_hist) < 5 or len(gt_items) < 1:
                continue
            
            # Find best recaller and collect scores
            # Mask all interacted items except gt_items for fair evaluation
            recaller_scores = {}
            best_ndcg, best_recaller = -1, None
            for recaller_name, recaller in recallers.items():
                items = recaller.recall(uid, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                item_ids = [item[0] for item in items] if items else []
                ndcg = ndcg_at_k(item_ids, gt_items, k=final_k)
                metrics[recaller_name]["ndcg"].append(ndcg)
                recaller_scores[recaller_name] = ndcg
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_recaller = recaller_name
            
            # Create sample
            user_profile = profile_agent.forward(uid, eval_hist, cut_off=profile_cutoff)
            # 确定 hint: use_self_hint 优先使用当前 best_recaller，否则使用 user_hint_map
            if use_self_hint:
                hint = best_recaller
            elif user_hint_map:
                hint = user_hint_map.get(uid, "")
            else:
                hint = ""
            
            base_prompt = build_prompt(user_profile, available_models=recaller_names, type='classification')
            if hint:
                prompt = base_prompt.replace("Your response:", f"\nNote: Based on this user's historical interactions, {hint} has shown the best performance.\nYour response:")
            else:
                prompt = base_prompt
            
            sample = {
                "text": prompt,
                "prompt": prompt,  # For GRPO compatibility
                "labels": label2id[best_recaller],
                "best_recaller": best_recaller,
                "best_ndcg": best_ndcg,
                "user_id": uid,
                "history": eval_hist,  # For GRPO reward computation
                "target_items": gt_items,  # For GRPO reward computation
                "full_hist": full_hist,  # All interacted items for masking during evaluation
                "history_len_used": len(eval_hist),
            }
            
            # Compute soft labels if enabled
            if use_soft_label:
                scores = torch.tensor([recaller_scores[name] for name in recaller_names])
                # Temperature-scaled softmax: higher temp = sharper, lower = smoother
                soft_labels = torch.softmax(scores * soft_label_temperature, dim=0).tolist()
                sample["soft_labels"] = soft_labels
            
            dataset.append(sample)
    
    # Print statistics
    print(f"\nDataset created: {len(dataset)} samples from {len(set(d['user_id'] for d in dataset))} users")
    for recaller_name in recallers.keys():
        if metrics[recaller_name]["ndcg"]:
            avg_ndcg = np.mean(metrics[recaller_name]["ndcg"])
            print(f"{recaller_name}: avg NDCG = {avg_ndcg:.4f}")
    
    # Best model distribution
    best_model_counts = defaultdict(int)
    for item in dataset:
        best_model_counts[item["best_recaller"]] += 1
    for model, count in sorted(best_model_counts.items()):
        print(f"{model}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # 构建 user -> best_recaller 映射 (取每个用户最后一条样本的 best_recaller)
    user_best_recaller = {}
    for item in dataset:
        user_best_recaller[item["user_id"]] = item["best_recaller"]
    
    return Dataset.from_list(dataset), label2id, id2label, user_best_recaller


def compute_metrics(eval_pred):
    """Compute metrics for classification"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(eval_pred.label_ids, predictions),
        'f1_macro': f1_score(eval_pred.label_ids, predictions, average='macro'),
        'f1_weighted': f1_score(eval_pred.label_ids, predictions, average='weighted')
    }


class SoftLabelTrainer(Trainer):
    """Trainer that supports soft labels using KL divergence loss instead of hard CE loss"""
    
    def __init__(self, *args, use_soft_label=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_soft_label = use_soft_label
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_soft_label and "soft_labels" in inputs:
            soft_labels = inputs.pop("soft_labels")  # (batch, num_classes)
            # Keep hard labels for metrics computation
            _ = inputs.pop("labels", None)
            
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, num_classes)
            
            # KL divergence: KL(soft_target || model_output)
            log_probs = torch.log_softmax(logits, dim=-1)
            soft_labels_tensor = torch.tensor(soft_labels, device=logits.device, dtype=logits.dtype)
            
            # Soft cross entropy: -sum(soft_target * log_prob)
            loss = -(soft_labels_tensor * log_probs).sum(dim=-1).mean()
            
            if return_outputs:
                return loss, outputs
            return loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def compute_metrics_seq2seq(eval_pred, tokenizer, label2id):
    """Compute metrics for seq2seq generation"""
    predictions, labels = eval_pred
    
    # If predictions are logits, take argmax to get token ids
    if predictions.ndim == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)
    
    # Replace -100 with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert text predictions to label ids
    pred_label_ids = []
    true_label_ids = []
    
    for pred_text, true_text in zip(decoded_preds, decoded_labels):
        # Clean up text
        pred_text = pred_text.strip()
        true_text = true_text.strip()
        
        # Map to label ids
        pred_id = label2id.get(pred_text, -1)  # -1 for unknown predictions
        true_id = label2id.get(true_text, -1)
        
        pred_label_ids.append(pred_id)
        true_label_ids.append(true_id)
    
    pred_label_ids = np.array(pred_label_ids)
    true_label_ids = np.array(true_label_ids)
    
    # Only evaluate on valid predictions
    valid_mask = (pred_label_ids != -1) & (true_label_ids != -1)
    
    if np.sum(valid_mask) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'valid_predictions': 0}
    
    valid_preds = pred_label_ids[valid_mask]
    valid_labels = true_label_ids[valid_mask]
    
    return {
        'accuracy': accuracy_score(valid_labels, valid_preds),
        'f1_macro': f1_score(valid_labels, valid_preds, average='macro'),
        'f1_weighted': f1_score(valid_labels, valid_preds, average='weighted'),
        'valid_predictions': np.sum(valid_mask) / len(pred_label_ids)
    }


def evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers=None, final_k=50):
    """Evaluate the pure classification model and generate recommendations"""
    device = model.device
    model.eval()
    
    all_predictions, all_labels = [], []
    recommendation_results = []
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, 
                             max_length=1536, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            all_predictions.append(prediction)
            all_labels.append(example["labels"])
            
            # Generate recommendations if recallers are provided
            if recallers is not None:
                predicted_recaller_name = id2label[prediction]
                true_recaller_name = example["best_recaller"]
                
                # Get data directly from example (already computed in create_sft_dataset)
                user_id = example["user_id"]
                eval_hist = example["history"]
                gt_items = example["target_items"]
                full_hist = example.get("full_hist", None)  # For masking interacted items
                    
                # Generate recommendations using predicted recaller
                if predicted_recaller_name in recallers and len(eval_hist) >= 5:
                    predicted_recaller = recallers[predicted_recaller_name]
                    pred_items = predicted_recaller.recall(user_id, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                    pred_item_ids = [item[0] for item in pred_items] if pred_items else []
                    pred_ndcg = ndcg_at_k(pred_item_ids, gt_items, k=final_k)
                    pred_recall = recall_at_k(pred_item_ids, gt_items, k=final_k)
                else:
                    print(f"Predicted recaller {predicted_recaller_name} not in recallers")
                    pred_item_ids = []
                    pred_ndcg = 0.0
                    pred_recall = 0.0
                
                # Generate recommendations using true recaller for comparison
                if true_recaller_name in recallers and len(eval_hist) >= 5:
                    true_recaller = recallers[true_recaller_name]
                    true_items = true_recaller.recall(user_id, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                    true_item_ids = [item[0] for item in true_items] if true_items else []
                    true_ndcg = ndcg_at_k(true_item_ids, gt_items, k=final_k)
                    true_recall = recall_at_k(true_item_ids, gt_items, k=final_k)
                else:
                    print(f"True recaller {true_recaller_name} not in recallers")
                    true_item_ids = []
                    true_ndcg = 0.0
                    true_recall = 0.0
                
                recommendation_results.append({
                    "user_id": user_id,
                    "predicted_recaller": predicted_recaller_name,
                    "true_recaller": true_recaller_name,
                    "predicted_items": pred_item_ids[:10],  # Top 10 for display
                    "true_items": true_item_ids[:10],  # Top 10 for display
                    "ground_truth": gt_items,
                    "eval_hist": eval_hist,  # Store eval_hist for base model evaluation
                    "full_hist": full_hist,  # All interacted items for masking
                    "predicted_ndcg": pred_ndcg,
                    "true_ndcg": true_ndcg,
                    "predicted_recall": pred_recall,
                    "true_recall": true_recall,
                    "correct_prediction": predicted_recaller_name == true_recaller_name,
                    "history_length": len(eval_hist)
                })
    
    # Calculate classification metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    print(f"\nAccuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=list(id2label.values())))
    
    result = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist()
    }

    # Add recommendation metrics if available
    if recommendation_results:
        avg_pred_ndcg = np.mean([r["predicted_ndcg"] for r in recommendation_results])
        avg_true_ndcg = np.mean([r["true_ndcg"] for r in recommendation_results])
        avg_pred_recall = np.mean([r["predicted_recall"] for r in recommendation_results])
        avg_true_recall = np.mean([r["true_recall"] for r in recommendation_results])
        correct_predictions = sum([r["correct_prediction"] for r in recommendation_results])
        
        # Calculate improvement/degradation metrics
        ndcg_improvement = avg_pred_ndcg - avg_true_ndcg
        recall_improvement = avg_pred_recall - avg_true_recall
        
        print(f"\nRecommendation Metrics:")
        print(f"Average Predicted NDCG@{final_k}: {avg_pred_ndcg:.4f}")
        print(f"Average True Best NDCG@{final_k}: {avg_true_ndcg:.4f}")
        print(f"NDCG Improvement: {ndcg_improvement:+.4f}")
        print(f"Average Predicted Recall@{final_k}: {avg_pred_recall:.4f}")
        print(f"Average True Best Recall@{final_k}: {avg_true_recall:.4f}")
        print(f"Recall Improvement: {recall_improvement:+.4f}")
        print(f"Correct Recaller Predictions: {correct_predictions}/{len(recommendation_results)} ({correct_predictions/len(recommendation_results)*100:.1f}%)")
        
        # Show some examples
        print(f"\nSample Recommendation Results (first 3):")
        for i, res in enumerate(recommendation_results[:3]):
            print(f"User {res['user_id']}: Predicted={res['predicted_recaller']}, True={res['true_recaller']}")
            print(f"  Predicted NDCG: {res['predicted_ndcg']:.4f}, True NDCG: {res['true_ndcg']:.4f}")
            print(f"  Predicted items: {res['predicted_items'][:5]}")
            print(f"  Ground truth: {res['ground_truth']}")
            print()
        
        result.update({
            "recommendation_results": recommendation_results,
            "avg_predicted_ndcg": avg_pred_ndcg,
            "avg_true_ndcg": avg_true_ndcg,
            "avg_predicted_recall": avg_pred_recall,
            "avg_true_recall": avg_true_recall,
            "ndcg_improvement": ndcg_improvement,
            "recall_improvement": recall_improvement,
            "recaller_prediction_accuracy": correct_predictions / len(recommendation_results)
        })
    
    # Evaluate all base recall models for comprehensive comparison
    if recallers is not None and recommendation_results:
        print(f"\n" + "="*60)
        print(f"Base Recall Models Performance on Test Set")
        print(f"="*60)
        
        base_model_results = {}
        
        # Collect all evaluation instances with valid histories
        valid_eval_instances = []
        for res in recommendation_results:
            if res['history_length'] >= 5:  # Only include instances with sufficient history
                valid_eval_instances.append(res)
        
        print(f"Evaluating {len(valid_eval_instances)} valid instances on {len(recallers)} base models...")
        
        for recaller_name, recaller in recallers.items():
            model_ndcgs = []
            model_recalls = []
            
            for res in tqdm(valid_eval_instances, desc=f"Evaluating {recaller_name}", leave=False):
                user_id = res['user_id']
                eval_hist = res['eval_hist']  # Use stored eval_hist
                gt_items = res['ground_truth']  # Use stored ground_truth
                full_hist = res.get('full_hist', None)  # All interacted items for masking
                
                # Generate recommendations
                items = recaller.recall(user_id, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                item_ids = [item[0] for item in items] if items else []
                
                # Calculate metrics
                ndcg = ndcg_at_k(item_ids, gt_items, k=final_k)
                recall = recall_at_k(item_ids, gt_items, k=final_k)
                
                model_ndcgs.append(ndcg)
                model_recalls.append(recall)
            
            if model_ndcgs:
                avg_ndcg = np.mean(model_ndcgs)
                avg_recall = np.mean(model_recalls)
                base_model_results[recaller_name] = {
                    "avg_ndcg": avg_ndcg,
                    "avg_recall": avg_recall,
                    "num_evaluations": len(model_ndcgs)
                }
                print(f"{recaller_name:>12}: NDCG@{final_k}={avg_ndcg:.4f}, Recall@{final_k}={avg_recall:.4f} ({len(model_ndcgs)} evals)")
        
        # Add comparison with model predictions
        print(f"\n" + "-"*60)
        print(f"Model Selection Performance Summary:")
        print(f"-"*60)
        
        if base_model_results:
            # Find best base model
            best_base_ndcg = max(base_model_results.values(), key=lambda x: x["avg_ndcg"])["avg_ndcg"]
            best_base_recall = max(base_model_results.values(), key=lambda x: x["avg_recall"])["avg_recall"]
            best_ndcg_model = max(base_model_results.keys(), key=lambda k: base_model_results[k]["avg_ndcg"])
            best_recall_model = max(base_model_results.keys(), key=lambda k: base_model_results[k]["avg_recall"])
            
            print(f"Best Base Model (NDCG): {best_ndcg_model} = {best_base_ndcg:.4f}")
            print(f"Best Base Model (Recall): {best_recall_model} = {best_base_recall:.4f}")
            print(f"Model Predicted NDCG: {avg_pred_ndcg:.4f}")
            print(f"Model Predicted Recall: {avg_pred_recall:.4f}")
            print(f"True Best Selection NDCG: {avg_true_ndcg:.4f}")
            print(f"True Best Selection Recall: {avg_true_recall:.4f}")
            
            # Calculate gaps
            ndcg_gap_vs_best_base = avg_pred_ndcg - best_base_ndcg
            recall_gap_vs_best_base = avg_pred_recall - best_base_recall
            
            print(f"\nModel vs Best Base Model:")
            print(f"NDCG Gap: {ndcg_gap_vs_best_base:+.4f}")
            print(f"Recall Gap: {recall_gap_vs_best_base:+.4f}")
            
            result.update({
                "base_model_results": base_model_results,
                "best_base_ndcg": best_base_ndcg,
                "best_base_recall": best_base_recall,
                "best_ndcg_model": best_ndcg_model,
                "best_recall_model": best_recall_model,
                "ndcg_gap_vs_best_base": ndcg_gap_vs_best_base,
                "recall_gap_vs_best_base": recall_gap_vs_best_base
            })
    
    return result


def multi_channel_recall_average(
    recallers: Dict,
    recaller_names: List[str],
    user_id: int,
    history: List[int],
    total_k: int,
    full_hist: List[int] = None,
    gt_items: List[int] = None
) -> List[Tuple[int, float]]:
    """
    Multi-channel recall using average (uniform) weights.
    Each recaller has equal weight = 1/n.
    """
    candidates = defaultdict(float)
    num_recallers = len(recaller_names)
    weight = 1.0 / num_recallers
    
    for name in recaller_names:
        name_lower = name.lower()
        if name_lower in recallers:
            items = recallers[name_lower].recall(user_id, total_k, history, full_hist=full_hist, gt_items=gt_items)
            for item_id, score in items:
                candidates[item_id] += score * weight
    
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:total_k]


def evaluate_multi_channel_recall(
    model, 
    tokenizer, 
    test_dataset, 
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    final_k: int = 50,
    use_softmax_weights: bool = True
):
    """
    Evaluate model using multi-channel recall with softmax weights.
    
    Instead of selecting a single recaller, uses the softmax output
    as weights for aggregating candidates from all recallers.
    """
    device = model.device
    model.eval()
    
    metrics = {
        "single_select": defaultdict(list),  # Best single recaller
        "multi_channel": defaultdict(list),   # Weighted multi-channel (softmax)
        "avg_score_weight": defaultdict(list),  # Average score-weight (uniform)
    }
    # Add metrics for each base recaller
    for recaller_name in recaller_names:
        metrics[recaller_name] = defaultdict(list)
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_dataset, desc="Evaluating Multi-Channel")):
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, 
                             max_length=1536, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits[0]  # (num_classes,)
            
            # Get softmax weights
            softmax_weights = torch.softmax(logits, dim=-1)
            
            # Get data directly from example (already computed in create_sft_dataset)
            user_id = example.get("user_id", idx)
            eval_hist = example["history"]
            gt_items = example["target_items"]
            full_hist = example.get("full_hist", None)  # For masking interacted items
            
            if len(eval_hist) < 5:
                continue
            
            # 1. Single recaller selection (argmax)
            pred_idx = logits.argmax().item()
            pred_recaller = recaller_names[pred_idx]
            # Use original name for lookup (consistent with evaluate_pure_model)
            if pred_recaller in recallers:
                items = recallers[pred_recaller].recall(user_id, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                single_rec = [item[0] for item in items]
                for k in [10, 20, 50]:
                    metrics["single_select"][f"ndcg@{k}"].append(ndcg_at_k(single_rec, gt_items, k))
                    metrics["single_select"][f"recall@{k}"].append(recall_at_k(single_rec, gt_items, k))
            
            # 2. Multi-channel recall with softmax weights
            if use_softmax_weights:
                candidates = multi_channel_recall_softmax(
                    softmax_weights, recallers, recaller_names,
                    user_id, eval_hist, final_k,
                    full_hist=full_hist, gt_items=gt_items
                )
                multi_rec = [item_id for item_id, _ in candidates]
                for k in [10, 20, 50]:
                    metrics["multi_channel"][f"ndcg@{k}"].append(ndcg_at_k(multi_rec, gt_items, k))
                    metrics["multi_channel"][f"recall@{k}"].append(recall_at_k(multi_rec, gt_items, k))
            
            # 3. Multi-channel recall with average (uniform) score-weight
            avg_candidates = multi_channel_recall_average(
                recallers, recaller_names, user_id, eval_hist, final_k,
                full_hist=full_hist, gt_items=gt_items
            )
            avg_rec = [item_id for item_id, _ in avg_candidates]
            for k in [10, 20, 50]:
                metrics["avg_score_weight"][f"ndcg@{k}"].append(ndcg_at_k(avg_rec, gt_items, k))
                metrics["avg_score_weight"][f"recall@{k}"].append(recall_at_k(avg_rec, gt_items, k))
            
            # 4. Evaluate each base recaller
            for recaller_name in recaller_names:
                if recaller_name in recallers:
                    items = recallers[recaller_name].recall(user_id, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                    base_rec = [item[0] for item in items] if items else []
                    for k in [10, 20, 50]:
                        metrics[recaller_name][f"ndcg@{k}"].append(ndcg_at_k(base_rec, gt_items, k))
                        metrics[recaller_name][f"recall@{k}"].append(recall_at_k(base_rec, gt_items, k))
    
    # Aggregate results
    results = {}
    for method, method_metrics in metrics.items():
        results[method] = {}
        for metric_name, values in method_metrics.items():
            if values:
                results[method][metric_name] = np.mean(values)
    
    # Print comparison
    print("\n" + "="*60)
    print("Multi-Channel Recall Evaluation")
    print("="*60)
    
    # Print base recaller performance
    print(f"\n--- Base Recaller Performance ---")
    header = f"{'Metric':<15}" + "".join([f"{name:>15}" for name in recaller_names])
    print(header)
    print("-" * (15 + 15 * len(recaller_names)))
    for k in [10, 20, 50]:
        for metric in ['ndcg', 'recall']:
            key = f"{metric}@{k}"
            row = f"{key:<15}"
            for recaller_name in recaller_names:
                val = results.get(recaller_name, {}).get(key, 0)
                row += f"{val:>15.4f}"
            print(row)
    
    # Find best base model
    best_base_ndcg = 0
    best_base_name = None
    for recaller_name in recaller_names:
        ndcg50 = results.get(recaller_name, {}).get("ndcg@50", 0)
        if ndcg50 > best_base_ndcg:
            best_base_ndcg = ndcg50
            best_base_name = recaller_name
    
    # Print single select vs multi-channel comparison
    if results.get("single_select") and results.get("multi_channel"):
        print(f"\n--- Model Selection vs Multi-Channel vs Avg Score-Weight ---")
        print(f"{'Metric':<12} {'Single':>12} {'Multi-Ch':>12} {'Avg-SW':>12} {'Best Base':>12} {'Multi Impr':>12} {'Avg Impr':>12}")
        print("-" * 84)
        for k in [10, 20, 50]:
            for metric in ['ndcg', 'recall']:
                key = f"{metric}@{k}"
                single = results["single_select"].get(key, 0)
                multi = results["multi_channel"].get(key, 0)
                avg_sw = results["avg_score_weight"].get(key, 0)
                best_base = max(results.get(name, {}).get(key, 0) for name in recaller_names)
                multi_impr = multi - best_base
                avg_impr = avg_sw - best_base
                print(f"{key:<12} {single:>12.4f} {multi:>12.4f} {avg_sw:>12.4f} {best_base:>12.4f} {multi_impr:>+12.4f} {avg_impr:>+12.4f}")
    
    return results


def tokenize_function(examples, tokenizer, max_length=1536, seq2seq=False):
    if seq2seq:
        # For seq2seq: create input-output pairs
        # Input: user profile text + special prompt
        # Output: recaller name
        inputs = []
        targets = []
        
        for text, recaller in zip(examples["text"], examples["best_recaller"]):
            # Create input-output format for causal LM
            input_text = text + f"\n\nBest recaller:"
            target_text = f" {recaller}"
            
            inputs.append(input_text)
            targets.append(input_text + target_text)  # Full sequence for causal LM
        
        # Tokenize full sequences
        model_inputs = tokenizer(targets, padding="max_length", truncation=True, max_length=max_length)
        
        # Create labels: copy input_ids and set input portion to -100
        labels = []
        input_lengths = []
        
        for input_text in inputs:
            input_tokens = tokenizer(input_text, truncation=True, max_length=max_length)
            input_lengths.append(len(input_tokens["input_ids"]))
        
        for i, length in enumerate(input_lengths):
            label = model_inputs["input_ids"][i].copy()
            # Set input portion to -100 (ignored in loss)
            label[:length] = [-100] * length
            # Set padding tokens to -100
            for j, token_id in enumerate(label):
                if token_id == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        
        model_inputs["labels"] = labels
        return model_inputs
    else:
        # For classification: keep numeric labels
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        tokenized["labels"] = examples["labels"]
        return tokenized


def get_paths(args):
    """Get standardized paths"""
    model_name = args.model_name.split('/')[-1]
    base = f"{args.output_dir}/{args.dataset}/{model_name}"
    recbole_models = args.recbole_models
    recbole_models.sort()
    recbole_models = '_'.join(recbole_models)
    return {
        "model_name": args.model_name,
        "sft": f"{base}_pure_{f'seq2seq_' if args.seq2seq else ''}sft_{recbole_models}",
        "data": f"{base}_pure_sft_data_{recbole_models}_{args.profile_cutoff}",
        "grpo": f"{base}_pure_grpo_{recbole_models}",
        "grpo_data": f"{base}_pure_grpo_data_{recbole_models}_{args.profile_cutoff}"
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Pure Text SFT Training for Model Selection')
    # Data
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='GRPO/pure_models')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'LightGCN'])
    # Tasks
    parser.add_argument('--gen_sft_data', action='store_true')
    parser.add_argument('--do_sft', action='store_true')
    parser.add_argument('--do_test_sft', action='store_true')
    parser.add_argument('--do_test_grpo', action='store_true')
    # Training
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=1536)
    # Other
    parser.add_argument('--final_k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--padding_side', type=str, default='right', choices=['left', 'right'],
                       help='Padding side for tokenizer. Use "left" for decoder-only models in classification.')
    parser.add_argument('--profile_cutoff', type=int, default=20,
                       help='Cutoff length for user profiles and minimum history length for augmentation.')
    parser.add_argument('--seq2seq', action='store_true',
                       help='Use seq2seq model instead of classification model.')
    # GRPO training
    parser.add_argument('--do_grpo', action='store_true',
                       help='Run SofT-GRPO training on top of SFT model.')
    parser.add_argument('--tau_gumbel', type=float, default=1.0,
                       help='Gumbel-Softmax temperature for SofT-GRPO.')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling threshold for SofT-GRPO.')
    parser.add_argument('--noise_scale', type=float, default=0.3,
                       help='Scale factor for Gumbel noise (0.0=no noise, 1.0=full noise).')
    parser.add_argument('--use_soft_grpo_loss', action='store_true', default=True,
                       help='Use SofT-GRPO loss (Gumbel reparameterization). Default: True.')
    parser.add_argument('--use_ppo_loss', action='store_true',
                       help='Use PPO-style loss instead of SofT-GRPO loss.')
    parser.add_argument('--epsilon', type=float, default=0.2,
                       help='PPO clipping coefficient.')
    parser.add_argument('--beta', type=float, default=0.01,
                       help='KL penalty weight.')
    parser.add_argument('--sync_ref_model', action='store_true',
                       help='Periodically sync ref_model with current model.')
    parser.add_argument('--ref_model_sync_steps', type=int, default=100,
                       help='Steps between ref_model syncs.')
    parser.add_argument('--num_generations', type=int, default=4,
                       help='Number of generations per prompt (G in GRPO).')
    parser.add_argument('--grpo_lr', type=float, default=1e-6,
                       help='Learning rate for GRPO training.')
    parser.add_argument('--grpo_epochs', type=int, default=3,
                        help='Number of GRPO training epochs.')
    # Soft label options
    parser.add_argument('--use_soft_label', action='store_true',
                        help='Use soft labels based on score difference instead of hard winner labels.')
    parser.add_argument('--soft_label_temperature', type=float, default=5.0,
                        help='Temperature for soft label: higher = sharper distribution, lower = smoother.')
    parser.add_argument('--random_history_selection', action='store_true',
                        help='Randomly select profile_cutoff items from history instead of using the most recent ones.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    paths = get_paths(args)
    

    grpo_config = GRPOConfig(
        output_dir=paths["grpo"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.grpo_epochs,
        learning_rate=args.grpo_lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,
        bf16=True,
        # num_generations=2,
        num_generations=args.num_generations,
        epsilon=args.epsilon,
        beta=args.beta,
        sync_ref_model=args.sync_ref_model,
        ref_model_sync_steps=args.ref_model_sync_steps,
        scale_rewards="group",
        report_to="wandb",
        run_name=paths["grpo"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        seed=args.seed,
    )
    # Initialize components if needed
    if args.gen_sft_data or args.do_test_sft or args.do_test_grpo:
        inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
        profile_agent = UserProfileAgent(inter_dataset, args.dataset)
        recallers = initialize_recallers(
            model_names=args.recbole_models,
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            data_path=args.data_path,
            seed=args.seed,
            use_latest_checkpoint=True,
            num_items=inter_dataset.ds.item_num
        )
    
    # Generate SFT data
    if args.gen_sft_data:
        # Step 1: 先生成 Eval dataset，获取每个用户的最佳 recaller 作为 hint
        print("="*60)
        print("Step 1: Generating Eval dataset to get user hints...")
        print("="*60)
        eval_dataset, label2id, id2label, eval_user_hint_map = create_sft_dataset(
            profile_agent,
            inter_dataset.eval_user_ids[:1000],
            inter_dataset.eval_histories[:1000],
            inter_dataset.eval_target_items[:1000],
            recallers, args.final_k, args.profile_cutoff,
            use_augmentation=False,
            use_soft_label=args.use_soft_label,
            soft_label_temperature=args.soft_label_temperature,
            user_hint_map=None,
            use_self_hint=True,  # eval set 使用自己的 best_recaller 作为 hint
            random_history_selection=args.random_history_selection,
        )
        print(f"Collected hints for {len(eval_user_hint_map)} users from eval set")
        
        # Step 2: 使用 eval set 的 hint 生成 Train dataset
        print("\n" + "="*60)
        print("Step 2: Generating Train dataset with eval hints...")
        print("="*60)
        train_dataset, _, _, _ = create_sft_dataset(
            profile_agent, 
            inter_dataset.train_user_ids,
            inter_dataset.train_histories,
            inter_dataset.train_target_items,
            recallers, args.final_k, args.profile_cutoff,
            use_soft_label=args.use_soft_label,
            soft_label_temperature=args.soft_label_temperature,
            user_hint_map=eval_user_hint_map,  # 使用 eval set 的 hint
            random_history_selection=args.random_history_selection,
        )
        
        # Step 3: 使用 eval set 的 hint 生成 Test dataset
        print("\n" + "="*60)
        print("Step 3: Generating Test dataset with eval hints...")
        print("="*60)
        test_dataset, _, _, _ = create_sft_dataset(
            profile_agent,
            inter_dataset.test_user_ids,
            inter_dataset.test_histories,
            inter_dataset.test_target_items,
            recallers, args.final_k, args.profile_cutoff,
            use_augmentation=False,
            use_soft_label=args.use_soft_label,
            soft_label_temperature=args.soft_label_temperature,
            user_hint_map=eval_user_hint_map,  # 使用 eval set 的 hint
            random_history_selection=args.random_history_selection,
        )
        
        # Statistics: eval set label distribution
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        
        eval_label_counts = defaultdict(int)
        for sample in eval_dataset:
            eval_label_counts[sample["best_recaller"]] += 1
        print("\nEval set label distribution:")
        for label, count in sorted(eval_label_counts.items()):
            print(f"  {label}: {count} ({count/len(eval_dataset)*100:.1f}%)")
        
        # Statistics: eval/test label agreement per user
        # Build user_id -> label mapping for eval and test
        eval_user_labels = {sample["user_id"]: sample["best_recaller"] for sample in eval_dataset}
        test_user_labels = {sample["user_id"]: sample["best_recaller"] for sample in test_dataset}
        
        # Find common users
        common_users = set(eval_user_labels.keys()) & set(test_user_labels.keys())
        if common_users:
            same_label_count = sum(1 for uid in common_users if eval_user_labels[uid] == test_user_labels[uid])
            print(f"\nEval/Test label agreement:")
            print(f"  Common users: {len(common_users)}")
            print(f"  Same label: {same_label_count} ({same_label_count/len(common_users)*100:.1f}%)")
        else:
            print(f"\nNo common users between eval and test sets")
        
        # Save
        os.makedirs(paths["data"], exist_ok=True)
        train_dataset.save_to_disk(f'{paths["data"]}/train')
        eval_dataset.save_to_disk(f'{paths["data"]}/eval')
        test_dataset.save_to_disk(f'{paths["data"]}/test')
        with open(f'{paths["data"]}/label_mapping.json', 'w') as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        # 保存 eval set 的 user hint map
        with open(f'{paths["data"]}/eval_user_hint_map.json', 'w') as f:
            json.dump({str(k): v for k, v in eval_user_hint_map.items()}, f, indent=2)
        
        print(f"\nSaved: Train={len(train_dataset)}, Eval={len(eval_dataset)}, Test={len(test_dataset)}")
        return
    
    # Train model
    if args.do_sft:
        # Load data
        train_dataset = Dataset.load_from_disk(f'{paths["data"]}/train')
        eval_dataset = Dataset.load_from_disk(f'{paths["data"]}/eval')
        with open(f'{paths["data"]}/label_mapping.json', 'r') as f:
            labels = json.load(f)
            label2id = labels["label2id"]
            id2label = {int(k): v for k, v in labels["id2label"].items()}
        
        # Setup model
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        if args.seq2seq:
            # Try seq2seq first, fallback to causal LM
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            except:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                torch_dtype=dtype,
                device_map="auto"
            )
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = args.padding_side
        
        # Tokenize
        tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length, seq2seq=args.seq2seq)
        if args.seq2seq:
            columns_to_remove = ["text", "labels", "best_ndcg", "user_id", "history_len_used"]
        else:
            columns_to_remove = ["text", "best_recaller", "best_ndcg", "user_id", "history_len_used"]
        
        # Remove soft_labels column if not using soft label training
        if not args.use_soft_label and "soft_labels" in train_dataset.column_names:
            columns_to_remove.append("soft_labels")
        
        tokenized_train = train_dataset.map(tokenize_fn, batched=True, 
                                           remove_columns=columns_to_remove)
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, 
                                         remove_columns=columns_to_remove)
        
        # Train
        if args.seq2seq:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            compute_metrics_fn = partial(compute_metrics_seq2seq, tokenizer=tokenizer, label2id=label2id)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            compute_metrics_fn = compute_metrics
        
        # Use SoftLabelTrainer if soft labels are enabled
        TrainerClass = SoftLabelTrainer if args.use_soft_label else Trainer
        trainer = TrainerClass(
            model=model,
            args=TrainingArguments(
                output_dir=paths["sft"],
                save_total_limit=1,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                save_steps=1000,
                eval_steps=args.eval_steps,
                logging_steps=args.logging_steps,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                bf16=args.bf16,
                fp16=args.fp16 and not args.bf16,
                gradient_checkpointing=args.gradient_checkpointing,
                eval_strategy="steps",
                save_strategy="steps",
                metric_for_best_model="eval_loss",
                greater_is_better=False,  # loss 越低越好
                load_best_model_at_end=True,  # 训练结束时加载最佳模型
                report_to="wandb",
                run_name=paths["sft"] + ("_soft" if args.use_soft_label else "") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                seed=args.seed
            ),
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            **({"use_soft_label": args.use_soft_label} if args.use_soft_label else {}),
        )
        
        print("Training...")
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(paths["sft"])
    
    # GRPO Training
    if args.do_grpo:
        
        print("\n" + "="*60)
        print("Starting Pure GRPO Training")
        print("="*60)
        
        # Initialize recallers if not already done (only needed when running --do_grpo alone)
        if 'recallers' not in dir() or recallers is None:
            inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
            recallers = initialize_recallers(
                model_names=args.recbole_models,
                dataset_name=args.dataset,
                checkpoint_dir=args.checkpoint_dir,
                data_path=args.data_path,
                seed=args.seed,
                use_latest_checkpoint=True,
                num_items=inter_dataset.ds.item_num
            )
        
        # Load label mapping
        with open(f'{paths["data"]}/label_mapping.json', 'r') as f:
            labels = json.load(f)
            label2id = labels["label2id"]
            id2label = {int(k): v for k, v in labels["id2label"].items()}
        
        recaller_names = sorted(args.recbole_models)
        
        # Load SFT model as starting point
        sft_model_path = args.model_name
        # sft_model_path = paths["sft"]
        # if not os.path.exists(sft_model_path):
        #     print(f"Error: SFT model not found at {sft_model_path}. Run --do_sft first.")
        #     return
        
        # # Get last checkpoint if needed
        # if not os.path.exists(os.path.join(sft_model_path, "config.json")):
        #     last_ckpt = get_last_checkpoint(sft_model_path)
        #     if last_ckpt:
        #         sft_model_path = last_ckpt
        
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            sft_model_path,
            num_labels=len(label2id),
            torch_dtype=dtype,
            device_map="cuda"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        # Load SFT dataset directly for GRPO (already contains prompt, history, target_items)
        grpo_train_dataset = Dataset.load_from_disk(f'{paths["data"]}/train')
        grpo_eval_dataset_full = Dataset.load_from_disk(f'{paths["data"]}/eval')
        grpo_eval_dataset = grpo_eval_dataset_full.select(range(min(1000, len(grpo_eval_dataset_full))))
        print(f"GRPO training dataset: {len(grpo_train_dataset)} samples")
        print(f"GRPO eval dataset: {len(grpo_eval_dataset)} samples")
        
        # Reward function (placeholder - actual rewards computed in trainer)
        def grpo_reward_fn(prompts, completions, completion_ids, **kwargs):
            return [0.0] * len(prompts)
        
        # GRPO Config
        
        # Initialize GRPOTrainer with pure classification mode (like main_soft.py)
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=grpo_reward_fn,
            args=grpo_config,
            train_dataset=grpo_train_dataset,
            eval_dataset=grpo_eval_dataset,
            processing_class=tokenizer,
        )
        
        # Create reference model for KL penalty (override the default CausalLM ref_model)
        if args.beta > 0:
            ref_model = AutoModelForSequenceClassification.from_pretrained(
                sft_model_path,
                num_labels=len(label2id),
                torch_dtype=dtype,
                device_map="cuda"
            )
            ref_model.config.pad_token_id = tokenizer.eos_token_id
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            # Use standard ref_model attribute (enables SyncRefModelCallback)
            grpo_trainer.ref_model = ref_model
        
        # Enable pure classification GRPO mode (similar to use_beta_sampling in main_soft.py)
        grpo_trainer.use_pure_classification = True
        grpo_trainer.pure_recallers = recallers
        grpo_trainer.pure_recaller_names = recaller_names
        # Use k=5 for reward ndcg@k during GRPO (keep other stages unchanged)
        grpo_trainer.pure_final_k = 5
        grpo_trainer.pure_noise_scale = args.noise_scale
        # SofT-GRPO loss (default) vs PPO-style loss
        grpo_trainer.use_soft_grpo_loss = not args.use_ppo_loss
        # Keep extra columns for reward computation (not removed by _remove_unused_columns)
        grpo_trainer._signature_columns = ["prompt", "user_id", "history", "target_items"]
        
        print("Starting GRPO training...")
        grpo_trainer.train()
        grpo_trainer.save_model()
        tokenizer.save_pretrained(paths["grpo"])
        print(f"GRPO model saved to {paths['grpo']}")
    
    # Test model
    if args.do_test_sft or args.do_test_grpo:
        # Load model
        if args.do_test_sft:
            model_path = paths["sft"] if os.path.exists(paths["sft"]) else args.model_name
        elif args.do_test_grpo:
            model_path = paths["grpo"] if os.path.exists(paths["grpo"]) else args.model_name
        
        # Try to get the last checkpoint if main model directory exists
        if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
            last_checkpoint = get_last_checkpoint(model_path)
            if last_checkpoint:
                model_path = last_checkpoint
                print(f"Loading from last checkpoint: {model_path}")
            else:
                print(f"Warning: No checkpoint found in {model_path}")
        with open(f'{paths["data"]}/label_mapping.json', 'r') as f:
            labels = json.load(f)
            id2label = {int(k): v for k, v in labels["id2label"].items()}
        
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        # ///
        # model_path = args.model_name
        if args.seq2seq:
            # Try seq2seq first, fallback to causal LM
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
            except:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = args.padding_side
        
        # Load test dataset from disk (generated in gen_sft_data)
        if os.path.exists(f'{paths["data"]}/test'):
            test_dataset = Dataset.load_from_disk(f'{paths["data"]}/test')
            print(f"Loaded test dataset: {len(test_dataset)} samples")
        else:
            test_dataset, _, _, _ = create_sft_dataset(
                profile_agent,
                inter_dataset.test_user_ids,
                inter_dataset.test_histories,
                inter_dataset.test_target_items,
                recallers, args.final_k, args.profile_cutoff,
                use_augmentation=False,
                random_history_selection=args.random_history_selection,
            )
        
        # Evaluate
        if not args.seq2seq:
            results = evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers, args.final_k)
            
            # Also run multi-channel recall evaluation
            # Use recallers.keys() to ensure consistent lowercase naming
            recaller_names = sorted(recallers.keys())
            multi_results = evaluate_multi_channel_recall(
                model, tokenizer, test_dataset, recallers, recaller_names,
                args.final_k, use_softmax_weights=True
            )
            results["multi_channel_evaluation"] = multi_results
            
            # Add configuration info to results
            recaller_combo = "_".join(sorted(args.recbole_models))
            results["config"] = {
                "dataset": args.dataset,
                "recbole_models": args.recbole_models,
                "recaller_combo": recaller_combo,
                "model_name": args.model_name,
                "final_k": args.final_k,
                "profile_cutoff": args.profile_cutoff,
                "test_samples": len(test_dataset),
                "model_path": model_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Save results with recaller combo in filename
            os.makedirs("results", exist_ok=True)
            result_filename = f"results/pure_results_{args.dataset}_{recaller_combo}.json"
            with open(result_filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {result_filename}")
            
            # Also print a summary for easy comparison
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Dataset: {args.dataset}")
            print(f"Recaller Combo: {recaller_combo}")
            print(f"Classification Accuracy: {results['accuracy']:.4f}")
            print(f"Classification F1 Macro: {results['f1_macro']:.4f}")
            if 'avg_predicted_ndcg' in results:
                print(f"Predicted NDCG@{args.final_k}: {results['avg_predicted_ndcg']:.4f}")
                print(f"True Best NDCG@{args.final_k}: {results['avg_true_ndcg']:.4f}")
            if 'avg_predicted_recall' in results:
                print(f"Predicted Recall@{args.final_k}: {results['avg_predicted_recall']:.4f}")
                print(f"True Best Recall@{args.final_k}: {results['avg_true_recall']:.4f}")
            if 'base_model_results' in results:
                print(f"\nBase Model Performance (NDCG@{args.final_k}):")
                for name, res in results['base_model_results'].items():
                    print(f"  {name}: {res['avg_ndcg']:.4f}")
        else:
            print("Seq2seq evaluation not implemented yet. Model saved successfully.")


if __name__ == "__main__":
    main()