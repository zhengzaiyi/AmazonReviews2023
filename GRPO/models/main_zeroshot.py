"""
Zero-shot evaluation for recaller selection using instruction prompts.
No training required - directly use LLM to predict the best recaller.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Reuse from main_pure.py
from GRPO.models.main_pure import (
    create_sft_dataset,
    evaluate_baseline_recallers,
)
from GRPO.models.main import initialize_recallers
from GRPO.core.data import load_dataset
from GRPO.core.agents import UserProfileAgent
from GRPO.core.utils import set_seed, ndcg_at_k, recall_at_k
from GRPO.models.evaluation_utils import extract_eval_data
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    """Simplified args for zero-shot evaluation"""
    parser = argparse.ArgumentParser(description='Zero-shot Recaller Selection Evaluation')
    # Data
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='GRPO/pure_models')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--recbole_models', type=str, nargs='+', default=['itemknn', 'lightgcn', 'pop'])
    # Evaluation settings
    parser.add_argument('--train_k', type=int, default=50)
    parser.add_argument('--eval_k', type=int, default=50)
    parser.add_argument('--profile_cutoff', type=int, default=20)
    parser.add_argument('--prompt_top_k', type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to evaluate (for debugging)')
    # Flags to match main_pure.py interface (used by create_sft_dataset)
    parser.add_argument('--use_soft_label', action='store_true', default=False)
    parser.add_argument('--soft_label_temperature', type=float, default=5.0)
    parser.add_argument('--random_history_selection', action='store_true', default=False)
    parser.add_argument('--autoregressive', action='store_true', default=True,
                       help='Always True for zero-shot (use instruction prompt)')
    return parser.parse_args()


def extract_recaller_from_response(response: str, recaller_names: List[str]) -> str:
    """Extract recaller name from model response"""
    response_lower = response.lower().strip()
    # Try exact match first
    for name in recaller_names:
        if name.lower() == response_lower:
            return name
    # Try substring match
    for name in recaller_names:
        if name.lower() in response_lower:
            return name
    return None


def evaluate_zeroshot(
    model,
    tokenizer,
    test_dataset,
    recallers: Dict,
    recaller_names: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    eval_k: int = 50,
    max_new_tokens: int = 32,
):
    """Zero-shot evaluation: generate recaller name and compute metrics"""
    device = model.device
    model.eval()
    
    all_predictions = []
    all_labels = []
    recommendation_results = []
    raw_responses = []
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_dataset, desc="Zero-shot Evaluation")):
            prompt = example["text"]
            
            # Convert to chat format
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            chat_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only the generated part (assistant response)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            raw_responses.append(response)
            
            # Extract predicted recaller
            predicted_recaller_name = extract_recaller_from_response(response, recaller_names)
            true_recaller_name = example["best_recaller"]
            
            # Convert to label ids
            pred_id = label2id.get(predicted_recaller_name, -1) if predicted_recaller_name else -1
            true_id = label2id.get(true_recaller_name, -1)
            
            all_predictions.append(pred_id)
            all_labels.append(true_id)
            
            # Skip recommendation evaluation if prediction is invalid
            if predicted_recaller_name is None:
                continue
            
            # Get eval data
            user_id, eval_hist, gt_items, full_hist = extract_eval_data(example, idx)
            
            
            # Generate recommendations using predicted recaller
            pred_recaller = recallers.get(predicted_recaller_name)
            if pred_recaller:
                pred_items = pred_recaller.recall(user_id, eval_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                pred_item_ids = [item[0] for item in pred_items] if pred_items else []
                pred_ndcg = ndcg_at_k(pred_item_ids, gt_items, k=eval_k)
                pred_recall = recall_at_k(pred_item_ids, gt_items, k=eval_k)
            else:
                pred_item_ids, pred_ndcg, pred_recall = [], 0.0, 0.0
            
            # Generate recommendations using true recaller
            true_recaller = recallers.get(true_recaller_name)
            if true_recaller:
                true_items = true_recaller.recall(user_id, eval_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                true_item_ids = [item[0] for item in true_items] if true_items else []
                true_ndcg = ndcg_at_k(true_item_ids, gt_items, k=eval_k)
                true_recall = recall_at_k(true_item_ids, gt_items, k=eval_k)
            else:
                true_item_ids, true_ndcg, true_recall = [], 0.0, 0.0
            
            recommendation_results.append({
                "user_id": user_id,
                "predicted_recaller": predicted_recaller_name,
                "true_recaller": true_recaller_name,
                "raw_response": response,
                "predicted_ndcg": pred_ndcg,
                "true_ndcg": true_ndcg,
                "predicted_recall": pred_recall,
                "true_recall": true_recall,
                "correct_prediction": predicted_recaller_name == true_recaller_name,
                "history_length": len(eval_hist),
            })
    
    # Compute classification metrics (only on valid predictions)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    valid_mask = (all_predictions != -1) & (all_labels != -1)
    
    print(f"\nValid predictions: {np.sum(valid_mask)}/{len(all_predictions)} ({np.sum(valid_mask)/len(all_predictions)*100:.1f}%)")
    
    # Sample invalid responses
    invalid_indices = np.where(all_predictions == -1)[0][:5]
    if len(invalid_indices) > 0:
        print(f"\nSample invalid responses:")
        for i in invalid_indices:
            print(f"  [{i}] '{raw_responses[i][:100]}...'")
    
    if np.sum(valid_mask) > 0:
        valid_preds = all_predictions[valid_mask]
        valid_labels = all_labels[valid_mask]
        
        accuracy = accuracy_score(valid_labels, valid_preds)
        f1_macro = f1_score(valid_labels, valid_preds, average='macro', labels=list(id2label.keys()))
        
        print(f"\nClassification Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(valid_labels, valid_preds, 
                                    labels=list(id2label.keys()),
                                    target_names=list(id2label.values()),
                                    zero_division=0))
    else:
        accuracy, f1_macro = 0.0, 0.0
    
    result = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "valid_predictions_ratio": np.sum(valid_mask) / len(all_predictions),
        "total_samples": len(all_predictions),
    }
    
    # Recommendation metrics
    if recommendation_results:
        avg_pred_ndcg = np.mean([r["predicted_ndcg"] for r in recommendation_results])
        avg_true_ndcg = np.mean([r["true_ndcg"] for r in recommendation_results])
        avg_pred_recall = np.mean([r["predicted_recall"] for r in recommendation_results])
        avg_true_recall = np.mean([r["true_recall"] for r in recommendation_results])
        correct = sum([r["correct_prediction"] for r in recommendation_results])
        
        print(f"\nRecommendation @{eval_k}: Pred NDCG={avg_pred_ndcg:.4f} Recall={avg_pred_recall:.4f}")
        print(f"                     TrueBest NDCG={avg_true_ndcg:.4f} Recall={avg_true_recall:.4f}")
        print(f"                     Acc={correct/len(recommendation_results):.2%} ({correct}/{len(recommendation_results)})")
        
        result.update({
            "recommendation_results": recommendation_results,
            "avg_predicted_ndcg": avg_pred_ndcg,
            "avg_true_ndcg": avg_true_ndcg,
            "avg_predicted_recall": avg_pred_recall,
            "avg_true_recall": avg_true_recall,
            "recaller_prediction_accuracy": correct / len(recommendation_results),
        })
    
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Force autoregressive=True for instruction prompt
    args.autoregressive = True
    
    print("="*60)
    print("Zero-shot Recaller Selection Evaluation")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Recallers: {args.recbole_models}")
    
    # Load data and recallers
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
    
    recaller_names = sorted([name.lower() for name in args.recbole_models])
    label2id = {name: i for i, name in enumerate(recaller_names)}
    id2label = {i: name for name, i in label2id.items()}
    
    # Create test dataset with instruction prompt
    print("\nCreating test dataset with instruction prompts...")
    test_dataset, _, _, _, _ = create_sft_dataset(
        profile_agent,
        inter_dataset.test_user_ids,
        inter_dataset.test_histories,
        inter_dataset.test_target_items,
        recallers, args,
        use_augmentation=False,
        min_thres=0.001,
        prompt_type='instruction',
    )
    
    # if args.max_samples:
    #     test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Zero-shot evaluation
    print("\n" + "="*60)
    print("Running Zero-shot Evaluation")
    print("="*60)
    results = evaluate_zeroshot(
        model, tokenizer, test_dataset,
        recallers, recaller_names,
        label2id, id2label,
        eval_k=args.eval_k,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Baseline evaluation
    print("\n" + "="*60)
    print("Baseline Recallers Evaluation")
    print("="*60)
    baseline_results = evaluate_baseline_recallers(
        test_dataset, recallers, recaller_names, args.eval_k,
        save_predictions=False,
    )
    
    # Save results
    model_name_short = args.model_name.split("/")[-1]
    recaller_combo = "_".join(recaller_names)
    os.makedirs("results", exist_ok=True)
    
    output = {
        "config": {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "recbole_models": recaller_names,
            "eval_k": args.eval_k,
            "max_new_tokens": args.max_new_tokens,
            "test_samples": len(test_dataset),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "zeroshot_results": results,
        "baseline_results": baseline_results,
    }
    
    # Remove large nested data for cleaner output
    if "recommendation_results" in output["zeroshot_results"]:
        output["zeroshot_results"].pop("recommendation_results")
    if "predictions" in output["baseline_results"]:
        output["baseline_results"].pop("predictions")
    
    result_file = f"results/zeroshot_results_{args.dataset}_{model_name_short}_{recaller_combo}.json"
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
