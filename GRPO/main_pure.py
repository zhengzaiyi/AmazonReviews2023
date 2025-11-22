import argparse
import json
import os
import random
from functools import partial
from typing import Any, List, Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding, Trainer
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

from GRPO.agents import UserProfileAgent
from GRPO.data import load_dataset
from GRPO.main import initialize_recallers
from GRPO.recallers import RecBoleRecaller
from GRPO.utils import set_seed, build_prompt, ndcg_at_k, recall_at_k
from collections import defaultdict


def create_sft_dataset(
    profile_agent: UserProfileAgent, 
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: List[RecBoleRecaller],
    final_k: int,
):
    """Create dataset where the model predicts the best recaller class"""
    dataset = []
    metrics = {recaller: defaultdict[Any, list](list) for recaller in recallers.keys()}
    
    # Ground-truth configuration options
    USE_MULTIPLE_GT = True  # Set to True to use multiple ground-truth items
    GT_RATIO = 0.2  # Use last 20% of history as ground-truth
    MIN_GT_ITEMS = 1  # Minimum number of ground-truth items
    MAX_GT_ITEMS = 5  # Maximum number of ground-truth items
    
    # Data augmentation configuration
    PROFILE_CUTOFF = 20  # Number of interactions to show in profile
    MIN_HISTORY_FOR_AUGMENTATION = 30  # Minimum history length to enable augmentation
    AUGMENTATION_STEP = 10  # Step size for creating multiple samples
    
    # Create label mapping
    recaller_names = sorted(list(recallers.keys()))
    label2id = {name: i for i, name in enumerate(recaller_names)}
    id2label = {i: name for name, i in label2id.items()}
    
    for i, uid in enumerate(user_ids):
        hist = histories[i]
        if 0 in hist:
            hist = hist[:hist.index(0)]
        
        # Determine if we should create multiple samples for this user
        if len(hist) >= MIN_HISTORY_FOR_AUGMENTATION:
            # Create multiple samples with different history lengths
            history_lengths = range(PROFILE_CUTOFF, len(hist), AUGMENTATION_STEP)
            # Ensure we include the full history as well
            history_lengths = list(history_lengths) + [len(hist)]
        else:
            # Single sample with full history
            history_lengths = [len(hist)]
        
        # Create samples for each history length
        for hist_len in history_lengths:
            # Use only the first hist_len interactions
            current_hist = hist[:hist_len]
            
            # Determine ground-truth items
            if USE_MULTIPLE_GT and len(current_hist) > MIN_GT_ITEMS:
                # Calculate number of ground-truth items from history
                n_gt = max(MIN_GT_ITEMS, min(MAX_GT_ITEMS, int(len(current_hist) * GT_RATIO)))
                # Use last n_gt items from history as ground-truth
                gt_items = current_hist[-n_gt:] + [target_items[i]]  # Include the next item too
                # Update history to exclude ground-truth items
                eval_hist = current_hist[:-n_gt]
            else:
                # Original behavior: only use the next item as ground-truth
                gt_items = [target_items[i]]
                eval_hist = current_hist
            
            # Skip if eval_hist is too short
            if len(eval_hist) < 5:
                continue
            
            # Find the best recaller based on NDCG with multiple ground-truth
            best_ndcg, best_recaller = -1, None
            for recaller_name in recallers.keys():
                items = recallers[recaller_name].recall(uid, int(final_k), eval_hist)  # Use eval_hist
                item_ids = [item[0] for item in items] if items else []
                ndcg = ndcg_at_k(item_ids, gt_items, k=final_k)  # Use gt_items list
                metrics[recaller_name]["ndcg"].append(ndcg)
                metrics[recaller_name]["recall@10"].append(recall_at_k(item_ids, gt_items, k=10))
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_recaller = recaller_name
            
            assert best_recaller is not None
            
            # Create prompt and label for classification with increased cutoff
            prompt = build_prompt(
                profile_agent.forward(uid, eval_hist, cut_off=PROFILE_CUTOFF), 
                available_models=list(recallers.keys()), 
                type='classification'
            )
            label = label2id[best_recaller]
            
            dataset.append({
                "text": prompt,  # For classification, use 'text' field
                "labels": label,  # Numeric label for classification
                "best_recaller": best_recaller,  # Keep for analysis
                "best_ndcg": best_ndcg,  # Store for analysis
                "num_gt_items": len(gt_items),  # Store number of ground-truth items
                "user_id": uid,  # Keep track of user
                "history_len_used": len(eval_hist),  # Track history length used
            })
    
    # Print dataset statistics
    print("\nDataset statistics:")
    for recaller_name in recallers.keys():
        avg_ndcg = sum(metrics[recaller_name]["ndcg"]) / len(metrics[recaller_name]["ndcg"])
        print(f"{recaller_name}: avg NDCG = {avg_ndcg:.4f}")
    
    # Count best model distribution
    best_model_counts = defaultdict(int)
    gt_counts = defaultdict(int)
    unique_users = set()
    history_len_distribution = defaultdict(int)
    
    for item in dataset:
        best_model_counts[item["best_recaller"]] += 1
        gt_counts[item["num_gt_items"]] += 1
        unique_users.add(item["user_id"])
        history_len_bucket = (item["history_len_used"] // 10) * 10
        history_len_distribution[history_len_bucket] += 1
    
    print("\nBest model distribution:")
    for model, count in best_model_counts.items():
        print(f"{model}: {count} ({count/len(dataset)*100:.1f}%)")
    
    print("\nGround-truth items distribution:")
    for num_gt, count in sorted(gt_counts.items()):
        print(f"{num_gt} ground-truth items: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    print(f"\nData augmentation statistics:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Unique users: {len(unique_users)}")
    print(f"  - Average samples per user: {len(dataset)/len(unique_users):.2f}")
    
    print("\nHistory length distribution in samples:")
    for hist_len, count in sorted(history_len_distribution.items()):
        print(f"  {hist_len}-{hist_len+9}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    if USE_MULTIPLE_GT:
        print(f"\nGround-truth configuration:")
        print(f"  - Using multiple ground-truth: {USE_MULTIPLE_GT}")
        print(f"  - Ground-truth ratio: {GT_RATIO:.1%}")
        print(f"  - Min ground-truth items: {MIN_GT_ITEMS}")
        print(f"  - Max ground-truth items: {MAX_GT_ITEMS}")
    
    print(f"\nData augmentation configuration:")
    print(f"  - Profile cutoff: {PROFILE_CUTOFF}")
    print(f"  - Min history for augmentation: {MIN_HISTORY_FOR_AUGMENTATION}")
    print(f"  - Augmentation step: {AUGMENTATION_STEP}")
    
    return Dataset.from_list(dataset), label2id, id2label


def compute_metrics(eval_pred):
    """Compute metrics for classification"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def evaluate_pure_model(
    model,
    tokenizer,
    test_dataset,
    id2label,
    device="cuda"
):
    """Evaluate the pure classification model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating"):
            # Prepare input
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=1536, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get prediction
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            all_predictions.append(prediction)
            all_labels.append(example["labels"])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_predictions, target_names=list(id2label.values())))
    
    # Print confusion matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_predictions)
    print("Labels:", [id2label[i] for i in range(len(id2label))])
    print(cm)
    
    # Class distribution statistics
    print("\n=== Class Distribution ===")
    pred_counts = defaultdict(int)
    true_counts = defaultdict(int)
    
    for pred in all_predictions:
        pred_counts[id2label[pred]] += 1
    for label in all_labels:
        true_counts[id2label[label]] += 1
        
    print("\nTrue Label Distribution:")
    total = len(all_labels)
    for model_name in sorted(true_counts.keys()):
        count = true_counts[model_name]
        print(f"{model_name}: {count} ({count/total*100:.1f}%)")
        
    print("\nPredicted Label Distribution:")
    for model_name in sorted(pred_counts.keys()):
        count = pred_counts[model_name]
        print(f"{model_name}: {count} ({count/total*100:.1f}%)")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(all_labels, all_predictions, target_names=list(id2label.values()), output_dict=True)
    }


def tokenize_function(examples, tokenizer, max_length=1536):
    """Tokenize the examples for classification"""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)


def parse_args():
    ap = argparse.ArgumentParser(description='Pure Text SFT Training for Model Selection')
    ap.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    ap.add_argument('--data_path', type=str, default='./dataset')
    ap.add_argument('--final_k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', ])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--output_dir', type=str, default='GRPO/pure_models')
    ap.add_argument('--gen_sft_data', action='store_true')
    ap.add_argument('--do_sft', action='store_true')
    ap.add_argument('--do_test', action='store_true')
    ap.add_argument('--use_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    ap.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    ap.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    ap.add_argument('--num_train_samples', type=int, default=1000, help='Number of training samples')
    ap.add_argument('--per_device_train_batch_size', type=int, default=4)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=5e-5)
    ap.add_argument('--num_train_epochs', type=int, default=3)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--logging_steps', type=int, default=10)
    ap.add_argument('--save_steps', type=int, default=500)
    ap.add_argument('--eval_steps', type=int, default=250)
    ap.add_argument('--max_length', type=int, default=1536)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--gradient_checkpointing', action='store_true')
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    
    model_save_dir = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Initialize dataset and recallers when needed
    if args.gen_sft_data or args.do_test:
        inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
        profile_agent = UserProfileAgent(inter_dataset, args.dataset)
        cut_off_users = min(len(inter_dataset.train_user_ids), 100000)
        recallers = initialize_recallers(
            model_names=args.recbole_models,
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            data_path=args.data_path,
            seed=args.seed,
            use_latest_checkpoint=True,
            num_items=inter_dataset.ds.item_num
        )
    
    if args.gen_sft_data:
        # Generate SFT data
        assert not args.do_sft and not args.do_test
        
        # Get training data
        user_ids = inter_dataset.train_user_ids[:cut_off_users]
        histories = inter_dataset.train_histories[:cut_off_users]
        target_items = inter_dataset.train_target_items[:cut_off_users]
        
        train_dataset, label2id, id2label = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=user_ids,
            histories=histories,
            target_items=target_items,
            recallers=recallers,
            final_k=args.final_k,
        )
        
        # Get test data for evaluation
        test_cut_off = min(len(inter_dataset.test_user_ids), 5000)  # Limit test size
        test_user_ids = inter_dataset.test_user_ids[:test_cut_off]
        test_histories = inter_dataset.test_histories[:test_cut_off]
        test_target_items = inter_dataset.test_target_items[:test_cut_off]
        
        eval_dataset, _, _ = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=test_user_ids,
            histories=test_histories,
            target_items=test_target_items,
            recallers=recallers,
            final_k=args.final_k,
        )
        
        # Save datasets and label mappings
        dataset_path = f'{model_save_dir}_pure_sft_data'
        if os.path.exists(dataset_path):
            import shutil
            shutil.rmtree(dataset_path)
        
        # Save train dataset
        train_dataset_path = f'{dataset_path}/train'
        train_dataset.save_to_disk(train_dataset_path)
        
        # Save eval dataset
        eval_dataset_path = f'{dataset_path}/eval'
        eval_dataset.save_to_disk(eval_dataset_path)
        
        # Save label mappings
        label_mapping = {
            "label2id": label2id,
            "id2label": id2label
        }
        with open(f'{dataset_path}/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"Train dataset saved to {train_dataset_path}")
        print(f"Eval dataset saved to {eval_dataset_path}")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of eval samples: {len(eval_dataset)}")
        print(f"Number of classes: {len(label2id)}")
        print(f"Classes: {list(label2id.keys())}")
        return
    
    if args.do_sft:
        model_save_dir = f"{model_save_dir}_pure_sft"
        

        dataset_path = f'{args.output_dir}/{args.dataset}/{args.model_name.split("/")[-1]}_pure_sft_data'
        # Load train and eval datasets
        train_dataset = Dataset.load_from_disk(f'{dataset_path}/train')
        eval_dataset = Dataset.load_from_disk(f'{dataset_path}/eval')
            
        # Load label mappings
        with open(f'{dataset_path}/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            label2id = label_mapping["label2id"]
            id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
            
        # Tokenize datasets
        tokenized_train_dataset = train_dataset.map(
            partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length),
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["labels"]]
        )
        
        tokenized_eval_dataset = eval_dataset.map(
            partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length),
            batched=True,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["labels"]]
        )
            
        print(f"Total training samples: {len(tokenized_train_dataset)}")
        print(f"Total evaluation samples: {len(tokenized_eval_dataset)}")
        print(f"Number of classes: {len(label2id)}")
        print(f"\nTraining set class distribution:")
        train_class_counts = defaultdict(int)
        for item in train_dataset:
            train_class_counts[item['best_recaller']] += 1
        for cls, count in sorted(train_class_counts.items()):
            print(f"  {cls}: {count} ({count/len(train_dataset)*100:.1f}%)")
            
            # Set up training arguments
        training_args = TrainingArguments(
                output_dir=model_save_dir,
                save_total_limit=2,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                save_strategy="steps",
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                eval_strategy="steps",  # Changed from evaluation_strategy
                eval_steps=1500,
                bf16=args.bf16,
                fp16=args.fp16 and not args.bf16,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
                gradient_checkpointing=args.gradient_checkpointing,
                run_name=f"pure_sft_{args.dataset}_lr{args.learning_rate}",
                report_to="none",
                metric_for_best_model="loss",
                greater_is_better=False,
                remove_unused_columns=True,
                dataloader_drop_last=True,
        )
            
            # Create data collator for classification
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
            
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
            
            # Train
        print("Starting training...")
        trainer.train()
            
        # Save final model
        print("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(model_save_dir)
        
        # Update model name for testing
        model_name = get_last_checkpoint(model_save_dir) or model_save_dir
        print(f"Using model: {model_name}")
    
    if args.do_test:
        # Determine model path
        if args.do_sft:
            # Use the just-trained model
            pass
        else:
            # Use specified model or find checkpoint
            model_save_dir_test = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_pure_sft"
            if os.path.exists(model_save_dir_test):
                model_name = get_last_checkpoint(model_save_dir_test) or model_save_dir_test
            else:
                model_name = args.model_name
        
        print(f"Testing model: {model_name}")
        
        # Load label mappings
        dataset_path = f'{args.output_dir}/{args.dataset}/{args.model_name.split("/")[-1]}_pure_sft_data'
        with open(f'{dataset_path}/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            label2id = label_mapping["label2id"]
            id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create test dataset
        cut_off_test_users = min(len(inter_dataset.test_user_ids), 100000)
        test_dataset, _, _ = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.test_user_ids[:cut_off_test_users],
            histories=inter_dataset.test_histories[:cut_off_test_users],
            target_items=inter_dataset.test_target_items[:cut_off_test_users],
            recallers=recallers,
            final_k=args.final_k,
        )
        
        # No need to apply chat template for classification
        # The test_dataset already has the correct format
        
        # Evaluate
        results = evaluate_pure_model(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            id2label=id2label,
            device=model.device
        )
        
        # Save results
        os.makedirs("results", exist_ok=True)
        result_file = f"results/pure_results_{args.dataset}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_file}")


if __name__ == "__main__":
    # Note: --gen_sft_data, --do_sft, and --do_test should be run separately
    # They cannot be combined in a single run due to GPU resource conflicts
    main()
