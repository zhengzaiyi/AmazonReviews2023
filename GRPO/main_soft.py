import argparse
import json
import os
import random
from functools import partial
from typing import List, Dict

import outlines
import torch
from datasets import Dataset
from huggingface_hub import InferenceClient
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, SFTTrainer, SFTConfig
from vllm.sampling_params import GuidedDecodingParams

from GRPO.agents import UserProfileAgent, LLMRouterAgent, create_model_config
from GRPO.data import load_dataset
from GRPO.main import initialize_recallers
from GRPO.recallers import RecBoleRecaller
from GRPO.trl_trainer import GRPOTrainer
from GRPO.utils import set_seed, build_prompt, multi_channel_recall, ndcg_at_k, evaluate, evaluate_recallers, recall_at_k
from accelerate import Accelerator
from collections import defaultdict
from GRPO.soft_model import get_model_and_tokenizer, get_mixed_model_and_tokenizer
from GRPO.soft_sft_trainer import SoftSFTTrainer

def create_sft_dataset(
    profile_agent: UserProfileAgent, 
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: List[RecBoleRecaller],
    final_k: int,
    norm_type: str = 'static', # oracle, static
):
    dataset = []
    metrics = {recaller: defaultdict(list) for recaller in recallers.keys()}
    for i, uid in enumerate(user_ids):
        hist = histories[i]
        if 0 in hist:
            hist = hist[:hist.index(0)]
        best_ndcg, best_recaller = -1, None
        for recaller_name in recallers.keys():
            items = recallers[recaller_name].recall(uid, int(final_k), histories[i])
            item_ids = [item[0] for item in items] if items else []
            metrics[recaller_name]["ndcg"].append(ndcg_at_k(item_ids, [target_items[i]], k=final_k))
            metrics[recaller_name]["recall@10"].append(recall_at_k(item_ids, [target_items[i]], k=10))
            if metrics[recaller_name]["ndcg"][-1] > best_ndcg:
                best_ndcg = metrics[recaller_name]["ndcg"][-1]
                best_recaller = recaller_name
        assert best_recaller is not None
        prompt = build_prompt(profile_agent.forward(uid, hist), available_models=recallers.keys())
        if norm_type == 'oracle':
            ground_truth_output = {
                recaller_name: {
                    "top-k": 1.0 if recaller_name == best_recaller else 0,
                    "score-weight": 1.0 if recaller_name == best_recaller else 0.0
                } for recaller_name in recallers.keys()
            }
        elif norm_type == 'static':
            recaller_scores = {name: ndcg_at_k([item[0] for item in items], [target_items[i]], k=final_k) 
                            if (items := recallers[name].recall(uid, final_k, histories[i])) else 0
                            for name in recallers.keys()}
            
            import numpy as np
            scores = np.array(list(recaller_scores.values()))
            weights = np.exp(scores * 2) / np.exp(scores * 2).sum()  # softmax with temperature=0.5
            weights = weights * 0.9 + np.random.dirichlet(np.ones(len(weights))) * 0.1  # 10% noise

            total_k = int(final_k * 1.5)
            k_values = np.maximum(5, (weights * total_k).astype(int))
            k_values = (k_values * min(1, total_k / k_values.sum())).astype(int)
            k_values = k_values / final_k
            
            # k_values is already normalized by final_k, so it should be in a reasonable range
            # We keep the normalized k_values (not converting back to int) for Beta distribution
            ground_truth_output = {
                name: {"top-k": float(k), "score-weight": float(w)}
                for name, k, w in zip(recaller_scores.keys(), k_values, weights/weights.sum())
            }
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        
        completion_lines = ["{"]
        value_targets = []
        for recaller_name in recallers.keys():
            completion_lines.append(f"  {recaller_name}: {{")
            inner_lines = []
            hparams = ['top-k', 'score-weight']
            for hp_idx, hp in enumerate(hparams):
                suffix = "," if hp_idx < len(hparams) - 1 else ""
                inner_lines.append(f"    {hp}: [num][soft_token]{suffix}")
                value_targets.append(float(ground_truth_output[recaller_name][hp]))
            completion_lines.extend(inner_lines)
            completion_lines.append("  },")
        completion_lines[-1] = completion_lines[-1].rstrip(",")
        completion_lines.append("}")
        completion = "\n".join(completion_lines)
        dataset.append({
            "prompt": prompt + "\n\n",
            "completion": completion,
            "value_targets": value_targets,
        })
    return Dataset.from_list(dataset)

def get_soft_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["[num]", "[soft_token]"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


class SoftDataCollator:
    """Custom data collator that supports mixed token-level supervision.
    
    For each feature we expect:
    - prompt: base prompt text
    - completion: completion text containing placeholders like [num][soft_token]
    - value_targets: list of floats aligned with the occurrences of [soft_token] in completion
    """
    
    def __init__(self, tokenizer, max_length=1536):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = -100  # Standard PyTorch ignore index for cross-entropy
        self.soft_token_id = tokenizer.convert_tokens_to_ids("[soft_token]")
    
    def __call__(self, features):
        batch_size = len(features)
        
        prompts = [f["prompt"] for f in features]
        completions = [f["completion"] for f in features]
        value_targets_list = [f.get("value_targets", []) for f in features]
        
        prompt_encodings = []
        for prompt in prompts:
            encoding = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                # max_length=self.max_length,
                return_tensors=None,
            )
            prompt_encodings.append(encoding)
        
        full_texts = [
            prompt + completion for prompt, completion in zip(prompts, completions)
        ]
        
        model_inputs = self.tokenizer(
            full_texts,
            # max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        value_labels = torch.zeros_like(input_ids, dtype=torch.float32)
        value_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for idx in range(batch_size):
            prompt_length = len(prompt_encodings[idx]["input_ids"])
            labels[idx, :prompt_length] = self.ignore_index
            
            soft_positions = (input_ids[idx] == self.soft_token_id).nonzero(as_tuple=True)[0]
            targets = value_targets_list[idx]
            if len(targets) != len(soft_positions):
                raise ValueError(
                    f"Mismatch between value_targets ({len(targets)}) and [soft_token] occurrences ({len(soft_positions)})"
                )
            if len(soft_positions) > 0:
                value_mask[idx, soft_positions] = True
                value_labels[idx, soft_positions] = torch.tensor(targets, dtype=torch.float32)
                labels[idx, soft_positions] = self.ignore_index
        
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        model_inputs["labels"] = labels
        model_inputs["value_labels"] = value_labels
        model_inputs["value_mask"] = value_mask
        
        return model_inputs
    
    def __call__alternative(self, features):
        """
        Alternative implementation using token search instead of length calculation.
        This might be more accurate for some tokenizers.
        """
        batch_size = len(features)
        
        # Extract data
        prompts = [f['prompt'] for f in features]
        completions = [f['completion'] for f in features]
        value_labels = [f['value_labels'] for f in features]
        
        # Get special token ids
        num_token_id = self.tokenizer.convert_tokens_to_ids('[num]')
        soft_token_id = self.tokenizer.convert_tokens_to_ids('[soft_token]')
        
        # Tokenize full sequences
        full_texts = [p + c for p, c in zip(prompts, completions)]
        model_inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels
        labels = model_inputs["input_ids"].clone()
        
        # Find completion tokens and mask everything before them
        for idx in range(batch_size):
            input_ids = model_inputs["input_ids"][idx]
            
            # Find the position of [num] or [soft_token]
            num_pos = (input_ids == num_token_id).nonzero(as_tuple=True)[0]
            soft_pos = (input_ids == soft_token_id).nonzero(as_tuple=True)[0]
            
            if len(num_pos) > 0:
                # Mask everything before [num]
                completion_start = num_pos[0].item()
                labels[idx, :completion_start] = self.ignore_index
            elif len(soft_pos) > 0:
                # Mask everything before [soft_token]
                completion_start = soft_pos[0].item()
                labels[idx, :completion_start] = self.ignore_index
            else:
                # If neither token found, mask everything (shouldn't happen)
                print(f"Warning: No completion token found in example {idx}")
                labels[idx, :] = self.ignore_index
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        model_inputs["labels"] = labels
        model_inputs["value_labels"] = torch.tensor(value_labels, dtype=torch.float32)
        
        return model_inputs


def parse_args():
    ap = argparse.ArgumentParser(description='Soft Token SFT Training for Recommendation Systems')
    ap.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    ap.add_argument('--data_path', type=str, default='./dataset')
    ap.add_argument('--final_k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'FPMC', 'Pop', 'ItemKNN'])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--output_dir', type=str, default='GRPO/soft_models')
    ap.add_argument('--gen_sft_data', action='store_true')
    ap.add_argument('--do_sft', action='store_true')
    ap.add_argument('--do_test', action='store_true')
    ap.add_argument('--use_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    ap.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    ap.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    ap.add_argument('--norm_type', type=str, default='static', choices=['oracle', 'static'])
    ap.add_argument('--num_train_samples', type=int, default=1000, help='Number of training samples')
    ap.add_argument('--per_device_train_batch_size', type=int, default=2)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=5e-5)
    ap.add_argument('--num_train_epochs', type=int, default=3)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--logging_steps', type=int, default=10)
    ap.add_argument('--save_steps', type=int, default=500)
    ap.add_argument('--eval_steps', type=int, default=100)
    ap.add_argument('--max_length', type=int, default=1536)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gradient_checkpointing', action='store_true')
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    
    model_save_dir = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load dataset
    
    
    # Generate or load SFT data - only initialize recallers when generating data or testing
    if args.gen_sft_data or args.do_test:
        inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
        profile_agent = UserProfileAgent(inter_dataset, args.dataset)
        cut_off_users = min(len(inter_dataset.train_user_ids), args.num_train_samples)
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
        # Generate SFT data separately to avoid GPU conflicts
        assert not args.do_sft and not args.do_test
        
        # Get training data
        user_ids = inter_dataset.train_user_ids[:cut_off_users]
        histories = inter_dataset.train_histories[:cut_off_users]
        target_items = inter_dataset.train_target_items[:cut_off_users]
        
        sft_dataset = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=user_ids,
            histories=histories,
            target_items=target_items,
            recallers=recallers,
            final_k=args.final_k,
            norm_type=args.norm_type
        )
        
        if os.path.exists(f'{model_save_dir}_sft_data'):
            import shutil
            shutil.rmtree(f'{model_save_dir}_sft_data')
        sft_dataset.save_to_disk(f'{model_save_dir}_sft_data')
        print(f"SFT dataset saved to {model_save_dir}_sft_data")
        return
    
    if args.do_sft:
        model_save_dir = f"{model_save_dir}_sft"
        if not os.path.exists(model_save_dir) or len(os.listdir(model_save_dir)) == 0:
            # Load model and tokenizer
            print("Loading model and tokenizer...")
            model, tokenizer = get_mixed_model_and_tokenizer(args.model_name)
            
            # Add padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Enable gradient checkpointing if requested
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Apply LoRA if requested
            if args.use_lora:
                print("Applying LoRA...")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    modules_to_save=["value_head"]  # Also train the value head
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            
            # Load dataset
            sft_dataset = Dataset.load_from_disk(f'{model_save_dir[:-4]}_sft_data')
            eval_dataset = sft_dataset.select(range(min(100, len(sft_dataset))))
            
            print(f"Total training samples: {len(sft_dataset)}")
            print(f"Sample data point: {sft_dataset[0]}")
            
            # Set up training arguments
            sft_config = TrainingArguments(
                output_dir=model_save_dir,
                save_total_limit=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                save_strategy="steps",
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                eval_steps=args.eval_steps,
                bf16=True,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
                gradient_checkpointing=args.gradient_checkpointing,
                run_name=f"soft_sft_{args.dataset}_lr{args.learning_rate}",
                report_to="none",
                metric_for_best_model="loss",
                greater_is_better=False,
                remove_unused_columns=False,  # Important for custom columns
                label_names=["labels", "value_labels", "value_mask"],  # Explicitly declare our label fields
                fp16=args.fp16 and not args.bf16,
                dataloader_drop_last=True,
            )
            
            # Create data collator
            data_collator = SoftDataCollator(tokenizer=tokenizer, max_length=args.max_length)
            
            # Create trainer
            trainer = SoftSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            print("Starting training...")
            trainer.train()
            
            # Save final model
            print("Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(model_save_dir)
        
        model_name = get_last_checkpoint(model_save_dir)
        print(f"Using checkpoint: {model_name}")
    
    if args.do_test:
        # Test the model
        if args.do_sft:
            model_save_dir = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_sft"
            model_name = get_last_checkpoint(model_save_dir)
        else:
            model_name = args.model_name
        
        print(f"Testing model: {model_name}")
        # Load test data and evaluate
        # TODO: Add evaluation logic here


if __name__ == "__main__":
    main()