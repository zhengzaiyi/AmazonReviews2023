import argparse
import os
import random
import json
import torch
from typing import List
from functools import partial
from datasets import Dataset

from GRPO.utils import set_seed, build_prompt, multi_channel_recall, ndcg_at_k, evaluate
from GRPO.data import load_dataset
from GRPO.agents import UserProfileAgent, LLMRouterAgent, create_model_config
from trl import GRPOConfig
from GRPO.trl_trainer import GRPOTrainer
from GRPO.main import initialize_recallers
from transformers import AutoModelForCausalLM, AutoTokenizer
from GRPO.recallers import BaseRecaller
from vllm.sampling_params import GuidedDecodingParams
from peft import LoraConfig, get_peft_model, TaskType


def create_trl_dataset(
    profile_agent, 
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: List[BaseRecaller]
):
    """Create TRL dataset containing information needed for reward calculation"""
    dataset = []
    for i, uid in enumerate(user_ids):
        hist = histories[i]
        prof_json = profile_agent.forward(uid, hist).profile_json
        content = build_prompt(prof_json, list(recallers.keys()))
        dataset.append({
            "prompt": [{"role": "user", "content": content}],
            "uid": uid,                           # User ID
            "histories": hist,                 # User history
            "target_items": [target_items[i]] if type(target_items[i]) == int else target_items[i]  # Test items
        })
    return Dataset.from_list(dataset)


def parse_args():
    ap = argparse.ArgumentParser(description='GRPO Training (TRL) for Recommendation Systems')
    ap.add_argument('--dataset', type=str, default='ml-100k')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--final_k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['SASRec', 'BPR'])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--use_hf_local', action='store_true')
    ap.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    ap.add_argument('--output_dir', type=str, default='./grpo_models')
    ap.add_argument('--do_train', action='store_true')
    ap.add_argument('--do_eval', action='store_true')
    ap.add_argument('--do_test', action='store_true')
    ap.add_argument('--use_vllm', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    ap.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    ap.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    ap.add_argument('--use_peft', action='store_true', help='Whether to use PEFT')
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    if not args.use_hf_local:
        raise RuntimeError("TRL entrypoint requires --use_hf_local to provide a local HF model.")
    
    
    set_seed(args.seed)

    assert args.do_train or args.do_eval or args.do_test, "At least one of --do_train, --do_eval, or --do_test must be True"
    assert not (args.do_train and args.do_test), "Only one of --do_train or --do_test can be True"

    inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=True
    )
    data_map_path = os.path.join(args.data_path, f'{args.dataset}', f'{args.dataset}.data_maps')
    reviews_path = os.path.join(args.data_path, f'{args.dataset}', f'{args.dataset}.reviews')
    with open(data_map_path, 'r') as f:
        data_maps = json.load(f)
    with open(reviews_path, 'r') as f:
        reviews = json.load(f)
    profile_agent = UserProfileAgent(inter_dataset.num_items, data_maps, reviews)

    model_config = create_model_config(
        available_models=list(recallers.keys()),
        default_configs=None,
        class_name="ModelConfigs"
    )
    
    # Initialize HF models
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.do_train:
        if not args.use_vllm:
            model = AutoModelForCausalLM.from_pretrained(
                args.hf_model, 
                trust_remote_code=True,
                # device_map="auto"  
            )
        def reward_func(completions, uid, histories, target_items, **kwargs):
            rewards = []
            recall_results = multi_channel_recall(
                completions=completions, 
                uid=uid, 
                histories=histories, 
                recallers=recallers, 
                final_k=args.final_k
            )
            for i, recall_result in enumerate(recall_results):
                ndcg = ndcg_at_k(recall_result, target_items[i], k=args.final_k)
                rewards.append(ndcg)
            return rewards
    elif args.do_test:
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            trust_remote_code=True,
            # device_map="auto"  
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    # Create trainer config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_model_impl="vllm",
        vllm_tensor_parallel_size=1,  # Each process uses only its assigned GPU
        vllm_guided_decoding_json=model_config.model_json_schema(),
        # vllm_gpu_memory_utilization=0.5,  # Increase from default 0.3 to 0.8
    )

    # Run training, evaluation, and testing
    if args.do_train:
        trl_train_dataset = create_trl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.train_user_ids,
            histories=inter_dataset.train_histories,
            target_items=inter_dataset.train_target_items,
            recallers=recallers
        )
        trainer = GRPOTrainer(
            model=args.hf_model if args.use_vllm else model,
            reward_funcs=reward_func,
            args=grpo_config,
            train_dataset=trl_train_dataset,
            peft_config=peft_config if args.use_peft else None,
        )
        trainer.train()
    if args.do_eval:
        trl_eval_dataset = create_trl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.eval_user_ids,
            histories=inter_dataset.eval_histories,
            target_items=inter_dataset.eval_target_items,
            recallers=recallers
        )
        completions = trainer.predict(trl_eval_dataset)
        metrics = evaluate(
            completions=completions,
            uid=trl_eval_dataset["uid"],
            histories=trl_eval_dataset["histories"],
            recallers=recallers,
            final_k=args.final_k,
        )
        print(metrics)
    if args.do_test:
        trl_test_dataset = create_trl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.test_user_ids,
            histories=inter_dataset.test_histories,
            target_items=inter_dataset.test_target_items,
            recallers=recallers
        )
        completions = trainer.predict(trl_test_dataset)
        metrics = evaluate(
            completions=completions,
            uid=trl_test_dataset["uid"],
            histories=trl_test_dataset["histories"],
            recallers=recallers,
            final_k=args.final_k,
        )
        print(metrics)


    
if __name__ == "__main__":
    main()