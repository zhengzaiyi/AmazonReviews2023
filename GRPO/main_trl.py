import argparse
import os
import random
import json
from typing import List
from functools import partial
from datasets import Dataset

from GRPO.utils import set_seed, build_prompt, ndcg_rewards
from GRPO.data import load_dataset
from GRPO.agents import UserProfileAgent, LLMRouterAgent
from trl import GRPOConfig
from GRPO.trl_trainer import GRPOTrainer
from GRPO.main import initialize_recallers
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_trl_dataset(profile_agent, histories, user2items_test, recallers):
    """Create TRL dataset containing information needed for reward calculation"""
    dataset = []
    for uid in histories.keys():
        hist = histories[uid]
        prof_json = profile_agent.forward(uid, hist).profile_json
        content = build_prompt(prof_json, list(recallers.keys()))
        dataset.append({
            "prompt": [{"role": "user", "content": content}],
            "uid": uid,                           # User ID
            "histories": hist,                 # User history
            "user2items_test": user2items_test.get(uid, [])  # Test items
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
    ap.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    ap.add_argument('--output_dir', type=str, default='./grpo_models')
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    if not args.use_hf_local:
        raise RuntimeError("TRL entrypoint requires --use_hf_local to provide a local HF model.")
    # TODO: Peft config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
    )

    set_seed(args.seed)
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
    trl_dataset = create_trl_dataset(
        profile_agent=profile_agent,
        histories=inter_dataset.user2item_list_train,
        user2items_test=inter_dataset.user2item_list_test,
        recallers=recallers
    )
    
    # Initialize HF models
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, 
        trust_remote_code=True,
        device_map="auto"
    )

    def reward_func(completions, uid, histories, user2items_test, **kwargs):
        return partial(
            ndcg_rewards,
            recallers=recallers, 
            final_k=args.final_k
        )(completions, uid, histories, user2items_test)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=trl_dataset,
    )
    
    trainer.train()

    return trainer

    
if __name__ == "__main__":
    main()