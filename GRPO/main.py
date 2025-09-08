import argparse
import os
import random
import numpy as np
from typing import List

from .utils import set_seed
from .data import load_dataset
from .recallers import RecBoleRecaller
from .agents import UserProfileAgent, LLMRouterAgent, HFLocalGenerator
from .selector import PreferenceSelectorNet, GRPOTrainer, GRPOConfig, run_router_only


def find_latest_checkpoint(model_name: str, checkpoint_dir: str) -> str:
    import glob
    pattern = os.path.join(checkpoint_dir, f"{model_name}-*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return ""
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def create_recaller(model_name: str, dataset_name: str, checkpoint_dir: str, data_path: str, seed: int, use_latest_checkpoint: bool = True) -> RecBoleRecaller:
    """Create a recaller with proper configuration for the specified model"""
    
    # Find checkpoint if requested
    checkpoint_path = ""
    if use_latest_checkpoint:
        checkpoint_path = find_latest_checkpoint(model_name, checkpoint_dir)
    
    # Model-specific configurations
    model_configs = {
        'BPR': {
            'seed': seed,
            'epochs': 20,
            'learning_rate': 0.001,
            'embedding_size': 64,
            'train_batch_size': 2048,
        },
        'SASRec': {
            'seed': seed,
            'epochs': 20,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'max_seq_length': 50,
            'train_batch_size': 256,
        },
        'Pop': {
            'seed': seed,
            'epochs': 1,  # Pop doesn't need training
        },
        'ItemKNN': {
            'seed': seed,
            'k': 50,
            'shrink': 0.0,
        },
        'FPMC': {
            'seed': seed,
            'epochs': 20,
            'learning_rate': 0.001,
            'embedding_size': 64,
        },
        'GRU4Rec': {
            'seed': seed,
            'epochs': 20,
            'learning_rate': 0.001,
            'embedding_size': 64,
            'hidden_size': 128,
        }
    }
    
    # Get configuration for the specific model
    config_dict = model_configs.get(model_name, {'seed': seed})
    
    # Create the recaller
    recaller = RecBoleRecaller(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        config_dict=config_dict
    )
    
    return recaller


def initialize_recallers(model_names: List[str], dataset_name: str, checkpoint_dir: str, data_path: str, seed: int, use_latest_checkpoint: bool = True) -> dict:
    """Initialize all recallers with proper error handling"""
    
    recallers = {}
    failed_models = []
    
    # Supported models in RecBole
    supported_models = ['BPR', 'SASRec', 'Pop', 'ItemKNN', 'FPMC', 'GRU4Rec']
    
    for model_name in model_names:
        # Normalize model name
        normalized_name = model_name.upper() if model_name.upper() in supported_models else model_name
        
        # Handle legacy model name mappings
        if model_name.lower() == 'itemcf':
            normalized_name = 'ItemKNN'
        
        print(f"Initializing {normalized_name} recaller...")
        
        if normalized_name not in supported_models:
            print(f"Warning: Model {model_name} not supported, skipping...")
            failed_models.append(model_name)
            continue
        
        recaller = create_recaller(
            model_name=normalized_name,
            dataset_name=dataset_name,
            checkpoint_dir=checkpoint_dir,
            data_path=data_path,
            seed=seed,
            use_latest_checkpoint=use_latest_checkpoint
        )
        
        recallers[model_name.lower()] = recaller
        print(f"✅ Successfully initialized {normalized_name} recaller")
    
    # Ensure we have at least one recaller
    if not recallers:
        print("No recallers successfully initialized, creating Pop model as fallback...")
        fallback_recaller = create_recaller(
            model_name='Pop',
            dataset_name=dataset_name,
            checkpoint_dir=checkpoint_dir,
            data_path=data_path,
            seed=seed,
            use_latest_checkpoint=False
        )
        recallers['pop'] = fallback_recaller
        print("✅ Fallback Pop recaller initialized")
    
    if failed_models:
        print(f"⚠️  Failed to initialize models: {failed_models}")
    
    print(f"📊 Total recallers initialized: {len(recallers)} ({list(recallers.keys())})")
    return recallers

def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description='GRPO Training for Recommendation Systems')
    ap.add_argument('--dataset', type=str, default='ml-100k')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    ap.add_argument('--users_per_batch', type=int, default=128)
    ap.add_argument('--group_size', type=int, default=4)
    ap.add_argument('--final_k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['SASRec', 'BPR'])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--use_latest_checkpoint', action='store_true')
    ap.add_argument('--router_only', action='store_true', help='Run in router-only mode')
    ap.add_argument('--train_router', action='store_true', default=True, help='Train router using GRPO when in router_only mode (default: True)')
    ap.add_argument('--eval_only', action='store_true', help='Only evaluate router without training (overrides --train_router)')
    ap.add_argument('--router_strategy', type=str, default='first', choices=['first','oracle'])
    ap.add_argument('--save_router_json', type=str, default='')
    ap.add_argument('--router_batch_size', type=int, default=32, help='Batch size for router training')
    ap.add_argument('--use_hf_local', action='store_true')
    ap.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.2-1B')
    ap.add_argument('--hf_dtype', type=str, default='auto')
    ap.add_argument('--hf_device', type=str, default='auto')
    ap.add_argument('--api_base', type=str, default='')
    ap.add_argument('--api_key', type=str, default='')
    ap.add_argument('--api_model', type=str, default='')
    args = ap.parse_args(argv)

    set_seed(args.seed)
    inter = load_dataset(args.dataset, args.data_path, seed=args.seed)

    # Initialize recallers using the new simplified approach
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=args.use_latest_checkpoint
    )

    profile_agent = UserProfileAgent(inter.num_items)
    local_hf = None
    if args.use_hf_local:
        local_hf = HFLocalGenerator(args.hf_model, dtype=args.hf_dtype, device=args.hf_device)

    router = LLMRouterAgent(n_per_user=args.group_size, local_hf=local_hf, available_models=list(recallers.keys()))

    if args.router_only:
        users = list(range(inter.num_users))
        save_path = args.save_router_json if args.save_router_json else None
        
        # Determine if we should train or just evaluate
        should_train_router = args.train_router and not args.eval_only
        
        if should_train_router:
            # Training mode for router
            print("Training router using GRPO...")
            
            # Create trainer for router training (no selector needed in router-only mode)
            trainer = GRPOTrainer(None, GRPOConfig(beta=1.0, group_size=args.group_size, lr=3e-4), router=router)
            
            for ep in range(args.epochs):
                print(f"Epoch {ep+1}/{args.epochs}")
                random.shuffle(users)
                
                res = run_router_only(
                    users, 
                    inter.user2item_list_train, 
                    inter.user2items_test, 
                    profile_agent, 
                    router, 
                    recallers, 
                    final_k=args.final_k, 
                    group_size=args.group_size, 
                    strategy=args.router_strategy, 
                    save_router_json=save_path,
                    grpo_trainer=trainer,
                    train_router=True,
                    batch_size=args.router_batch_size
                )
                print(f"[Epoch {ep+1}] Router Training - Loss: {res.get('avg_loss', 0.0):.4f}, Avg Recall@{args.final_k}: {res['avg_recall']:.4f}")
            
            print(f"Done. Router training completed. Final avg Recall@{args.final_k} = {res['avg_recall']:.4f}")
        else:
            # Evaluation mode for router (original behavior)
            print("Evaluating router (no training)...")
            res = run_router_only(
                users, 
                inter.user2item_list_train, 
                inter.user2items_test, 
                profile_agent, 
                router, 
                recallers, 
                final_k=args.final_k, 
                group_size=args.group_size, 
                strategy=args.router_strategy, 
                save_router_json=save_path,
                train_router=False
            )
            print(f"Done. Router-only evaluation avg Recall@{args.final_k} = {res['avg_recall']:.4f}")
        return

    selector = PreferenceSelectorNet(available_models=list(recallers.keys()))
    trainer = GRPOTrainer(selector, GRPOConfig(beta=1.0, group_size=args.group_size, lr=3e-4))

    all_users = list(range(inter.num_users))
    for ep in range(args.epochs):
        random.shuffle(all_users)
        batches = [all_users[i:i+args.users_per_batch] for i in range(0, len(all_users), args.users_per_batch)]
        logs = []
        for bu in batches:
            out = trainer.step(bu, inter.user2item_list_train, inter.user2items_test, profile_agent, router, recallers, final_k=args.final_k)
            logs.append(out)
        avg_loss = np.mean([x['loss'] for x in logs if isinstance(x['loss'], (int,float))])
        avg_recall = np.mean([x['avg_recall'] for x in logs]) if logs else 0.0
        print(f"[Epoch {ep+1}] loss={avg_loss:.4f} avg Recall@{args.final_k}={avg_recall:.4f}")

    print("Done. Selector trained.")


if __name__ == '__main__':
    main()
