from typing import Dict, List, Optional, Union
import os
import torch
import numpy as np

from .data import InteractionData

class BaseRecaller:
    def __init__(self, name: str, num_items: int):
        self.name = name
        self.num_items = num_items
    def recall(self, user_id: int, topk: int, history: List[int]) -> List[int]:
        raise NotImplementedError


class RecBoleRecaller(BaseRecaller):
    def __init__(self, model_name: str, dataset_name: str, checkpoint_path: str,
                 data_path: str = "./data", config_dict: dict = None):
        super().__init__(model_name.lower(), 0)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.config_dict = config_dict or {}
        self._init_recbole_model()

    def _get_model_config(self):
        """Get specific configuration for each model type"""
        
        # Base configuration
        base_config = {
            'data_path': self.data_path,
            'benchmark_filename': ['train', 'valid', 'test'],
            'epochs': 10,
            'stopping_step': 10,
            'eval_step': 1,
            'metrics': ['Recall', 'NDCG'],
            'topk': [10, 20],
            'valid_metric': 'NDCG@10',
            'checkpoint_dir': './checkpoints/',
            'show_progress': True
        }
        
        # Model-specific configurations based on RecBole documentation
        if self.model_name == 'BPR':
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id']},
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1,
                    'alpha': 1.0,
                    'dynamic': False,
                    'candidate_num': 0
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'learning_rate': 0.001,
                'train_batch_size': 2048,
            }
            
        elif self.model_name == 'SASRec':
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id_list', 'item_id']},
                'alias_of_item_id': ['item_id_list'],
                'train_neg_sample_args': None,
                'loss_type': 'CE',
                'learning_rate': 0.001,
                'train_batch_size': 256,
                'max_seq_length': 50,
                'hidden_size': 64,
                'n_layers': 2,
                'n_heads': 2,
                'inner_size': 256,
                'hidden_dropout_prob': 0.5,
                'attn_dropout_prob': 0.5,
                'hidden_act': 'gelu',
                'layer_norm_eps': 1e-12,
                'initializer_range': 0.02,
            }
            
        elif self.model_name == 'Pop':
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id']},
                'train_neg_sample_args': None,
                'epochs': 1,  # Pop model doesn't need many epochs
            }
            
        elif self.model_name == 'ItemKNN':
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id']},
                'train_neg_sample_args': None,
                'k': 50,  # Number of similar items
                'shrink': 0.0,  # Shrinkage parameter
            }
            
        elif self.model_name == 'FPMC':
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id_list', 'item_id']},
                'alias_of_item_id': ['item_id_list'],
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'learning_rate': 0.001,
            }
            
        elif self.model_name == 'GRU4Rec':
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id_list', 'item_id']},
                'alias_of_item_id': ['item_id_list'],
                'train_neg_sample_args': None,
                'loss_type': 'CE',
                'embedding_size': 64,
                'hidden_size': 128,
                'num_layers': 1,
                'dropout_prob': 0.3,
                'learning_rate': 0.001,
            }
            
        else:
            # Default configuration for other models
            model_config = {
                **base_config,
                'load_col': {'inter': ['user_id', 'item_id']},
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'learning_rate': 0.001,
            }
        
        # Update with user-provided config
        model_config.update(self.config_dict)
        return model_config

    def _init_recbole_model(self):
        # Import RecBole modules (no try-catch, assume RecBole is installed)
        from recbole.utils import get_trainer, get_model as get_recbole_model
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed as recbole_init_seed

        # Get model class
        self.model_class = get_recbole_model(self.model_name)
        
        # Get model-specific configuration
        model_config = self._get_model_config()

        # Create RecBole config
        self.config = Config(
            model=self.model_name, 
            dataset=self.dataset_name, 
            config_dict=model_config
        )
        
        # Initialize seed
        recbole_init_seed(self.config['seed'], self.config['reproducibility'])
        
        # Create dataset and prepare data
        self.dataset = create_dataset(self.config)
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)
        self.num_items = self.dataset.item_num
        
        # Initialize model
        self.model = self.model_class(self.config, self.dataset).to(self.config['device'])

        # Load checkpoint if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.config['device'], weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.eval()
            
        # Initialize trainer
        self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)
        
        # Build user-item mappings
        self._build_user_item_mappings()

    def _build_user_item_mappings(self):
        """Build user-item mappings following the same logic as data.py"""
        self.user2items_train = {}
        self.user2items_test = {}
        self.user2item_list_train = {}
        self.user2item_list_test = {}
        
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        
        # Build training mappings
        if hasattr(self.train_data.dataset.inter_feat, 'interaction'):
            # For sequential models with item_id_list
            interaction = self.train_data.dataset.inter_feat.interaction
            train_user_ids = interaction['user_id']
            train_item_id_lists = interaction['item_id_list'] if 'item_id_list' in interaction.keys() else [[]]*len(train_user_ids)
            train_item_ids = interaction['item_id']
                
            for i in range(len(self.train_data.dataset.inter_feat)):
                user_id = int(train_user_ids[i])
                item_id = int(train_item_ids[i])
                item_id_list = train_item_id_lists[i]

                # Build historical sequence (item_id_list)
                self.user2item_list_train[user_id] = [int(it) for it in item_id_list if it > 0]

                # Build target items (item_id)
                self.user2items_train.setdefault(user_id, []).append(item_id)
        else:
            # Fallback for older RecBole versions
            for inter in self.train_data.dataset.inter_feat:
                user_id = int(inter[uid_field])
                item_id = int(inter[iid_field])
                self.user2items_train.setdefault(user_id, []).append(item_id)
                self.user2item_list_train[user_id] = self.user2items_train[user_id][:-1]
        
        # Build test mappings  
        if hasattr(self.test_data.dataset.inter_feat, 'interaction'):
            # For sequential models with item_id_list
            inter = self.test_data.dataset.inter_feat.interaction
            test_user_ids = inter['user_id']
            test_item_id_lists = inter['item_id_list'] if 'item_id_list' in inter.keys() else [[]]*len(test_user_ids)
            test_item_ids = inter['item_id']
                
            for i in range(len(self.test_data.dataset.inter_feat)):
                user_id = int(test_user_ids[i])
                item_id = int(test_item_ids[i])
                item_id_list = test_item_id_lists[i]
                
                # Build historical sequence (item_id_list)
                self.user2item_list_test[user_id] = [int(it) for it in item_id_list if it > 0]
                
                # Build target items (item_id)
                self.user2items_test.setdefault(user_id, []).append(item_id)
        else:
            # Fallback for older RecBole versions
            for inter in self.test_data.dataset.inter_feat:
                user_id = int(inter[uid_field])
                item_id = int(inter[iid_field])
                self.user2items_test.setdefault(user_id, []).append(item_id)
                self.user2item_list_test[user_id] = self.user2items_train.get(user_id, [])

    def _create_user_interaction(self, user_id: int) -> Optional[object]:
        """Create a proper Interaction object for the given user"""
        try:
            from recbole.data.interaction import Interaction
            
            # Create interaction dict with user information
            interaction_dict = {}
            uid_field = self.dataset.uid_field
            interaction_dict[uid_field] = torch.tensor([user_id], device=self.config['device'])
            
            # For sequential models, add item_id_list if needed
            if hasattr(self.dataset, 'field2type') and 'item_id_list' in self.dataset.field2type:
                # Get user's historical sequence
                hist_items = self.user2item_list_train.get(user_id, [])
                if hist_items:
                    # Pad or truncate to max_seq_length if needed
                    max_seq_len = 50
                    if len(hist_items) > max_seq_len:
                        hist_items = hist_items[-max_seq_len:]
                    # Ensure proper tensor shape [batch_size, seq_len]
                    hist_tensor = torch.tensor([hist_items], dtype=torch.long, device=self.config['device'])
                else:
                    # Empty sequence with proper shape
                    hist_tensor = torch.tensor([[0]], dtype=torch.long, device=self.config['device'])
                interaction_dict['item_id_list'] = hist_tensor
            
            # Add other fields that might be required
            # Length field for sequential models
            if 'item_length' in getattr(self.dataset, 'field2type', {}):
                hist_len = len(self.user2item_list_train.get(user_id, []))
                interaction_dict['item_length'] = torch.tensor([hist_len], device=self.config['device'])
            
            # Create the Interaction object
            return Interaction(interaction_dict)
            
        except Exception as e:
            print(f"Warning: Failed to create interaction for user {user_id}: {e}")
            return None

    def recall(self, user_id: int, topk: int, history: List[int]) -> List[int]:
        """Generate recommendations for a user
        
        Args:
            user_id: The user ID to generate recommendations for
            topk: Number of recommendations to return
            history: Optional explicit history list. If not provided, will use internal user history
            
        Returns:
            List of recommended item IDs
        """
        if not hasattr(self, 'model') or self.model is None:
            return []
            
        # Check if user exists in training data
        if user_id not in self.user2items_train:
            return []
            
        self.model.eval()
        with torch.no_grad():
            # Check if model supports full sort prediction
            if not hasattr(self.model, 'full_sort_predict'):
                return []
            
            # Create proper Interaction object for this user
            interaction = self._create_user_interaction(user_id)
            if interaction is None:
                return []
            
            # Generate scores for all items using proper Interaction object
            scores = self.model.full_sort_predict(interaction)
            
            # Determine which history to use for filtering
            if history:
                # Use explicitly provided history
                history_set = set(history)
            else:
                # Use internal training history + historical sequence
                user_train_items = set(self.user2items_train.get(user_id, []))
                user_history_items = set(self.user2item_list_train.get(user_id, []))
                history_set = user_train_items.union(user_history_items)
            
            # Convert scores to numpy and filter out history items
            if scores.dim() > 1:
                scores_np = scores[0].cpu().numpy().flatten()  # Take first batch element
            else:
                scores_np = scores.cpu().numpy().flatten()
            
            # Create candidate list excluding history items
            candidates = []
            for item_id in range(len(scores_np)):
                if item_id not in history_set:
                    candidates.append((item_id, scores_np[item_id]))
            
            # Sort by score and return top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:topk]

    def get_user_history(self, user_id: int, split: str = 'train') -> List[int]:
        """Get user's historical interaction sequence
        
        Args:
            user_id: The user ID
            split: 'train' or 'test' split
            
        Returns:
            List of historical item IDs
        """
        if split == 'train':
            return self.user2item_list_train.get(user_id, [])
        elif split == 'test':
            return self.user2item_list_test.get(user_id, [])
        else:
            return []


# PopularityRecaller and ItemCFRecaller are now unified through RecBoleRecaller
# Usage: RecBoleRecaller(model_name='Pop', ...) or RecBoleRecaller(model_name='ItemKNN', ...)
