from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import os
import copy
from GRPO.core.constant import dataset2item_feat_fields, dataset2user_feat_fields, dataset2inter_feat_fields

try:
    from recbole.config import Config
    from recbole.data.dataset import SequentialDataset
    from recbole.data import create_dataset, data_preparation
    from recbole.data.dataloader import TrainDataLoader, AbstractDataLoader, FullSortEvalDataLoader
    from recbole.utils import init_seed as recbole_init_seed
    from recbole.utils import get_model as get_recbole_model
    RECB = True
except Exception:
    RECB = False
from datasets import Dataset


@dataclass
class InteractionData:
    ds: SequentialDataset
    train_user_ids: List[int]
    train_histories: List[List[int]]
    train_target_items: List[int]
    eval_user_ids: List[int]
    eval_histories: List[List[int]]
    eval_target_items: List[int]
    test_user_ids: List[int]
    test_histories: List[List[int]]
    test_target_items: List[int]

def filter_dataset_one_per_user(
    dataloader,
    dataset: SequentialDataset,
    config: Config = None,
    use_all_gt: bool = True,
):
    """
    Filter dataset to keep only one record per user (the one with maximum sequence length).
    
    Args:
        dataloader: DataLoader to filter (can be TrainDataLoader, FullSortEvalDataLoader, etc.)
        dataset: SequentialDataset object (for field names)
        config: Config object (not used anymore, kept for compatibility)
        use_all_gt: Whether to use all ground truth items (not used, kept for compatibility)
        
    Returns:
        Tuple of (user_ids, item_id_lists, item_ids) numpy arrays
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    iid_list_field = f'{iid_field}_list'
    len_field = dataset.item_list_length_field  # 一般是 f"{iid_field}_list_length"
    
    # Get the sub-dataset from dataloader
    sub_dataset = dataloader.dataset
    inter = sub_dataset.inter_feat  # Interaction 对象（增广后的多条样本）
    
    # 取出 uid 与序列长度（numpy 数组）
    uids = inter[uid_field]
    lengths = inter[len_field]
    n = len(uids)
    idx = np.arange(n)
    
    # 为了"每个用户保留长度最大的那条"，我们按 (uid, length, idx) 升序排序，
    # 然后取每个 uid 在排序后出现的最后一个位置（即最大 length；若并列取最后一条）
    order = np.lexsort((idx, lengths, uids))  # 先按 uid，再按 length，再按原始 idx
    uids_sorted = uids[order]
    # 获取每个用户的起始位置和计数
    unique_uids, first_pos, counts = np.unique(uids_sorted, return_index=True, return_counts=True)
    
    last_pos = []
    for i, (start, count) in enumerate(zip(first_pos, counts)):
        pos_in_user = min(int(count), count - 1)
        last_pos.append(start + pos_in_user)
    
    keep_indices_sorted = order[last_pos]  # 映射回原始样本下标
    
    # 构造布尔掩码并切片出过滤后的 Interaction
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_indices_sorted] = True
    print("sum of keep_mask", sum(keep_mask))
    # 直接返回过滤后的三元组
    filtered_inter = inter[keep_mask]
    user_ids = filtered_inter[uid_field]
    item_id_lists = filtered_inter[iid_list_field]
    item_ids = filtered_inter[iid_field]
    
    return user_ids, item_id_lists, item_ids


def get_base_config_dict(
    dataset_name: str, 
    data_path: str = 'dataset', 
    seed: int = 42,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """
    Get base config dict for RecBole dataset loading.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset directory
        seed: Random seed
        train_ratio: Ratio for training set (default: 0.8, matching main_pure.py)
        eval_ratio: Ratio for validation set (default: 0.1, matching main_pure.py)
        test_ratio: Ratio for test set (default: 0.1, matching main_pure.py)
    
    Returns:
        Config dictionary for RecBole
    """
    # Normalize ratios to ensure they sum to 1.0
    total = train_ratio + eval_ratio + test_ratio
    train_ratio = train_ratio / total
    eval_ratio = eval_ratio / total
    test_ratio = test_ratio / total
    
    config_dict = {
        'data_path': data_path,
        'seed': seed,
        'load_col': {
            'inter': dataset2inter_feat_fields[dataset_name], # required
            # 'item': dataset2item_feat_fields[dataset_name], # required
        },
        'user_inter_num_interval': "[5,inf)",
        'item_inter_num_interval': "[5,inf)",
        'train_neg_sample_args': None,
        'loss_type': 'CE',
        'val_interval': {
            'rating': '[3,inf)'
        },
        'eval_args': {
            # 'split': {'RS': [train_ratio, eval_ratio, test_ratio]},  
            'split': {'LS': "valid_and_test"},  
            'order': 'TO',
            'group_by': 'user'
        },
        'save_dataset': True,
        'save_dataloaders': True,
    }
    if dataset2user_feat_fields[dataset_name] is not None:
        config_dict['load_col']['user'] = dataset2user_feat_fields[dataset_name] # optional
    if dataset_name == 'steam':
        config_dict.update({
            'ITEM_ID_FIELD': 'product_id',
        })
        del config_dict['val_interval']
    elif dataset_name == 'anime':
        del config_dict['val_interval']
    return config_dict

def load_dataset(
    dataset: str, 
    data_path: str, 
    seed: int = 42, 
    filter_train: bool = False,
    filter_eval: bool = False,
    filter_test: bool = False,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> InteractionData:
    """
    Load dataset with train/eval/test split (matching main_pure.py style).
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset directory
        seed: Random seed
        filter_train: Whether to filter train set (keep one record per user)
        filter_eval: Whether to filter eval set (keep one record per user)
        filter_test: Whether to filter test set (keep one record per user)
        train_ratio: Ratio for training set (default: 0.8, matching main_pure.py)
        eval_ratio: Ratio for validation set (default: 0.1, matching main_pure.py)
        test_ratio: Ratio for test set (default: 0.1, matching main_pure.py)
    
    Returns:
        InteractionData with train/eval/test splits
    """
    config_dict = get_base_config_dict(dataset, data_path, seed, train_ratio, eval_ratio, test_ratio)
    model = 'SASRec'
    cfg = Config(
        model=model, 
        dataset=dataset, 
        config_dict=config_dict,
    )
    recbole_init_seed(cfg['seed'], reproducibility=True) # TODO: check if this is correct
    ds = create_dataset(cfg)
    print(ds)
    raw_ds = copy.deepcopy(ds)
    train, valid, test = data_preparation(cfg, ds)

    uid_field, iid_field = ds.uid_field, ds.iid_field
    iid_list_field = f'{iid_field}_list'

    # Extract data from dataloaders (with optional filtering)
    if filter_train and model != "BPR":
        train_user_ids, train_item_id_lists, train_item_ids = filter_dataset_one_per_user(train, ds, cfg)
    else:
        train_user_ids = train.dataset.inter_feat.interaction[uid_field]
        train_item_id_lists = train.dataset.inter_feat.interaction[iid_list_field]
        train_item_ids = train.dataset.inter_feat.interaction[iid_field]
    
    if filter_eval and model != "BPR":
        eval_user_ids, eval_item_id_lists, eval_item_ids = filter_dataset_one_per_user(valid, ds, cfg)
    else:
        eval_user_ids = valid.dataset.inter_feat.interaction[uid_field]
        eval_item_id_lists = valid.dataset.inter_feat.interaction[iid_list_field]
        eval_item_ids = valid.dataset.inter_feat.interaction[iid_field]
    
    if filter_test and model != "BPR":
        test_user_ids, test_item_id_lists, test_item_ids = filter_dataset_one_per_user(test, ds, cfg)
    else:
        test_user_ids = test.dataset.inter_feat.interaction[uid_field]
        test_item_id_lists = test.dataset.inter_feat.interaction[iid_list_field]
        test_item_ids = test.dataset.inter_feat.interaction[iid_field]
    return InteractionData(
        raw_ds, 
        train_user_ids.tolist(), 
        train_item_id_lists.tolist(), 
        train_item_ids.tolist(), 
        eval_user_ids.tolist(), 
        eval_item_id_lists.tolist(), 
        eval_item_ids.tolist(), 
        test_user_ids.tolist(), 
        test_item_id_lists.tolist(), 
        test_item_ids.tolist(),
    )

