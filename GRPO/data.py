from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import os
import copy
from GRPO.constant import dataset2item_feat_fields, dataset2user_feat_fields, dataset2inter_feat_fields

try:
    from recbole.config import Config
    from recbole.data.dataset import SequentialDataset
    from recbole.data import create_dataset, data_preparation
    from recbole.data.dataloader import TrainDataLoader
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

def get_base_config_dict(dataset_name: str, data_path: str = 'dataset', seed: int = 42) -> dict:
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
            'split': {'RS': [0.8, 0.1, 0.1]},  # Leave-One-Out
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

def load_dataset(dataset: str, data_path: str, seed: int = 42, filter_train: bool = False) -> InteractionData:
    config_dict = get_base_config_dict(dataset, data_path, seed)
    cfg = Config(
        model="SASRec", 
        dataset=dataset, 
        config_dict=config_dict,
    )
    recbole_init_seed(cfg['seed'], reproducibility=True) # TODO: check if this is correct
    ds = create_dataset(cfg)
    raw_ds = copy.deepcopy(ds)
    train, valid, test = data_preparation(cfg, ds)

    uid_field, iid_field = ds.uid_field, ds.iid_field
    iid_list_field = f'{iid_field}_list'


    if filter_train:
        len_field = ds.item_list_length_field  # 一般是 f"{iid_field}_list_length"

        # ===== 仅对“训练子集”做每用户一条的过滤 =====
        train_dataset = train.dataset           # 注意：这是 train 的子数据集，不是原始 ds
        inter = train_dataset.inter_feat        # Interaction 对象（增广后的多条样本）

        # 取出 uid 与序列长度（numpy 数组）
        uids = inter[uid_field]
        lengths = inter[len_field]
        n = len(uids)
        idx = np.arange(n)

        # 为了“每个用户保留长度最大的那条”，我们按 (uid, length, idx) 升序排序，
        # 然后取每个 uid 在排序后出现的最后一个位置（即最大 length；若并列取最后一条）
        order = np.lexsort((idx, lengths, uids))           # 先按 uid，再按 length，再按原始 idx
        uids_sorted = uids[order]
        # 获取每个用户的起始位置和计数
        unique_uids, first_pos, counts = np.unique(uids_sorted, return_index=True, return_counts=True)
        
        selected_positions = []
        for i, (start, count) in enumerate(zip(first_pos, counts)):
            pos_in_user = min(int(count), count - 1)
            selected_positions.append(start + pos_in_user)
        
        keep_indices_sorted = order[selected_positions]     # 映射回原始样本下标

        # 构造布尔掩码并切片出过滤后的 Interaction
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep_indices_sorted] = True
        filtered_inter = inter[keep_mask]

        # 用过滤后的 Interaction 替换训练子集，并重建 TrainDataLoader
        train_dataset.inter_feat = filtered_inter
        # 一些 Dataset 的缓存/索引依赖 inter_feat，更新后建议调用 rebuild_cache（有的版本没有则忽略）
        if hasattr(train_dataset, "rebuild_cache"):
            train_dataset.rebuild_cache()

        train = TrainDataLoader(config=cfg, dataset=train_dataset, sampler=train.sampler)
    train_user_ids, train_item_id_lists, train_item_ids = train.dataset.inter_feat.interaction[uid_field], train.dataset.inter_feat.interaction[iid_list_field], train.dataset.inter_feat.interaction[iid_field]
    eval_user_ids, eval_item_id_lists, eval_item_ids = valid.dataset.inter_feat.interaction[uid_field], valid.dataset.inter_feat.interaction[iid_list_field], valid.dataset.inter_feat.interaction[iid_field]
    test_user_ids, test_item_id_lists, test_item_ids = test.dataset.inter_feat.interaction[uid_field], test.dataset.inter_feat.interaction[iid_list_field], test.dataset.inter_feat.interaction[iid_field]
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

