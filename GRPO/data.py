from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import os

try:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed as recbole_init_seed
    from recbole.utils import get_model as get_recbole_model
    RECB = True
except Exception:
    RECB = False


@dataclass
class InteractionData:
    num_users: int
    num_items: int
    train_user_ids: List[int]
    train_histories: List[List[int]]
    train_target_items: List[int]
    eval_user_ids: List[int]
    eval_histories: List[List[int]]
    eval_target_items: List[int]
    test_user_ids: List[int]
    test_histories: List[List[int]]
    test_target_items: List[int]


def load_dataset(dataset: str, data_path: str, seed: int = 42) -> InteractionData:
    if RECB:
        cfg = Config(
            model="Pop", 
            dataset=dataset, 
            config_dict={
                "data_path": data_path, 
                "seed": seed,
                'load_col': {
                    'inter': ['user_id', 'item_id_list', 'item_id']
                },
                'benchmark_filename': ['train', 'valid', 'test'],
                'alias_of_item_id': ['item_id_list'],
                'train_neg_sample_args': None,
                'loss_type': 'CE',
            },     
        )
        recbole_init_seed(cfg["seed"], reproducibility=True) # TODO: check if this is correct
        ds = create_dataset(cfg)
        train, valid, test = data_preparation(cfg, ds)

        uid_field, iid_field = ds.uid_field, ds.iid_field

        train_user_ids, train_item_id_lists, train_item_ids = train.dataset.inter_feat.interaction[uid_field], train.dataset.inter_feat.interaction['item_id_list'], train.dataset.inter_feat.interaction[iid_field]
        eval_user_ids, eval_item_id_lists, eval_item_ids = valid.dataset.inter_feat.interaction[uid_field], valid.dataset.inter_feat.interaction['item_id_list'], valid.dataset.inter_feat.interaction[iid_field]
        test_user_ids, test_item_id_lists, test_item_ids = test.dataset.inter_feat.interaction[uid_field], test.dataset.inter_feat.interaction['item_id_list'], test.dataset.inter_feat.interaction[iid_field]
        return InteractionData(
            ds.user_num, 
            ds.item_num, 
            train_user_ids.tolist(), 
            train_item_id_lists.tolist(), 
            train_item_ids.tolist(), 
            eval_user_ids.tolist(), 
            eval_item_id_lists.tolist(), 
            eval_item_ids.tolist(), 
            test_user_ids.tolist(), 
            test_item_id_lists.tolist(), 
            test_item_ids.tolist()
        )
    # Fallback synthetic
    rng = np.random.default_rng(seed)
    num_users, num_items = 500, 2000
    u2t, u2e = {}, {}
    for u in range(num_users):
        topic = rng.integers(0, 20)
        pool = [i for i in range(num_items) if i % 20 == topic]
        hist = rng.choice(pool, size=30, replace=False).tolist()
        test = rng.choice(pool, size=5, replace=False).tolist()
        u2t[u] = hist; u2e[u] = test
    return InteractionData(u2t, u2e, num_users, num_items)
