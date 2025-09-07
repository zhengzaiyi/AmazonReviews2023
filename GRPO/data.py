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
    user2item_list_train: Dict[int, List[int]]
    user2item_list_test: Dict[int, List[int]]
    user2items_train: Dict[int, List[int]]
    user2items_test: Dict[int, List[int]]
    num_users: int
    num_items: int


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
                }
            )
        recbole_init_seed(cfg["seed"], reproducibility=True)
        ds = create_dataset(cfg)
        train, valid, test = data_preparation(cfg, ds)
        uid_field, iid_field = ds.uid_field, ds.iid_field
        user2item_list_train, user2item_list_test = {}, {}
        user2items_train, user2items_test = {}, {}

        train_user_ids, train_item_id_lists, train_item_ids = train.dataset.inter_feat.interaction['user_id'], train.dataset.inter_feat.interaction['item_id_list'], train.dataset.inter_feat.interaction['item_id']
        for i in range(len(train.dataset.inter_feat)):
            user = int(train_user_ids[i])
            item_id_list = train_item_id_lists[i].detach()
            item = int(train_item_ids[i])
            
            user2item_list_train[user] = [int(it) for it in item_id_list if it > 0]
            user2items_train.setdefault(user, []).append(item)
        test_user_ids, test_item_id_lists, test_item_ids = test.dataset.inter_feat.interaction['user_id'], test.dataset.inter_feat.interaction['item_id_list'], test.dataset.inter_feat.interaction['item_id']

        for i in range(len(test.dataset.inter_feat)):
            user = int(test_user_ids[i])
            item_id_list = test_item_id_lists[i].detach()
            item = int(test_item_ids[i])
            user2item_list_test[user] = [int(it) for it in item_id_list if it > 0]
            user2items_test.setdefault(user, []).append(item)
        return InteractionData(user2item_list_train, user2item_list_test, user2items_train, user2items_test, ds.user_num, ds.item_num)
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
