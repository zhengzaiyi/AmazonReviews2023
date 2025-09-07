import copy
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import recall_at_k, merge_candidates


class PreferenceSelectorNet(nn.Module):
    def __init__(self, available_models: List[str] = None):
        super().__init__()
        self.available_models = available_models or ['sasrec', 'bpr', 'pop']
        self.model_to_idx = {model: idx for idx, model in enumerate(self.available_models)}
        self.uid_emb = nn.Embedding(100000, 16)
        self.mlp = nn.Sequential(nn.Linear(16 + 3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.model_emb = nn.Embedding(len(self.available_models), 8)
        self.cand_mlp = nn.Sequential(nn.Linear(8*2 + 3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.out = nn.Linear(64 + 32, 1)

    def encode_profile(self, profile: Dict) -> torch.Tensor:
        uid = torch.tensor([profile["uid"]], dtype=torch.long)
        feats = torch.tensor([[profile["len_hist"], profile["last_item"], profile["pop_bias"]]], dtype=torch.float32)
        h = torch.cat([self.uid_emb(uid), feats], dim=-1).squeeze(0)
        return self.mlp(h)

    def score_candidates(self, prof_vec: torch.Tensor, routes: List[dict]) -> torch.Tensor:
        scores = []
        for r in routes:
            model_1_idx = self.model_to_idx.get(r["model_1"], 0)
            model_2_idx = self.model_to_idx.get(r["model_2"], 0)
            model_1_emb = self.model_emb(torch.tensor([model_1_idx], dtype=torch.long)).squeeze(0)
            model_2_emb = self.model_emb(torch.tensor([model_2_idx], dtype=torch.long)).squeeze(0)
            cf = torch.tensor([[r["k_1"], r["k_2"], r["w_1"]]], dtype=torch.float32)
            route_vec = torch.cat([model_1_emb, model_2_emb, cf.squeeze(0)], dim=0)
            z = self.cand_mlp(route_vec.unsqueeze(0)).squeeze(0)
            s = self.out(torch.cat([prof_vec, z], dim=-1))
            scores.append(s)
        return torch.cat(scores, dim=0).squeeze(-1)

    def logprob_of_index(self, prof_vec: torch.Tensor, routes: List[dict], idx: int) -> torch.Tensor:
        scores = self.score_candidates(prof_vec, routes)
        return F.log_softmax(scores, dim=0)[idx]


@dataclass
class GRPOConfig:
    beta: float = 1.0
    group_size: int = 4
    lr: float = 3e-4
    polyak_tau: float = 0.01


class GRPOTrainer:
    def __init__(self, selector: PreferenceSelectorNet, cfg: GRPOConfig):
        self.selector = selector
        self.ref_selector = copy.deepcopy(selector).eval()
        for p in self.ref_selector.parameters(): p.requires_grad = False
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.selector.parameters(), lr=cfg.lr)

    def step(self,
             batch_users: List[int],
             histories: Dict[int, List[int]],
             user2items_test: Dict[int, List[int]],
             profile_agent,
             router,
             recallers: Dict[str, object],
             final_k: int = 50) -> Dict[str, float]:
        losses, rewards_all = [], []
        for uid in batch_users:
            hist = histories.get(uid, [])
            prof_json = profile_agent.forward(uid, hist).profile_json
            profile = json.loads(prof_json)
            routes = router.forward(prof_json, n_candidates=self.cfg.group_size)
            if len(routes) < 2:
                continue
            rewards = []
            for r in routes:
                model_1_name = r["model_1"]
                model_2_name = r["model_2"]
                list_1 = recallers.get(model_1_name, None).recall(uid, int(r["k_1"]), hist) if recallers.get(model_1_name, None) else []
                list_2 = recallers.get(model_2_name, None).recall(uid, int(r["k_2"]), hist) if recallers.get(model_2_name, None) else []
                merged = merge_candidates(list_1, list_2, float(r["w_1"]), final_k)
                rr = recall_at_k(merged, user2items_test.get(uid, []), k=final_k)
                rewards.append(rr)
            rewards_all.extend(rewards)
            best_idx = int(np.argmax(rewards))
            prof_vec = self.selector.encode_profile(profile)
            with torch.no_grad():
                ref_prof_vec = self.ref_selector.encode_profile(profile)
            for j in range(len(routes)):
                if j == best_idx: continue
                lp_plus = self.selector.logprob_of_index(prof_vec, routes, best_idx)
                lp_minus = self.selector.logprob_of_index(prof_vec, routes, j)
                ref_lp_plus = self.ref_selector.logprob_of_index(ref_prof_vec, routes, best_idx)
                ref_lp_minus = self.ref_selector.logprob_of_index(ref_prof_vec, routes, j)
                diff = (lp_plus - ref_lp_plus) - (lp_minus - ref_lp_minus)
                loss_pair = -F.logsigmoid(self.cfg.beta * diff)
                losses.append(loss_pair)
        if not losses:
            return {"loss": 0.0, "avg_recall": float(np.mean(rewards_all) if rewards_all else 0.0)}
        loss = torch.stack(losses).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.selector.parameters(), 1.0)
        self.opt.step()
        with torch.no_grad():
            tau = self.cfg.polyak_tau
            for p, q in zip(self.ref_selector.parameters(), self.selector.parameters()):
                p.data.mul_(1 - tau).add_(tau * q.data)
        return {"loss": float(loss.item()), "avg_recall": float(np.mean(rewards_all) if rewards_all else 0.0)}


def run_router_only(
    users: List[int],
    histories: Dict[int, List[int]],
    user2items_test: Dict[int, List[int]],
    profile_agent,
    router,
    recallers: Dict[str, object],
    final_k: int = 50,
    group_size: int = 4,
    strategy: str = "first",
    save_router_json: Optional[str] = None,
) -> Dict[str, float]:
    logs: List[float] = []
    routes_dump: Dict[str, List[dict]] = {}
    for uid in users:
        hist = histories.get(uid, [])
        prof_json = profile_agent.forward(uid, hist).profile_json
        routes = router.forward(prof_json, n_candidates=group_size)
        if save_router_json is not None:
            routes_dump[str(uid)] = routes
        if not routes:
            continue
        chosen = routes[0]
        if strategy == "oracle" and len(routes) > 1:
            scores = []
            for r in routes:
                model_1_name = r["model_1"]
                model_2_name = r["model_2"]
                list_1 = recallers.get(model_1_name, None).recall(uid, int(r["k_1"]), hist) if recallers.get(model_1_name, None) else []
                list_2 = recallers.get(model_2_name, None).recall(uid, int(r["k_2"]), hist) if recallers.get(model_2_name, None) else []
                merged = merge_candidates(list_1, list_2, float(r["w_1"]), final_k)
                sc = recall_at_k(merged, user2items_test.get(uid, []), k=final_k)
                scores.append(sc)
            if scores:
                best_idx = int(np.argmax(scores))
                chosen = routes[best_idx]
        model_1_name = chosen["model_1"]
        model_2_name = chosen["model_2"]
        list_1 = recallers.get(model_1_name, None).recall(uid, int(chosen["k_1"]), hist) if recallers.get(model_1_name, None) else []
        list_2 = recallers.get(model_2_name, None).recall(uid, int(chosen["k_2"]), hist) if recallers.get(model_2_name, None) else []
        merged = merge_candidates(list_1, list_2, float(chosen["w_1"]), final_k)
        rr = recall_at_k(merged, user2items_test.get(uid, []), k=final_k)
        logs.append(rr)
    if save_router_json is not None:
        try:
            with open(save_router_json, "w", encoding="utf-8") as f:
                json.dump(routes_dump, f, ensure_ascii=False)
        except Exception:
            pass
    avg_recall = float(np.mean(logs)) if logs else 0.0
    return {"avg_recall": avg_recall}
