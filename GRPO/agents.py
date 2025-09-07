import json
import math
from typing import List, Optional

try:
    import dspy
    DSPY_OK = True
except Exception:
    DSPY_OK = False

import torch

class HFLocalGenerator:
    def __init__(self, model_name: str, dtype: str = "auto", device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype_map = {
            "auto": None,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_kwargs = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    @staticmethod
    def _build_prompt(profile_json: str, n: int) -> str:
        return (
            "You are a routing assistant. Given a user profile JSON and an integer n, "
            "output ONLY a JSON array with length n. Each element must be an object with keys: "
            "\"model_1\" (string: model name like 'sasrec', 'bpr', 'pop'), "
            "\"k_1\" (int 1..500: number of items from model_1), "
            "\"model_2\" (string: model name like 'sasrec', 'bpr', 'pop'), "
            "\"k_2\" (int 1..500: number of items from model_2), "
            "\"w_1\" (float 0..1: weight for model_1).\n\n"
            f"Profile:\n{profile_json}\n"
            f"n={n}\n\nJSON:"
        )

    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[dict]]:
        try:
            s = text.find("["); e = text.rfind("]")
            if s != -1 and e != -1 and e > s:
                js = text[s:e+1]
                data = json.loads(js)
                if isinstance(data, list):
                    return data
        except Exception:
            return None
        return None

    def generate(self, profile_json: str, n: int) -> List[dict]:
        prompt = self._build_prompt(profile_json, n)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        data = self._extract_json_array(text)
        return data if isinstance(data, list) else []


class UserProfileAgent:
    def __init__(self, num_items: int): self.num_items = num_items
    def forward(self, user_id: int, history: List[int]):
        feats = {"uid": int(user_id), "len_hist": len(history), "last_item": int(history[-1]) if history else -1, "pop_bias": float(1.0 / (1.0 + math.exp(-(len(history) - 5))))}
        class _Out:
            def __init__(self, js): self.profile_json = js
        return _Out(json.dumps(feats))


class LLMRouterAgent:
    def __init__(self, n_per_user: int = 4, local_hf: Optional[HFLocalGenerator] = None, available_models: List[str] = None):
        self.n_per_user = n_per_user
        self.local_hf = local_hf
        self.available_models = available_models or ['sasrec', 'bpr', 'pop']

    def forward(self, profile_json: str, n_candidates: int = None) -> List[dict]:
        n = n_candidates or self.n_per_user
        if self.local_hf is not None:
            try:
                routes = self.local_hf.generate(profile_json, n)
            except Exception:
                routes = []
        else:
            routes = []
        if not routes:
            fallback = []
            for j in range(n):
                model_1 = self.available_models[j % len(self.available_models)] if self.available_models else "sasrec"
                model_2 = self.available_models[(j+1) % len(self.available_models)] if self.available_models else "bpr"
                fallback.append({
                    "model_1": model_1, "k_1": 10*(j+1),
                    "model_2": model_2, "k_2": 10*(j+2),
                    "w_1": max(0.1, min(0.9, 0.5 + 0.1*(j-1)))
                })
            routes = fallback

        cleaned = []
        for j in range(min(n, len(routes))):
            r = routes[j] if isinstance(routes[j], dict) else {}
            model_1 = str(r.get("model_1", self.available_models[0] if self.available_models else "sasrec")).lower()
            model_2 = str(r.get("model_2", self.available_models[1] if len(self.available_models) > 1 else model_1)).lower()
            if model_1 not in self.available_models: model_1 = self.available_models[0]
            if model_2 not in self.available_models: model_2 = self.available_models[0]
            k_1 = int(max(1, min(500, int(r.get("k_1", 50)))))
            k_2 = int(max(1, min(500, int(r.get("k_2", 50)))))
            w_1 = float(r.get("w_1", 0.5)); w_1 = max(0.0, min(1.0, w_1))
            cleaned.append({
                "model_1": model_1, "k_1": k_1,
                "model_2": model_2, "k_2": k_2,
                "w_1": w_1, "w_2": 1.0 - w_1
            })

        while len(cleaned) < n:
            j = len(cleaned)
            model_1 = self.available_models[j % len(self.available_models)] if self.available_models else "sasrec"
            model_2 = self.available_models[(j+1) % len(self.available_models)] if self.available_models else "bpr"
            cleaned.append({
                "model_1": model_1, "k_1": 10*(j+1),
                "model_2": model_2, "k_2": 10*(j+2),
                "w_1": max(0.1, min(0.9, 0.5 + 0.1*(j-1))), "w_2": 0.5
            })
        return cleaned
