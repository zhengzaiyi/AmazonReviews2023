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
    def _build_prompt(profile_json: str, n: int, available_models: List[str] = None) -> str:
        if available_models is None:
            available_models = ['sasrec', 'bpr', 'pop']
        
        models_str = "', '".join(available_models)
        
        # Create an example output format
        import random
        example_output = [{
            "name": available_models[i], 
            "k": random.randint(1, 100), 
            "weight": round(random.uniform(0, 1), 4)
        } for i in range(min(2, len(available_models)))]

        
        example_json = json.dumps(example_output[:min(n, 2)], indent=2)
        
        return (
            "You are a multi-channel recall assistant in a recommendation system. Given a user profile JSON, "
            "output ONLY a JSON file describe the usage of different models during the multi-channel recall. Each element must be an object with keys: "
            "\"name\" (string: model name like '"
            + models_str + "'), \"k\" (int 1..500: number of items), \"weight\" (float 0..1: model weight)\n\n"
            f"Available models: {available_models}\n"
            f"Profile:\n{profile_json}\n"
            # f"Expected output format example:\n{example_json}\n\n"
            "Your JSON response:"
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

    def generate(self, profile_json: str, n: int, available_models: List[str] = None, 
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, 
                 num_return_sequences: int = 1) -> List[dict]:
        prompt = self._build_prompt(profile_json, n, available_models)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        all_routes = []
        
        for _ in range(num_return_sequences):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    min_new_tokens=10,
                    do_sample=True, 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
            if text[0] != "[":
                text = "[" + text
            if text[-1] != "]":
                text = text + "]"
            data = self._extract_json_array(text)
            if isinstance(data, list):
                all_routes.extend(data)
        
        return all_routes


class UserProfileAgent:
    def __init__(self, num_items: int): self.num_items = num_items
    def forward(self, user_id: int, history: List[int]):
        feats = {
            "uid": int(user_id), 
            "len_hist": len(history), 
            "last_item": int(history[-1]) if history else -1, 
            # "pop_bias": float(1.0 / (1.0 + math.exp(-(len(history) - 5))))
        }
        class _Out:
            def __init__(self, js): self.profile_json = js
        return _Out(json.dumps(feats))


class LLMRouterAgent:
    def __init__(self, n_per_user: int = 4, local_hf: Optional[HFLocalGenerator] = None, available_models: List[str] = None):
        self.n_per_user = n_per_user
        self.local_hf = local_hf
        self.available_models = available_models or ['sasrec', 'bpr', 'pop']

    def _select_diverse_routes(self, candidate_routes: List[dict], n: int) -> List[dict]:
        """
        choose n different routes from candidate routes
        """
        if not candidate_routes:
            return []
        
        unique_routes = []
        seen_keys = set()
        
        for route in candidate_routes:
            if "models" in route and isinstance(route["models"], list):
                models_key = tuple(
                    (m.get("name", ""), m.get("k", 0), round(float(m.get("weight", 0.0)), 3))
                    for m in route["models"]
                )
                route_key = ("new_format", models_key)
            else:
                route_key = (
                    "old_format",
                    route.get("model_1", ""),
                    route.get("model_2", ""),
                    route.get("k_1", 0),
                    route.get("k_2", 0),
                    round(float(route.get("w_1", 0.0)), 3)
                )
            
            if route_key not in seen_keys:
                seen_keys.add(route_key)
                unique_routes.append(route)
                
                if len(unique_routes) >= n:
                    break
        
        return unique_routes

    def forward(self, profile_json: str, n_candidates: int = None, 
                temperature: float = 0.8, ensure_diversity: bool = True) -> List[dict]:
        n = n_candidates or self.n_per_user
        routes = []
        
        if self.local_hf is not None:
            try:
                if ensure_diversity:
                    candidate_routes = self.local_hf.generate(
                        profile_json, 
                        n * 2,
                        self.available_models,
                        temperature=temperature,
                        num_return_sequences=2
                    )
                    routes = self._select_diverse_routes(candidate_routes, n)
                else:
                    routes = self.local_hf.generate(profile_json, n, self.available_models, temperature=temperature)
            except Exception:
                routes = []

        if len(routes) < n:
            # Fallback: create routing strategies using all available models
            fallback = []
            for j in range(n):
                # Create diverse routing strategies by varying model combinations
                num_models = len(self.available_models) if self.available_models else 3
                
                # Strategy 1: Use 2-3 models per route for diversity
                models_per_route = min(3, max(2, num_models))
                selected_models = []
                
                # Select models in a round-robin fashion with some variation
                start_idx = j % num_models
                for i in range(models_per_route):
                    model_idx = (start_idx + i) % num_models
                    model_name = self.available_models[model_idx] if self.available_models else f"model_{model_idx}"
                    k_value = 20 + (i * 15) + (j * 5)  # Vary k values
                    weight = 1.0 / models_per_route  # Equal weights initially
                    
                    # Adjust weights based on position and strategy
                    if i == 0:  # First model gets slightly higher weight
                        weight = 0.4 if models_per_route == 2 else 0.5
                    elif i == 1 and models_per_route == 2:
                        weight = 0.6
                    elif models_per_route == 3:
                        weight = 0.3 if i == 1 else 0.2
                    
                    selected_models.append({
                        "name": model_name,
                        "k": min(500, max(1, k_value)),
                        "weight": weight
                    })
                
                # Normalize weights to sum to 1.0
                total_weight = sum(m["weight"] for m in selected_models)
                for model in selected_models:
                    model["weight"] = model["weight"] / total_weight
                
                fallback.append({"models": selected_models})
            routes = fallback

        # Clean and validate the routes
        cleaned = []
        for j in range(min(n, len(routes))):
            r = routes[j] if isinstance(routes[j], dict) else {}
            
            # Handle both old format (model_1, model_2) and new format (models)
            if "models" in r and isinstance(r["models"], list):
                # New format with multiple models
                models = []
                total_weight = 0.0
                
                for model_info in r["models"]:
                    if not isinstance(model_info, dict):
                        continue
                        
                    model_name = str(model_info.get("name", "")).lower()
                    if not model_name or model_name not in self.available_models:
                        model_name = self.available_models[0] if self.available_models else "sasrec"
                    
                    k_value = int(max(1, min(500, int(model_info.get("k", 50)))))
                    weight = float(max(0.0, min(1.0, model_info.get("weight", 1.0))))
                    
                    models.append({
                        "name": model_name,
                        "k": k_value,
                        "weight": weight
                    })
                    total_weight += weight
                
                # Normalize weights
                if total_weight > 0:
                    for model in models:
                        model["weight"] = model["weight"] / total_weight
                else:
                    # Assign equal weights if all weights are 0
                    for model in models:
                        model["weight"] = 1.0 / len(models)
                
                cleaned.append({"models": models})
                
            else:
                # Convert old format to new format for backward compatibility
                model_1 = str(r.get("model_1", self.available_models[0] if self.available_models else "sasrec")).lower()
                model_2 = str(r.get("model_2", self.available_models[1] if len(self.available_models) > 1 else model_1)).lower()
                
                if model_1 not in self.available_models: 
                    model_1 = self.available_models[0] if self.available_models else "sasrec"
                if model_2 not in self.available_models: 
                    model_2 = self.available_models[1] if len(self.available_models) > 1 else self.available_models[0]
                
                k_1 = int(max(1, min(500, int(r.get("k_1", 50)))))
                k_2 = int(max(1, min(500, int(r.get("k_2", 50)))))
                w_1 = float(max(0.0, min(1.0, r.get("w_1", 0.5))))
                w_2 = 1.0 - w_1
                
                models = [
                    {"name": model_1, "k": k_1, "weight": w_1},
                    {"name": model_2, "k": k_2, "weight": w_2}
                ]
                cleaned.append({"models": models})

        # Ensure we have enough routes
        while len(cleaned) < n:
            j = len(cleaned)
            num_models = len(self.available_models) if self.available_models else 2
            models = []
            
            # Create a simple 2-model strategy for additional routes
            model_1 = self.available_models[j % num_models] if self.available_models else "sasrec"
            model_2 = self.available_models[(j+1) % num_models] if self.available_models else "bpr"
            
            models.append({"name": model_1, "k": 10*(j+1), "weight": 0.6})
            models.append({"name": model_2, "k": 10*(j+2), "weight": 0.4})
            
            cleaned.append({"models": models})
            
        return cleaned
