import json
import math
from typing import List, Optional

try:
    import dspy
    DSPY_OK = True
except Exception:
    DSPY_OK = False

import torch
from typing import Dict
from collections import defaultdict

TOOLS_DESCRIPTION = {
    "sasrec": {
        "description": "A Transformer-based sequential recommendation model (Self-Attentive Sequential Recommendation). It captures the order and short-term interest patterns from the user's recent interactions.",
        "when_to_use": "Use when the task requires modeling the sequence of recent user interactions or short-term preferences. For example, predicting the next item a user might click or watch.",
        "input": "A chronologically ordered sequence of user-item interactions.",
        "output": "Top-K candidate items predicted as the next likely interactions."
    },
    "bpr": {
        "description": "Bayesian Personalized Ranking, a classic pairwise ranking method based on matrix factorization. It focuses on modeling user preference orderings.",
        "when_to_use": "Use when the task involves general recommendation based on long-term user preferences, without considering sequence order. Suitable for implicit feedback like clicks or likes.",
        "input": "A user-item interaction matrix or embeddings representing user and item factors.",
        "output": "Top-K candidate items ranked by the user's overall preference."
    },
    "lightgcn": {
        "description": "A graph-based recommendation model using Graph Convolutional Networks. It propagates embeddings over a user-item bipartite graph to capture high-order connectivity.",
        "when_to_use": "Use when the task involves graph structures, such as leveraging user-item relations or higher-order neighbor connections. Suitable for social or community-driven recommendations.",
        "input": "A user-item bipartite graph.",
        "output": "Top-K candidate items derived from graph-based user and item embeddings."
    }
}

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
        if device in ["auto", "balanced"]:
            model_kwargs["device_map"] = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **model_kwargs
        )
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **model_kwargs
        )
        for p in self.ref_model.parameters(): p.requires_grad = False
        self.ref_model.eval()
        

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
            f"Available models: \n{[json.dumps(TOOLS_DESCRIPTION[m], indent=2) for m in available_models]}\n"
            f"Profile:\n{profile_json}\n"
            f"Expected output format example:\n{example_json}\n\n"
            "Your JSON response (only return the JSON file, no other text):"
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


    def get_logprob(self, text: str, label: str) -> dict:
        """Compute logprob using current model (policy model)"""
        full_text = text + label
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            avg_logprob = -loss.item()
            total_logprob = avg_logprob * (labels != -100).sum().item()

        return {"avg_logprob": avg_logprob, "total_logprob": total_logprob}

    def generate(self, profile_json: str, n: int, available_models: List[str] = None, 
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, 
                 num_return_sequences: int = 1,
                 ref_mode: bool = False) -> dict:
        prompt = self._build_prompt(profile_json, n, available_models)
        
        all_routes = []
        all_logprobs = []
        
        # Use HuggingFace model for generation
        model = self.model if not ref_mode else self.ref_model
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=10,
                do_sample=True, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
                )
        
        text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        data = [self._extract_json_array(t[len(prompt):]) for t in text]
        all_routes.extend(data)
        
        # Calculate total logprobs directly
        for i in range(num_return_sequences):
            total_logprob = 0.0
            for j, score in enumerate(outputs.scores):
                log_probs = torch.log_softmax(score[i], dim=-1)
                token_id = outputs.sequences[i][len(inputs.input_ids[0]) + j]
                total_logprob += log_probs[token_id].item()
            all_logprobs.append(total_logprob)
            

        
        return {
            "routes": all_routes,
            "logprobs": all_logprobs,
            "ref_mode": ref_mode
        }


class UserProfileAgent:
    # TODO: add summary of the user
    def __init__(
        self, num_items: int, data_maps: Dict[str, Dict[str, str]], reviews: Dict[str, List[dict]]): 
        self.num_items = num_items
        self.data_maps = data_maps
        self.item2id = data_maps['item2id']
        self.id2meta = data_maps['id2meta']
        self.reviews = self._preprocess_reviews(reviews)

    def _preprocess_reviews(self, reviews: List[dict]):
        processed_reviews = defaultdict(list)
        for user, user_reviews in reviews.items():
            for review in user_reviews:
                processed_reviews[user].append({
                    "rating": review["rating"],
                    "item": self.id2meta[str(review["item_id"])],
                    "review title": review["title"],
                    "review text": review["text"],
                    "purchased": review["verified_purchase"],
                })
        return processed_reviews

    def _get_item_metadata(self, item_id: int):
        return self.id2meta[str(item_id)]

    def forward(self, user_id: int, history: List[int]):
        user_profile_dict = {
            "user id": int(user_id), 
            "purchased item numbers": len(history),
            "purchased items": [self._get_item_metadata(item_id) for item_id in history],
            "last purchased item": self._get_item_metadata(history[-1]) if history else -1, 
            "reviews": self.reviews[user_id],
        }
        class _Out:
            def __init__(self, js): self.profile_json = js
        return _Out(json.dumps(user_profile_dict))


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

    def _create_fallback_routes(self, n: int) -> List[dict]:
        """Create fallback routes when LLM generation fails"""
        fallback = []
        num_models = len(self.available_models) or 3
        
        for j in range(n):
            models_per_route = min(3, max(2, num_models))
            selected_models = []
            
            start_idx = j % num_models
            for i in range(models_per_route):
                model_idx = (start_idx + i) % num_models
                model_name = self.available_models[model_idx] if self.available_models else f"model_{model_idx}"
                k_value = min(500, max(1, 20 + (i * 15) + (j * 5)))
                
                # Simple weight assignment
                weight = 0.5 if i == 0 else 0.5 / (models_per_route - 1)
                
                selected_models.append({
                    "name": model_name,
                    "k": k_value,
                    "weight": weight
                })
            
            fallback.append({"models": selected_models})
        return fallback

    def _clean_and_validate_routes(self, routes: List[dict], n: int) -> List[dict]:
        """Clean and validate route formats"""
        cleaned = []
        for j in range(min(n, len(routes))):
            route = routes[j] if isinstance(routes[j], dict) else {}
            
            if "models" in route and isinstance(route["models"], list):
                # New format - just validate
                models = []
                for model_info in route["models"]:
                    if isinstance(model_info, dict):
                        model_name = str(model_info.get("name", "")).lower()
                        if model_name not in self.available_models:
                            model_name = self.available_models[0] if self.available_models else "sasrec"
                        
                        models.append({
                            "name": model_name,
                            "k": max(1, min(500, int(model_info.get("k", 50)))),
                            "weight": max(0.0, min(1.0, model_info.get("weight", 1.0)))
                        })
                
                # Normalize weights
                total_weight = sum(m["weight"] for m in models)
                if total_weight > 0:
                    for model in models:
                        model["weight"] /= total_weight
                
                cleaned.append({"models": models})
            else:
                # Old format - convert to new format
                model_1 = self.available_models[0] if self.available_models else "sasrec"
                model_2 = self.available_models[1] if len(self.available_models) > 1 else model_1
                
                models = [
                    {"name": model_1, "k": 50, "weight": 0.5},
                    {"name": model_2, "k": 50, "weight": 0.5}
                ]
                cleaned.append({"models": models})
        
        # Fill remaining slots
        while len(cleaned) < n:
            j = len(cleaned)
            num_models = len(self.available_models) or 2
            model_1 = self.available_models[j % num_models] if self.available_models else "sasrec"
            model_2 = self.available_models[(j+1) % num_models] if self.available_models else "bpr"
            
            models = [
                {"name": model_1, "k": 10*(j+1), "weight": 0.6},
                {"name": model_2, "k": 10*(j+2), "weight": 0.4}
            ]
            cleaned.append({"models": models})
            
        return cleaned

    def forward(self, profile_json: str, n_candidates: int = None, 
                temperature: float = 0.8, ensure_diversity: bool = True) -> List[dict]:
        n = n_candidates or self.n_per_user
        
        # Try LLM generation
        routes = []
        if self.local_hf:
            result = self.local_hf.generate(
                profile_json, 
                n * 2 if ensure_diversity else n,
                self.available_models,
                temperature=temperature,
                num_return_sequences=2 if ensure_diversity else 1
            )
            candidate_routes = result.get("routes", []) if isinstance(result, dict) else result or []
            routes = self._select_diverse_routes(candidate_routes, n) if ensure_diversity else candidate_routes
        
        # Use fallback if needed
        if len(routes) < n:
            routes = self._create_fallback_routes(n)
        
        # Clean and validate
        return self._clean_and_validate_routes(routes, n)
