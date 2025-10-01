import json
import math
from typing import List, Optional, Union
from dataclasses import dataclass
from typing import Dict, List, Optional

from GRPO.utils import ndcg_at_k
from collections import defaultdict
from pydantic.config import ConfigDict
import outlines

try:
    import dspy
    DSPY_OK = True
except Exception:
    DSPY_OK = False

import torch
import torch.nn as nn
from typing import Dict
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, AutoModel
from GRPO.utils import build_prompt
from pydantic import BaseModel, Field, create_model
from typing import Dict, List, Type


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
        item_id_history = [self.item2id[item] for item in history]
        user_profile_dict = {
            "user id": int(user_id), 
            "purchased item numbers": len(history),
            "purchased items": [self._get_item_metadata(item_id) for item_id in history],
            "last purchased item": self._get_item_metadata(history[-1]) if history else -1, 
            "reviews": [review for review in self.reviews[user_id] if review["item"] in item_id_history],
        }
        class _Out:
            def __init__(self, js): self.profile_json = js
        return _Out(json.dumps(user_profile_dict))


class LLMRouterAgent:
    def __init__(
        self, 
        n_per_user: int = 4, 
        available_models: List[str] = None
    ):
        self.n_per_user = n_per_user
        self.available_models = available_models or ['sasrec', 'bpr', 'pop']
            
    
    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[dict]]:
        """Extract JSON array from generated text"""
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
    
    def get_logprob(self, text: str, label: str, model, tokenizer) -> dict:
        """Compute logprob using provided model (policy model)"""
        full_text = text + label
        
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            avg_logprob = -loss.item()
            total_logprob = avg_logprob * (labels != -100).sum().item()

        return {"avg_logprob": avg_logprob, "total_logprob": total_logprob}
    
    def generate(self, profile_json: str, model, tokenizer, 
                 available_models: List[str] = None,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, 
                 num_return_sequences: int = 1) -> dict:
        """Generate routes using provided HuggingFace model"""
        prompt = build_prompt(profile_json, available_models or self.available_models)
        
        all_routes = []
        all_logprobs = []
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=10,
                do_sample=True, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
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
            "logprobs": all_logprobs
        }

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
        return routes[:n] # TODO: validate routes

    def forward(self, profile_json: str, model=None, tokenizer=None, n_candidates: int = None, 
                temperature: float = 0.8) -> List[dict]:
        n = n_candidates or self.n_per_user
        
        # Try LLM generation if model is provided
        routes = []
        if model and tokenizer:
            result = self.generate(
                profile_json, 
                model,
                tokenizer,
                self.available_models,
                temperature=temperature,
                num_return_sequences=n_candidates
            )
            routes = result.get("routes", []) if isinstance(result, dict) else result or []
        
        # Use fallback if needed
        if len(routes) < n:
            routes = self._create_fallback_routes(n)
        
        # Clean and validate
        return self._clean_and_validate_routes(routes, n)


def generate_diverse_routes(
    router: LLMRouterAgent, 
    prof_json: str, 
    n_candidates: int, 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    temperature=0.8, 
    include_logprobs=False
):
    """Generate diverse routes using a single model"""
    if not include_logprobs:
        # Simple generation without logprobs
        return router.forward(
            prof_json, 
            model=model,
            tokenizer=tokenizer,
            n_candidates=n_candidates, 
            temperature=temperature, 
            ensure_diversity=True
        )
    
    # Generate with logprobs
    result = router.generate(
        prof_json,
        n_candidates,
        model,
        tokenizer,
        router.available_models,
        temperature=temperature,
        num_return_sequences=n_candidates
    )
    
    routes = result["routes"]
    logprobs = result["logprobs"]
    
    # Apply diversity selection if needed
    if len(routes) > n_candidates:
        diverse_routes = router._select_diverse_routes(routes, n_candidates)
        route_indices = [routes.index(r) for r in diverse_routes]
        logprobs = [logprobs[i] for i in route_indices]
        routes = diverse_routes
    
    return {
        "routes": routes,
        "logprobs": logprobs
    }


def calculate_route_logprobs(
    router: LLMRouterAgent,
    prof_json: str,
    routes: List[dict],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase
) -> List[float]:
    """Calculate log probabilities for given routes"""
    logprobs = []
    for route in routes:
        result = router.get_logprob(prof_json, json.dumps(route), model, tokenizer)
        logprobs.append(result['total_logprob'])
    return logprobs

class PerModelConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        # populate_by_name=True
    )
    k: int = Field(default=50, alias='top-k', description="Number of top items to retrieve")
    w: float = Field(default=1.0, alias='score-weight', description="Weight for scoring")

def create_model_config(
    available_models: List[str], 
    default_configs: Optional[Dict[str, BaseModel]] = None,
    class_name: str = "ModelConfigs"
) -> Type[BaseModel]:
    """
    Create an advanced dynamic Pydantic model with support for custom default configurations
    
    Args:
        available_models: List of available model names
        default_configs: Optional dictionary of default configurations
        class_name: Name of the generated class
    
    Returns:
        Dynamically created Pydantic model class
    """
    default_configs = default_configs or {}
    
    fields = {}
    for model_name in available_models:
        default_config = default_configs.get(model_name, PerModelConfig()) if default_configs else PerModelConfig()
        
        fields[model_name] = (
            PerModelConfig, 
            Field(default=default_config, description=f"Configuration for {model_name} model")
        )
    
    ModelConfigs = create_model(class_name, **fields, __base__=BaseModel)
    
    return ModelConfigs

# Example usage for GRPO:
# 1. Generate routes with ref_model
# ref_result = generate_diverse_routes(router, prof_json, n_candidates, ref_model, tokenizer, include_logprobs=True)
# ref_routes = ref_result["routes"]
# ref_logprobs = ref_result["logprobs"]
#
# 2. Calculate policy model logprobs for the same routes
# policy_logprobs = calculate_route_logprobs(router, prof_json, ref_routes, policy_model, tokenizer)


