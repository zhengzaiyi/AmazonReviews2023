"""
Utilities for soft token generation and beta sampling.
Shared between main_soft.py and trl_trainer.py to avoid code duplication.
"""

import torch
from typing import List
from tqdm import tqdm


def build_soft_template(model_names: List[str]) -> str:
    """Build JSON template with [num][soft_token] placeholders.
    
    Args:
        model_names: List of model names to include in template
        
    Returns:
        JSON string template with placeholders
    """
    lines = ["{"]
    for i, name in enumerate(model_names):
        lines.append(f'  "{name}": {{')
        lines.append('    "top-k": [num][soft_token],')
        lines.append('    "score-weight": [num][soft_token]')
        lines.append("  }" + ("," if i < len(model_names) - 1 else ""))
    lines.append("}")
    return "\n".join(lines)


def sample_from_logits(logit, clamp_range=(1e-6, 1.0 - 1e-6), sample=True):
    """Sample from logits using sigmoid activation.
    
    Args:
        logit: Raw logit value (unbounded real number)
        clamp_range: Range to clamp sampled values
        sample: Whether to sample from Bernoulli or return sigmoid probability
        
    Returns:
        Sampled value in [0, 1] range
    """
    # Convert logit to probability
    prob = torch.sigmoid(logit).clamp(*clamp_range)
    
    if sample:
        # Sample from Bernoulli distribution
        dist = torch.distributions.Bernoulli(prob)
        return dist.sample().item()
    else:
        # Return probability directly
        return prob.item()


def sample_from_beta_params(alpha, beta, clamp_range=(1e-6, 1.0 - 1e-6), sample=True, apply_transform=True):
    """Sample from Beta distribution with given parameters.
    
    Args:
        alpha: Alpha parameter
        beta: Beta parameter  
        clamp_range: Range to clamp sampled values
        sample: Whether to sample or return mean/mode
        apply_transform: Whether to apply softplus + 1.0 transform to alpha/beta
        
    Returns:
        Sampled value from Beta distribution
    """
    # Process parameters if needed
    if apply_transform:
        alpha_processed = torch.nn.functional.softplus(alpha) + 1.0
        beta_processed = torch.nn.functional.softplus(beta) + 1.0
    else:
        alpha_processed = alpha
        beta_processed = beta
    
    # Sample and clamp
    if sample:
        dist = torch.distributions.Beta(alpha_processed, beta_processed)
        return dist.sample().clamp(*clamp_range).item()
    else:
        # return mean value
        # return alpha_processed / (alpha_processed + beta_processed)
        return (alpha_processed - 1) / (alpha_processed + beta_processed - 2)
        # TODO: try (alpha_processed - 1) / (alpha_processed + beta_processed - 2)


def replace_placeholders_with_values(template: str, values: List[float], precision: int = 6) -> str:
    """Replace [num][soft_token] placeholders with numeric values.
    
    Args:
        template: Template string containing placeholders
        values: List of numeric values to substitute
        precision: Number of decimal places for formatting
        
    Returns:
        Template with placeholders replaced by values
    """
    result = template
    for val in values:
        result = result.replace("[num][soft_token]", f"{float(val):.{precision}f}", 1)
    
    # Fill remaining placeholders with default value
    while "[num][soft_token]" in result:
        result = result.replace("[num][soft_token]", f"{0.0:.{precision}f}", 1)
    
    return result


def generate_soft_completions(model, tokenizer, test_dataset, model_names: List[str], max_length: int = 1536) -> List[str]:
    """Generate completions using soft token beta sampling.
    
    Args:
        model: Model with value_head for beta parameter prediction
        tokenizer: Tokenizer with [num] and [soft_token] special tokens
        test_dataset: Dataset containing prompts
        model_names: List of model names for template
        max_length: Maximum sequence length
        
    Returns:
        List of generated completion strings
    """
    completions = []
    device = next(model.parameters()).device
    model.eval()
    model.to(device)
    soft_token_id = tokenizer.convert_tokens_to_ids("[soft_token]")
    num_token_id = tokenizer.convert_tokens_to_ids("[num]")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Generating completions"):
            # Extract prompt
            prompt = test_dataset[i]['prompt']
            prompt_text = prompt[0]['content'] if isinstance(prompt, list) and isinstance(prompt[0], dict) else str(prompt)
            
            # Build full text with template
            template = build_soft_template(model_names)
            full_text = prompt_text + "\n\n" + template
            
            # Tokenize and forward pass
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            # Need to explicitly request hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Sample values from logits (following new BCE logic)
            sampled_values = []
            if hasattr(model, 'value_head') and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                input_ids = inputs['input_ids'][0]
                num_positions = (input_ids == num_token_id).nonzero(as_tuple=True)[0]
                soft_positions = (input_ids == soft_token_id).nonzero(as_tuple=True)[0]
                
                if len(num_positions) > 0 and len(soft_positions) > 0:
                    # Get hidden states at [num] positions
                    hidden_states = outputs.hidden_states[-1][0, num_positions, :]  # (K, H)
                    value_preds = model.value_head(hidden_states)  # (K, 2)
                    
                    # Sample from logits (only use first output as logit)
                    for pred in value_preds:
                        logit = pred[0]  # Only use the first output (mu)
                        sampled_values.append(sample_from_logits(logit))
            
            # Use default values if no sampling occurred
            if not sampled_values:
                num_soft_tokens = template.count("[num][soft_token]")
                sampled_values = [0.5] * num_soft_tokens
            
            # Replace placeholders with sampled values
            completion_text = replace_placeholders_with_values(template, sampled_values)
            completions.append(completion_text)
    
    return completions
