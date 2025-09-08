# Router Training Usage Guide

The `main.py` now supports GRPO training functionality in router-only mode.

## Usage

### 1. Train Router (Default Behavior)

```bash
# Train router using GRPO (training enabled by default)
python -m GRPO.main --router_only --epochs 5 --router_batch_size 32

# Explicitly specify training parameters
python -m GRPO.main --router_only --train_router --epochs 10 --router_batch_size 64 --group_size 8
```

### 2. Evaluate Router Only (No Training)

```bash
# Evaluation only, no training
python -m GRPO.main --router_only --eval_only --router_strategy oracle

# Or explicitly disable training
python -m GRPO.main --router_only --train_router=False
```

### 3. Train Selector (Original Functionality)

```bash
# Train selector (original functionality remains unchanged)
python -m GRPO.main --epochs 5 --users_per_batch 128
```

## New Parameter Descriptions

- `--train_router`: Whether to train router in router-only mode (default: True)
- `--eval_only`: Evaluation only mode, overrides `--train_router` setting
- `--router_batch_size`: Batch size for router training (default: 32)
- `--epochs`: Number of training epochs, applies to both selector and router training

## Output Examples

### Training Mode Output:
```
Training router using GRPO...
Epoch 1/5
[Epoch 1] Router Training - Loss: 0.6234, Avg Recall@50: 0.1234
Epoch 2/5
[Epoch 2] Router Training - Loss: 0.5123, Avg Recall@50: 0.1567
...
Done. Router training completed. Final avg Recall@50 = 0.2345
```

### Evaluation Mode Output:
```
Evaluating router (no training)...
Done. Router-only evaluation avg Recall@50 = 0.1234
```

## Important Notes

1. **Router Compatibility**: Ensure your router implements the `get_route_logprob(prof_json, route)` method to support GRPO training
2. **Memory Usage**: Training mode creates reference models, requiring more memory
3. **Performance Monitoring**: Training process outputs loss and recall metrics to monitor training progress
4. **Router-Only Architecture**: In router-only mode, no selector model is created - only the router is trained using GRPO

## Advanced Usage

### Using Local HuggingFace Models:
```bash
python -m GRPO.main --router_only --train_router \
    --use_hf_local --hf_model meta-llama/Llama-3.1-8B-Instruct \
    --epochs 5 --router_batch_size 16
```

### Save Router Outputs:
```bash
python -m GRPO.main --router_only --train_router \
    --save_router_json router_outputs.json \
    --epochs 3
```
