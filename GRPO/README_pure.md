# Pure SFT Training for Model Selection

This module (`main_pure.py`) implements a simplified version of the GRPO system where the model learns to predict only the best recommendation model name instead of generating complex JSON outputs.

## Key Differences from `main_soft.py`

1. **Simple Text Output**: Predicts only the model name (e.g., "SASRec") instead of JSON with soft tokens
2. **Standard SFT Training**: Uses standard `SFTTrainer` instead of custom soft token trainer
3. **Accuracy-based Evaluation**: Evaluates based on classification accuracy
4. **No RL Support**: Currently supports only supervised fine-tuning

## Usage

### Command Line

Run the steps separately (cannot be combined due to GPU resource conflicts):

```bash
# Step 1: Generate SFT training data
python -m GRPO.main_pure --gen_sft_data --dataset Amazon_All_Beauty --num_train_samples 1000

# Step 2: Train the model with LoRA
python -m GRPO.main_pure --do_sft --dataset Amazon_All_Beauty --use_lora

# Step 3: Test the trained model
python -m GRPO.main_pure --do_test --dataset Amazon_All_Beauty
```

### Using the Shell Script

```bash
# Run full pipeline with default settings
./GRPO/pure_sft.sh

# Customize parameters
./GRPO/pure_sft.sh --dataset ml-1m --gpu 1 --lr 1e-4 --epochs 5 --samples 2000
```

### Using VS Code Launch Configurations

Available debug configurations in VS Code:
- **GRPO Pure: Generate SFT Data** - Generate training data
- **GRPO Pure: Train SFT** - Train on ml-1m dataset
- **GRPO Pure: Train SFT (Amazon Beauty)** - Train on Amazon Beauty dataset
- **GRPO Pure: Test Model** - Evaluate trained model

## Parameters

### Data Generation
- `--gen_sft_data`: Generate SFT training data
- `--num_train_samples`: Number of training samples to generate (default: 1000)
- `--final_k`: Top-k for evaluation (default: 50)

### Training
- `--do_sft`: Perform supervised fine-tuning
- `--use_lora`: Enable LoRA training (recommended)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--warmup_steps`: Warmup steps (default: 100)
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency

### Testing
- `--do_test`: Evaluate the model
- `--max_length`: Maximum sequence length (default: 1536)

### Model & Data
- `--dataset`: Dataset name (e.g., 'Amazon_All_Beauty', 'ml-1m')
- `--model_name`: Base model to use (default: 'Qwen/Qwen2.5-0.5B-Instruct')
- `--data_path`: Path to dataset files (default: './dataset')
- `--output_dir`: Output directory for models (default: 'GRPO/pure_models')

## Output

- Training data: `{output_dir}/{dataset}/{model_name}_pure_sft_data/`
- Trained model: `{output_dir}/{dataset}/{model_name}_pure_sft/checkpoint-*/`
- Test results: `results/pure_results_{dataset}.json`

## Dataset Statistics

The script prints helpful statistics during data generation:
- Average NDCG for each recaller model
- Distribution of best models in the training set

## Example Results

After testing, you'll see:
- Accuracy: Percentage of correct model predictions
- Prediction distribution: Which models the system predicts and how often

