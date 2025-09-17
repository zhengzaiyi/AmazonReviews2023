# RecBole Dataset Configuration Guide

## Data Format Requirements

Your dataset needs to follow RecBole's standard format. Main file structure:

```
data/
├── All_Beauty/
│   ├── All_Beauty.inter  # Interaction data
│   ├── All_Beauty.item   # Item features (optional)
│   └── All_Beauty.user   # User features (optional)
```

## Interaction Data Format (.inter file)

Required fields:
- `user_id:token` - User ID
- `item_id:token` - Item ID  
- `timestamp:float` - Timestamp (for temporal ordering)

Example:
```
user_id:token	item_id:token	timestamp:float
0	0	1234567890.0
0	1	1234567891.0
1	0	1234567892.0
```

## Model Configuration

### Supported RecBole Model Types:

1. **Sequential Recommendation Models**：
   - SASRec - Self-Attentive Sequential Recommendation
   - GRU4Rec - GRU-based sequential recommendation
   - BERT4Rec - BERT for sequential recommendation

2. **Collaborative Filtering Models**：  
   - BPR - Bayesian Personalized Ranking
   - ItemKNN - Item-based collaborative filtering
   - Pop - Popularity-based recommendation

3. **Matrix Factorization Models**：
   - NMF - Non-negative Matrix Factorization
   - MF - Matrix Factorization

## Configuring GRPO to Use RecBole Models

### 1. Specify models to use
```bash
--recbole_models SASRec BPR Pop
```

### 2. Set checkpoint directory
```bash
--checkpoint_dir ./checkpoints
--use_latest_checkpoint
```

### 3. Routing configuration format

New routing configuration format:
```json
{
  "model_1": "sasrec",    # First model name
  "k_1": 100,             # Number of items recalled from first model
  "model_2": "bpr",       # Second model name  
  "k_2": 50,              # Number of items recalled from second model
  "w_1": 0.7              # Weight of first model
}
```

## Usage Steps

1. **Prepare data**: Ensure data format follows RecBole requirements
2. **Train models**: Use RecBole to train base recommendation models
3. **Save checkpoints**: Ensure model weights are saved in checkpoints directory
4. **Run GRPO**: Use modified GRPO.py for routing learning

## Troubleshooting

### Common Issues:

1. **RecBole import error**：
   ```bash
   pip install recbole
   ```

2. **Checkpoint not found**：
   - Check file name format: `{ModelName}-{timestamp}.pth`
   - Ensure file path is correct

3. **Data format error**：
   - Check field separators (tab)
   - Ensure field type declarations are correct
   - Verify timestamp format

4. **Out of memory**：
   - Reduce batch_size
   - Use float16 precision
   - Limit number of candidate models
