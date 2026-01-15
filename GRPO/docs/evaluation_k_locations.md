# Evaluation@k 的 k 值设置位置汇总

## 1. 命令行参数设置
- **位置**: `main_pure.py:1101`
- **参数**: `--final_k`
- **默认值**: `50`
- **用途**: 控制召回阶段返回的候选物品数量，以及部分评估指标使用的 k 值

```python
parser.add_argument('--final_k', type=int, default=50)
```

## 2. 硬编码的评估 k 值列表
以下位置硬编码了 `[10, 20, 50]` 作为评估指标：

### 2.1 `evaluate_baseline_recallers` 函数
- **位置**: `main_pure.py:746, 763`
- **代码**:
  ```python
  for k in [10, 20, 50]:
      metrics["avg_score_weight"][f"ndcg@{k}"].append(ndcg_at_k(avg_rec, gt_items, k))
      metrics["avg_score_weight"][f"recall@{k}"].append(recall_at_k(avg_rec, gt_items, k))
  ```
- **用途**: 评估 baseline recallers 和平均融合方法

### 2.2 `evaluate_multi_channel_recall` 函数
- **位置**: `main_pure.py:897, 910, 929, 946, 968, 991`
- **代码**:
  ```python
  for k in [10, 20, 50]:
      metrics["single_select"][f"ndcg@{k}"].append(ndcg_at_k(single_rec, gt_items, k))
      metrics["single_select"][f"recall@{k}"].append(recall_at_k(single_rec, gt_items, k))
  ```
- **用途**: 评估单选择、多通道、平均融合和各个基础 recaller

### 2.3 打印结果时的 k 值
- **位置**: `main_pure.py:785, 799, 819, 968, 991`
- **代码**:
  ```python
  for k in [10, 20, 50]:
      for metric in ['ndcg', 'recall']:
          key = f"{metric}@{k}"
  ```
- **用途**: 打印和比较不同方法的性能

## 3. GRPO 训练时的 k 值（硬编码）
- **位置**: `main_pure.py:1502-1503`
- **代码**:
  ```python
  # Use k=5 for reward ndcg@k during GRPO (keep other stages unchanged)
  grpo_trainer.pure_final_k = 5
  ```
- **用途**: GRPO 训练时计算 reward 使用的 NDCG@k 的 k 值
- **注意**: 这是硬编码为 5，与其他阶段的 k 值不同
- **在 GRPO trainer 中的使用** (`trl_trainer.py:1633, 1679, 1714`):
  - 用于召回物品数量: `recallers[name_lower].recall(user_id, final_k, history)`
  - 用于计算 reward: `compute_ndcg_at_k(rec_list, ground_truth, final_k)`
  - 默认值: 如果未设置 `pure_final_k`，则使用 50

## 4. 使用 `final_k` 参数的评估位置

### 4.1 `create_sft_dataset` 函数
- **位置**: `main_pure.py:143`
- **代码**:
  ```python
  ndcg = ndcg_at_k(item_ids, gt_items, k=final_k)
  ```
- **用途**: 在创建 SFT 数据集时，评估每个 recaller 的 NDCG@final_k

### 4.2 `evaluate_pure_model` 函数
- **位置**: `main_pure.py:419, 455, 456, 468, 469, 525, 526, 528, 529, 584, 585, 598`
- **代码**:
  ```python
  def evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers=None, final_k=50):
      # ...
      pred_ndcg = ndcg_at_k(pred_item_ids, gt_items, k=final_k)
      pred_recall = recall_at_k(pred_item_ids, gt_items, k=final_k)
      true_ndcg = ndcg_at_k(true_item_ids, gt_items, k=final_k)
      true_recall = recall_at_k(true_item_ids, gt_items, k=final_k)
      # ...
      print(f"Average Predicted NDCG@{final_k}: {avg_pred_ndcg:.4f}")
      print(f"Average True Best NDCG@{final_k}: {avg_true_ndcg:.4f}")
  ```
- **用途**: 评估纯分类模型的推荐性能

### 4.3 `evaluate_baseline_recallers` 函数
- **位置**: `main_pure.py:709, 761`
- **代码**:
  ```python
  def evaluate_baseline_recallers(
      test_dataset,
      recallers: Dict[str, RecBoleRecaller],
      recaller_names: List[str],
      final_k: int = 50,
      ...
  ):
      # ...
      items = recallers[recaller_name].recall(user_id, final_k, eval_hist, ...)
  ```
- **用途**: 评估 baseline recallers，使用 final_k 作为召回数量

### 4.4 `evaluate_multi_channel_recall` 函数
- **位置**: `main_pure.py:840, 895, 905, 924, 944`
- **代码**:
  ```python
  def evaluate_multi_channel_recall(
      model, 
      tokenizer, 
      test_dataset, 
      recallers: Dict[str, RecBoleRecaller],
      recaller_names: List[str],
      final_k: int = 50,
      ...
  ):
      # ...
      items = recallers[pred_recaller].recall(user_id, final_k, eval_hist, ...)
  ```
- **用途**: 评估多通道召回方法，使用 final_k 作为召回数量

## 5. 函数默认参数中的 k 值
以下函数的默认参数中设置了 `final_k=50`:
- `evaluate_pure_model`: `final_k=50` (line 419)
- `evaluate_baseline_recallers`: `final_k=50` (line 709)
- `evaluate_multi_channel_recall`: `final_k=50` (line 840)

## 总结

### k 值的不同用途：
1. **召回阶段的 k**: 使用 `args.final_k` (默认 50)，控制从每个 recaller 召回多少物品
2. **评估阶段的 k**: 
   - 硬编码的 `[10, 20, 50]` 用于计算和报告 NDCG@k 和 Recall@k
   - `args.final_k` 也用于部分评估指标
3. **GRPO 训练时的 k**: 硬编码为 `5`，用于计算 reward 的 NDCG@k

### 潜在问题：
- GRPO 训练时使用的 k=5 是硬编码的，与其他阶段不一致
- 评估时使用的 k 值列表 `[10, 20, 50]` 是硬编码的，无法通过参数配置
- `final_k` 参数既用于召回数量，也用于部分评估指标，可能造成混淆

### 建议：
1. 将 GRPO 的 k 值改为可配置参数
2. 将评估 k 值列表改为可配置参数
3. 明确区分召回 k 值和评估 k 值的用途
