# RecBole数据集配置指南

## 数据格式要求

你的数据集需要遵循RecBole的标准格式。主要文件结构：

```
data/
├── All_Beauty/
│   ├── All_Beauty.inter  # 交互数据
│   ├── All_Beauty.item   # 物品特征（可选）
│   └── All_Beauty.user   # 用户特征（可选）
```

## 交互数据格式 (.inter文件)

必须包含的字段：
- `user_id:token` - 用户ID
- `item_id:token` - 物品ID  
- `timestamp:float` - 时间戳（用于时序排序）

示例：
```
user_id:token	item_id:token	timestamp:float
0	0	1234567890.0
0	1	1234567891.0
1	0	1234567892.0
```

## 模型配置

### 支持的RecBole模型类型：

1. **序列推荐模型**：
   - SASRec - Self-Attentive Sequential Recommendation
   - GRU4Rec - GRU-based sequential recommendation
   - BERT4Rec - BERT for sequential recommendation

2. **协同过滤模型**：  
   - BPR - Bayesian Personalized Ranking
   - ItemKNN - Item-based collaborative filtering
   - Pop - Popularity-based recommendation

3. **矩阵分解模型**：
   - NMF - Non-negative Matrix Factorization
   - MF - Matrix Factorization

## 配置GRPO使用RecBole模型

### 1. 指定要使用的模型
```bash
--recbole_models SASRec BPR Pop
```

### 2. 设置checkpoint目录
```bash
--checkpoint_dir ./checkpoints
--use_latest_checkpoint
```

### 3. 路由配置格式

新的路由配置格式：
```json
{
  "model_1": "sasrec",    # 第一个模型名称
  "k_1": 100,             # 从第一个模型召回的物品数
  "model_2": "bpr",       # 第二个模型名称  
  "k_2": 50,              # 从第二个模型召回的物品数
  "w_1": 0.7              # 第一个模型的权重
}
```

## 使用步骤

1. **准备数据**：确保数据格式符合RecBole要求
2. **训练模型**：使用RecBole训练基础推荐模型
3. **保存checkpoints**：确保模型权重保存在checkpoints目录
4. **运行GRPO**：使用修改后的GRPO.py进行路由学习

## 故障排除

### 常见问题：

1. **RecBole导入错误**：
   ```bash
   pip install recbole
   ```

2. **找不到checkpoint**：
   - 检查文件名格式：`{ModelName}-{timestamp}.pth`
   - 确保文件路径正确

3. **数据格式错误**：
   - 检查字段分隔符（制表符）
   - 确保字段类型声明正确
   - 验证时间戳格式

4. **内存不足**：
   - 减少batch_size
   - 使用float16精度
   - 限制候选模型数量
