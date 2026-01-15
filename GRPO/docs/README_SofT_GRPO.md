# SofT-GRPO: Soft Token Group Relative Policy Optimization

## 概述

SofT-GRPO是基于分布的软采样与奖励的GRPO算法实现，主要特点：

1. **Gumbel-Softmax软采样**: 使用分布样本而非离散token
2. **噪声密度比**: 基于Gumbel噪声的重要性比率计算
3. **多路召回奖励**: 使用分类模型softmax输出作为不同recaller的权重

## 快速开始

### 1. VS Code调试配置

已添加完整的`launch.json`配置，包含以下调试选项：

- **SofT-GRPO: Generate SFT Data (ml-1m)**: 生成SFT训练数据
- **SofT-GRPO: SFT Training (ml-1m)**: SFT分类模型训练
- **SofT-GRPO: GRPO Training (ml-1m)**: SofT-GRPO训练
- **SofT-GRPO: Test Model (ml-1m)**: 模型评估
- **SofT-GRPO: Full Pipeline (steam)**: 完整流水线
- **Standalone SofT-GRPO**: 独立SofT-GRPO脚本
- **SofT-GRPO: Debug Small Dataset**: 小数据集调试
- **SofT-GRPO: Hyperparameter Tuning**: 超参数调优
- **SofT-GRPO: Multi-Recaller Experiment**: 多召回器实验

### 2. 命令行脚本

使用快速运行脚本：

```bash
# 完整流水线 (数据生成 -> SFT -> GRPO -> 测试)
./GRPO/run_soft_grpo.sh ml-1m full small

# 仅生成数据
./GRPO/run_soft_grpo.sh ml-1m data small

# 仅SFT训练
./GRPO/run_soft_grpo.sh ml-1m sft small

# 仅GRPO训练
./GRPO/run_soft_grpo.sh ml-1m grpo small

# 仅测试
./GRPO/run_soft_grpo.sh ml-1m test small

# 独立SofT-GRPO
./GRPO/run_soft_grpo.sh ml-1m standalone small

# 调试模式
./GRPO/run_soft_grpo.sh ml-1m debug small
```

**参数说明:**
- `dataset`: ml-1m, steam, Amazon_All_Beauty
- `stage`: data, sft, grpo, test, full, standalone, debug  
- `model_size`: small (Qwen2.5-0.5B), large (Llama-3.2-1B)

### 3. 手动运行

#### 步骤1: 生成SFT数据
```bash
python GRPO/main_pure.py \
    --dataset ml-1m \
    --recbole_models BPR SASRec \
    --gen_sft_data \
    --seed 42
```

#### 步骤2: SFT训练
```bash
python GRPO/main_pure.py \
    --dataset ml-1m \
    --recbole_models BPR SASRec \
    --do_sft \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --bf16
```

#### 步骤3: SofT-GRPO训练
```bash
python GRPO/main_pure.py \
    --dataset ml-1m \
    --recbole_models BPR SASRec \
    --do_grpo \
    --tau_gumbel 1.0 \
    --top_p 0.9 \
    --epsilon 0.2 \
    --beta 0.01 \
    --num_generations 4 \
    --grpo_lr 1e-6 \
    --bf16
```

#### 步骤4: 模型评估
```bash
python GRPO/main_pure.py \
    --dataset ml-1m \
    --recbole_models BPR SASRec \
    --do_test \
    --bf16
```

## 核心参数

### SofT-GRPO算法参数
- `--tau_gumbel`: Gumbel-Softmax温度 (默认: 1.0)
- `--top_p`: Nucleus采样阈值 (默认: 0.9) 
- `--epsilon`: PPO裁剪系数 (默认: 0.2)
- `--beta`: KL惩罚权重 (默认: 0.01)
- `--num_generations`: 每个prompt的生成数量G (默认: 4)

### 训练参数
- `--grpo_lr`: GRPO学习率 (默认: 1e-6)
- `--grpo_epochs`: GRPO训练轮数 (默认: 1)
- `--per_device_train_batch_size`: 每设备批大小
- `--gradient_accumulation_steps`: 梯度累积步数

## 实验配置

### 数据集支持
- **ml-1m**: BPR + SASRec
- **steam**: BPR + SASRec + LightGCN
- **Amazon_All_Beauty**: BPR + SASRec + Pop

### 模型规模
- **Small**: Qwen2.5-0.5B-Instruct (batch_size=4)
- **Large**: Llama-3.2-1B-Instruct (batch_size=2)

## 输出结果

### 1. 模型保存路径
```
GRPO/pure_models/{dataset}/{model_name}_pure_sft_{recallers}/    # SFT模型
GRPO/pure_models/{dataset}/{model_name}_pure_grpo_{recallers}/   # GRPO模型
```

### 2. 评估结果
- 单一召回器选择性能
- 多路召回加权性能
- 性能提升对比

### 3. 日志输出
- 训练损失和指标
- 重要性比率统计
- 裁剪比率
- 奖励统计

## 调试建议

1. **内存不足**: 减少batch_size，增加gradient_accumulation_steps
2. **训练不稳定**: 降低learning_rate，增加warmup_steps
3. **收敛缓慢**: 调整tau_gumbel和epsilon参数
4. **奖励稀疏**: 检查recaller配置和数据质量

## 算法细节

### 软采样过程
1. 对分类logits应用top-p过滤
2. 采样独立Gumbel噪声
3. 计算q' = log(p) + ε
4. 应用Gumbel-Softmax得到软分布

### 重要性比率
- 基于Gumbel噪声密度比率而非策略比率
- r_soft = exp(log_p_new_noise - log_p_old_noise)

### 多路召回奖励  
- 使用softmax权重聚合不同recaller的候选
- 通过NDCG@k计算奖励信号

## 相关文件

- `soft_grpo.py`: 核心算法实现
- `soft_utils.py`: 工具函数
- `trl_trainer.py`: SoftGRPOTrainer类
- `main_pure.py`: 集成训练脚本
- `main_soft_grpo.py`: 独立训练脚本
