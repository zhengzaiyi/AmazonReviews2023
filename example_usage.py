#!/usr/bin/env python3
"""
使用RecBole模型的GRPO示例
"""

import os
import sys

def main():
    # 示例1: 使用RecBole模型进行路由器推荐（无训练selector）
    print("示例1: Router-only模式，使用已训练的RecBole模型")
    
    cmd1 = [
        "python", "GRPO.py",
        "--dataset", "All_Beauty",  # 根据你的数据集名称调整
        "--data_path", "./data",
        "--recbole_models", "SASRec", "BPR", "Pop",  # 使用的RecBole模型
        "--checkpoint_dir", "./checkpoints",
        "--use_latest_checkpoint",  # 使用最新的checkpoint
        "--router_only",  # 只使用router，不训练selector
        "--router_strategy", "first",  # 使用router返回的第一个路由
        "--save_router_json", "router_outputs.json",  # 保存路由决策
        "--final_k", "50",
        "--group_size", "4",
        "--seed", "42"
    ]
    
    print("命令1:", " ".join(cmd1))
    print()
    
    # 示例2: 训练selector
    print("示例2: 训练模式，使用GRPO训练selector")
    
    cmd2 = [
        "python", "GRPO.py",
        "--dataset", "All_Beauty",
        "--data_path", "./data", 
        "--recbole_models", "SASRec", "BPR",
        "--checkpoint_dir", "./checkpoints",
        "--use_latest_checkpoint",
        "--epochs", "5",
        "--users_per_batch", "64",
        "--group_size", "4", 
        "--final_k", "50",
        "--seed", "42"
    ]
    
    print("命令2:", " ".join(cmd2))
    print()
    
    # 示例3: 使用HuggingFace本地模型做路由
    print("示例3: 使用本地HuggingFace模型进行路由")
    
    cmd3 = [
        "python", "GRPO.py",
        "--dataset", "All_Beauty",
        "--data_path", "./data",
        "--recbole_models", "SASRec", "BPR", "Pop",
        "--checkpoint_dir", "./checkpoints", 
        "--use_latest_checkpoint",
        "--router_only",
        "--use_hf_local",  # 使用本地HF模型
        "--hf_model", "microsoft/DialoGPT-medium",  # 指定模型
        "--hf_dtype", "float16",
        "--hf_device", "auto",
        "--final_k", "50"
    ]
    
    print("命令3:", " ".join(cmd3))
    print()
    
    print("注意事项:")
    print("1. 确保你的数据集在./data目录下，格式符合RecBole要求")
    print("2. 确保./checkpoints目录下有对应模型的训练好的权重文件")
    print("3. 模型名称需要与checkpoint文件名前缀匹配，如'SASRec-Aug-26-2025_16-28-37.pth'")
    print("4. 如果没有checkpoint，系统会使用未训练的模型")
    print("5. 可用的RecBole模型包括: SASRec, BPR, Pop, ItemKNN等")

if __name__ == "__main__":
    main()
