#!/usr/bin/env python3
"""
脚本用于检查 neural_model.Net 的参数内容、维度和 require_grad 属性
"""

import torch
from neural_model import Net

def analyze_net_parameters():
    """分析网络参数"""
    # 创建网络 (dim=3072 for SVHN 32x32x3, width=1024, depth=5)
    net = Net(dim=3072, width=1024, depth=5, num_classes=10)
    
    print("=== Net Parameters Analysis ===")
    print(f"Total number of parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    
    print("\n=== Layer-wise Parameter Details ===")
    for name, param in net.named_parameters():
        print(f"Layer: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Size: {param.numel():,} elements")
        print(f"  Requires grad: {param.requires_grad}")
        print(f"  Data type: {param.dtype}")
        print(f"  Device: {param.device}")
        print()
    
    print("=== Parameters() vs Named_parameters() ===")
    params_list = list(net.parameters())
    named_params_list = list(net.named_parameters())
    
    print(f"net.parameters() returns {len(params_list)} tensors")
    print(f"net.named_parameters() returns {len(named_params_list)} (name, tensor) pairs")
    
    print("\n=== Frozen Network Comparison ===")
    # Note: Net class doesn't have freeze parameter, freeze is handled in trainer
    print("Note: freeze is handled in trainer.py, not in Net constructor")
    
    print("\n=== First Few Parameter Values (sample) ===")
    for i, param in enumerate(net.parameters()):
        if i >= 2:  # 只显示前两层
            break
        print(f"Parameter {i} (shape {param.shape}):")
        print(f"  Mean: {param.data.mean().item():.6f}")
        print(f"  Std: {param.data.std().item():.6f}")
        print(f"  Min: {param.data.min().item():.6f}")
        print(f"  Max: {param.data.max().item():.6f}")
        print()

if __name__ == "__main__":
    analyze_net_parameters()
