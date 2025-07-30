#!/usr/bin/env python3
"""
测试完整的 torch.compile 兼容性修复
"""

import torch
import neural_model
import sys
sys.path.append('.')
from trainer import get_clean_state_dict, clean_compiled_state_dict

def test_complete_fix():
    """测试完整的修复方案"""
    print("=== Testing Complete torch.compile Fix ===")
    
    # 创建网络
    net = neural_model.Net(dim=3072, width=1024, depth=5, num_classes=10)
    
    # 模拟编译
    compiled_net = torch.compile(net, mode='reduce-overhead')
    
    print("1. Original model parameter names:")
    for name, _ in net.named_parameters():
        print(f"   {name}")
    
    print("\n2. Compiled model parameter names:")
    for name, _ in compiled_net.named_parameters():
        print(f"   {name}")
    
    # 测试我们的清理函数
    print("\n3. Testing get_clean_state_dict function:")
    clean_state = get_clean_state_dict(compiled_net)
    print("   Cleaned state dict keys:")
    for key in clean_state.keys():
        print(f"   {key}")
    
    # 测试能否加载到新的模型
    print("\n4. Testing loading cleaned state dict:")
    try:
        new_net = neural_model.Net(dim=3072, width=1024, depth=5, num_classes=10)
        new_net.load_state_dict(clean_state)
        print("   ✅ Successfully loaded cleaned state dict!")
        return True
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_fix()
    if success:
        print("\n🎉 Complete fix works! Models can be saved and loaded correctly.")
    else:
        print("\n💥 Fix needs more work.")
