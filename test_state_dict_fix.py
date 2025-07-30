#!/usr/bin/env python3
"""
测试脚本：验证编译模型状态字典的清理功能
"""

import torch
import neural_model

def clean_compiled_state_dict(state_dict):
    """
    Remove '_orig_mod.' prefix from compiled model state dict keys.
    This fixes the loading issue when models are saved after torch.compile().
    """
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_key = key[len('_orig_mod.'):]
            cleaned_state_dict[cleaned_key] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict

def test_state_dict_cleaning():
    """测试状态字典清理功能"""
    print("=== Testing State Dict Cleaning ===")
    
    # 创建模拟的编译模型状态字典
    mock_compiled_state = {
        '_orig_mod.first.0.weight': torch.randn(1024, 3072),
        '_orig_mod.middle.0.1.weight': torch.randn(1024, 1024),
        '_orig_mod.middle.1.1.weight': torch.randn(1024, 1024),
        '_orig_mod.middle.2.1.weight': torch.randn(1024, 1024),
        '_orig_mod.middle.3.1.weight': torch.randn(1024, 1024),
        '_orig_mod.last.1.weight': torch.randn(10, 1024),
    }
    
    print("Original keys (with _orig_mod prefix):")
    for key in mock_compiled_state.keys():
        print(f"  {key}")
    
    # 清理状态字典
    cleaned_state = clean_compiled_state_dict(mock_compiled_state)
    
    print("\nCleaned keys (without _orig_mod prefix):")
    for key in cleaned_state.keys():
        print(f"  {key}")
    
    # 测试是否能正确加载到模型
    try:
        net = neural_model.Net(dim=3072, width=1024, depth=5, num_classes=10)
        net.load_state_dict(cleaned_state)
        print("\n✅ Successfully loaded cleaned state dict into model!")
        return True
    except Exception as e:
        print(f"\n❌ Failed to load cleaned state dict: {e}")
        return False

if __name__ == "__main__":
    success = test_state_dict_cleaning()
    if success:
        print("\n🎉 State dict cleaning fix works correctly!")
    else:
        print("\n💥 State dict cleaning fix needs more work!")
