#!/usr/bin/env python3
"""
测试 verify_deep_NFA.py 的完整修复
"""

import torch
import neural_model
import sys
import os
sys.path.append('.')

def create_mock_model_files():
    """创建模拟的模型文件来测试 verify_deep_NFA.py"""
    print("Creating mock model files for testing...")
    
    # 确保 saved_nns 目录存在
    os.makedirs('saved_nns', exist_ok=True)
    
    # 创建一个网络
    net = neural_model.Net(dim=3072, width=1024, depth=5, num_classes=10)
    
    # 模拟编译后的状态字典（带 _orig_mod 前缀）
    compiled_state = {}
    for key, value in net.state_dict().items():
        compiled_state[f'_orig_mod.{key}'] = value
    
    # 保存模拟的模型文件
    mock_filename = 'svhn:num_epochs:100:learning_rate:0.01:weight_decay:0:init:default:optimizer:muon:freeze:False:width:1024:depth:5:act:relu:val_interval:20:nn.pth'
    init_filename = 'init_' + mock_filename
    
    # 保存训练后的模型
    torch.save({'state_dict': compiled_state}, f'saved_nns/{mock_filename}')
    
    # 保存初始化模型  
    torch.save({'state_dict': compiled_state}, f'saved_nns/{init_filename}')
    
    print(f"✅ Created: saved_nns/{mock_filename}")
    print(f"✅ Created: saved_nns/{init_filename}")
    
    return f'./saved_nns/{mock_filename}'

def test_verify_nfa_functions():
    """测试 verify_deep_NFA.py 中的关键函数"""
    print("\n=== Testing verify_deep_NFA functions ===")
    
    try:
        import verify_deep_NFA
        
        # 创建模拟文件
        model_path = create_mock_model_files()
        
        # 测试 read_configs 函数
        print("\n1. Testing read_configs...")
        width, depth, act_name = verify_deep_NFA.read_configs(model_path)
        print(f"   Width: {width}, Depth: {depth}, Act: {act_name}")
        assert width == 1024 and depth == 5 and act_name == 'relu'
        print("   ✅ read_configs works correctly")
        
        # 测试 load_nn 函数
        print("\n2. Testing load_nn...")
        net, M = verify_deep_NFA.load_nn(model_path, width=1024, depth=5, 
                                         dim=3072, num_classes=10, layer_idx=0)
        print(f"   Matrix shape: {M.shape}")
        print("   ✅ load_nn works correctly")
        
        # 测试 load_init_nn 函数
        print("\n3. Testing load_init_nn...")
        init_net, M0 = verify_deep_NFA.load_init_nn(model_path, width=1024, depth=5,
                                                    dim=3072, num_classes=10, layer_idx=0)
        print(f"   Init matrix shape: {M0.shape}")
        print("   ✅ load_init_nn works correctly")
        
        print("\n🎉 All functions work correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理模拟文件
        try:
            os.remove('saved_nns/svhn:num_epochs:100:learning_rate:0.01:weight_decay:0:init:default:optimizer:muon:freeze:False:width:1024:depth:5:act:relu:val_interval:20:nn.pth')
            os.remove('saved_nns/init_svhn:num_epochs:100:learning_rate:0.01:weight_decay:0:init:default:optimizer:muon:freeze:False:width:1024:depth:5:act:relu:val_interval:20:nn.pth')
            print("\n🧹 Cleaned up mock files")
        except:
            pass

if __name__ == "__main__":
    success = test_verify_nfa_functions()
    if success:
        print("\n✨ verify_deep_NFA.py is ready to use!")
    else:
        print("\n💥 There are still issues to fix.")
