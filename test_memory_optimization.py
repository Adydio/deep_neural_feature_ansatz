#!/usr/bin/env python3
"""
测试内存优化版本的 verify_deep_NFA.py
"""

import torch
import neural_model
import sys
import os
sys.path.append('.')

def test_memory_optimized_egop():
    """测试内存优化的 EGOP 计算"""
    print("=== Testing Memory-Optimized EGOP ===")
    
    try:
        import verify_deep_NFA
        
        # 创建小型测试网络和数据
        net = neural_model.Net(dim=32, width=64, depth=3, num_classes=2)
        
        # 创建小型测试数据
        test_data = torch.randn(500, 32)  # 500 samples, 32 features
        
        print(f"Test network parameters: {sum(p.numel() for p in net.parameters())}")
        print(f"Test data shape: {test_data.shape}")
        
        # 测试新的 EGOP 函数
        print("\n1. Testing egop with centering=False...")
        G1 = verify_deep_NFA.egop(net, test_data, centering=False)
        print(f"   G matrix shape: {G1.shape}")
        
        print("\n2. Testing egop with centering=True...")
        G2 = verify_deep_NFA.egop(net, test_data, centering=True)
        print(f"   G matrix shape: {G2.shape}")
        
        print("\n3. Memory usage test - processing larger dataset...")
        large_data = torch.randn(2000, 32)  # 2000 samples
        G3 = verify_deep_NFA.egop(net, large_data, centering=True)
        print(f"   Large dataset G matrix shape: {G3.shape}")
        
        print("\n✅ All EGOP tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_functorch_migration():
    """测试 functorch 到 torch.func 的迁移"""
    print("\n=== Testing functorch Migration ===")
    
    try:
        import verify_deep_NFA
        
        # 创建简单网络
        net = neural_model.Net(dim=10, width=16, depth=2, num_classes=2)
        test_data = torch.randn(5, 10)
        
        print("Testing get_jacobian function...")
        J = verify_deep_NFA.get_jacobian(net, test_data)
        print(f"Jacobian shape: {J.shape}")
        print("✅ Jacobian computation works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing memory-optimized verify_deep_NFA.py...\n")
    
    success1 = test_functorch_migration()
    success2 = test_memory_optimized_egop()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Memory-optimized version is ready to use.")
        print("\nUsage:")
        print("python verify_deep_NFA.py --path MODEL_PATH --max_samples 5000")
    else:
        print("\n💥 Some tests failed. Check the errors above.")
