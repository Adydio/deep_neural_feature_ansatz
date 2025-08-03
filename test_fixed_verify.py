#!/usr/bin/env python3
"""
测试修复后的 verify_deep_NFA.py
"""

import torch
import neural_model
import sys
import os
sys.path.append('.')

def test_data_dimensions():
    """测试数据维度处理是否正确"""
    print("=== Testing Data Dimensions ===")
    
    try:
        import verify_deep_NFA
        
        # 创建模拟的 SVHN 数据加载器
        class MockDataLoader:
            def __init__(self, num_batches=3, batch_size=128):
                self.num_batches = num_batches
                self.batch_size = batch_size
                
            def __iter__(self):
                for i in range(self.num_batches):
                    # SVHN data: 32x32x3 = 3072 features
                    data = torch.randn(self.batch_size, 3, 32, 32)
                    labels = torch.randint(0, 10, (self.batch_size,))
                    yield data, labels
        
        # 创建网络
        net = neural_model.Net(dim=3072, width=1024, depth=5, num_classes=10)
        mock_loader = MockDataLoader()
        
        print("Testing layer 0 (input layer)...")
        out0 = verify_deep_NFA.get_layer_output(net, mock_loader, layer_idx=0, max_samples=200)
        print(f"Layer 0 output shape: {out0.shape}")
        assert out0.dim() == 2, f"Expected 2D output, got {out0.dim()}D"
        assert out0.shape[1] == 3072, f"Expected 3072 features, got {out0.shape[1]}"
        
        print("Testing layer 1...")
        out1 = verify_deep_NFA.get_layer_output(net, mock_loader, layer_idx=1, max_samples=200)
        print(f"Layer 1 output shape: {out1.shape}")
        assert out1.dim() == 2, f"Expected 2D output, got {out1.dim()}D"
        assert out1.shape[1] == 1024, f"Expected 1024 features, got {out1.shape[1]}"
        
        print("✅ All dimension tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jacobian_computation():
    """测试 Jacobian 计算是否正常"""
    print("\n=== Testing Jacobian Computation ===")
    
    try:
        import verify_deep_NFA
        
        # 创建小型网络
        net = neural_model.Net(dim=10, width=16, depth=2, num_classes=2)
        test_data = torch.randn(5, 10)  # 5 samples, 10 features
        
        print(f"Test data shape: {test_data.shape}")
        print("Computing Jacobian...")
        
        J = verify_deep_NFA.get_jacobian(net, test_data)
        print(f"Jacobian shape: {J.shape}")
        
        # Expected shape: (output_dim, input_dim, batch_size)
        expected_output_dim = 2  # num_classes
        expected_input_dim = 10  # input features
        expected_batch_size = 5
        
        assert J.shape[0] == expected_output_dim, f"Wrong output dim: {J.shape[0]} vs {expected_output_dim}"
        assert J.shape[1] == expected_input_dim, f"Wrong input dim: {J.shape[1]} vs {expected_input_dim}"
        assert J.shape[2] == expected_batch_size, f"Wrong batch size: {J.shape[2]} vs {expected_batch_size}"
        
        print("✅ Jacobian computation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_egop():
    """测试小规模的 EGOP 计算"""
    print("\n=== Testing Small EGOP ===")
    
    try:
        import verify_deep_NFA
        
        # 创建小型网络和数据
        net = neural_model.Net(dim=20, width=32, depth=2, num_classes=2)
        test_data = torch.randn(100, 20)  # 100 samples, 20 features
        
        print(f"Network parameters: {sum(p.numel() for p in net.parameters())}")
        print(f"Test data shape: {test_data.shape}")
        
        # 测试不带中心化的 EGOP
        print("Computing EGOP without centering...")
        G1 = verify_deep_NFA.egop(net, test_data, centering=False)
        print(f"G1 shape: {G1.shape}")
        
        # 测试带中心化的 EGOP
        print("Computing EGOP with centering...")
        G2 = verify_deep_NFA.egop(net, test_data, centering=True)
        print(f"G2 shape: {G2.shape}")
        
        print("✅ Small EGOP test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing fixed verify_deep_NFA.py...\n")
    
    success1 = test_data_dimensions()
    success2 = test_jacobian_computation()
    success3 = test_small_egop()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed! Fixed version is ready to use.")
        print("\nUsage examples:")
        print("1. Use all training samples (memory intensive but accurate):")
        print("   python verify_deep_NFA.py --path MODEL_PATH")
        print("\n2. Limit samples for memory efficiency:")
        print("   python verify_deep_NFA.py --path MODEL_PATH --max_samples 10000")
        print("\n3. Test single layer:")
        print("   python verify_deep_NFA.py --path MODEL_PATH --max_samples 5000 --layers 0")
    else:
        print("\n💥 Some tests failed. Check the errors above.")
