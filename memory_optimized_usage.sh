#!/bin/bash
# 使用修复和优化版本运行 verify_deep_NFA.py

echo "=== Fixed & Optimized verify_deep_NFA.py Usage ==="
echo "✅ Fixed dimension bugs"
echo "✅ Fixed torch.compile compatibility"  
echo "✅ Fixed deprecated functorch APIs"
echo "✅ Memory optimized EGOP computation"
echo

echo "=== Recommended Usage ==="
echo
echo "1. Full accuracy (use ALL training samples, memory intensive):"
echo "python verify_deep_NFA.py --path MODEL_PATH"
echo

echo "2. Memory efficient (limit samples for Colab/limited memory):"
echo "python verify_deep_NFA.py --path MODEL_PATH --max_samples 10000"
echo

echo "3. Quick test (single layer, small sample):"
echo "python verify_deep_NFA.py --path MODEL_PATH --max_samples 5000 --layers 0"
echo

echo "4. Your specific model:"
echo "python verify_deep_NFA.py --path ./saved_nns/svhn:num_epochs:100:learning_rate:0.01:weight_decay:0:init:default:optimizer:muon:freeze:False:width:1024:depth:5:act:relu:val_interval:20:nn.pth"
echo

echo "5. Same model with memory limit:"
echo "python verify_deep_NFA.py --path ./saved_nns/svhn:num_epochs:100:learning_rate:0.01:weight_decay:0:init:default:optimizer:muon:freeze:False:width:1024:depth:5:act:relu:val_interval:20:nn.pth --max_samples 15000"
echo

echo "=== Key Fixes & Optimizations ==="
echo "🔧 Fixed RuntimeError: matmul dimension mismatch"
echo "🔧 Fixed data shape issues (ensures 2D tensors)"
echo "🔧 Uses SAME training samples as in training (when --max_samples not specified)"
echo "⚡ Streaming EGOP computation (no storing all Jacobians)"
echo "⚡ Reduced batch size from 1000 to 200"
echo "⚡ Automatic memory cleanup"
echo "⚡ Progress reporting every 10 batches"
echo "🆕 Modern PyTorch APIs (torch.func instead of functorch)"
echo

echo "=== Memory Usage Guidelines ==="
echo "No limit (full dataset): Use for final accurate results"
echo "15000-20000 samples: Good balance of accuracy and memory (~8-12 GB)"
echo "10000 samples: Default for memory-limited environments (~4-8 GB)"
echo "5000 samples: Quick testing (~2-4 GB)"
