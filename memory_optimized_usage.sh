#!/bin/bash
# 使用内存优化版本运行 verify_deep_NFA.py

echo "=== Memory-Optimized verify_deep_NFA.py Usage ==="
echo "For Colab or memory-limited environments"
echo

echo "1. Light memory usage (5000 samples):"
echo "python verify_deep_NFA.py --path MODEL_PATH --max_samples 5000"
echo

echo "2. Medium memory usage (10000 samples, default):"
echo "python verify_deep_NFA.py --path MODEL_PATH --max_samples 10000"
echo

echo "3. Single layer testing (fastest):"
echo "python verify_deep_NFA.py --path MODEL_PATH --max_samples 5000 --layers 0"
echo

echo "4. Your original command (optimized):"
echo "python verify_deep_NFA.py --path ./saved_nns/svhn:num_epochs:100:learning_rate:0.01:weight_decay:0:init:default:optimizer:muon:freeze:False:width:1024:depth:5:act:relu:val_interval:20:nn.pth --max_samples 5000"
echo

echo "=== Key Optimizations ==="
echo "✅ Reduced batch size from 1000 to 200"
echo "✅ Streaming processing (no storing all Jacobians)"
echo "✅ Automatic memory cleanup with torch.cuda.empty_cache()"
echo "✅ Sample limit option (--max_samples)"
echo "✅ Progress reporting every 10 batches"
echo "✅ Fixed deprecated functorch APIs"
echo

echo "=== Memory Estimates ==="
echo "5000 samples:  ~2-4 GB RAM"
echo "10000 samples: ~4-8 GB RAM"
echo "20000 samples: ~8-16 GB RAM"
