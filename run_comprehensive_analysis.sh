#!/bin/bash
# 综合训练和分析脚本 - 适用于Colab

# 安装可视化依赖
echo "Installing visualization dependencies..."
pip install matplotlib seaborn

# 创建实验目录
mkdir -p experiments

echo "=== Starting Comprehensive Training Analysis ==="
echo "This will train models with SGD, Adam, and Muon optimizers"
echo "Each training will take significant time (500 epochs each)"

# 运行SGD训练
echo "=== 1/3 Running SGD Training ==="
python advanced_training_analysis.py --optimizer sgd --lr 0.1 --epochs 500 --val_interval 20

# 运行Adam训练
echo "=== 2/3 Running Adam Training ==="
python advanced_training_analysis.py --optimizer adam --lr 0.001 --epochs 500 --val_interval 20

# 运行Muon训练
echo "=== 3/3 Running Muon Training ==="
python advanced_training_analysis.py --optimizer muon --lr 0.01 --epochs 500 --val_interval 20

echo "=== All Training Complete ==="
echo "Results are saved in the experiments/ directory"
echo "Each optimizer has its own timestamped subdirectory with:"
echo "  - models/ (saved checkpoints)"
echo "  - plots/ (visualization graphs)"
echo "  - results.json (raw data)"

# 显示结果目录结构
echo "=== Experiment Directory Structure ==="
find experiments -type f -name "*.png" | head -20
find experiments -type f -name "*.json" | head -10
