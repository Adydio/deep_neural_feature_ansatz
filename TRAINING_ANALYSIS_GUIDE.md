# 综合训练与AGOP分析使用指南

## 脚本说明

我为您创建了两个脚本来完成复杂的训练和分析任务：

### 1. `advanced_training_analysis.py` - 完整版本
- 适用于有充足内存的环境
- 支持单个优化器训练
- 生成高质量可视化图表

### 2. `colab_training_analysis.py` - Colab优化版本 ⭐ **推荐**
- 专门为Google Colab优化
- 自动安装依赖包
- 内存管理优化
- 一键运行所有三个优化器

## 快速开始 - Colab使用

### 方式1：一键运行所有优化器（推荐）
```python
# 在Colab中直接运行
python colab_training_analysis.py
```

这将依次训练：
- SGD (lr=0.1)
- Adam (lr=0.001) 
- Muon (lr=0.01)

每个训练500个epoch，每20个epoch分析一次。

### 方式2：单独运行每个优化器
```python
# SGD训练
python advanced_training_analysis.py --optimizer sgd --lr 0.1 --epochs 500 --val_interval 20

# Adam训练  
python advanced_training_analysis.py --optimizer adam --lr 0.001 --epochs 500 --val_interval 20

# Muon训练
python advanced_training_analysis.py --optimizer muon --lr 0.01 --epochs 500 --val_interval 20
```

## 本地环境使用

### 安装依赖
```bash
pip install matplotlib seaborn
```

### 运行脚本
```bash
# 使用bash脚本一键运行所有
chmod +x run_comprehensive_analysis.sh
./run_comprehensive_analysis.sh

# 或者手动运行每个
python advanced_training_analysis.py --optimizer sgd --lr 0.1
python advanced_training_analysis.py --optimizer adam --lr 0.001  
python advanced_training_analysis.py --optimizer muon --lr 0.01
```

## 输出结果

每次训练会创建如下目录结构：
```
experiments/
├── sgd_20250803_123456/
│   ├── models/
│   │   ├── init_model.pth
│   │   ├── model_epoch_0.pth
│   │   ├── model_epoch_20.pth
│   │   └── ...
│   ├── plots/
│   │   ├── training_analysis.png      # 5层综合图
│   │   ├── training_analysis.pdf
│   │   ├── layer_0_analysis.png       # 单层详细图
│   │   └── ...
│   └── results.json                   # 原始数据
├── adam_20250803_123457/
└── muon_20250803_123458/
```

## 可视化图表说明

### 主要图表：`training_analysis.png`
- **2×3子图布局**：5个层（Layer 0-4）+ 1个空位
- **每个子图包含**：
  - 蓝线：Train Loss（左y轴）
  - 红线：Val Loss（左y轴）
  - 绿线：AGOP/NFM Correlation（右y轴，带圆点标记）
- **双y轴设计**：自动调整不同scale的数据

### 单层详细图：`layer_X_analysis.png`
- 每层单独的大图
- 相同的双y轴设计
- 更清晰的细节展示

## 数据记录

### `results.json` 包含：
```json
{
  "epochs": [0, 20, 40, 60, ...],
  "train_losses": [...],
  "val_losses": [...],
  "layer_correlations": {
    "0": [...],
    "1": [...],
    "2": [...],
    "3": [...],
    "4": [...]
  }
}
```

## 参数说明

### 命令行参数
- `--optimizer`: sgd, adam, muon
- `--lr`: 学习率（默认：sgd=0.1, adam=0.001, muon=0.01）
- `--epochs`: 训练轮数（默认：500）
- `--val_interval`: 分析间隔（默认：20）
- `--max_samples`: AGOP计算样本数限制（内存管理）

### 内存优化说明
- Colab版本默认限制AGOP计算使用5000个样本
- 可通过`max_samples`参数调整
- 自动GPU内存清理
- 流式处理大数据集

## 预期运行时间

### 在Colab环境：
- 单个优化器：约2-3小时
- 全部三个优化器：约6-9小时

### 时间分布：
- 训练：70%（每epoch约30-60秒）
- AGOP计算：25%（每次分析约5-10分钟）
- 可视化：5%

## 故障排除

### 内存不足
```python
# 使用更少样本进行AGOP计算
python advanced_training_analysis.py --optimizer sgd --max_samples 2000
```

### Torch.compile错误（Muon）
- 脚本会自动fallback到常规模式
- 不影响结果准确性

### 可视化包缺失
```bash
pip install matplotlib seaborn
```

## Colab专用代码

在Colab中可以直接复制粘贴运行：

```python
# 克隆代码（如果需要）
!git clone [your_repo_url]
%cd deep_neural_feature_ansatz

# 安装依赖
!pip install matplotlib seaborn

# 运行分析
!python colab_training_analysis.py

# 查看结果
import os
print("Generated experiments:")
for root, dirs, files in os.walk("experiments"):
    for file in files:
        if file.endswith('.png'):
            print(os.path.join(root, file))
```

## 核心功能实现

✅ **训练500个epoch**：完整训练循环
✅ **每20个epoch分析**：自动保存模型和计算指标  
✅ **记录train/val loss**：每个分析点记录
✅ **计算AGOP/NFM correlation**：每层每个分析点
✅ **双y轴可视化**：loss和correlation在同一图
✅ **5层分析**：Layer 0-4全覆盖
✅ **三种优化器**：SGD, Adam, Muon
✅ **自动学习率**：每种优化器的最佳学习率
✅ **内存优化**：适配Colab环境限制

立即开始使用 `colab_training_analysis.py` 获得完整的训练分析结果！
