# 100 Epochs 训练脚本使用指南

## 脚本特点

新创建的 `training_100epochs.py` 脚本具有以下特点：

### 训练配置
- **总epochs**: 100
- **分析计划**: 
  - Epochs 0-20: 每个epoch都保存模型并分析
  - Epochs 40, 60, 80, 100: 里程碑式保存和分析
- **AGOP分析**: 使用全部训练样本确保准确性

### 保存的模型
- **详细阶段**: epochs 0, 1, 2, 3, ..., 20 (21个模型)
- **里程碑阶段**: epochs 40, 60, 80, 100 (4个模型)
- **总计**: 25个保存的模型每个优化器

## 运行命令

### 1. 运行所有优化器 (推荐)
```bash
python3 training_100epochs.py --optimizer all
```

### 2. 运行单个优化器
```bash
# SGD
python3 training_100epochs.py --optimizer sgd

# Adam  
python3 training_100epochs.py --optimizer adam

# Muon
python3 training_100epochs.py --optimizer muon
```

### 3. 自定义学习率的单个优化器
```bash
# SGD with custom LR
python3 training_100epochs.py --optimizer sgd --lr 0.05

# Adam with custom LR
python3 training_100epochs.py --optimizer adam --lr 0.002

# Muon with custom LR
python3 training_100epochs.py --optimizer muon --lr 0.005
```

## 默认学习率

脚本会自动使用以下默认学习率：
- **SGD**: 0.1
- **Adam**: 0.001  
- **Muon**: 0.01

## 输出结果

### 目录结构
```
experiments/
├── sgd_100epochs_20250803_HHMMSS/
│   ├── models/
│   │   ├── model_epoch_0:svhn:width:1024:depth:5:act:relu:nn.pth
│   │   ├── model_epoch_1:svhn:width:1024:depth:5:act:relu:nn.pth
│   │   ├── ...
│   │   ├── model_epoch_20:svhn:width:1024:depth:5:act:relu:nn.pth
│   │   ├── model_epoch_40:svhn:width:1024:depth:5:act:relu:nn.pth
│   │   ├── model_epoch_60:svhn:width:1024:depth:5:act:relu:nn.pth
│   │   ├── model_epoch_80:svhn:width:1024:depth:5:act:relu:nn.pth
│   │   └── model_epoch_100:svhn:width:1024:depth:5:act:relu:nn.pth
│   ├── plots/
│   │   ├── training_analysis_100epochs.png
│   │   └── training_analysis_100epochs.pdf
│   └── results.json
├── adam_100epochs_20250803_HHMMSS/
│   └── ... (同样结构)
└── muon_100epochs_20250803_HHMMSS/
    └── ... (同样结构)
```

### 可视化特点
- **5层网络**: 每层一个子图 (2x3布局)
- **双Y轴**: 左侧Loss，右侧Correlation (0-1统一刻度)
- **标记**: 在epoch 20处添加虚线，标示从详细分析到里程碑分析的转换
- **数据点**: 在分析的epoch处显示correlation数据点

## 预期运行时间

### 单个优化器 (100 epochs)
- **详细分析期** (epochs 0-20): ~30-45分钟
- **里程碑分析期** (epochs 21-100): ~15-20分钟
- **总计**: ~45-65分钟每个优化器

### 全部三个优化器
- **总时间**: ~2.5-3.5小时

## 内存使用

- **AGOP分析**: 使用全部训练样本 (~73k samples)
- **建议**: 确保有足够内存 (8GB+ RAM推荐)
- **GPU**: 如果可用会自动使用CUDA加速

## 快速验证

如果想快速测试脚本功能，可以先运行单个优化器：
```bash
python3 training_100epochs.py --optimizer sgd
```

## 结果分析

完成后查看：
1. **可视化图表**: `experiments/*/plots/training_analysis_100epochs.png`
2. **数值结果**: `experiments/*/results.json`
3. **模型文件**: `experiments/*/models/model_epoch_*.pth`

每个图表将显示：
- Train/Val Loss的演化过程
- 每层AGOP/NFM correlation的变化
- 详细期和里程碑期的清晰分界
