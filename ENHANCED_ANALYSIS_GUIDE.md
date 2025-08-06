# Enhanced Analysis Visualization 使用指南

## 概述

`enhanced_analysis_visualization.py` 脚本用于为已训练好的模型生成增强的分析图表，包含：

1. **原有指标**：
   - AGOP/NFM correlation（AGOP与神经特征矩阵的余弦相似度）
   - Train/Val Loss（训练和验证损失）

2. **新增指标**：
   - $G_i$ vs $H_i$：梯度矩阵与表示矩阵的余弦相似度
   - $G_i$ vs $M_i$：梯度矩阵与权重矩阵的余弦相似度  
   - $H_i$ vs $M_i$：表示矩阵与权重矩阵的余弦相似度

## 数学定义

- **$H_i := \mathbb{E}[h_i h_i^T]$**：表示矩阵（Representation Matrix）
- **$G_i := \mathbb{E}[g_i g_i^T]$，其中 $g_i := -\nabla_{h_i} \ell$**：梯度矩阵（Gradient Matrix）
- **$M_i := W_i^T W_i$**：权重矩阵（Weight Matrix，与NFM定义相同）

## 基本用法

### 运行指定的三个模型
```bash
python enhanced_analysis_visualization.py --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335
```

### 自定义输出目录
```bash
python enhanced_analysis_visualization.py \
  --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335 \
  --output_dir my_enhanced_analysis
```

### 指定实验目录
```bash
python enhanced_analysis_visualization.py \
  --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335 \
  --experiments_dir /path/to/experiments \
  --output_dir enhanced_results
```

## 参数说明

- `--models`：必需参数，指定要分析的模型目录名称列表
- `--experiments_dir`：可选，实验目录路径（默认：experiments）
- `--output_dir`：可选，输出目录路径（默认：enhanced_analysis）

## 输出文件

运行完成后，将在输出目录中生成：

### 每个模型的输出文件
- `{model_name}_enhanced_analysis.png`：增强分析图表
- `{model_name}_similarities.json`：矩阵相似度原始数据

### 图表内容
- **5个子图**：对应Layer 0-4的分析
- **双y轴**：
  - 左轴：训练和验证损失
  - 右轴：各种余弦相似度（0-1范围）
- **6条线**：
  - Train Loss（蓝色）
  - Val Loss（红色）
  - AGOP/NFM Corr（绿色）
  - G vs H（紫色）
  - G vs M（青色）
  - H vs M（橙色）

## 示例输出结构

```
enhanced_analysis/
├── adam_20250803_173338_enhanced_analysis.png
├── adam_20250803_173338_similarities.json
├── muon_20250804_041908_enhanced_analysis.png
├── muon_20250804_041908_similarities.json
├── sgd_20250803_164335_enhanced_analysis.png
└── sgd_20250803_164335_similarities.json
```

## 运行时间估计

- **单个模型**：约30-60分钟（取决于保存的检查点数量）
- **三个模型**：约1.5-3小时

## 注意事项

1. **内存使用**：脚本使用全部训练样本计算矩阵，确保有足够内存
2. **依赖关系**：需要matplotlib和seaborn用于可视化
3. **模型路径**：确保指定的模型目录存在且包含必要的检查点文件
4. **数据一致性**：使用与训练相同的数据集和随机种子

## 故障排除

### 模型目录不存在
```
Model directory not found: experiments/model_name
```
**解决方案**：检查模型目录名称和实验目录路径

### 检查点文件缺失
```
Model not found for epoch X
```
**解决方案**：检查`models/`子目录中是否存在对应的检查点文件

### 内存不足
**解决方案**：可以修改脚本中的`max_samples`参数限制样本数量

## 高级用法

### 修改要分析的模型
编辑脚本中的模型列表：
```python
# 在命令行中直接指定
python enhanced_analysis_visualization.py --models new_model1 new_model2 new_model3
```

### 自定义层数
修改脚本中的`layer_indices`：
```python
layer_indices = [0, 1, 2, 3, 4]  # 可根据需要调整
```

立即开始使用该脚本分析您的训练模型！
