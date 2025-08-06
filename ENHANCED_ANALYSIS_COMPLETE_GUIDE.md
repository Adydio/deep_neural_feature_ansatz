# 增强分析系统完整指南

## 📊 系统概述

本增强分析系统为您提供了三个层次的分析脚本：

1. **原始分析**：AGOP/NFM correlation + Train/Val Loss  
2. **增强分析**：添加 $G_i$, $H_i$, $M_i$ 矩阵间的余弦相似度分析
3. **完整测试**：验证系统功能的演示脚本

## 🔬 数学理论基础

### 矩阵定义
- **$H_i := \mathbb{E}[h_i h_i^T]$**：表示矩阵（Representation Matrix）
- **$G_i := \mathbb{E}[g_i g_i^T]$，其中 $g_i := -\nabla_{h_i} \ell$**：梯度矩阵（Gradient Matrix）  
- **$M_i := W_i^T W_i$**：权重矩阵（Weight Matrix，与NFM定义相同）

### 相似度计算
所有相似度均使用**余弦相似度**（Cosine Similarity）：
$$\text{similarity} = \frac{\langle \text{vec}(A), \text{vec}(B) \rangle}{||A||_F \cdot ||B||_F}$$

## 🚀 快速开始

### 1. 测试系统功能（推荐第一步）
```bash
# 生成演示数据和可视化
python test_enhanced_analysis.py
```
这将创建：
- `experiments/` 目录（包含模拟的三个模型）
- `enhanced_analysis/demo_enhanced_analysis.png`（演示图表）

### 2. 在真实训练模型上运行增强分析
```bash
# 分析指定的三个模型
python enhanced_analysis_visualization.py --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335
```

## 📁 文件结构

```
deep_neural_feature_ansatz/
├── enhanced_analysis_visualization.py  # 主增强分析脚本
├── test_enhanced_analysis.py          # 功能测试脚本  
├── ENHANCED_ANALYSIS_GUIDE.md         # 详细使用指南
├── experiments/                       # 训练模型目录
│   ├── adam_20250803_173338/
│   ├── muon_20250804_041908/
│   └── sgd_20250803_164335/
└── enhanced_analysis/                 # 分析结果目录
    ├── demo_enhanced_analysis.png
    ├── adam_20250803_173338_enhanced_analysis.png
    ├── adam_20250803_173338_similarities.json
    └── ...
```

## 🎯 核心功能

### 原有指标（已存在）
- ✅ AGOP/NFM Correlation：AGOP与神经特征矩阵的余弦相似度
- ✅ Train/Val Loss：训练和验证损失曲线

### 新增指标（本次添加）
- 🆕 **G vs H**：梯度矩阵与表示矩阵的余弦相似度
- 🆕 **G vs M**：梯度矩阵与权重矩阵的余弦相似度  
- 🆕 **H vs M**：表示矩阵与权重矩阵的余弦相似度

## 📊 可视化输出

每个模型生成一张图表，包含：
- **5个子图**：对应Layer 0-4
- **双y轴设计**：
  - 左轴：训练和验证损失（对数尺度）
  - 右轴：各种余弦相似度（0-1范围）
- **6条曲线**：
  - Train Loss（蓝色●）
  - Val Loss（红色■）  
  - AGOP/NFM Corr（绿色▲）
  - G vs H（紫色♦）
  - G vs M（青色▼）
  - H vs M（橙色★）

## ⚙️ 自定义使用

### 修改模型列表
```bash
# 分析不同的模型
python enhanced_analysis_visualization.py --models model1 model2 model3
```

### 自定义目录
```bash
# 指定自定义路径
python enhanced_analysis_visualization.py \
  --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335 \
  --experiments_dir /path/to/experiments \
  --output_dir my_results
```

### 内存优化版本
如果内存不足，可以修改脚本中的 `max_samples` 参数：
```python
# 在 compute_matrix_similarities 函数中
similarities = compute_matrix_similarities(model_path, layer_indices, max_samples=5000)
```

## ⏱️ 运行时间估计

- **测试脚本**：< 1分钟
- **单个模型增强分析**：30-60分钟（取决于检查点数量）
- **三个模型完整分析**：1.5-3小时

## 🔧 故障排除

### 常见问题
1. **模型目录不存在**
   ```
   Model directory not found: experiments/model_name
   ```
   **解决**：检查模型目录名称和路径

2. **检查点文件缺失**
   ```
   Model not found for epoch X  
   ```
   **解决**：确保 `models/` 子目录中有对应的 `.pt` 文件

3. **内存不足**
   **解决**：修改 `max_samples` 参数限制样本数量

### 依赖检查
```bash
pip install torch matplotlib seaborn numpy
```

## 📈 实际使用工作流

### 标准工作流
1. **训练模型**：
   ```bash
   python colab_training_analysis.py  # 或其他训练脚本
   ```

2. **测试功能**：
   ```bash
   python test_enhanced_analysis.py
   ```

3. **增强分析**：
   ```bash
   python enhanced_analysis_visualization.py --models your_model1 your_model2 your_model3
   ```

4. **结果分析**：
   - 查看 `enhanced_analysis/` 目录中的PNG图表
   - 分析 `*_similarities.json` 文件中的原始数据

## 🎓 理论意义

### 分析价值
- **G vs H**：揭示梯度流与表示学习的关系
- **G vs M**：理解梯度如何塑造权重结构
- **H vs M**：表示空间与参数空间的对应关系

### 预期模式
- 训练初期：相似度较低，各矩阵相对独立
- 训练中期：相似度逐渐上升，开始形成结构对应
- 训练后期：相似度趋于稳定，反映学习到的特征结构

## 📝 引用和扩展

该系统基于神经特征假设（Neural Feature Ansatz）理论，支持：
- 多优化器比较（SGD, Adam, Muon）
- 多层分析（Layer 0-4）
- 完整训练轨迹跟踪

可扩展用于研究：
- 不同网络架构的行为差异
- 优化器对特征学习的影响  
- 训练动力学的理论验证

---

**立即开始**：运行 `python test_enhanced_analysis.py` 体验增强分析系统！
