# Colab执行代码示例

## 重要说明：AGOP分析使用全部训练样本

从2025年8月3日起，脚本已修改为使用**全部训练样本**进行AGOP分析，确保理论正确性：
- ✅ **准确性**：AGOP分析现在使用与训练完全相同的数据集
- ⚠️ **内存消耗**：内存使用将会增加，但确保分析的准确性
- 📊 **理论一致**：符合AGOP理论要求的数据一致性

## 方式1：直接在Colab中运行完整分析（推荐）

```python
# 第一步：安装依赖包
!pip install matplotlib seaborn

# 第二步：运行完整的训练分析（包含所有三个优化器）
# 注意：现在使用全部训练样本进行AGOP分析
!python colab_training_analysis.py

# 第三步：查看生成的结果
import os
print("=== 生成的实验结果 ===")
for root, dirs, files in os.walk("experiments"):
    level = root.replace("experiments", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

# 第四步：显示生成的图片
from IPython.display import Image, display
import glob

print("\n=== 生成的可视化图表 ===")
png_files = glob.glob("experiments/*/plots/*.png")
for png_file in png_files:
    print(f"\n{png_file}:")
    display(Image(png_file))
```

## 方式2：分别运行每个优化器

```python
# 安装依赖
!pip install matplotlib seaborn

# 运行SGD
!python advanced_training_analysis.py --optimizer sgd --lr 0.1 --epochs 500 --val_interval 20

# 运行Adam  
!python advanced_training_analysis.py --optimizer adam --lr 0.001 --epochs 500 --val_interval 20

# 运行Muon
!python advanced_training_analysis.py --optimizer muon --lr 0.01 --epochs 500 --val_interval 20
```

## 方式3：快速测试（验证功能）

```python
# 安装依赖
!pip install matplotlib seaborn

# 运行快速测试（只需几分钟）
!python test_functionality.py

# 查看测试结果
from IPython.display import Image, display
display(Image("test_experiment/test_plot.png"))
```

## 方式4：自定义参数运行

```python
# 安装依赖
!pip install matplotlib seaborn

# 运行较短的测试版本（100个epoch，每10个epoch分析一次）
!python advanced_training_analysis.py --optimizer sgd --lr 0.1 --epochs 100 --val_interval 10 --max_samples 3000

# 内存受限环境（更少样本）
!python advanced_training_analysis.py --optimizer adam --lr 0.001 --epochs 50 --val_interval 5 --max_samples 1000
```

## 结果文件说明

运行完成后，您将得到：

### 目录结构
```
experiments/
├── sgd_20250803_123456/
│   ├── models/          # 保存的模型检查点
│   ├── plots/           # 可视化图表
│   └── results.json     # 原始数据
├── adam_20250803_123457/
└── muon_20250803_123458/
```

### 关键文件
- `training_analysis.png` - 5层综合分析图
- `layer_X_analysis.png` - 单层详细图  
- `results.json` - 所有训练数据和相关性数据

## 预期运行时间

- **快速测试**: 5-10分钟
- **单个优化器完整训练**: 2-3小时  
- **全部三个优化器**: 6-9小时

## 注意事项

1. **内存管理**: Colab版本已优化内存使用
2. **长时间运行**: 建议分批运行或在运行期间保持Colab活跃
3. **结果保存**: 重要结果会自动保存到文件
4. **中断恢复**: 如果中断，可以从最后保存的检查点继续

立即开始使用推荐的**方式1**获得完整的训练分析结果！
