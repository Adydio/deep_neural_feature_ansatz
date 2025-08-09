# Training 100 Epochs - 多数据集支持

`training_100epochs.py` 现已支持多个数据集，可以通过命令行参数指定要使用的数据集。

## 支持的数据集

- **svhn**: SVHN数据集（默认），10个类别，32x32图像
- **cifar**: CIFAR-10数据集，10个类别，32x32图像
- **cifar_mnist**: CIFAR-10和MNIST合并数据集，10个类别，32x32图像
- **celeba**: CelebA数据集（男性/女性分类），2个类别，96x96图像
- **stl_star**: STL-10星形检测数据集，2个类别，96x96图像

## 使用示例

### 1. 使用默认设置（SVHN，所有优化器）
```bash
python training_100epochs.py
```

### 2. 指定数据集运行所有优化器
```bash
# CIFAR-10数据集
python training_100epochs.py --dataset cifar

# CelebA数据集
python training_100epochs.py --dataset celeba

# STL-Star数据集
python training_100epochs.py --dataset stl_star
```

### 3. 指定数据集和优化器
```bash
# 在CIFAR-10上使用SGD
python training_100epochs.py --dataset cifar --optimizer sgd --lr 0.1

# 在CelebA上使用Adam
python training_100epochs.py --dataset celeba --optimizer adam --lr 0.001

# 在STL-Star上使用Muon
python training_100epochs.py --dataset stl_star --optimizer muon --lr 0.01
```

### 4. 自定义学习率
```bash
# 在CIFAR-10上使用SGD，自定义学习率
python training_100epochs.py --dataset cifar --optimizer sgd --lr 0.05

# 在CelebA上使用Adam，自定义学习率
python training_100epochs.py --dataset celeba --optimizer adam --lr 0.0001
```

## 输出组织

实验结果将保存在以下目录结构中：
```
experiments/
├── {dataset}_{optimizer}_100epochs_{timestamp}/
│   ├── models/
│   │   ├── init_model.pth
│   │   ├── model_epoch_0_{dataset}_width_1024_depth_5_act_relu_nn.pth
│   │   ├── model_epoch_1_{dataset}_width_1024_depth_5_act_relu_nn.pth
│   │   └── ...
│   ├── plots/
│   │   ├── training_analysis_100epochs.png
│   │   └── training_analysis_100epochs.pdf
│   └── results.json
```

## 分析时间表

- **0-20周期**: 每个周期都进行分析（详细分析）
- **40, 60, 80, 100周期**: 里程碑分析

## 重要说明

1. **AGOP分析**: 使用所有训练样本进行精确的相关性计算
2. **内存使用**: 较大的数据集（如CelebA, STL-Star）可能需要更多内存
3. **训练时间**: 100个周期的训练需要较长时间，建议在GPU上运行
4. **数据下载**: 首次使用数据集时会自动下载到 `~/datasets/` 目录

## 常见用法

```bash
# 快速测试：在SVHN上只运行SGD
python training_100epochs.py --optimizer sgd

# 比较优化器：在CIFAR-10上运行所有优化器
python training_100epochs.py --dataset cifar

# 针对特定任务：在CelebA上进行二分类
python training_100epochs.py --dataset celeba --optimizer adam
```
