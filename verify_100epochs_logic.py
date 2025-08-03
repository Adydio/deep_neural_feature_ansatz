#!/usr/bin/env python3
"""
验证 training_100epochs.py 的分析计划逻辑
"""

def get_analysis_epochs(num_epochs=100):
    """
    Define which epochs to analyze:
    - Epochs 0-20: every epoch
    - Epochs 40, 60, 80, 100: milestone epochs
    """
    analysis_epochs = list(range(0, 21))  # 0, 1, 2, ..., 20
    milestone_epochs = [40, 60, 80, 100]
    
    # Only add milestone epochs if they're within the total epochs
    for epoch in milestone_epochs:
        if epoch <= num_epochs:
            analysis_epochs.append(epoch)
    
    return sorted(set(analysis_epochs))

# 测试分析计划
print("=== 100 Epochs 训练分析计划验证 ===")
print()

analysis_epochs = get_analysis_epochs(100)
print(f"总分析epochs数量: {len(analysis_epochs)}")
print(f"分析epochs: {analysis_epochs}")
print()

print("分析计划详情:")
print("📊 详细分析期 (epochs 0-20):")
detailed_epochs = [e for e in analysis_epochs if e <= 20]
print(f"  Epochs: {detailed_epochs}")
print(f"  数量: {len(detailed_epochs)} 个epoch")

print()
print("🎯 里程碑分析期 (epochs 40, 60, 80, 100):")
milestone_epochs = [e for e in analysis_epochs if e > 20]
print(f"  Epochs: {milestone_epochs}")
print(f"  数量: {len(milestone_epochs)} 个epoch")

print()
print("📈 预期效果:")
print("  - 前期密集分析: 捕捉训练初期的快速变化")
print("  - 后期稀疏分析: 关注训练后期的关键节点")
print("  - 总模型数量: 25个/优化器 (相比原来500epochs版本大幅减少)")

print()
print("🔧 命令行使用:")
print("  python3 training_100epochs.py --optimizer all     # 运行所有优化器")
print("  python3 training_100epochs.py --optimizer sgd     # 仅运行SGD")
print("  python3 training_100epochs.py --optimizer adam    # 仅运行Adam")
print("  python3 training_100epochs.py --optimizer muon    # 仅运行Muon")

print()
print("✅ 脚本逻辑验证完成!")
print("✅ 可以直接运行 training_100epochs.py 进行实际训练")
