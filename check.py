import torch

# 1. 检查MPS (Apple Silicon GPU) 是否可用
if torch.backends.mps.is_available():
    # 如果可用，则将设备设置为 "mps"
    device = torch.device("mps")
    print("✅ PyTorch将使用Apple Silicon GPU (MPS)进行计算。")
else:
    # 如果不可用，则回退到CPU
    device = torch.device("cpu")
    print("❌ Apple Silicon GPU (MPS) 不可用，PyTorch将使用CPU。")

# 2. 将您的模型和数据移动到所选设备
# 例如，创建一个张量并将其发送到GPU
try:
    x = torch.rand(3, 5).to(device)
    print(f"\n成功在设备 '{device}' 上创建了一个张量:")
    print(x)
    
    # 同样，在训练时，您需要将模型和数据都移动到该设备
    # model.to(device)
    # data = data.to(device)
    
except Exception as e:
    print(f"\n在设备 '{device}' 上创建张量时出错: {e}")