import torch
import os

# Çevre değişkenlerini kontrol et
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

# GPU durumunu kontrol et
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Her GPU'da tensor oluşturmayı dene
for i in range(torch.cuda.device_count()):
    try:
        x = torch.ones(2, 3).to(f"cuda:{i}")
        print(f"Successfully created tensor on GPU {i}")
    except Exception as e:
        print(f"Failed to create tensor on GPU {i}: {e}")

# CUDA sürüm bilgilerini yazdır
print(f"PyTorch CUDA Version: {torch.version.cuda}")
