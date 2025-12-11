import torch
import sys

print("=" * 50)
print("PyTorch CUDA Check")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA built: {torch.cuda.is_built()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\nCUDA is NOT available!")
    print("This means PyTorch was installed without CUDA support (CPU-only version).")
    print("\nTo install PyTorch with CUDA support, run:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nOr for CUDA 11.8:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("=" * 50)

