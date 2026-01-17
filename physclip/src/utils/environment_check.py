"""
GPU Diagnostics for PHYSCLIP

Checks PyTorch, CUDA, and GPU availability.
"""

import torch
import sys

print("="*70)
print("GPU DIAGNOSTICS")
print("="*70)

# PyTorch version
print(f"\nPyTorch Version: {torch.__version__}")
print(f"Python Version: {sys.version}")

# CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version (torch): {torch.version.cuda}")

# GPU info
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("\n⚠️  CUDA not available. Install GPU-enabled PyTorch:")
    print("\nFor CUDA 11.8 (Windows/Linux):")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor CUDA 12.1 (Windows/Linux):")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nFor CPU-only (fallback):")
    print("  pip install torch torchvision torchaudio")

# Check NVIDIA driver
print("\n" + "="*70)
print("NVIDIA DRIVER CHECK")
print("="*70)

try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("\n✓ nvidia-smi found:")
        print(result.stdout[:500])  # Print first 500 chars
    else:
        print("\n✗ nvidia-smi not found or failed")
except Exception as e:
    print(f"\n✗ nvidia-smi check failed: {e}")

print("\n" + "="*70)
