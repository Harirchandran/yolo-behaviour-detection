import torch
import sys

print("--- GPU Verification ---")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ CUDA is AVAILABLE!")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    try:
        x = torch.rand(5, 3).cuda()
        print(f"✅ Tensor allocation on GPU successful: {x.device}")
    except Exception as e:
        print(f"❌ Tensor allocation FAILED: {e}")
else:
    print("❌ CUDA is NOT available.")
    sys.exit(1)
