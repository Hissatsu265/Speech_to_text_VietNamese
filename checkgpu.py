import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.cuda.empty_cache()

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    name = torch.cuda.get_device_name(gpu_id)
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3

    print(f"GPU dang dung: {gpu_id} - {name}")
    print(f"VRAM da allocate: {allocated:.2f} GB")
    print(f"VRAM da reserve: {reserved:.2f} GB")
    print(f"Tong VRAM: {total:.2f} GB")
else:
    print("Khong co GPU CUDA.")
