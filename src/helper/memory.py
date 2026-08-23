import torch 
import gc 
def cleanup():
    """
    Release CPU/GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def print_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated : {allocated:.2f} GB")
    print(f"Reserved  : {reserved:.2f} GB")