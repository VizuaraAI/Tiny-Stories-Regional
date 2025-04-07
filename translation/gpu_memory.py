import torch

def print_gpu_memory_usage():
    """
    Prints the current GPU memory usage for all available GPUs.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n------------ GPU -------------")
    print(f"Number of GPUs available: {num_gpus}")

    for gpu_id in range(num_gpus):
        # Get memory usage for each GPU
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        reserved_memory = torch.cuda.memory_reserved(gpu_id)

        # Convert to GB for easier readability
        allocated_memory_GB = allocated_memory / (1024 ** 3)
        reserved_memory_GB = reserved_memory / (1024 ** 3)

        # Print memory usage for the GPU
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  Allocated Memory: {allocated_memory_GB:.2f} GB")
        print(f"  Reserved Memory: {reserved_memory_GB:.2f} GB")
    print(f"------------------------------\n")