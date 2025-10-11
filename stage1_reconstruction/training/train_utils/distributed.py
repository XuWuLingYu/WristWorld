import os
import time
import torch

def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = os.environ.get("LOCAL_RANK", None)
    distributed_rank = os.environ.get("RANK", None)
    
    # Handle single GPU case where these variables may not be set
    if local_rank is None or distributed_rank is None:
        # Try to infer from CUDA_VISIBLE_DEVICES or default to 0
        if local_rank is None:
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            local_rank = int(cuda_devices.split(",")[0]) if cuda_devices else 0
        else:
            local_rank = int(local_rank)
            
        if distributed_rank is None:
            distributed_rank = 0
        else:
            distributed_rank = int(distributed_rank)
    else:
        local_rank = int(local_rank)
        distributed_rank = int(distributed_rank)
        
    return local_rank, distributed_rank
