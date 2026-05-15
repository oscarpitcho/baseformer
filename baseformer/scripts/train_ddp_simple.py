import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from baseformer.main import load_model



if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus
    mp.spawn(fn=train_ddp_simple, args=(world_size, ), nprocs=world_size, join=True)