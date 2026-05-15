import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    stream = torch.cuda.current_stream()
    print(f"Rank {rank}: Stream = {stream}, Device = {stream.device}")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
