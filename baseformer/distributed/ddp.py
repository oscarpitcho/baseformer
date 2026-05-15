import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, model: nn.Module, world_size: int, rank: int):
        super().__init__()
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.params = []
        self.handles = []

        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.reduce_gradients_hook)


    def forward(self, *args, **kwargs):
        pass

    def reduce_gradients_hook(self, tensor: torch.Tensor): 
        handle = dist.all_reduce(tensor, async_op=True)
        self.handles.append(handle)







