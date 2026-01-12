"""
Sequential container for chaining modules in order.
"""

from torch import Tensor
from torch.nn import Module, Parameter
from jaxtyping import Float


class Sequential(Module):
    """
    Sequential container that chains modules in order.

    Passes input through each module in sequence, where the output of one
    module becomes the input to the next. Modules are registered as submodules
    so their parameters are included in state_dict().
    """

    def __init__(self, *args: Module):
        """
        Args:
            *args: Variable number of modules to chain together.
        """
        super().__init__()

        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def forward(self, x, *args, **kwargs) -> Tensor:
        """Pass input through all modules sequentially.

        Args:
            x: Input tensor to the first module.
            *args: Additional positional arguments passed to each module.
            **kwargs: Additional keyword arguments passed to each module.

        Returns:
            Output tensor from the last module.
        """
        h = x
        for module in self.children():
            h = module(h, *args, **kwargs)
        return h



    