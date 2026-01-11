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
    module becomes the input to the next.

    Attributes:
        modules: List of (index, module) tuples in execution order.
    """

    def __init__(self, *args: Module):
        """
        Args:
            *args: Variable number of modules to chain together.
        """
        super().__init__()
        self.modules = []

        for i, module in enumerate(args):
            self.modules.append((i, module))

    def forward(self, x, **args) -> Tensor:
        """Pass input through all modules sequentially.

        Args:
            x: Input tensor to the first module.

        Returns:
            Output tensor from the last module.
        """
        h = x
        for _, module in self.modules:
            h = module(h, **args)
        return h



    