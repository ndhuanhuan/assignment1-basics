import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Numerically stable softmax implementation.
    Args:
        x: Input tensor of any shape.
        dim: Dimension to apply softmax over.
    Returns:
        Tensor of same shape as x, with softmax applied along `dim`.
    """
    # Subtract max for numerical stability
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum