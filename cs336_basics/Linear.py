import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import einsum

# https://zhuanlan.zhihu.com/p/720840508 => good explanation of 为什么 self.weight 的权重矩阵 shape is (out_features, in_features)
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Custom Linear layer without bias, similar to nn.Linear but no bias term.

        Args:
            in_features: int
                Size of each input sample (last dimension of input).
            out_features: int
                Size of each output sample (last dimension of output).
            device: torch.device | None
                Device to store the parameters on.
            dtype: torch.dtype | None
                Data type of the parameters.
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.device = device
        self.dtype = dtype

        # Weight matrix of shape (out_features, in_features).
        # Each output feature is a linear combination of all input features.
        # Jaxtyping annotation documents the expected shape and dtype.
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=True
        )

        # Initialize weights using a truncated normal distribution:
        # mean=0, std=sqrt(2/(in_features+out_features)), truncated at [-3*std, 3*std].
        # This initialization is recommended for Transformer layers to avoid extreme values.
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Applies the linear transformation to the input tensor.

        Args:
            x: Float[Tensor, "... d_in"]
                Input tensor of shape (..., in_features).

        Returns:
            Float[Tensor, "... d_out"]
                Output tensor of shape (..., out_features).
        """
        # einsum performs batched matrix multiplication over the last dimension.
        # For each batch and leading dimension, multiplies x by the weight matrix.
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        # Note: einsum automatically aligns dimensions based on the provided labels.
        # The einsum string "... d_in, d_out d_in -> ... d_out" contracts over d_in,
        # so you do NOT need to manually transpose the weight matrix.
        # This is equivalent to x @ self.weight.t() in standard