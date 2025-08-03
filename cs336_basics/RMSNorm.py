"""
RMSNorm Layer Summary

This file implements Root Mean Square Layer Normalization (RMSNorm) as a PyTorch nn.Module.
RMSNorm normalizes each input vector by its root mean square (RMS) value, then scales each feature
by a learnable weight. This technique is similar to LayerNorm but does not subtract the mean, making it
more efficient and sometimes more stable for deep learning models like Transformers.

- Input: Tensor of shape (..., d_model)
- Normalization: Each vector in the last dimension is divided by its RMS value (sqrt(mean(x^2)))
- Scaling: Each feature is multiplied by a learnable parameter (gamma) of shape (d_model,)
- Output: Tensor of the same shape as input, normalized and scaled

Detailed comments and examples are provided in the code below to illustrate each step and broadcasting behavior.

Summary:
- Use * for elementwise scaling with broadcasting.
- Use @ for matrix multiplication, which may require transposing for correct dimensions.
"""
# https://zhuanlan.zhihu.com/p/685181632
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Root Mean Square Layer Normalization (RMSNorm).

        Args:
            d_model: int
                Hidden dimension of the model (last dimension of input).
            eps: float
                Epsilon value for numerical stability.
            device: torch.device | None
                Device to store the parameters on.
            dtype: torch.dtype | None
                Data type of the parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # Learnable scaling parameter (gamma), shape: (d_model,)
        # If d_model = 8, it would look like:
        # tensor([1., 1., 1., 1., 1., 1., 1., 1.], requires_grad=True)
        self.weight: nn.Parameter = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Args:
            x: FloatTensor of shape (..., d_model)
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        # Upcast to float32 for numerical stability
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Compute inverse RMS over last dimension
        # 
        # Detailed explanation:
        # 1. x.pow(2): Squares every element in the input tensor x.
        # 2. .mean(-1, keepdim=True): Computes the mean of the squared values along the last dimension (d_model).
        #    This gives the mean squared value for each vector in the last dimension, shape (..., 1).
        # 3. + self.eps: Adds a small epsilon for numerical stability (prevents division by zero).
        # 4. torch.rsqrt(...): Computes the reciprocal square root of the result, i.e., 1 / sqrt(mean(x^2) + eps).
        #    This is the inverse of the root mean square (RMS) of the vector.
        #
        # Example:
        # If x = [2.0, 4.0, 4.0, 4.0], then:
        #   x.pow(2) = [4.0, 16.0, 16.0, 16.0]
        #   mean = (4 + 16 + 16 + 16) / 4 = 13.0
        #   rsqrt = 1 / sqrt(13.0) ≈ 0.277
        # So, rms = 0.277 for this vector.
        #
        # Visualization:
        # For x = [
        #   [2.0, 4.0, 4.0, 4.0],   # First vector
        #   [1.0, 1.0, 1.0, 1.0]    # Second vector
        # ]
        # rms = [
        #   [0.277],   # For first vector
        #   [0.5]      # For second vector
        # ]
        # Shape: (2, 1)
        #
        # This normalization rescales each vector so its RMS becomes 1 (up to the scaling parameter).
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        
        # Scale by learnable weight and restore original dtype
        # 
        # Detailed explanation:
        # self.weight has shape (d_model,) and acts as a learnable scaling factor for each feature in the last dimension.
        # When you multiply self.weight * x, PyTorch broadcasts self.weight across all leading dimensions of x.
        # For example, if x has shape (batch_size, seq_len, d_model) and self.weight is (d_model,),
        # the result is also (batch_size, seq_len, d_model), where each feature in the last dimension is scaled by its corresponding weight.
        #
        # Example:
        # If self.weight = tensor([1.0, 0.5, 2.0, 1.5]) and
        # x = tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), then:
        # self.weight * x =
        # [
        #   [1.0*1.0, 0.5*2.0, 2.0*3.0, 1.5*4.0],   # [1.0, 1.0, 6.0, 6.0]
        #   [1.0*5.0, 0.5*6.0, 2.0*7.0, 1.5*8.0]    # [5.0, 3.0, 14.0, 12.0]
        # ]
        # Result shape: (2, 4)
        #
        # If x is (batch_size, seq_len, d_model), for every position in batch and sequence,
        # you multiply the last dimension by self.weight.
        # .to(in_dtype) converts the result back to the original input dtype.
        return (self.weight * x).to(in_dtype)