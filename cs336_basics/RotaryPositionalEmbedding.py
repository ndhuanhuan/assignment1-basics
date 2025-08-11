import torch
from torch import LongTensor, nn
from torch import Tensor
from jaxtyping import Float, Int

# Rotary Positional Embedding (RoPE) Detailed Notes
#
# RoPE encodes positional information by rotating pairs of features in the query/key vectors using position-dependent sine and cosine functions.
# This rotation is mathematically equivalent to multiplying the vector by a rotation matrix whose angle depends on the token’s position (m) and a frequency (f).
#
# freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
# - freq is the frequency for each feature pair.
# - torch.arange(0, d_k, 2, device=device).float() generates [0, 2, 4, ..., d_k-2] (length d_k // 2).
# - Each value is divided by d_k and exponentiated by theta.
# - Result: freq is a vector of length d_k // 2, containing the inverse frequencies for each feature pair.
#
# positions = torch.arange(max_seq_len, device=device).float()
# - positions is a tensor of all possible position indices: [0, 1, 2, ..., max_seq_len-1].
# - This represents the position index (m) in the original RoPE formula.
#
# freqs = torch.outer(positions, freq)
# - torch.outer(a, b) computes the outer product of two 1D tensors a and b.
# - The result is a 2D tensor where each element is the product of an element from a and an element from b.
# - Here, freqs computes m * f for every position and every frequency.
# - Shape: (max_seq_len, d_k // 2)
#
# self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
# self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
# - For each position and feature pair, stores the cosine and sine of the rotation angle.
# - These are cached as buffers (not parameters, but saved with the model).
# - Shape: (max_seq_len, d_k // 2)
# - Purpose: Fast lookup of cos/sin values for any position during forward pass.
#
# In the forward pass:
# - For each token position, you look up the corresponding cos/sin values.
# - You split the input vector into even and odd channels (feature pairs).
# - You rotate each pair using the cached cos/sin values:
#   out_even = x_even * cos_pos - x_odd * sin_pos
#   out_odd  = x_even * sin_pos + x_odd * cos_pos
# - You re-interleave the results to reconstruct the rotated vector.
#
# torch.arange:
# - torch.arange(start, end, step) creates a 1D tensor with evenly spaced values from start (inclusive) to end (exclusive), with a given step.
# - Example: torch.arange(0, 10, 2) -> tensor([0, 2, 4, 6, 8])
#
# torch.outer:
# - torch.outer(a, b) computes the outer product of two 1D tensors a and b.
# - Example:
#   a = torch.tensor([1, 2, 3])
#   b = torch.tensor([4, 5])
#   torch.outer(a, b) -> tensor([[ 4,  5], [ 8, 10], [12, 15]])
#
# In short:
# - m = position index (positions)
# - f = frequency (freq)
# - sin(m * f) = torch.sin(freqs) where freqs = torch.outer(positions, freq)
# - This code precomputes and caches the cos/sin tables needed for efficient rotary positional embedding, enabling fast and flexible position encoding for attention mechanisms in Transformers.

# https://zhuanlan.zhihu.com/p/1932785030888952719
# https://www.bilibili.com/video/BV1F1421B7iv/?spm_id_from=333.337.search-card.all.click&vd_source=48e4bed01dd155111c1b309b768743f6
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")

        self.d_k = d_k
        # ---- pre-compute inverse frequencies ----
        # freq[k] = 1 / theta ** (2k / d_k)          (k = 0,1,…,d_k/2-1)
        freq= 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # shape: (max_seq_len, d_k // 2)
        # Note: This shape comes from combining every possible position (max_seq_len rows)
        # with every possible frequency/feature pair (d_k // 2 columns).
        # Each entry represents the rotation angle for a specific position and feature pair,
        # which is exactly what RoPE needs to encode positional information for all tokens and all
        # frequencies.
        positions = torch.arange(max_seq_len, device=device).float()
        # Outer product: (max_seq_len, d_k//2
        freqs = torch.einsum('i,j->ij', positions, freq)  # (max_seq_len, d_k//2)

        # cache cos/sin; no gradients needed → persistent=False
        # cos_cached and sin_cached are 2D tables of shape (max_seq_len, d_k // 2)
        # - Each row represents a token position in the sequence (0 to max_seq_len-1)
        # - Each column represents a frequency/feature pair (0 to d_k//2-1)
        # - Each entry is the cosine or sine of the rotation angle for that position and frequency:
        #     cos_cached[pos, freq] = cos(positions[pos] * freq[freq])
        #     sin_cached[pos, freq] = sin(positions[pos] * freq[freq])
        #
        # Example:
        # If max_seq_len = 4 and d_k = 6 (so d_k // 2 = 3):
        # positions = [0, 1, 2, 3]
        # freq = [f0, f1, f2]
        # cos_cached =
        # [
        #   [cos(0*f0), cos(0*f1), cos(0*f2)],  # position 0
        #   [cos(1*f0), cos(1*f1), cos(1*f2)],  # position 1
        #   [cos(2*f0), cos(2*f1), cos(2*f2)],  # position 2
        #   [cos(3*f0), cos(3*f1), cos(3*f2)],  # position 3
        # ]
        # sin_cached =
        # [
        #   [sin(0*f0), sin(0*f1), sin(0*f2)],  # position 0
        #   [sin(1*f0), sin(1*f1), sin(1*f2)],  # position 1
        #   [sin(2*f0), sin(2*f1), sin(2*f2)],  # position 2
        #   [sin(3*f0), sin(3*f1), sin(3*f2)],  # position 3
        # ]
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
        
    def forward(
        self,
        x: torch.Tensor,              # (..., seq_len, d_k)
        token_positions: torch.Tensor # (..., seq_len)
    ) -> torch.Tensor:
        """
        Apply RoPE to `x`.  Works with any batch shape prefix.
        """
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")

        # Gather the cached tables for the required positions.
        # Resulting shape: (..., seq_len, d_k // 2)
        cos_pos = self.cos_cached[token_positions] # since token_positions length is seq_len, so shape is (..., seq_len, d_k // 2)
        sin_pos = self.sin_cached[token_positions]

        # Split even / odd channels.
        # x[..., ::2] selects all even-indexed features in the last dimension (feature pairs).
        # x[..., 1::2] selects all odd-indexed features in the last dimension.
        # This is necessary because RoPE rotates each pair of features (even, odd) together using the cached cos/sin values.
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        # Apply the 2-D rotation to each pair.
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd  = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave.
        # This is a computational trick for efficiency:
        # - After rotating each feature pair (even, odd) using RoPE, we need to reconstruct the original feature order.
        # - out_even contains all rotated even-indexed features, out_odd contains all rotated odd-indexed features.
        # - By assigning out[..., ::2] = out_even and out[..., 1::2] = out_odd,
        #   we efficiently restore the full tensor with positional information encoded, without explicit loops.
        # - This leverages fast, vectorized tensor operations for hardware acceleration.
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out

    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply rotary positional embedding to the input tensor.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
    #         token_positions (torch.Tensor): Tensor of shape (..., seq_len) specifying token positions.

    #     Returns:
    #         torch.Tensor: Tensor of the same shape as x with RoPE
    #     """
    #     *batch_dims,seq_len,d_k = x.shape



    #     assert d_k == self.d_k, " d_k must match"
        
    #     #make last dim into pairs
    #     x_=x.view(*batch_dims, seq_len,d_k//2,2)
    #     x1,x2=x_[...,0],x_[...,1]

    #     cos=self.cos_cached[token_positions]
    #     sin=self.sin_cached[token_positions]

    #     #rotation
    #     x_rotated_0=x1 * cos - x2 *sin 
    #     x_rotated_1=x1*sin +  x2 * cos

    #     x_rotated_stacked=torch.stack([x_rotated_0,x_rotated_1],dim=-1)
    #     x_rotated_stacked=x_rotated_stacked.view(*batch_dims,seq_len,d_k)

    #     return x_rotated_stacked