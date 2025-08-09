import torch
from torch import LongTensor, nn
from torch import Tensor
from jaxtyping import Float, Int

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
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        # cache cos/sin; no gradients needed → persistent=False
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
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        # Split even / odd channels.
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        # Apply the 2-D rotation to each pair.
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd  = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave.
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out