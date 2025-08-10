from einops import rearrange
import torch
import torch.nn as nn

from cs336_basics.Attention import scaled_dot_product_attention
from cs336_basics.Linear import Linear
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            use_rope (bool): Whether to use RoPE        
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            if use_rope else None
        )
        self.token_positions = token_positions
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
    
    def forward(self, in_features: torch.Tensor):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        seq_len = in_features.shape[-2]
        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        qkv = in_features @ qkv_proj.T
        q, k, v = qkv.chunk(3, -1)

        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )
        return self.o_proj(output)