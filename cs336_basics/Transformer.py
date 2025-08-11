import torch
import torch.nn as nn

from cs336_basics.MultiheadSelfAttention import MultiheadSelfAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.SwiGLU import SwiGLU

# check PDF page 14 figure 2
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        """
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention. 
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache.
            theta (float): RoPE parameter.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.rms_norm2 = RMSNorm(d_model=d_model)

        self.attn = MultiheadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            use_rope=True, 
            max_seq_len=max_seq_len,
            theta=theta
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """   
        x_attn = self.attn(self.rms_norm1(x))
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.rms_norm2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn

        return ffn_sublayer_output
