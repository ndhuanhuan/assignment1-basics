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
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len) # here d_k = d_model // num_heads
            if use_rope else None
        )
        self.token_positions = token_positions
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k) # d_model, d_model 
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.o_proj = Linear(self.num_heads * self.d_v, self.d_model)

    def forward(self, in_features: torch.Tensor):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        *b, seq_len, d_model = in_features.size()
        assert d_model == self.d_model

        # Project input features into query, key, and value spaces using learned linear layers:
        # - These projections allow the model to learn different representations for Q, K, and V,
        #   which are essential for the attention mechanism.
        # - Each projection uses a separate weight matrix, so the model can "ask" (query), "store" (key),
        #   and "provide" (value) information differently for each token.
        # - Output shapes: (..., seq_len, num_heads * d_k) for q, k, v.
        q = self.q_proj(in_features)  # (..., seq_len, d_model)
        k = self.k_proj(in_features)  # (..., seq_len, d_model)
        v = self.v_proj(in_features)  # (..., seq_len, d_model) => (batch, seq_len, num_heads * d_k)

        # old version:
        # q = rearrange(
        #     q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        # )
        # k = rearrange(
        #     k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        # )
        # v = rearrange(
        #     v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        # )

        # Rearranging Q, K, V tensors:
        # - After projection, Q, K, V have shape (..., seq_len, num_heads * d_k).
        # - We need to split the last dimension into (num_heads, d_k) so each head can process its own chunk.
        # - rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads) transforms the shape to (..., num_heads, seq_len, d_k).
        # - This enables parallel computation of attention for each head independently.
        q, k, v = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (q, k, v)
        ) 

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        # Create a causal (look-ahead) mask to prevent attention to future tokens:
        # - torch.ones(seq_len, seq_len) creates a square matrix of ones.
        # - torch.triu(..., diagonal=1) keeps only the upper triangle above the main diagonal,
        #   marking positions where j > i (future tokens) as True.
        # - .bool() converts the mask to boolean type.
        # - casual_mask[None, None, :, :] adds two leading singleton dimensions so the mask can be broadcasted
        #   across batch and head dimensions.
        # - ~casual_mask inverts the mask so True means "can attend" (current and past tokens),
        #   and False means "cannot attend" (future tokens).
        # - This mask ensures that, for each token position, the model can only attend to itself and previous tokens,
        #   never future ones. This is essential for autoregressive language modeling and next-word prediction.
        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)

        # rearrange the output back to the original shape:
        # - The output from scaled_dot_product_attention has shape (..., num_heads, seq_len, d_k).
        # - We need to combine the num_heads and d_k dimensions back into a single dimension.
        # - rearrange(output, "... h seq_len d_head -> ... seq_len (h d_head)") transforms it to (..., seq_len, num_heads * d_k).
        # - This is the expected output shape for the next linear layer.
        # - Finally, we apply the output projection layer to map it back to the original d_model dimension.
        # - The output projection layer combines the information from all heads into a single representation.
        # - The final output has shape (..., seq_len, d_model), ready for further processing.
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )
        return self.o_proj(output)