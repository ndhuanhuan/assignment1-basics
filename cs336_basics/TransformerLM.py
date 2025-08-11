import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Transformer import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
            self, 
            vocab_size: int,         # The number of unique items in the output vocabulary to be predicted.
            context_length: int,     # The maximum number of tokens to process at once.
            d_model: int,            # The dimensionality of the model embeddings and sublayer outputs.
            num_layers: int,         # The number of Transformer layers to use.
            num_heads: int,          # Number of heads to use in multi-headed attention. d_model must be evenly divisible by num_heads.
            d_ff: int,               # Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta: float        # The angle rate for rotary positional encoding.
            ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)


    def forward(
            self, 
            in_indices: Int[Tensor, "batch_size sequence_length"]
            ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        # in_indices: (batch_size, sequence_length)
        # Output shape: (batch_size, sequence_length, d_model)
        x = self.token_embeddings(in_indices)
        
        # x shape: (batch_size, sequence_length, d_model)
        # Pass through each Transformer block
        # output shape after each block: (batch_size, sequence_length, d_model)
        for layer in self.layers:
            x = layer(x)

        # x shape before is (batch_size, sequence_length, d_model)
        # Apply final layer normalization and linear layer
        # x shape after ln_final: (batch_size, sequence_length, d_model)
        # logits shape: (batch_size, sequence_length, vocab_size)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
