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
            vocab_size: int, 
            context_length: int, 
            d_model: int, 
            num_layers: int, 
            num_heads: int, 
            d_ff: int,
            rope_theta: float):
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
        x=self.token_embeddings(in_indices) #bs,sq_len,d_model

        for layer in self.layers:
            x=layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) #bs seq_len vs

        return logits