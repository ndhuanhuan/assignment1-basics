"""
Embedding Layer Detailed Explanation:

This module implements a custom embedding layer for a Transformer model. The core idea is to map integer token IDs
to dense embedding vectors using an embedding matrix.

How it works:
- self.weight is a tensor of shape (num_embeddings, embedding_dim), where each row is the embedding vector for a token ID.
- token_ids is an integer tensor (e.g., shape (batch_size, sequence_length)), where each value is a token ID in [0, num_embeddings-1].
- The line `self.weight[token_ids, :]` uses advanced indexing to select the embedding vector for each token ID.
  - For each integer in token_ids, it returns the corresponding row from self.weight.
  - The output tensor has shape (*token_ids.shape, embedding_dim), e.g., (batch_size, sequence_length, embedding_dim).
- This operation efficiently looks up the embedding vector for every token in the batch and sequence, forming the input to the Transformer.

Example:
If self.weight is (10000, 768) and token_ids is (2, 5):
    self.weight[token_ids, :] returns (2, 5, 768), where each [i, j, :] is the embedding for token_ids[i, j].

Detailed Example:
Suppose token_ids is:
    [[4, 7, 2, 9, 1],
     [3, 8, 6, 0, 5]]
and self.weight is (10000, 768).
Then self.weight[token_ids, :] returns a tensor of shape (2, 5, 768), where:
    - The first row contains the embeddings for token IDs [4, 7, 2, 9, 1]
    - The second row contains the embeddings for token IDs [3, 8, 6, 0, 5]
    - Each [i, j, :] is the embedding vector for token_ids[i, j]

This is the standard way embedding layers work in deep learning frameworks.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int

# As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs
# into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from
# torch.nn.Module (so you should not use nn.Embedding). The forward method should select the embedding
# vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
# torch.LongTensor of token IDs with shape (batch_size, sequence_length).
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Args:
            num_embeddings: int, size of the vocabulary
            embedding_dim: int, dimension of the embedding vectors (d_model)
            device: torch.device | None, device to store the parameters on
            dtype: torch.dtype | None, data type of the parameters
        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            requires_grad=True
        )
        # Initialize weights using a truncated normal distribution:
        # mean=0, std=1, truncated at [-3, 3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        # token_ids shape: (batch_size, sequence_length)
        # Output shape: (batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids, :]