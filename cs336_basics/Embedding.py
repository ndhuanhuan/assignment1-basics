import torch
import torch.nn as nn

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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids shape: (batch_size, sequence_length)
        # Output shape: (batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids]