import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on 
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.device = device
        self.dtype = dtype
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Initialize weights using a truncated normal distribution:
        # mean=0, std=sqrt(2/(in_features+out_features)), truncated at [-3*std, 3*std]
        # This helps avoid extreme values and is recommended for Transformer layers.
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        # Output shape: (..., out_features)
        return x @ self.weight.t()