import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

# https://www.bilibili.com/video/BV1jmquYpEhn/?spm_id_from=333.337.search-card.all.click&vd_source=48e4bed01dd155111c1b309b768743f6
# https://github.com/HongYan-L/cs336_ass1/blob/main/adapters.py#L1063
# https://github.com/yaozile123/assignment1-basics/blob/main/cs336_basics/layers.py#L81
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        SwiGLU Feed-Forward Network: SiLU activation + GLU gating.
        dff is set to 8/3 * d_model, rounded up to nearest multiple of 64 by default,
        but can be overridden for testing.
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
            device (torch.device, optional): Device to place the parameters on.
            dtype (torch.dtype, optional): Data type of the parameters.
        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        assert d_model % 64 == 0, "d_model must be divisible by 64"
        if d_ff is None:
            self.d_ff = d_model * 8 // 3
        # self.dff = self.adjust_dff(d_model) if d_ff < self.adjust_dff(d_model) else d_ff
        # Three linear layers to match the test adapter's expected weights
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # GLU: (SiLU(W1 x) * sigmoid(W3 x)) then project back with W2
        # Shape analysis for this line:
        # output = self.w2(self.silu(self.w1(x)) * self.w3(x))
        #
        # Step-by-step shapes:
        # 1. x: (..., d_model)
        # 2. self.w1(x): (..., d_ff)         # Linear projection from d_model to d_ff
        # 3. self.w3(x): (..., d_ff)         # Another linear projection from d_model to d_ff
        # 4. self.silu(self.w1(x)): (..., d_ff)  # SiLU activation applied elementwise
        # 5. self.silu(self.w1(x)) * self.w3(x): (..., d_ff)  # Elementwise gating
        # 6. self.w2(...): (..., d_model)    # Final linear projection back to d_model
        # 7. output: (..., d_model)
        #
        # Example:
        # If x has shape (batch_size=2, seq_len=5, d_model=768) and d_ff=2048:
        #   self.w1(x): (2, 5, 2048)
        #   self.w3(x): (2, 5, 2048)
        #   self.silu(self.w1(x)): (2, 5, 2048)
        #   Elementwise product: (2, 5, 2048)
        #   self.w2(...): (2, 5, 768)
        #   output: (2, 5, 768)
        output = self.w2(self.silu(self.w1(x)) * self.w3(x))
        return output
    
    def adjust_dff(self, d_model: int) -> int:
        return (int(4 * d_model) + 63) // 64 * 64

    def silu(self, in_features: Float[Tensor, "... d_model"]) -> Float[Tensor, "... "]:
        """Given a tensor of inputs, return the output of applying SiLU
        to each element.

        Based on the formula: SiLU(x) = x * sigmoid(x).

        Args:
            in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

        Returns:
            Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
            SiLU to each element.
        """
        return in_features * torch.sigmoid(in_features)