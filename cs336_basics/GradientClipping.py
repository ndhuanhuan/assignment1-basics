
from typing import Iterable
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6  # for numerical stability
    # ---
    # Gradient clipping explanation:
    # 1. Collect all gradients from parameters (ignore if grad is None).
    # 2. Flatten and concatenate all gradients into a single vector.
    # 3. Compute the L2 norm of this vector (overall gradient norm).
    #    L2 norm formula: sqrt(g1**2 + g2**2 + ... + gn**2)
    # 4. If the norm exceeds max_l2_norm, compute a scaling factor:
    #       scale = max_l2_norm / (grad_norm + eps)
    #    (eps prevents division by zero)
    # 5. Multiply each gradient by this scale in-place, so the total norm is just under max_l2_norm.
    # 6. If the norm is already small enough, gradients are unchanged.
    # ---
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # Flatten and concatenate all gradients into a single vector
    grad_vector = torch.cat([g.view(-1) for g in grads])
    grad_norm = torch.norm(grad_vector, p=2)
    if grad_norm > max_l2_norm:
        scale = max_l2_norm / (grad_norm + eps)
        for g in grads:
            g.mul_(scale)

    # alternative implementation using torch.sqrt and torch.sum
    # grads = []
    # for pt in parameters:
    #     if pt.grad is not None:
    #         grads.append(pt.grad)
    # grads_l2norm = 0.0
    # for gd in grads:
    #     grads_l2norm += (gd ** 2).sum()
    # grads_l2norm = torch.sqrt(grads_l2norm)
    # if grads_l2norm >= max_l2_norm:
    #     ft = max_l2_norm / (grads_l2norm + 1e-6)
    #     for gd in grads:
    #         gd *= ft