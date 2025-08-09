import math
from einops import einsum
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Numerically stable softmax implementation.
    Args:
        x: Input tensor of any shape.
        dim: Dimension to apply softmax over.
    Returns:
        Tensor of same shape as x, with softmax applied along `dim`.
    """
    # Subtract max for numerical stability
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    # Compute attention scores:
    # - einsum computes the dot product between each query and each key for all batch-like dimensions.
    # - Resulting shape: (..., query, key)
    # - Divide by sqrt(d_k) for scaling, which stabilizes gradients and prevents large values.
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    # Apply mask (if provided):
    # - If a mask is given, positions where mask == False are set to -inf.
    # - This ensures that after softmax, those positions get zero probability.
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    # Softmax over the key dimension:
    # - Converts attention scores into probabilities for each query over all keys.
    # - Ensures that for each query, the probabilities sum to 1.
    # Softmax is applied to the key dimension (last dimension) because:
    # - For each query, we want a probability distribution over all keys (i.e., which keys to pay attention to).
    # - This means, for every query position, we compute how much attention it should pay to each key position.
    # - Applying softmax over the key dimension ensures that the attention weights for each query sum to 1,
    #   forming a valid probability distribution.
    # - This lets each query "choose" how much to focus on each key, which is the core idea of attention.
    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    # Combine attention weights and values:
    # - attention_weights: (..., query, key), gives the probability of each key for each query.
    # - V: (..., key, d_v), each key has a corresponding value vector.
    # - einsum computes a weighted sum of the value vectors for each query, using the attention weights.
    # - Output shape: (..., query, d_v)
    # - Note: In attention, the number of keys and values always matches, so V's axis is labeled "key" here.
    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")