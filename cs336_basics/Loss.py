import torch
import numpy as np

def log_softmax(x, dim=-1):
    """
    Compute log-softmax in a numerically stable way.
    
    Mathematical formula: log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    
    The numerical stability trick:
    1. Subtract max value to prevent overflow when computing exp()
    2. This doesn't change the result since: log(exp(x-c) / sum(exp(x-c))) = log(exp(x) / sum(exp(x)))
    """
    # Find max value for numerical stability (prevents overflow in exp)
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    # Subtract max from all values
    x = x - x_max
    # Compute log-softmax: x_i - log(sum(exp(x_j)))
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy loss.
    
    Cross-entropy loss measures how "surprised" the model is by the correct answer:
    - High confidence in correct class → low loss
    - Low confidence in correct class → high loss  
    - Completely wrong prediction → very high loss
    
    Mathematical formula: CE = -log(P(correct_class))
    Where P(correct_class) is the softmax probability of the true class.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:

    Math equation:
        For a single example:
            CE = -log(P_{y}(x))
        For a batch of N examples:
            CE = -(1/N) * sum_{i=1}^N log(P_{y_i}(x_i))
        Where:
            - P_{y}(x) is the softmax probability of the true class y for input x
            - y_i is the true class for example i
            - x_i is the logits for example i

        In code, this means:
            1. For each example i, compute loss_i = -log(P_{y_i}(x_i))
            2. Final loss = (loss_1 + loss_2 + ... + loss_N) / N
            Or, using mean:
                final_loss = torch.mean(loss_per_example)

        Float[Tensor, ""]: The average cross-entropy loss across examples.
        
    Example:
        inputs = [[2.0, 1.0, 0.1]], targets = [0]
        After softmax: probabilities ≈ [0.66, 0.24, 0.10]
        Cross-entropy loss ≈ -log(0.66) ≈ 0.41

    Concrete torch.gather example:
        Suppose negative_log_softmax_logits = [[0.1, 0.5, 0.2, 0.8], [0.9, 0.3, 0.1, 0.6]]
        and targets = [2, 1]
        After targets.unsqueeze(-1): [[2], [1]]
        torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)) selects:
            Row 0: index 2 → 0.2
            Row 1: index 1 → 0.3
        Result: [[0.2], [0.3]]
        This means for each example, we pick the loss for the correct class.
    """
    # Get -log(P(class_i)) for each class
    # This is -log_softmax, which gives us the negative log probabilities
    negative_log_softmax_logits = -log_softmax(inputs)
    
    # Extract the loss for the correct class for each example
    # targets.unsqueeze(-1) converts shape (batch_size,) to (batch_size, 1)
    # torch.gather selects negative_log_softmax_logits[i, targets[i]] for each i
    # This gives us -log(P(correct_class)) for each example
    loss_per_example = torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1))
    
    # Return average loss across all examples in the batch
    return torch.mean(loss_per_example)