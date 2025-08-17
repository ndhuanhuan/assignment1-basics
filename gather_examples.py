import torch
import numpy as np

def log_softmax(x, dim=-1):
    """Same as your implementation"""
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

print("Understanding torch.gather in cross-entropy loss")
print("=" * 50)

# Example 1: Simple case with 1 example, 3 classes
print("\n🔍 Example 1: Single example, 3 classes")
print("-" * 40)

inputs1 = torch.tensor([[2.0, 1.0, 0.1]])  # Shape: (1, 3)
targets1 = torch.tensor([0])  # Shape: (1,) - correct class is index 0

print(f"Inputs (logits): {inputs1}")
print(f"Targets (correct class indices): {targets1}")

# Step by step breakdown
log_softmax_result = log_softmax(inputs1)
print(f"Log-softmax result: {log_softmax_result}")

negative_log_softmax = -log_softmax_result
print(f"Negative log-softmax: {negative_log_softmax}")

# Show what unsqueeze does
targets_unsqueezed = targets1.unsqueeze(-1)
print(f"Targets after unsqueeze(-1): {targets_unsqueezed} (shape: {targets_unsqueezed.shape})")

# Show what gather does
gathered = torch.gather(negative_log_softmax, -1, targets_unsqueezed)
print(f"After torch.gather: {gathered}")
print(f"This selected negative_log_softmax[0, {targets1[0]}] = {negative_log_softmax[0, targets1[0]]}")

final_loss = torch.mean(gathered)
print(f"Final loss (mean): {final_loss}")

print("\n" + "="*50)

# Example 2: Batch of 3 examples
print("\n🔍 Example 2: Batch of 3 examples, 4 classes each")
print("-" * 40)

inputs2 = torch.tensor([
    [3.0, 1.0, 0.5, 2.0],  # Example 0
    [1.0, 4.0, 2.0, 1.5],  # Example 1  
    [0.5, 1.0, 0.2, 3.5]   # Example 2
])  # Shape: (3, 4)

targets2 = torch.tensor([0, 1, 3])  # Correct classes: 0, 1, 3
print(f"Inputs shape: {inputs2.shape}")
print(f"Inputs:\n{inputs2}")
print(f"Targets: {targets2} (correct classes for each example)")

negative_log_softmax2 = -log_softmax(inputs2)
print(f"\nNegative log-softmax:\n{negative_log_softmax2}")

targets2_unsqueezed = targets2.unsqueeze(-1)
print(f"Targets after unsqueeze: {targets2_unsqueezed} (shape: {targets2_unsqueezed.shape})")

gathered2 = torch.gather(negative_log_softmax2, -1, targets2_unsqueezed)
print(f"After torch.gather: {gathered2}")

print("\nWhat gather selected:")
for i, target_class in enumerate(targets2):
    print(f"  Example {i}: selected class {target_class} → loss = {negative_log_softmax2[i, target_class]:.4f}")

final_loss2 = torch.mean(gathered2)
print(f"\nFinal average loss: {final_loss2}")

print("\n" + "="*50)

# Example 3: Show what happens with different confidence levels
print("\n🔍 Example 3: Different confidence levels")
print("-" * 40)

# High confidence (correct prediction)
high_conf = torch.tensor([[10.0, 1.0, 1.0]])  # Very confident about class 0
# Low confidence (less confident prediction)  
low_conf = torch.tensor([[2.0, 1.8, 1.9]])   # Less confident about class 0
# Wrong prediction
wrong_pred = torch.tensor([[1.0, 1.0, 10.0]]) # Confident about wrong class (2), but target is 0

targets3 = torch.tensor([0])  # All should predict class 0

examples = [
    ("High confidence", high_conf),
    ("Low confidence", low_conf), 
    ("Wrong prediction", wrong_pred)
]

for name, inputs in examples:
    neg_log_sm = -log_softmax(inputs)
    loss = torch.gather(neg_log_sm, -1, targets3.unsqueeze(-1))
    
    # Also show the actual probabilities for intuition
    probs = torch.softmax(inputs, dim=-1)
    
    print(f"{name}:")
    print(f"  Logits: {inputs[0].tolist()}")
    print(f"  Probabilities: {probs[0].tolist()}")
    print(f"  P(class_0) = {probs[0, 0]:.4f}")
    print(f"  Loss = -log({probs[0, 0]:.4f}) = {loss.item():.4f}")
    print()
