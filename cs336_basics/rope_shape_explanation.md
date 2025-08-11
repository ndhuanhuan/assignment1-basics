# Rotary Positional Embedding (RoPE) Shape Mismatch Example

This document explains, with a concrete example, why **shape mismatches** can occur in RoPE code and how to fix them by properly shaping `token_positions`.

---

## 1. **Background**

RoPE rotates pairs of features in a tensor using precomputed sine and cosine tables, indexed by token positions.  
The rotation is performed elementwise between the input tensor and the positional encoding tensors.

---

## 2. **Typical Shapes in RoPE**

Suppose you have:
- **Batch size**: 2
- **Sequence length**: 4
- **Feature dimension**: 6 (so `d_k = 6`)
- **RoPE operates on pairs, so `d_k // 2 = 3`**

### **Input tensor `x`**
Shape: `(batch, seq_len, d_k)`  
Example: `(2, 4, 6)`

### **After splitting even/odd features**
```python
x_even = x[..., ::2]  # shape: (2, 4, 3)
x_odd  = x[..., 1::2] # shape: (2, 4, 3)
```
- Now, both `x_even` and `x_odd` have shape `(2, 4, 3)`.

---

## 3. **Cosine/Sine Table Lookup**

### **Precomputed tables**
- `cos_cached` and `sin_cached` have shape `(max_seq_len, d_k // 2)`  
  Example: `(4, 3)`

### **Token positions**
Suppose you want to apply RoPE for every token in every batch.  
You need to index into the cached tables using `token_positions`.

#### **Case 1: Incorrect shape**
If `token_positions` is just `[0, 1, 2, 3]` (shape `(4,)`),  
then `cos_pos = cos_cached[token_positions]` gives shape `(4, 3)`.

But your input is `(2, 4, 3)`!  
**Shape mismatch:** You can't broadcast `(2, 4, 3)` and `(4, 3)` for elementwise multiplication.

**Why can't you multiply (2, 4, 3) and (4, 3)?**  
PyTorch tries to match dimensions from the right.  
The first dimension (batch) is 2 in `x_even`, but missing in `cos_pos`.  
So, PyTorch can't broadcast these together for elementwise multiplication.

#### **Case 2: Correct shape**
If you expand `token_positions` for batch:
```python
token_positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])  # shape (2, 4)
cos_pos = cos_cached[token_positions]  # shape (2, 4, 3)
```
Now, `cos_pos` matches the shape of `x_even` and `x_odd` (`(2, 4, 3)`), so elementwise operations work.

---

## 4. **Why Singleton Dimensions Matter**

Sometimes, you need to add singleton dimensions for broadcasting, especially if you have more batch-like dimensions (e.g., heads).  
For example:
```python
token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
```
This ensures that when you index into `cos_cached`, the resulting tensor can be broadcasted to match all batch/head dimensions.

---

## 5. **Concrete Example**

Suppose you have:
- `x.shape = (batch=2, seq_len=4, d_k=6)`
- `token_positions.shape = (2, 4)`
- `cos_cached.shape = (4, 3)`

```python
cos_pos = cos_cached[token_positions]  # shape: (2, 4, 3)
x_even = x[..., ::2]                   # shape: (2, 4, 3)
out_even = x_even * cos_pos            # works!
```

If `token_positions` was just `(4,)`,  
`cos_pos = cos_cached[token_positions]` would be `(4, 3)`,  
and you would get a shape mismatch error when trying to multiply with `(2, 4, 3)`.

---

## 6. **Summary Table**

| Tensor      | Shape (incorrect) | Shape (correct) | Notes                  |
|-------------|-------------------|-----------------|------------------------|
| x_even      | (2, 4, 3)         | (2, 4, 3)       | Input after split      |
| cos_pos     | (4, 3)            | (2, 4, 3)       | Needs batch dimension  |

---

## 7. **Key Takeaway**

**Always ensure that `token_positions` is shaped to match all batch-like dimensions of your input tensor before using it to index into cached positional encoding tables.**  
This avoids shape mismatches and ensures correct elementwise operations in RoPE.

---