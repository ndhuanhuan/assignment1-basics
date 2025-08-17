import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Randomly sample starting indices for each sequence in the batch
    # Each index is chosen so that there is enough room for a full context_length
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))

    # For each starting index, extract a sequence of length context_length
    # These are the input sequences for the language model
    x = torch.stack([
        torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
        for i in starting_idxs
    ])  # Shape: (batch_size, context_length)

    # For each starting index, extract the next sequence (shifted by 1)
    # These are the target labels for the language model (next token prediction)
    y = torch.stack([
        torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
        for i in starting_idxs
    ])  # Shape: (batch_size, context_length)

    # Move tensors to the specified device
    # If using CUDA, use pin_memory and non_blocking for efficient transfer
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    # Return input sequences and their corresponding labels
    return x, y