from collections import defaultdict
from random import shuffle
import numpy as np
import torch

# compile the dinucleotide edges
def prepare_edges(s_arr):
    """
    Args:
      s_arr: numpy array of shape (L, C) where L is sequence length, C is num characters
             (e.g. C=4 for one-hot DNA)
    """
    edges = defaultdict(list)
    for i in range(s_arr.shape[0] - 1):
        # Store the actual one-hot encoded vector as a tuple to be hashable for dict keys
        edges[tuple(s_arr[i])].append(s_arr[i + 1])
    return edges

def shuffle_edges(edges):
    # for each character (represented by its one-hot encoding tuple),
    # remove the last edge, shuffle, add edge back
    for char_key in edges:
        # Make sure there's more than one edge to shuffle if we pop one
        if len(edges[char_key]) > 1:
            last_edge = edges[char_key][-1]
            # Shuffle all but the last edge
            # Ensure that edges[char_key][:-1] is a list before shuffling
            shufflable_part = list(edges[char_key][:-1])
            shuffle(shufflable_part)
            # Reconstruct the list of edges for this char_key
            edges[char_key] = shufflable_part + [last_edge]
        # If only one edge, no shuffling is possible or needed for that part.
        # If the list is empty or has one element, random.shuffle doesn't do anything.
        # However, the original code implies that if edges[char_key] had one element,
        # it would become empty, then shuffled (no-op), then last_edge (the original single element) re-added.
        # If edges[char_key] was empty, it would remain empty.
        # The current logic handles these cases correctly (shuffle on empty or single-element list is fine).
    return edges

def traverse_edges(s_arr, edges):
    """
    Args:
      s_arr: numpy array of shape (L, C)
      edges: dict of dinucleotide edges
    """
    generated_list = [s_arr[0]] # Start with the first one-hot vector
    edges_queue_pointers = defaultdict(lambda: 0)
    
    for i in range(s_arr.shape[0] - 1):
        last_char_one_hot = generated_list[-1]
        # Use tuple of the one-hot vector for dict key
        last_char_key = tuple(last_char_one_hot)
        
        if not edges[last_char_key]: # Should not happen if s_arr had valid sequence
             raise ValueError(f"Ran out of edges for character {last_char_key} at index {i}. "
                              "This might happen if input sequence is too short or has issues.")

        next_char_one_hot = edges[last_char_key][edges_queue_pointers[last_char_key]]
        generated_list.append(next_char_one_hot)
        edges_queue_pointers[last_char_key] += 1
        
    return np.array(generated_list)


def dinuc_shuffle_numpy(s_one_hot_numpy):
    """
    Performs dinucleotide shuffle on a single one-hot encoded sequence.
    Args:
        s_one_hot_numpy (np.ndarray): A single one-hot encoded sequence, shape (L, C).
                                      L = sequence length, C = number of characters (e.g., 4 for DNA).
    Returns:
        np.ndarray: A dinucleotide shuffled version of the input sequence, same shape.
    """
    if not isinstance(s_one_hot_numpy, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if s_one_hot_numpy.ndim != 2:
        raise ValueError("Input array must be 2D (sequence_length, num_characters).")
    if s_one_hot_numpy.shape[0] < 2 :
        # Dinucleotide shuffle needs at least 2 bases to define a dinucleotide.
        # Return a copy or the original if too short.
        return s_one_hot_numpy.copy()

    prepared_edges = prepare_edges(s_one_hot_numpy)
    shuffled_prepared_edges = shuffle_edges(prepared_edges)
    shuffled_sequence = traverse_edges(s_one_hot_numpy, shuffled_prepared_edges)
    return shuffled_sequence

def dinuc_shuffle_pytorch(batch_one_hot_tensor):
    """
    Performs dinucleotide shuffle on a batch of one-hot encoded sequences.
    Args:
        batch_one_hot_tensor (torch.Tensor): Batch of one-hot encoded sequences,
                                             shape (N, L, C).
    Returns:
        torch.Tensor: Dinucleotide shuffled version of the input batch, same shape.
    """
    if not isinstance(batch_one_hot_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if batch_one_hot_tensor.ndim != 3:
        raise ValueError("Input tensor must be 3D (batch, sequence_length, num_characters).")

    device = batch_one_hot_tensor.device
    batch_one_hot_numpy = batch_one_hot_tensor.cpu().numpy()
    
    shuffled_batch_list = []
    for i in range(batch_one_hot_numpy.shape[0]):
        shuffled_seq = dinuc_shuffle_numpy(batch_one_hot_numpy[i])
        shuffled_batch_list.append(shuffled_seq)
    
    shuffled_batch_numpy = np.array(shuffled_batch_list)
    return torch.from_numpy(shuffled_batch_numpy).to(device)


# The original file had a `dinuc_shuffle(s)` that handled strings.
# For one-hot encoded sequences, the main function is `dinuc_shuffle_numpy` (single)
# and `dinuc_shuffle_pytorch` (batch).

if __name__ == '__main__':
    print("Testing Dinucleotide Shuffle for PyTorch...")

    # Example: A batch of 2 sequences, length 10, 4 channels (one-hot DNA)
    # Sequence 1: A C G T A C G T A C
    # Sequence 2: T G C A T G C A T G
    seq1_np = np.array([
        [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0],
        [0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]
    ])
    seq2_np = np.array([
        [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1],
        [0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]
    ])
    
    batch_np = np.stack([seq1_np, seq2_np])
    batch_tensor = torch.from_numpy(batch_np).float()

    print("Original batch tensor shape:", batch_tensor.shape)

    shuffled_batch_tensor = dinuc_shuffle_pytorch(batch_tensor)
    print("Shuffled batch tensor shape:", shuffled_batch_tensor.shape)

    # Verification (difficult to verify exact shuffle, but can check properties)
    # 1. Shape must be the same
    assert batch_tensor.shape == shuffled_batch_tensor.shape
    # 2. Sum of elements per one-hot vector must be 1
    assert torch.allclose(torch.sum(shuffled_batch_tensor, dim=2), torch.ones_like(shuffled_batch_tensor[:,:,0]))
    # 3. Content should be different (highly likely for non-trivial sequence)
    # This might fail for very short or repetitive sequences where shuffle has no effect
    if batch_tensor.numel() > 0 and batch_tensor.shape[1] > 2: # If not empty and long enough
      assert not torch.allclose(batch_tensor, shuffled_batch_tensor)

    # 4. Check dinucleotide counts (should be preserved)
    def get_dinuc_counts(one_hot_seq_numpy):
        counts = defaultdict(int)
        for i in range(one_hot_seq_numpy.shape[0] - 1):
            dinuc = (tuple(one_hot_seq_numpy[i]), tuple(one_hot_seq_numpy[i+1]))
            counts[dinuc] += 1
        return counts

    for i in range(batch_np.shape[0]):
        original_counts = get_dinuc_counts(batch_np[i])
        shuffled_counts = get_dinuc_counts(shuffled_batch_tensor[i].cpu().numpy())
        assert original_counts == shuffled_counts, f"Dinucleotide counts mismatch for sequence {i}"
        print(f"Dinucleotide counts preserved for sequence {i}.")

    # Test with a short sequence (length 1)
    short_seq_np = np.array([[1,0,0,0]])[np.newaxis, :, :] # Batch of 1, seq len 1
    short_tensor = torch.from_numpy(short_seq_np).float()
    shuffled_short_tensor = dinuc_shuffle_pytorch(short_tensor)
    assert torch.allclose(short_tensor, shuffled_short_tensor), "Short sequence shuffle failed"
    print("Short sequence (len 1) shuffle test passed.")

    # Test with a sequence of length 0 (should probably raise error or handle)
    # Current code dinuc_shuffle_numpy returns copy for seq_len < 2.
    # dinuc_shuffle_pytorch would process this fine.
    empty_len_seq_np = np.empty((1,0,4)) # Batch of 1, seq len 0
    empty_len_tensor = torch.from_numpy(empty_len_seq_np).float()
    shuffled_empty_len_tensor = dinuc_shuffle_pytorch(empty_len_tensor)
    assert shuffled_empty_len_tensor.shape == empty_len_tensor.shape
    print("Empty length sequence shuffle test passed (returns same shape).")


    print("Dinucleotide shuffle tests completed.")
