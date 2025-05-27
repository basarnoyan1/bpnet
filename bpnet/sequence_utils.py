"""
PyTorch-compatible sequence utilities to replace concise functions
"""
import numpy as np
import torch
from typing import List, Union

# DNA vocabulary
DNA = 'ACGT'

def pad_sequences(sequences: List[str], maxlen: int = None, align: str = "start", value: str = "N") -> List[str]:
    """
    Pad DNA sequences to the same length.
    
    Args:
        sequences: List of DNA sequence strings
        maxlen: Maximum length. If None, use the length of the longest sequence
        align: 'start' or 'end' - where to align sequences
        value: Character to use for padding
        
    Returns:
        List of padded sequences
    """
    if not sequences:
        return []
    
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded_seqs = []
    for seq in sequences:
        if len(seq) > maxlen:
            # Trim sequence
            if align == "start":
                seq = seq[:maxlen]
            else:  # align == "end"
                seq = seq[-maxlen:]
        elif len(seq) < maxlen:
            # Pad sequence
            padding_needed = maxlen - len(seq)
            padding = value * padding_needed
            if align == "start":
                seq = seq + padding
            else:  # align == "end"
                seq = padding + seq
        
        padded_seqs.append(seq)
    
    return padded_seqs


def encodeDNA(seq_vec: List[str], maxlen: int = None, seq_align: str = "start") -> np.ndarray:
    """
    Convert the DNA sequence into 1-hot-encoding numpy array
    
    This function replicates the behavior of concise.preprocessing.encodeDNA
    
    Args:
        seq_vec: List of DNA sequence strings that can have different lengths
        maxlen: int or None. Should we trim (subset) the resulting sequence. 
                If None don't trim. Note that trims wrt the align parameter.
                It should be smaller than the longest sequence.
        seq_align: str; 'end' or 'start'. To which end should we align sequences?
        
    Returns:
        3D numpy array of shape (len(seq_vec), trim_seq_len(or maximal sequence length if None), 4)
        
    Example:
        >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
        >>> X_seq = encodeDNA(sequence_vec, seq_align="end", maxlen=8)
        >>> X_seq.shape
        (2, 8, 4)
    """
    # Character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(DNA)}
    
    if not seq_vec:
        return np.array([])
    
    # Pad sequences to handle variable lengths
    padded_sequences = pad_sequences(seq_vec, maxlen=maxlen, align=seq_align, value="N")
    
    seq_len = len(padded_sequences[0])
    n_sequences = len(padded_sequences)
    
    # Initialize one-hot array
    one_hot = np.zeros((n_sequences, seq_len, 4), dtype=np.float32)
    
    for seq_idx, seq in enumerate(padded_sequences):
        seq = seq.upper()
        for pos, char in enumerate(seq):
            if char in char_to_idx:
                one_hot[seq_idx, pos, char_to_idx[char]] = 1.0
            # If character is not in ACGT (e.g., N), leave as all zeros
    
    return one_hot


def one_hot2string(one_hot_array: np.ndarray, vocab: str = DNA) -> List[str]:
    """
    Convert one-hot encoded arrays back to DNA sequence strings.
    
    Args:
        one_hot_array: numpy array of shape (n_sequences, seq_len, 4)
        vocab: DNA vocabulary string (default: 'ACGT')
        
    Returns:
        List of DNA sequence strings
    """
    if one_hot_array.size == 0:
        return []
    
    # Get the indices of maximum values along the last axis
    indices = np.argmax(one_hot_array, axis=-1)
    
    sequences = []
    for seq_indices in indices:
        seq = ''.join([vocab[idx] for idx in seq_indices])
        sequences.append(seq)
    
    return sequences


def encodeDNA_torch(sequences: List[str]) -> torch.Tensor:
    """
    PyTorch version of encodeDNA.
    
    Args:
        sequences: List of DNA sequence strings
        
    Returns:
        torch.Tensor of shape (n_sequences, seq_len, 4)
    """
    one_hot_numpy = encodeDNA(sequences)
    return torch.from_numpy(one_hot_numpy)


def one_hot2string_torch(one_hot_tensor: torch.Tensor, vocab: str = DNA) -> List[str]:
    """
    PyTorch version of one_hot2string.
    
    Args:
        one_hot_tensor: torch.Tensor of shape (n_sequences, seq_len, 4)
        vocab: DNA vocabulary string (default: 'ACGT')
        
    Returns:
        List of DNA sequence strings
    """
    one_hot_numpy = one_hot_tensor.cpu().numpy()
    return one_hot2string(one_hot_numpy, vocab)


# For plotting, we'll create a simple sequence logo placeholder
def seqlogo(pwm_array, ax=None, **kwargs):
    """
    Simple sequence logo placeholder.
    For full functionality, consider using logomaker or other packages.
    
    Args:
        pwm_array: Position weight matrix
        ax: matplotlib axis
        **kwargs: additional arguments
    """
    try:
        import matplotlib.pyplot as plt
        import logomaker
        
        # Convert to logomaker format if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create a simple sequence logo using logomaker
        if hasattr(pwm_array, 'shape') and len(pwm_array.shape) == 2:
            # Convert to pandas DataFrame for logomaker
            import pandas as pd
            df = pd.DataFrame(pwm_array, columns=list(DNA))
            logomaker.Logo(df, ax=ax, **kwargs)
        else:
            # Fallback: simple text representation
            ax.text(0.5, 0.5, 'Sequence Logo\n(install logomaker for full functionality)', 
                   ha='center', va='center', transform=ax.transAxes)
            
    except ImportError:
        # Fallback if logomaker not available
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 2))
        
        ax.text(0.5, 0.5, 'Sequence Logo\n(install logomaker for full functionality)', 
               ha='center', va='center', transform=ax.transAxes)


def seqlogo_fig(pwm_array, figsize=(10, 2), **kwargs):
    """
    Create a sequence logo figure.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        seqlogo(pwm_array, ax=ax, **kwargs)
        return fig
    except Exception as e:
        print(f"Error creating sequence logo: {e}")
        return None


# Utility function for getting from module (simple implementation)
def get_from_module(module, name, default=None):
    """
    Simple implementation of get_from_module.
    """
    try:
        return getattr(module, name, default)
    except AttributeError:
        return default


if __name__ == "__main__":
    # Test the functions
    sequences = ["ATCG", "GCTA", "TTAA"]
    
    # Test encoding
    one_hot = encodeDNA(sequences)
    print(f"One-hot shape: {one_hot.shape}")
    print(f"First sequence one-hot:\n{one_hot[0]}")
    
    # Test decoding
    decoded = one_hot2string(one_hot)
    print(f"Decoded sequences: {decoded}")
    
    # Test torch versions
    one_hot_torch = encodeDNA_torch(sequences)
    print(f"Torch tensor shape: {one_hot_torch.shape}")
    
    decoded_torch = one_hot2string_torch(one_hot_torch)
    print(f"Decoded from torch: {decoded_torch}")
    
    print("All tests passed!")
