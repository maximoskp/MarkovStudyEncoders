import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def generate_P_h_given_h(diagonal_probability: float, h_size: int, seed: int = None):
    """
    Generate first-order Markov transition matrix for h-space.

    Args:
        diagonal_probability: probability mass for self-transition (on diagonal).
        h_size: number of h-space tokens.
        seed: random seed for reproducibility.

    Returns:
        h_token_ids: list of token IDs for h-space.
        T_h: (h_size x h_size) numpy array transition matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    h_token_ids = list(range(h_size))

    # Start with diagonal-probability on the diagonal
    T_h = np.full((h_size, h_size), (1 - diagonal_probability) / (h_size - 1))
    np.fill_diagonal(T_h, diagonal_probability)

    # Normalize rows (safety)
    T_h = T_h / T_h.sum(axis=1, keepdims=True)

    return h_token_ids, T_h
# end generate_P_h_given_h

def generate_P_m_given_h(num_peaks: int, peaks_probability: float,
                         m_size: int, h_token_ids: list, seed: int = None):
    """
    Generate conditional distribution P(m|h).

    Args:
        num_peaks: number of high-probability peaks per h.
        peaks_probability: total probability mass assigned to peaks.
        m_size: number of m-space tokens.
        h_token_ids: list of h-space token IDs.
        seed: random seed.

    Returns:
        P_m_h: (len(h_token_ids) x m_size) numpy array conditional probabilities.
        m_token_ids: list of m-space token IDs (start after h_token_ids).
    """
    if seed is not None:
        np.random.seed(seed)

    m_token_ids = list(range(m_size))
    H = len(h_token_ids)

    P_m_h = np.zeros((H, m_size))

    for h in range(H):
        peaks = np.random.choice(m_size, size=num_peaks, replace=False)
        # Assign peaks evenly
        P_m_h[h, peaks] = peaks_probability / num_peaks
        # Assign noise uniformly to the rest
        remaining = m_size - num_peaks
        if remaining > 0:
            P_m_h[h, [i for i in range(m_size) if i not in peaks]] = (1 - peaks_probability) / remaining
        # Normalize
        P_m_h[h] /= P_m_h[h].sum()

    return P_m_h, m_token_ids
# end generate_P_m_given_h

def generate_sample(h_token_ids, m_token_ids, T_h, P_m_h, seq_len: int = 64):
    """
    Generate one (m_sequence, h_sequence) pair.

    Args:
        h_token_ids: list of h-space token IDs.
        m_token_ids: list of m-space token IDs.
        T_h: h_size x h_size transition matrix.
        P_m_h: h_size x m_size conditional probability matrix.
        seq_len: length of sequence.

    Returns:
        dict with fields:
            "h_sequence": list of h tokens
            "m_sequence": list of m tokens
    """
    h_size = len(h_token_ids)
    m_size = len(m_token_ids)

    # Sample H sequence
    h_seq = []
    current_h = np.random.choice(h_token_ids)
    h_seq.append(current_h)
    for _ in range(seq_len - 1):
        current_h = np.random.choice(h_token_ids, p=T_h[current_h])
        h_seq.append(current_h)

    # Sample M sequence from H sequence
    m_seq = []
    for h in h_seq:
        h_idx = h_token_ids.index(h)
        m_idx = np.random.choice(m_size, p=P_m_h[h_idx])
        m_seq.append(m_token_ids[m_idx])

    return {"h_sequence": h_seq, "m_sequence": m_seq}
# end generate_sample

def generate_dataset_pickle(file_path, num_samples, h_token_ids, m_token_ids, T_h, P_m_h, seq_len: int = 64):
    """
    Generate dataset and save as pickle.

    Args:
        file_path: where to save .pickle file.
        num_samples: number of samples to generate.
        h_token_ids, m_token_ids, T_h, P_m_h: dataset specification.
        seq_len: sequence length.
    """
    dataset = []
    for _ in tqdm( range(num_samples) ):
        dataset.append(generate_sample(h_token_ids, m_token_ids, T_h, P_m_h, seq_len=seq_len))

    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {num_samples} samples to {file_path}")
# end generate_dataset_pickle

class HM_Dataset(Dataset):
    """
    PyTorch Dataset for H-M sequences.
    Loads from a pickle file created with generate_dataset_pickle.
    """

    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Return as torch tensors
        return (
            torch.tensor(sample["h_sequence"], dtype=torch.long),
            torch.tensor(sample["m_sequence"], dtype=torch.long)
        )
# end HM_Dataset