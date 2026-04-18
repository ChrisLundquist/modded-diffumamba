"""Data pipeline: FineWeb-Edu tokenized and cached."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def tokenize_and_cache(num_tokens=1_100_000_000, split='train', cache_name='fineweb_1B'):
    """Download, tokenize, and cache FineWeb-Edu tokens to disk.

    Returns path to the .npy file.
    """
    cache_path = os.path.join(CACHE_DIR, f'{cache_name}.npy')
    if os.path.exists(cache_path):
        print(f'Cache exists: {cache_path}')
        return cache_path

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'Tokenizing {num_tokens/1e9:.1f}B tokens from FineWeb-Edu...')

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding('gpt2')
    ds = load_dataset('HuggingFaceFW/fineweb-edu', split=split, streaming=True)

    tokens = []
    total = 0
    for example in ds:
        t = enc.encode_ordinary(example['text'])
        tokens.extend(t)
        total += len(t)
        if total >= num_tokens:
            break
        if total % 10_000_000 < len(t):
            print(f'  {total/1e6:.0f}M tokens...')

    tokens = np.array(tokens[:num_tokens], dtype=np.uint16)
    np.save(cache_path, tokens)
    print(f'Saved {len(tokens)/1e6:.0f}M tokens to {cache_path}')
    return cache_path


def load_cached_tokens(cache_name='fineweb_1B'):
    """Load pre-tokenized tokens from cache."""
    cache_path = os.path.join(CACHE_DIR, f'{cache_name}.npy')
    return np.load(cache_path).astype(np.int64)


class TokenDataset(Dataset):
    """Simple dataset that returns fixed-length token sequences."""
    def __init__(self, tokens, seq_len=1024):
        self.tokens = tokens
        self.seq_len = seq_len
        # Number of complete sequences (with +1 for target)
        self.n_sequences = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(self.tokens[start:start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + 1:start + self.seq_len + 1].astype(np.int64))
        return x, y


def make_dataloader(tokens, seq_len=1024, batch_size=128, shuffle=True,
                    num_workers=2, seed=0):
    """Create a DataLoader from a token array."""
    ds = TokenDataset(tokens, seq_len)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True, generator=g)


def make_repeated_dataloader(tokens, unique_tokens, seq_len=1024,
                             batch_size=128, num_workers=2, seed=0):
    """Create a DataLoader that cycles through a subset of tokens.

    Args:
        tokens: Full token array
        unique_tokens: Number of unique tokens to use (rest is ignored)
        seq_len: Sequence length
        batch_size: Batch size
    """
    subset = tokens[:unique_tokens]
    return make_dataloader(subset, seq_len, batch_size, shuffle=True,
                           num_workers=num_workers, seed=seed)
