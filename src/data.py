"""
data.py — Data preparation and loading for DiffuMamba3

Supports:
  1. Pre-tokenized binary shards (modded-nanogpt format)
  2. HuggingFace datasets → tokenize once → cache as binary shard
  3. Simple random-chunk DataLoader

Binary shard format (from modded-nanogpt):
  Header: 256 int32s (magic=20240520, version=1, num_tokens)
  Body: uint16 tokens
"""

import os
import numpy as np
import torch
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data"
MAGIC = 20240520


def write_shard(path: str, tokens: np.ndarray):
    """Write tokens to a binary shard file."""
    tokens = tokens.astype(np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    print(f"  Saved {len(tokens)/1e6:.3f}M tokens to {path}")


def read_shard(path: str) -> torch.Tensor:
    """Read tokens from a binary shard file."""
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC, f"Bad magic in {path}"
    n = int(header[2])
    tokens = np.memmap(path, dtype=np.uint16, mode="r",
                       offset=256 * 4, shape=(n,))
    return torch.from_numpy(tokens.astype(np.int64))


def prepare_tiny_shakespeare() -> tuple[Path, Path]:
    """Download tiny_shakespeare once, tokenize with GPT-2, cache as shards."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_path = CACHE_DIR / "tiny_shakespeare_train.bin"
    val_path = CACHE_DIR / "tiny_shakespeare_val.bin"

    if train_path.exists() and val_path.exists():
        return train_path, val_path

    print("  Downloading and tokenizing tiny_shakespeare (one-time)...")
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("Trelis/tiny-shakespeare")
    text_key = "Text" if "Text" in ds["train"].column_names else "text"

    for split, path in [("train", train_path), ("test", val_path)]:
        all_tokens = []
        for example in ds[split]:
            all_tokens.extend(enc.encode_ordinary(example[text_key]))
        write_shard(str(path), np.array(all_tokens))

    return train_path, val_path


def load_tokens(path: str = None, split: str = "train",
                max_tokens: int = None) -> torch.Tensor:
    """Load tokens from a shard file, or prepare tiny_shakespeare as fallback."""
    if path and Path(path).exists():
        tokens = read_shard(path)
        print(f"  Loaded {len(tokens)/1e6:.3f}M tokens from {path}")
        return tokens[:max_tokens] if max_tokens else tokens

    # Fallback: tiny_shakespeare cached locally
    train_path, val_path = prepare_tiny_shakespeare()
    path = str(train_path if split == "train" else val_path)
    tokens = read_shard(path)
    print(f"  Loaded {len(tokens)/1e6:.3f}M tokens from {path}")
    return tokens[:max_tokens] if max_tokens else tokens


class DataLoader:
    """Simple random-chunk data loader from a flat token array."""
    def __init__(self, tokens: torch.Tensor, seq_len: int, batch_size: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n = len(tokens)
        assert self.n > seq_len, f"Need more tokens ({self.n}) than seq_len ({seq_len})"

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        starts = torch.randint(0, self.n - self.seq_len, (self.batch_size,))
        return torch.stack([self.tokens[s:s + self.seq_len] for s in starts])


if __name__ == "__main__":
    print("=== Preparing data ===")
    train_path, val_path = prepare_tiny_shakespeare()
    train_tokens = read_shard(str(train_path))
    val_tokens = read_shard(str(val_path))
    print(f"Train: {len(train_tokens)/1e6:.3f}M tokens")
    print(f"Val:   {len(val_tokens)/1e6:.3f}M tokens")

    loader = DataLoader(train_tokens, seq_len=256, batch_size=4)
    batch = next(loader)
    print(f"Batch shape: {batch.shape}, range: [{batch.min()}, {batch.max()}]")
