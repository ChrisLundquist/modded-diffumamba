"""Build fixed held-out prompt set for the generation harness.

Extracts 500 prefixes of 64 tokens + matching 128-token real continuations
from a region unseen by every model we evaluate:
  - 30M models trained on fineweb_1B.npy[:1e9], val [1e9:1.1e9]
  - 125M trained on fineweb_10B.npy[:9.5e9], val [9.5e9:]
  - fineweb_10B.npy last ~55M tokens are zero padding

Safe region: fineweb_10B.npy[9.5e9 : 9.9e9]. Unseen by all.

Output: nvidia/eval/gen_harness/prompts/fineweb_edu_held.pt
  { 'prefix_ids': LongTensor[N, P],
    'continuation_ids': LongTensor[N, C],
    'source': str, 'seed': int, 'prefix_len': int, 'cont_len': int }
"""

import os
import sys
import torch
import numpy as np

DATA_PATH = '/home/clundquist/muon_data/fineweb_10B.npy'
OUT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'fineweb_edu_held.pt')

N_PROMPTS = 500
PREFIX_LEN = 64
CONT_LEN = 128
SAFE_START = 9_500_000_000
SAFE_END = 9_900_000_000
SEED = 1729


def main():
    print(f'Loading {DATA_PATH} (mmap)...')
    tokens = np.load(DATA_PATH, mmap_mode='r')
    print(f'Total tokens: {len(tokens):,}')

    window = PREFIX_LEN + CONT_LEN
    region_len = SAFE_END - SAFE_START
    assert N_PROMPTS * window < region_len, "Not enough room in safe region"

    rng = np.random.default_rng(SEED)
    offsets = rng.integers(SAFE_START, SAFE_END - window, size=N_PROMPTS * 3)
    offsets = np.unique(offsets)[:N_PROMPTS]
    assert len(offsets) >= N_PROMPTS, f'Not enough unique offsets: {len(offsets)}'
    offsets = np.sort(offsets)

    prefix = np.zeros((N_PROMPTS, PREFIX_LEN), dtype=np.int64)
    cont = np.zeros((N_PROMPTS, CONT_LEN), dtype=np.int64)
    for i, off in enumerate(offsets):
        chunk = tokens[off:off + window].astype(np.int64)
        prefix[i] = chunk[:PREFIX_LEN]
        cont[i] = chunk[PREFIX_LEN:]

    n_zero_prefix = int((prefix == 0).all(axis=1).sum())
    assert n_zero_prefix == 0, f'{n_zero_prefix} all-zero prefixes hit padding'

    out = {
        'prefix_ids': torch.from_numpy(prefix),
        'continuation_ids': torch.from_numpy(cont),
        'source': DATA_PATH,
        'safe_region': (SAFE_START, SAFE_END),
        'seed': SEED,
        'prefix_len': PREFIX_LEN,
        'cont_len': CONT_LEN,
        'offsets': torch.from_numpy(offsets.astype(np.int64)),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(out, OUT_PATH)
    print(f'Saved {N_PROMPTS} prompts to {OUT_PATH}')
    print(f'Prefix shape: {out["prefix_ids"].shape}  Continuation shape: {out["continuation_ids"].shape}')

    # Sanity: decode the first one
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    print('\nSample #0:')
    print(f'  PREFIX: {enc.decode(prefix[0].tolist())!r}')
    print(f'  CONT:   {enc.decode(cont[0].tolist())!r}')


if __name__ == '__main__':
    main()
