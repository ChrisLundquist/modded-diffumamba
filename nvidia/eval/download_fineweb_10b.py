"""Download and tokenize FineWeb-Edu 10B with minimal RAM usage.

Same approach as download_tokens.py but 10x more tokens.
Writes tokens directly to a memory-mapped numpy file on disk.
Peak RAM: ~500MB. Output: ~20GB uint16 file.
"""

import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset

OUTPUT_DIR = '/home/clundquist/muon_data'
TOTAL_TOKENS = 10_000_000_000  # 10B
FINAL_PATH = os.path.join(OUTPUT_DIR, 'fineweb_10B.npy')
PROGRESS_PATH = os.path.join(OUTPUT_DIR, 'progress_10B.txt')
RAW_PATH = os.path.join(OUTPUT_DIR, 'tokens_raw_10B.bin')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(FINAL_PATH):
        size = os.path.getsize(FINAL_PATH)
        print(f'Final file exists: {FINAL_PATH} ({size/1e9:.2f} GB)')
        return

    # Resume support
    offset = 0
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            offset = int(f.read().strip())
        print(f'Resuming from token {offset/1e9:.1f}B')

    if not os.path.exists(RAW_PATH) or offset == 0:
        print(f'Allocating {TOTAL_TOKENS * 2 / 1e9:.1f} GB on disk...')
        fp = np.memmap(RAW_PATH, dtype=np.uint16, mode='w+', shape=(TOTAL_TOKENS,))
        del fp
        offset = 0

    fp = np.memmap(RAW_PATH, dtype=np.uint16, mode='r+', shape=(TOTAL_TOKENS,))
    enc = tiktoken.get_encoding('gpt2')

    # Use the 10BT sample if available, otherwise stream the full dataset
    try:
        ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                          split='train', streaming=True)
        print('Using fineweb-edu sample-10BT')
    except Exception:
        ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
        print('Using fineweb-edu full (streaming)')

    tokens_seen = 0
    skipping = offset > 0

    print(f'Tokenizing {TOTAL_TOKENS/1e9:.0f}B tokens to {RAW_PATH}...')
    cursor = offset

    for i, example in enumerate(ds):
        tokens = enc.encode_ordinary(example['text'])

        if skipping:
            tokens_seen += len(tokens)
            if tokens_seen >= offset:
                skipping = False
                overlap = tokens_seen - offset
                if overlap > 0:
                    tokens = tokens[-overlap:]
                else:
                    continue
            else:
                continue

        n = len(tokens)
        end = min(cursor + n, TOTAL_TOKENS)
        n_write = end - cursor

        if n_write > 0:
            fp[cursor:end] = np.array(tokens[:n_write], dtype=np.uint16)
            cursor = end

        # Save progress every 100M tokens
        if cursor % 100_000_000 < n:
            fp.flush()
            with open(PROGRESS_PATH, 'w') as f:
                f.write(str(cursor))
            print(f'  {cursor/1e9:.1f}B / {TOTAL_TOKENS/1e9:.0f}B tokens '
                  f'({100*cursor/TOTAL_TOKENS:.1f}%)', flush=True)

        if cursor >= TOTAL_TOKENS:
            break

    fp.flush()
    del fp

    # Reopen as read-only memmap for chunked conversion
    print('Writing .npy header (no full-RAM copy)...')
    import struct
    fp_read = np.memmap(RAW_PATH, dtype=np.uint16, mode='r', shape=(TOTAL_TOKENS,))
    header = {'descr': '<u2', 'fortran_order': False, 'shape': (TOTAL_TOKENS,)}
    header_bytes = str(header).encode('latin1')
    pad_len = 64 - ((10 + len(header_bytes)) % 64)
    if pad_len == 64:
        pad_len = 0
    header_bytes = header_bytes + b' ' * pad_len + b'\n'
    with open(FINAL_PATH, 'wb') as out:
        out.write(b'\x93NUMPY\x01\x00')
        out.write(struct.pack('<H', len(header_bytes)))
        out.write(header_bytes)
        CHUNK = 100_000_000  # 100M tokens = 200MB per chunk
        for start in range(0, TOTAL_TOKENS, CHUNK):
            end = min(start + CHUNK, TOTAL_TOKENS)
            out.write(fp_read[start:end].tobytes())
            print(f'  Writing {end/1e9:.1f}B / {TOTAL_TOKENS/1e9:.0f}B', flush=True)
    del fp_read

    os.remove(RAW_PATH)
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
    print(f'Done: {FINAL_PATH} ({os.path.getsize(FINAL_PATH)/1e9:.2f} GB)')


if __name__ == '__main__':
    main()
