"""
Tokenize raw parquet files into .bin training shards (modded-nanogpt format).

Alternative to get_data.py — use this if you want to control tokenization
or use a different dataset. Downloads nothing; operates on local parquet files.

Streams through data: never holds more than one shard (~200MB) in memory.
Output format: 256-int32 header (magic=20240520, version=1, n_tokens) + uint16 body.

Usage:
    python data/tokenize.py                                # default: data/fineweb-edu/ → data/fineweb10B/
    python data/tokenize.py --src data/my_parquets/ --dst data/my_tokens/ --tokens 10B
"""
import os
import argparse
import glob
import numpy as np
import tiktoken

SHARD_SIZE = 100_000_000  # 100M tokens per shard (~200MB as uint16)
MAGIC = 20240520

def write_shard(path, tokens_np):
    """Write .bin shard: 256-int32 header + uint16 body."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(tokens_np)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())


def tokenize_parquets(src_dir, dst_dir, target_tokens):
    """Stream-tokenize parquet files into .bin shards."""
    import pyarrow.parquet as pq

    os.makedirs(dst_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    parquet_files = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
    if not parquet_files:
        print(f"No .parquet files in {src_dir}/")
        print("Download parquets first, e.g.:")
        print(f"  huggingface-cli download HuggingFaceFW/fineweb-edu "
              f'--include "sample/10BT/00*" --local-dir {src_dir}')
        return

    print(f"Found {len(parquet_files)} parquet files in {src_dir}/")
    print(f"Target: {target_tokens / 1e9:.1f}B tokens -> {dst_dir}/")

    shard = np.zeros(SHARD_SIZE, dtype=np.uint16)
    shard_pos = 0
    shard_idx = 0
    total_tokens = 0
    done = False

    for pf in parquet_files:
        if done:
            break
        print(f"  {os.path.basename(pf)}...", flush=True)
        table = pq.read_table(pf, columns=["text"])

        for batch_start in range(0, len(table), 5000):
            if done:
                break
            texts = table.slice(batch_start, 5000).column("text").to_pylist()

            for text in texts:
                toks = np.array(enc.encode_ordinary(text) + [eot], dtype=np.uint16)

                # Copy into shard, splitting across boundary if needed
                src_pos = 0
                remaining = len(toks)
                while remaining > 0:
                    space = SHARD_SIZE - shard_pos
                    take = min(remaining, space)
                    shard[shard_pos:shard_pos + take] = toks[src_pos:src_pos + take]
                    shard_pos += take
                    src_pos += take
                    remaining -= take

                    if shard_pos >= SHARD_SIZE:
                        split = "val" if shard_idx == 0 else "train"
                        fname = os.path.join(dst_dir, f"fineweb_{split}_{shard_idx:06d}.bin")
                        write_shard(fname, shard)
                        total_tokens += SHARD_SIZE
                        print(f"    shard {shard_idx} ({split}): "
                              f"{total_tokens / 1e6:.0f}M tokens", flush=True)
                        shard_idx += 1
                        shard_pos = 0

                if total_tokens + shard_pos >= target_tokens:
                    done = True
                    break

    # Final partial shard
    if shard_pos > 0:
        split = "val" if shard_idx == 0 else "train"
        fname = os.path.join(dst_dir, f"fineweb_{split}_{shard_idx:06d}.bin")
        write_shard(fname, shard[:shard_pos])
        total_tokens += shard_pos
        print(f"    shard {shard_idx} ({split}): "
              f"{total_tokens / 1e6:.0f}M tokens (final)", flush=True)

    print(f"\nDone: {total_tokens / 1e9:.2f}B tokens in {shard_idx + 1} shards")


def parse_tokens(s):
    """Parse '1B', '100M', '10_000_000' into int."""
    s = s.strip().replace("_", "")
    if s[-1] in "Bb":
        return int(float(s[:-1]) * 1e9)
    if s[-1] in "Mm":
        return int(float(s[:-1]) * 1e6)
    return int(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tokenize parquet files into .bin shards")
    p.add_argument("--src", default="data/fineweb-edu",
                   help="Directory containing .parquet files")
    p.add_argument("--dst", default="data/fineweb10B",
                   help="Output directory for .bin shards")
    p.add_argument("--tokens", default="1B",
                   help="Target token count (e.g. 1B, 500M)")
    args = p.parse_args()
    tokenize_parquets(args.src, args.dst, parse_tokens(args.tokens))
