"""
Download pre-tokenized FineWeb-10B (GPT-2 tokenizer) from HuggingFace Hub.

Each .bin shard contains ~100M tokens as uint16 with a 256-int32 header.
This is the same data used by modded-nanogpt (kjj0/fineweb10B-gpt2).

Usage:
    python data/get_data.py          # 1B tokens (10 train shards + 1 val)
    python data/get_data.py 103      # full 10.3B tokens
"""
import os
import sys
from huggingface_hub import hf_hub_download

REPO = "kjj0/fineweb10B-gpt2"

def get(fname, local_dir):
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id=REPO, filename=fname,
                        repo_type="dataset", local_dir=local_dir)

local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fineweb10B")

# Val shard (always)
get("fineweb_val_000000.bin", local_dir)

# Train shards: default 10 (1B tokens)
num_chunks = 10
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])
print(f"Downloading {num_chunks} train shards ({num_chunks * 100}M tokens)...")

for i in range(1, num_chunks + 1):
    get("fineweb_train_%06d.bin" % i, local_dir)
    print(f"  {i}/{num_chunks}")

print(f"Done! Data in {local_dir}/")
