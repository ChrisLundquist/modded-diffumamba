# FineWeb-Edu 10B (GPT-2 tokenized .bin shards)

Source: https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards
Downloaded: 2026-04-15

First 10 training shards + 1 validation shard from the 100B token dataset.
Each shard is ~200MB containing ~100M GPT-2 tokens as uint16.

Format: modded-nanogpt .bin shard format
- Header: 256 int32s (magic=20240520, version=1, num_tokens)
- Body: uint16 tokens (GPT-2 tokenizer, vocab 50257)

Total: ~1B train tokens + ~100M val tokens

Usage:
```bash
python train.py --data_dir data/fineweb-edu-10B --config quokka
```

Note: This is FineWeb-Edu (education-filtered), not plain FineWeb.
The regular FineWeb-10B is at data/fineweb10B/ (from kjj0/fineweb10B-gpt2).
