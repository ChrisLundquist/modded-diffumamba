"""Large gen probe on a saved checkpoint.

Loads any model.state_dict() checkpoint and runs an n=128 gen probe for
a resolved rep_4 / distinct_4 / top10_share. Useful for post-hoc quality
snapshots of checkpoints trained without --gen_probe_final.

Usage:
    python probe_checkpoint.py --ckpt checkpoints/E_best_10k_s42.pt
    python probe_checkpoint.py --ckpt checkpoints/... --n_samples 256
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

from model import DiffuMamba3, CONFIGS
from train import rep_n, distinct_n, top_word_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="quokka")
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--num_steps", type=int, default=128)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=16,
                    help="Per-call batch size inside model.sample() to control peak VRAM")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = copy.deepcopy(CONFIGS[args.config])
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.d_model is not None:
        cfg.d_model = args.d_model

    model = DiffuMamba3(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded {args.ckpt}")
    print(f"Config: {args.config}  d_model={cfg.d_model}  n_layers={cfg.n_layers}")
    print(f"Probing: n={args.n_samples} × seq_len={args.seq_len} × "
          f"steps={args.num_steps}  top_k={args.top_k}")

    t0 = time.perf_counter()
    all_samples = []
    remaining = args.n_samples
    while remaining > 0:
        bs = min(args.chunk, remaining)
        with torch.no_grad():
            chunk_samples = model.sample(
                batch_size=bs,
                seq_len=args.seq_len,
                num_steps=args.num_steps,
                device=device.type,
                top_k=args.top_k,
            )
        all_samples.append(chunk_samples.cpu())
        torch.cuda.empty_cache()
        remaining -= bs
        print(f"  chunk done ({args.n_samples - remaining}/{args.n_samples})")
    samples = torch.cat(all_samples, dim=0)
    toks = [s.tolist() for s in samples]
    metrics = {
        "rep_4": rep_n(toks, 4),
        "rep_8": rep_n(toks, 8),
        "distinct_4": distinct_n(toks, 4),
        "top10_share": top_word_share(toks, 10),
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "num_steps": args.num_steps,
        "probe_time_s": time.perf_counter() - t0,
        "ckpt": args.ckpt,
    }
    print()
    print(f"rep_4       = {metrics['rep_4']:.4f}")
    print(f"rep_8       = {metrics['rep_8']:.4f}")
    print(f"distinct_4  = {metrics['distinct_4']:.4f}")
    print(f"top10_share = {metrics['top10_share']:.4f}")
    print(f"probe_time  = {metrics['probe_time_s']:.1f}s")

    out = args.ckpt + ".gen.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
