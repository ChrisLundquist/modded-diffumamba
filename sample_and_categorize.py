"""Generate text from checkpoints and categorize quality.

Categorization:
- coherent: grammatical English, topical flow
- grammatical: English-looking but disjointed
- repetitive: loops/copies of substrings
- garbage: random tokens, broken text

Also computes simple metrics:
- distinct_n: unique n-grams / total n-grams
- repetition: fraction of repeated 4-grams
- avg_word_len: sanity check
"""
import sys
import json
import copy
import time
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

import torch
import tiktoken

from model import DiffuMamba3, CONFIGS

CKPT_DIR = Path(__file__).parent / "checkpoints"
OUT_DIR = Path(__file__).parent / "samples"
OUT_DIR.mkdir(exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
MASK_ID = 50257


def generate_samples(ckpt_path, n_samples=4, seq_len=128, num_steps=128,
                     temperature=0.8, device="cuda", dtype=torch.bfloat16):
    """Load a checkpoint and generate samples."""
    cfg = copy.deepcopy(CONFIGS["quokka"])
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)

    model = DiffuMamba3(cfg).to(device, dtype=dtype)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    torch.manual_seed(0)
    tokens = model.sample(batch_size=n_samples, seq_len=seq_len,
                           num_steps=num_steps, device=device,
                           temperature=temperature)

    del model
    torch.cuda.empty_cache()

    texts = []
    for i in range(n_samples):
        # Replace any remaining mask tokens
        ids = tokens[i].cpu().tolist()
        ids = [t for t in ids if t != MASK_ID and t < 50257]
        text = enc.decode(ids)
        texts.append(text)
    return texts


def compute_metrics(text):
    """Simple quality metrics on generated text."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return {"n_words": 0, "distinct_4gram": 0, "repetition_4gram": 0,
                "avg_word_len": 0, "punct_ratio": 0}

    # 4-gram diversity
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    unique = len(set(ngrams))
    total = len(ngrams)
    distinct_4 = unique / max(total, 1)

    # Repetition: how many 4-grams appear more than once
    counts = Counter(ngrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    rep_4 = repeats / max(total, 1)

    avg_word_len = sum(len(w) for w in words) / len(words)

    # Punctuation ratio (sanity)
    punct = sum(1 for c in text if c in ".,;:!?")
    total_chars = max(len(text), 1)
    punct_ratio = punct / total_chars

    return {
        "n_words": len(words),
        "distinct_4gram": round(distinct_4, 3),
        "repetition_4gram": round(rep_4, 3),
        "avg_word_len": round(avg_word_len, 2),
        "punct_ratio": round(punct_ratio, 3),
    }


def categorize(text, metrics):
    """Heuristic category: coherent / grammatical / repetitive / garbage."""
    if metrics["n_words"] < 10:
        return "garbage"
    if metrics["repetition_4gram"] > 0.3:
        return "repetitive"
    if metrics["avg_word_len"] < 2.5 or metrics["avg_word_len"] > 9:
        return "garbage"
    if metrics["distinct_4gram"] < 0.5:
        return "repetitive"
    # Check word-like-ness: are words mostly alphabetic?
    words = re.findall(r"\w+", text.lower())
    alpha_words = sum(1 for w in words if w.isalpha())
    if alpha_words / max(len(words), 1) < 0.6:
        return "garbage"
    # If it has reasonable diversity and word stats, it's at least grammatical
    if metrics["distinct_4gram"] > 0.8 and metrics["punct_ratio"] > 0.01:
        return "coherent"
    return "grammatical"


def main():
    # Select representative checkpoints
    checkpoints = [
        # Best overall (3 seeds of new_best at 10k)
        ("final10k_new_best_s42",  "FineWeb-Edu + lr=0.01 (best seed, 4.976)"),
        ("final10k_new_best_s137", "FineWeb-Edu + lr=0.01 (typical)"),
        ("final10k_new_best_s2024","FineWeb-Edu + lr=0.01"),
        # Old best
        ("final10k_old_best_s42",  "FineWeb + lr=0.02"),
        # Mousse (best raw loss at 10k)
        ("opt10k_mousse_s42",      "Mousse optimizer (5.329)"),
        # Adam baseline for contrast
        ("best10k_adam_s42",       "Adam baseline (5.732)"),
    ]

    all_results = []
    print(f"Generating samples from {len(checkpoints)} checkpoints...")

    for ckpt_name, desc in checkpoints:
        ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_name} — not found")
            continue

        print(f"\n{'='*70}")
        print(f"  {ckpt_name}")
        print(f"  {desc}")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        texts = generate_samples(ckpt_path, n_samples=4, seq_len=128,
                                  num_steps=128, temperature=0.8)
        elapsed = time.perf_counter() - t0

        results = []
        for i, text in enumerate(texts):
            metrics = compute_metrics(text)
            category = categorize(text, metrics)
            results.append({
                "sample_idx": i, "text": text, "metrics": metrics,
                "category": category,
            })
            print(f"\n  Sample {i+1} [{category}]")
            print(f"  metrics: {metrics}")
            # Show first 250 chars
            preview = text[:250].replace("\n", " ")
            print(f"  text: {preview}...")

        # Summary per checkpoint
        from collections import Counter as C
        cat_counts = C(r["category"] for r in results)
        print(f"\n  Summary: {dict(cat_counts)} (generated in {elapsed:.1f}s)")

        all_results.append({
            "checkpoint": ckpt_name, "description": desc,
            "samples": results,
            "category_counts": dict(cat_counts),
        })

    # Save full results
    with open(OUT_DIR / "samples.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary table
    print(f"\n{'='*70}")
    print(f"CATEGORIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Checkpoint':<32s} {'coh':>4s} {'gram':>5s} {'rep':>4s} {'gar':>4s}  Distinct-4g")
    for r in all_results:
        c = r["category_counts"]
        avg_d4 = sum(s["metrics"]["distinct_4gram"] for s in r["samples"]) / len(r["samples"])
        print(f"  {r['checkpoint']:<32s} "
              f"{c.get('coherent',0):>4d} {c.get('grammatical',0):>5d} "
              f"{c.get('repetitive',0):>4d} {c.get('garbage',0):>4d}  "
              f"{avg_d4:.3f}")

    print(f"\n  Samples saved to {OUT_DIR}/samples.json")


if __name__ == "__main__":
    main()
