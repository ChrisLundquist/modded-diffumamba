"""Diversity metrics on model completions.

All metrics operate on the completion tokens only (not the prefix).

- distinct-n: unique n-grams / total n-grams, averaged across samples
- self-BLEU-4: BLEU of each sample against the rest of the set (lower = more diverse)
- zipf_slope: fit log(rank) vs log(freq) over the full corpus of completions
"""

import math
from collections import Counter

import torch


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(completion_list, n):
    """Mean distinct-n across samples. completion_list: list of list[int]."""
    ratios = []
    for toks in completion_list:
        grams = _ngrams(toks, n)
        if not grams:
            continue
        ratios.append(len(set(grams)) / len(grams))
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def self_bleu_4(completion_list, max_refs=20, seed=1729):
    """Approximate self-BLEU-4. For each sample, score against <=max_refs others.

    Uses a simple BLEU-4 with uniform weights (no brevity penalty for speed).
    Lower = more diverse. References are selected by deterministic random
    sampling (not first-N) so prompt-set ordering does not bias the metric.
    """
    if len(completion_list) < 2:
        return 0.0
    import random
    rng = random.Random(seed)
    scores = []
    n = len(completion_list)
    for i, hyp in enumerate(completion_list):
        others_idx = list(range(n))
        others_idx.remove(i)
        if len(others_idx) > max_refs:
            others_idx = rng.sample(others_idx, max_refs)
        refs = [completion_list[j] for j in others_idx]
        if not refs:
            continue
        precisions = []
        for n in (1, 2, 3, 4):
            hyp_grams = Counter(_ngrams(hyp, n))
            if not hyp_grams:
                precisions.append(0.0)
                continue
            ref_grams = Counter()
            for r in refs:
                ref_grams |= Counter(_ngrams(r, n))
            overlap = sum(min(c, ref_grams.get(g, 0)) for g, c in hyp_grams.items())
            total = sum(hyp_grams.values())
            precisions.append(overlap / total if total else 0.0)
        # Geometric mean, guarded
        if all(p > 0 for p in precisions):
            score = math.exp(sum(math.log(p) for p in precisions) / 4)
        else:
            score = 0.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def zipf_slope(completion_list):
    """Fit log(freq) = a + b * log(rank). Return -b (positive ~1 for natural text)."""
    counts = Counter()
    for toks in completion_list:
        counts.update(toks)
    if len(counts) < 10:
        return 0.0
    freqs = sorted(counts.values(), reverse=True)
    ranks = list(range(1, len(freqs) + 1))
    # Truncate to top 5000 to avoid long-tail noise
    freqs = freqs[:5000]
    ranks = ranks[:5000]
    xs = [math.log(r) for r in ranks]
    ys = [math.log(f) for f in freqs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = num / den
    return -slope


def uniq_token_ratio(completion_list):
    """Same as probes/generation_multi_seed.py: lowercase-whitespace-split unique ratio."""
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    ratios = []
    for toks in completion_list:
        text = enc.decode(toks).lower()
        words = text.split()
        if not words:
            continue
        ratios.append(len(set(words)) / len(words))
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)
