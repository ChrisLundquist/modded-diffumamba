"""Prefix-conditional MDLM sampler (top-k with pre-nucleus ordering).

Identical to the fixed demasking recipe in probes/generation_multi_seed.py
(2026-04-18 fix: order positions by pre-nucleus max prob) but starts from
a mixed sequence [real_prefix, MASK...], unmasks only completion positions.
"""

import torch
import torch.nn.functional as F

MASK_TOKEN = 50257


@torch.no_grad()
def demask_topk_prefix(
    model,
    prefix_ids,          # LongTensor [B, P]
    cont_len,            # int
    top_k=50,
    temperature=1.0,
    n_steps=None,        # default cont_len // 2
    device='cuda',
    autocast_dtype=torch.bfloat16,
):
    """Return LongTensor [B, P + cont_len] with prefix preserved, rest unmasked.

    Mirrors probes/generation_multi_seed.py: pre-nucleus max prob for ordering,
    top-k filter after temperature for sampling.
    """
    B, P = prefix_ids.shape
    T = P + cont_len
    if n_steps is None:
        n_steps = cont_len // 2
    n_per_step = max(1, cont_len // n_steps)

    x = torch.full((B, T), MASK_TOKEN, device=device, dtype=torch.long)
    x[:, :P] = prefix_ids.to(device)

    prefix_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    prefix_mask[:, :P] = True

    for _ in range(n_steps):
        with torch.autocast('cuda', dtype=autocast_dtype):
            logits = model(x, causal=False)[..., :MASK_TOKEN]  # strip MASK column

        still_masked = (x == MASK_TOKEN) & ~prefix_mask        # [B, T]
        if not still_masked.any():
            break

        # Per-batch-element unmask (vectorized across positions)
        for b in range(B):
            pos = still_masked[b].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            l_raw = logits[b, pos].float()
            pre_probs = F.softmax(l_raw, dim=-1)
            l_scaled = l_raw / temperature
            topk_vals, topk_idx = l_scaled.topk(top_k, dim=-1)
            filtered = torch.full_like(l_scaled, -float('inf'))
            filtered.scatter_(-1, topk_idx, topk_vals)
            probs = F.softmax(filtered, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            order_key = pre_probs.max(dim=-1).values
            _, order = order_key.sort(descending=True)
            k_this_step = min(n_per_step, len(pos))
            chosen = order[:k_this_step]
            x[b, pos[chosen]] = sampled[chosen]

    # Safety: any MASK tokens still present (shouldn't happen) → replace with samples
    still_masked = (x == MASK_TOKEN) & ~prefix_mask
    if still_masked.any():
        with torch.autocast('cuda', dtype=autocast_dtype):
            logits = model(x, causal=False)[..., :MASK_TOKEN]
        for b in range(B):
            pos = still_masked[b].nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                l = logits[b, pos].float() / temperature
                tv, ti = l.topk(top_k, dim=-1)
                ff = torch.full_like(l, -float('inf'))
                ff.scatter_(-1, ti, tv)
                p = F.softmax(ff, dim=-1)
                x[b, pos] = torch.multinomial(p, num_samples=1).squeeze(-1)

    return x
