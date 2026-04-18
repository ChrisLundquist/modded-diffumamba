"""MaskGIT-style iterative remasking sampler for prefix-conditional MDLM.

Difference from samplers/mdlm_topk:
  - mdlm_topk one-shot: each demask step *commits* tokens permanently.
    Once a position is unmasked it never changes. ~O(L/k) forwards.
  - MaskGIT iterative: each step samples ALL masked positions, scores
    confidence, then re-masks the lowest-confidence fraction. Allows the
    model to revise earlier guesses, often improves coherence.

Schedule (cosine, Chang 2022): fraction of positions to KEEP at step t/T is
    keep(t) = 1 - cos(π/2 · t/T)
so step 0 keeps 0 (all remasked), step T keeps 1 (all final).

Reference: MaskGIT (Chang et al. 2022, arXiv 2202.04200).
"""

import math
import torch
import torch.nn.functional as F

MASK_TOKEN = 50257


@torch.no_grad()
def maskgit_prefix(
    model,
    prefix_ids,          # LongTensor [B, P]
    cont_len,            # int
    n_steps=12,          # default MaskGIT image setting; lower = faster, higher = better
    top_k=50,
    temperature=1.0,
    confidence='max_prob',   # or 'gumbel': add Gumbel noise to confidence
    gumbel_tau=4.5,           # noise scale for Gumbel-perturbed confidence (Chang 2022)
    device='cuda',
    autocast_dtype=torch.bfloat16,
):
    """Return LongTensor [B, P + cont_len] with prefix preserved.

    Args:
      n_steps: number of demask iterations. MaskGIT paper used 8-12.
      confidence: 'max_prob' uses raw max-confidence; 'gumbel' adds annealed
        Gumbel noise to encourage exploration in early steps.
    """
    B, P = prefix_ids.shape
    T = P + cont_len
    x = torch.full((B, T), MASK_TOKEN, device=device, dtype=torch.long)
    x[:, :P] = prefix_ids.to(device)

    prefix_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    prefix_mask[:, :P] = True

    for step in range(n_steps):
        # Cosine schedule: fraction of cont positions kept after this step
        progress = (step + 1) / n_steps
        keep_frac = 1.0 - math.cos(0.5 * math.pi * progress)
        n_keep_this_step = int(round(keep_frac * cont_len))

        with torch.autocast('cuda', dtype=autocast_dtype):
            logits = model(x, causal=False)[..., :MASK_TOKEN]

        # Sample fresh tokens for ALL non-prefix positions (regardless of mask state)
        # — MaskGIT remasks each step, so previously-unmasked positions can change.
        non_prefix_pos = ~prefix_mask
        # top-k + temperature sampling per position
        scaled = logits.float() / temperature
        tv, ti = scaled.topk(top_k, dim=-1)
        filtered = torch.full_like(scaled, -float('inf'))
        filtered.scatter_(-1, ti, tv)
        probs = F.softmax(filtered, dim=-1)                              # [B, T, V]
        # multinomial expects 2D, flatten
        flat = probs.reshape(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1).squeeze(-1).reshape(B, T)

        # Confidence per position: probability the model assigns to its sampled token
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)        # [B, T]

        if confidence == 'gumbel':
            # Annealed Gumbel noise (Chang 2022) — high noise early, anneal toward 0
            anneal = gumbel_tau * (1.0 - progress)
            gumbel = -torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-9)).clamp_min(1e-9))
            score = conf.log() + anneal * gumbel
        else:
            score = conf

        # Prefix positions get -inf score so they're never "kept" (they stay)
        score = score.masked_fill(prefix_mask, -float('inf'))

        # For each batch element, pick the n_keep_this_step highest-conf positions to KEEP
        # All other non-prefix positions get re-masked
        new_x = x.clone()
        for b in range(B):
            cand_pos = non_prefix_pos[b].nonzero(as_tuple=True)[0]   # all cont positions
            cand_scores = score[b, cand_pos]
            k = min(n_keep_this_step, len(cand_pos))
            if k <= 0:
                # Nothing to keep yet — everything stays masked
                new_x[b, cand_pos] = MASK_TOKEN
                continue
            _, top_idx = cand_scores.topk(k)
            keep_pos = cand_pos[top_idx]
            # Re-mask everything not in keep_pos
            keep_mask = torch.zeros_like(non_prefix_pos[b])
            keep_mask[keep_pos] = True
            # Kept positions get the sampled token; others get MASK
            new_x[b, cand_pos] = torch.where(
                keep_mask[cand_pos], sampled[b, cand_pos],
                torch.full_like(sampled[b, cand_pos], MASK_TOKEN),
            )
        x = new_x

    return x
