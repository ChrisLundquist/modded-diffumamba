"""ReMDM / corrected-MaskGIT sampler for prefix-conditional MDLM.

Per Chang 2022 (MaskGIT, arXiv 2202.04200) and Wang 2025 (ReMDM, arXiv 2503.00307).

Difference from `mdlm_maskgit.py` (the buggy first attempt):
  - mdlm_maskgit.py re-samples ALL non-prefix positions every step (including
    already-committed ones). Empirically: monotonic degradation as n_steps grows.
  - This sampler samples ONLY at currently-MASKED positions, then uses
    confidence-based remasking on ALL non-prefix positions to decide what to
    commit vs. send back to MASK for the next step. This matches Chang 2022's
    actual algorithm.

Difference from `mdlm_topk.py`:
  - mdlm_topk: pre-nucleus ordering, ONE-SHOT commit. A position unmasked
    in step t stays committed forever.
  - This sampler: confidence-based, REVISABLE. A position committed in step t
    can be re-masked in step t+1 if its current confidence drops below the
    threshold.

Schedule (cosine, Chang 2022 Eq. 4): mask ratio at step (t+1)/T is
    γ((t+1)/T) = cos(π/2 · (t+1)/T)
so cumulative *kept* tokens = L · (1 - γ). Step T → all kept.

Algorithm per step:
  1. Forward pass on current x. Compute logits over vocab (excluding MASK).
  2. At currently-MASKED non-prefix positions, sample tokens via top-k.
  3. Build x_predicted (prefix + currently-committed + newly-sampled).
  4. For ALL non-prefix positions, compute confidence as the model's prob
     for the token currently held there in x_predicted. (For freshly-MASKED
     positions, this is the prob of the just-sampled token; for already-
     committed positions, it's the prob of the kept token under current logits.)
  5. Decide n_keep = round(cont_len · (1 - γ)). Among non-prefix positions,
     keep top n_keep by confidence; the rest become MASK in the next step.
  6. Repeat. At final step, n_keep = cont_len → all positions committed.
"""

import math
import torch
import torch.nn.functional as F

MASK_TOKEN = 50257


@torch.no_grad()
def remdm_prefix(
    model,
    prefix_ids,           # LongTensor [B, P]
    cont_len,             # int
    n_steps=12,
    top_k=50,
    temperature=1.0,
    device='cuda',
    autocast_dtype=torch.bfloat16,
    confidence_noise=0.0,  # 0 = pure confidence; >0 adds Gumbel jitter (Chang 2022)
):
    B, P = prefix_ids.shape
    T = P + cont_len
    x = torch.full((B, T), MASK_TOKEN, device=device, dtype=torch.long)
    x[:, :P] = prefix_ids.to(device)

    prefix_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    prefix_mask[:, :P] = True

    for step in range(n_steps):
        with torch.autocast('cuda', dtype=autocast_dtype):
            logits = model(x, causal=False)[..., :MASK_TOKEN]

        is_masked = (x == MASK_TOKEN) & ~prefix_mask     # [B, T]

        # Sample tokens at currently-masked non-prefix positions only
        scaled = logits.float() / temperature
        tv, ti = scaled.topk(top_k, dim=-1)
        filtered = torch.full_like(scaled, -float('inf'))
        filtered.scatter_(-1, ti, tv)
        probs = F.softmax(filtered, dim=-1)             # [B, T, V]
        flat = probs.reshape(-1, probs.size(-1))
        sampled_flat = torch.multinomial(flat, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B, T)

        # x_pred: prefix + already-committed + newly-sampled-at-masked
        x_pred = torch.where(is_masked, sampled, x)

        # Confidence per position: model's probability for the current token
        confidence = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)  # [B, T]

        if confidence_noise > 0:
            anneal = confidence_noise * (1.0 - (step + 1) / n_steps)
            gumbel = -torch.log(-torch.log(
                torch.rand_like(confidence).clamp_min(1e-9)).clamp_min(1e-9))
            score = confidence.log().clamp_min(-30) + anneal * gumbel
        else:
            score = confidence

        # Cosine mask ratio for step t+1: γ((t+1)/T) = cos(π/2 · (t+1)/T)
        progress = (step + 1) / n_steps
        gamma_next = math.cos(0.5 * math.pi * progress)
        n_keep = int(round(cont_len * (1.0 - gamma_next)))
        n_keep = max(0, min(n_keep, cont_len))

        # Prefix never competes for keep slots
        score = score.masked_fill(prefix_mask, -float('inf'))

        # Per-batch select top n_keep over non-prefix positions; rest go MASK
        new_x = x_pred.clone()
        for b in range(B):
            cand_pos = (~prefix_mask[b]).nonzero(as_tuple=True)[0]
            if n_keep <= 0:
                new_x[b, cand_pos] = MASK_TOKEN
                continue
            cand_score = score[b, cand_pos]
            k_eff = min(n_keep, len(cand_pos))
            _, top_idx = cand_score.topk(k_eff)
            keep_mask_row = torch.zeros_like(prefix_mask[b])
            keep_mask_row[cand_pos[top_idx]] = True
            # At kept positions, keep x_pred's token; at others, MASK
            new_x[b, cand_pos] = torch.where(
                keep_mask_row[cand_pos], x_pred[b, cand_pos],
                torch.full_like(x_pred[b, cand_pos], MASK_TOKEN),
            )
        x = new_x

    # Final safety: any remaining MASKs (shouldn't happen at progress=1.0
    # since gamma_next=cos(π/2)=0 → n_keep=cont_len) get the last sample.
    final_mask = (x == MASK_TOKEN) & ~prefix_mask
    if final_mask.any():
        x = torch.where(final_mask, sampled, x)
    return x
