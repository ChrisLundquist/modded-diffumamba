"""AR-teacher NLL of model completions.

Scores the completion (cont_len tokens after the prefix) under an AR teacher.
Length-normalized so numbers are comparable to per-token perplexities.

We prepend the prefix as context so the teacher sees the same starting state
the MDLM model did. We then compute NLL only over the completion tokens.

Teacher bias caveat: GPT-2 was trained on WebText, not FineWeb-Edu, so
absolute NLLs are biased toward a different distribution. Always pair with
the real-continuation ceiling (score real FineWeb-Edu continuations under the
same teacher) to read *gaps*, not absolutes.
"""

import os
import torch
import torch.nn.functional as F

# Cache
_TEACHER_CACHE = {}


def load_teacher(model_name='gpt2', device='cuda'):
    """Lazy-load an HF GPT-2 causal LM. Cached per process."""
    key = (model_name, device)
    if key in _TEACHER_CACHE:
        return _TEACHER_CACHE[key]
    from transformers import GPT2LMHeadModel
    # Attempt local HF cache first (no network) before falling back
    model = GPT2LMHeadModel.from_pretrained(
        model_name, local_files_only=False, torch_dtype=torch.float32,
    ).to(device).eval()
    _TEACHER_CACHE[key] = model
    return model


@torch.no_grad()
def teacher_nll(
    completion_ids,     # LongTensor [B, T_full]  (prefix + continuation concatenated)
    prefix_len,         # int: first P tokens are prefix, skipped for NLL
    teacher=None,
    teacher_name='gpt2',
    device='cuda',
    batch_size=16,
):
    """Return per-sample length-normalized NLL over continuation tokens.

    completion_ids: full sequence including prefix (prefix + N new tokens)
    prefix_len: how many tokens at the start are the given prefix
    """
    if teacher is None:
        teacher = load_teacher(teacher_name, device)
    ids = completion_ids.to(device).long()
    assert ids.dim() == 2, f'expected [B, T], got {ids.shape}'
    # GPT-2 vocab is 50257 (IDs 0..50256). Our MDLM uses 50258 with MASK=50257.
    # If a sample contains 50257, embedding OOBs.
    assert int(ids.max().item()) < 50257, (
        f'teacher_nll: input contains token id {int(ids.max().item())} '
        f'>= 50257 (MASK); cannot score under GPT-2-vocab teacher')
    B, T = ids.shape
    cont_len = T - prefix_len
    assert cont_len > 0

    nlls = torch.zeros(B, device=device)
    for i in range(0, B, batch_size):
        chunk = ids[i:i + batch_size]
        # GPT-2 AR: logits at position t predict token at t+1.
        # We want NLL of tokens at positions [prefix_len .. T-1] given all prior.
        logits = teacher(chunk).logits                       # [b, T, V]
        shifted_logits = logits[:, prefix_len - 1:-1, :]      # predicts tokens [P..T-1]
        targets = chunk[:, prefix_len:]                       # tokens at [P..T-1]
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            targets.reshape(-1),
            reduction='none',
        ).reshape(chunk.size(0), cont_len)
        nlls[i:i + chunk.size(0)] = loss.mean(dim=1)
    return nlls.cpu()
