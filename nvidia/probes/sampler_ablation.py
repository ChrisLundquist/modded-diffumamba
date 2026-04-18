"""Fixed sampler probe — tests multiple sampling strategies.

Fixes the primary bug identified by the reviewer:
  - OLD: max_probs computed from post-nucleus distribution → peaky positions
    (fillers) always unmask first, causing self-reinforcing repetition.
  - NEW: ordering by pre-nucleus max probability, sampling from post-nucleus.

Tests 3 sampling strategies to compare:
  A) Fixed nucleus (top_p=0.9, temp=1.0)
  B) Top-k (k=50, temp=1.0)
  C) Wider nucleus (top_p=0.95, temp=1.0)

Plus baseline strategies for comparison:
  D) Greedy (argmax)
  E) Buggy original (post-nucleus ordering) — to confirm the bug

Tests all 3 checkpoints: 125M D_modern, 30M D_modern, 30M Mamba3.
"""

import sys
import os
import torch
import torch.nn.functional as F
import tiktoken
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformer_v2 import TransformerV2
from hybrid_model import DiffuMambaH

MASK_TOKEN = 50257
enc = tiktoken.get_encoding('gpt2')


def load_model(name, device):
    if name == 'd_modern_125m':
        model = TransformerV2(
            vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
            use_rope=True, use_swiglu=True, use_qk_norm=True).to(device)
        ckpt_path = 'muon_exp/outputs/125m_10b_dmodern/checkpoint_72479.pt'
    elif name == 'd_modern_30m':
        model = TransformerV2(
            vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
            use_rope=True, use_swiglu=True).to(device)
        ckpt_path = 'muon_exp/outputs/transformer_converge_v3/checkpoint_56000.pt'
    elif name == 'mamba3_30m':
        model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                            attn_positions=set(), mamba_version=3, d_state=32).to(device)
        ckpt_path = 'muon_exp/outputs/mamba3_converge/checkpoint_56000.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded {name}: {model.param_count()/1e6:.1f}M params')
    del ckpt
    torch.cuda.empty_cache()
    return model


def demask(model, device, seq_len, strategy, temperature=1.0, top_p=0.9, top_k=50,
           n_steps=None, buggy_ordering=False):
    """Iterative demasking with configurable sampling.

    Args:
        strategy: 'greedy', 'nucleus', 'top_k'
        buggy_ordering: If True, reproduce the original bug (post-nucleus ordering)
    """
    if n_steps is None:
        n_steps = seq_len // 2
    x = torch.full((1, seq_len), MASK_TOKEN, device=device, dtype=torch.long)
    n_per_step = max(1, seq_len // n_steps)

    for step in range(n_steps):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, causal=False)[..., :MASK_TOKEN]
        masked_pos = (x[0] == MASK_TOKEN).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            break

        l_raw = logits[0, masked_pos].float()
        # Pre-nucleus probabilities — used for ORDERING (FIX)
        pre_probs = F.softmax(l_raw, dim=-1)

        # Apply temperature + sampling filter for TOKEN SELECTION
        l_scaled = l_raw / temperature
        if strategy == 'greedy':
            sampled = l_scaled.argmax(dim=-1)
        elif strategy == 'top_k':
            topk_vals, topk_idx = l_scaled.topk(top_k, dim=-1)
            filtered = torch.full_like(l_scaled, -float('inf'))
            filtered.scatter_(-1, topk_idx, topk_vals)
            probs = F.softmax(filtered, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        elif strategy == 'nucleus':
            sorted_l, sorted_idx = l_scaled.sort(dim=-1, descending=True)
            cumprobs = F.softmax(sorted_l, dim=-1).cumsum(dim=-1)
            mask_nuc = cumprobs > top_p
            mask_nuc[:, 0] = False
            sorted_l[mask_nuc] = -float('inf')
            filtered = torch.full_like(l_scaled, -float('inf'))
            filtered.scatter_(-1, sorted_idx, sorted_l)
            probs = F.softmax(filtered, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Ordering: pre-nucleus max prob (FIX) or post-nucleus (bug reproduction)
        if buggy_ordering and strategy != 'greedy':
            order_probs = probs.max(dim=-1).values
        else:
            order_probs = pre_probs.max(dim=-1).values

        _, order = order_probs.sort(descending=True)
        to_unmask = order[:min(n_per_step, len(masked_pos))]

        for idx in to_unmask:
            pos = masked_pos[idx]
            x[0, pos] = sampled[idx]

    return enc.decode(x[0].tolist())


def analyze(text):
    text = text.replace('\ufffd', '?')
    tokens = text.lower().split()
    if len(tokens) < 4:
        return {'(too short)': True}
    bigrams = Counter((tokens[i], tokens[i+1]) for i in range(len(tokens)-1))
    top_bg = bigrams.most_common(1)[0]
    prep = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'the', 'a', 'an'}
    prep_frac = sum(1 for t in tokens if t in prep) / len(tokens)
    unique_ratio = len(set(tokens)) / len(tokens)
    return {
        'unique': f'{unique_ratio:.0%}',
        'prep': f'{prep_frac:.0%}',
        'top_bigram': f'{top_bg[0]} ({top_bg[1]}x)',
    }


def main():
    device = 'cuda'
    torch.set_float32_matmul_precision('high')

    models = ['d_modern_125m', 'd_modern_30m', 'mamba3_30m']

    strategies = [
        ('greedy', dict(strategy='greedy')),
        ('nucleus_bugged', dict(strategy='nucleus', temperature=0.8, top_p=0.9, buggy_ordering=True)),
        ('nucleus_fixed', dict(strategy='nucleus', temperature=1.0, top_p=0.9)),
        ('nucleus_wider', dict(strategy='nucleus', temperature=1.0, top_p=0.95)),
        ('top_k_50', dict(strategy='top_k', temperature=1.0, top_k=50)),
    ]

    SEQ_LEN = 128

    for model_name in models:
        model = load_model(model_name, device)
        print(f'\n{"="*70}')
        print(f'  GENERATION: {model_name} (seq_len={SEQ_LEN})')
        print(f'{"="*70}')

        for strat_name, kwargs in strategies:
            torch.manual_seed(42)
            text = demask(model, device, SEQ_LEN, **kwargs)
            stats = analyze(text)
            print(f'\n  [{strat_name}]')
            print(f'  {text[:400]}')
            print(f'  stats: {stats}')

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
