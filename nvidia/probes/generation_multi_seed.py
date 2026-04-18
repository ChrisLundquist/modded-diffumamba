"""Multi-seed probe to check for content bias and mode diversity.

Generate 10 samples per model with different seeds, top-k=50, temp=1.0.
Analyze: subject/topic diversity, religious content frequency, repetition patterns.
"""

import sys
import os
import re
import torch
import torch.nn.functional as F
import tiktoken
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformer_v2 import TransformerV2
from hybrid_model import DiffuMambaH

MASK_TOKEN = 50257
enc = tiktoken.get_encoding('gpt2')

# Rough semantic categories to detect content bias
CATEGORIES = {
    'religious': ['god', 'father', 'heaven', 'heavens', 'earth', 'jesus', 'christ', 'holy',
                  'faith', 'church', 'prayer', 'bible', 'scripture', 'lord', 'prophet',
                  'divine', 'angel', 'spiritual', 'worship', 'creation', 'ministry'],
    'scientific': ['research', 'study', 'data', 'analysis', 'experiment', 'hypothesis',
                   'results', 'method', 'scientists', 'researchers', 'evidence'],
    'educational': ['students', 'school', 'teacher', 'learning', 'education', 'class',
                    'course', 'curriculum', 'academic', 'university', 'college'],
    'medical': ['disease', 'patient', 'treatment', 'symptoms', 'diagnosis', 'health',
                'medical', 'doctor', 'hospital', 'clinical', 'therapy'],
    'political': ['government', 'political', 'president', 'election', 'policy',
                  'congress', 'senate', 'party', 'democrat', 'republican'],
    'historical': ['century', 'war', 'history', 'ancient', 'empire', 'king', 'queen',
                   'kingdom', 'historical', 'revolution'],
}


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
    del ckpt
    torch.cuda.empty_cache()
    return model


def demask_topk(model, device, seq_len=128, top_k=50, temperature=1.0, n_steps=None):
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
        pre_probs = F.softmax(l_raw, dim=-1)
        l_scaled = l_raw / temperature
        topk_vals, topk_idx = l_scaled.topk(top_k, dim=-1)
        filtered = torch.full_like(l_scaled, -float('inf'))
        filtered.scatter_(-1, topk_idx, topk_vals)
        probs = F.softmax(filtered, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        order_probs = pre_probs.max(dim=-1).values
        _, order = order_probs.sort(descending=True)
        to_unmask = order[:min(n_per_step, len(masked_pos))]
        for idx in to_unmask:
            pos = masked_pos[idx]
            x[0, pos] = sampled[idx]
    return enc.decode(x[0].tolist())


def categorize(text):
    """Count category keyword occurrences in text."""
    text_lower = text.lower()
    counts = {}
    for cat, keywords in CATEGORIES.items():
        count = sum(len(re.findall(rf'\b{kw}\b', text_lower)) for kw in keywords)
        counts[cat] = count
    return counts


def main():
    device = 'cuda'
    torch.set_float32_matmul_precision('high')

    models = ['d_modern_125m', 'd_modern_30m', 'mamba3_30m']
    seeds = list(range(10))  # 10 samples per model
    SEQ_LEN = 128

    all_results = {}
    for model_name in models:
        print(f'\n{"="*70}')
        print(f'  {model_name} — 10 samples, seq_len={SEQ_LEN}, top-k=50, temp=1.0')
        print(f'{"="*70}')

        model = load_model(model_name, device)
        samples = []
        cat_totals = Counter()
        unique_ratios = []

        for seed in seeds:
            torch.manual_seed(seed)
            text = demask_topk(model, device, SEQ_LEN)
            cats = categorize(text)
            tokens = text.lower().split()
            ur = len(set(tokens)) / max(len(tokens), 1)
            unique_ratios.append(ur)
            for cat, c in cats.items():
                cat_totals[cat] += c
            samples.append((seed, text, cats, ur))
            # Print first 150 chars of each sample
            top_cat = max(cats, key=cats.get) if any(cats.values()) else 'none'
            print(f'  s{seed} [{top_cat}] ur={ur:.0%}: {text[:140]}')

        avg_ur = sum(unique_ratios) / len(unique_ratios)
        print(f'\n  Avg unique ratio: {avg_ur:.0%}')
        print(f'  Total category keywords across 10 samples:')
        for cat, count in cat_totals.most_common():
            print(f'    {cat}: {count}')

        all_results[model_name] = {
            'avg_unique_ratio': avg_ur,
            'category_totals': dict(cat_totals),
        }

        del model
        torch.cuda.empty_cache()

    print(f'\n{"="*70}')
    print('  SUMMARY')
    print(f'{"="*70}')
    print(f'{"Model":20s} {"Avg unique":>12s} {"Top cat":>12s}')
    for name, r in all_results.items():
        top_cat = max(r['category_totals'], key=r['category_totals'].get)
        top_count = r['category_totals'][top_cat]
        print(f'{name:20s} {r["avg_unique_ratio"]:12.0%} {top_cat+f"({top_count})":>12s}')


if __name__ == '__main__':
    main()
