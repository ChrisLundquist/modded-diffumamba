"""Profile peak VRAM for transformer and Mamba3 at various scales.

Runs one forward+backward pass and reports peak GPU memory.
"""

import sys
import os
import gc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

MASK_TOKEN = 50257
SEQ_LEN = 1024


def profile_model(name, model_fn, micro_batch, device):
    """Profile peak VRAM for one forward+backward pass."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    baseline_mem = torch.cuda.memory_allocated() / 1e9

    torch.manual_seed(42)
    model = model_fn().to(device)
    params = sum(p.numel() for p in model.parameters())
    model_mem = torch.cuda.memory_allocated() / 1e9

    # Dummy optimizer to allocate state
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy forward+backward
    x = torch.randint(0, MASK_TOKEN, (micro_batch, SEQ_LEN), device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x, causal=False)
        loss = logits[..., :MASK_TOKEN].mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    current_mem = torch.cuda.memory_allocated() / 1e9

    print(f'  {name:40s} | {params/1e6:7.1f}M | MB={micro_batch:2d} | '
          f'peak {peak_mem:5.1f}GB | current {current_mem:5.1f}GB | '
          f'headroom {32-peak_mem:5.1f}GB')

    del model, optimizer, x, logits, loss
    torch.cuda.empty_cache()
    gc.collect()
    return peak_mem, params


def main():
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print()

    # Transformer configs
    from model_v2 import TransformerV2

    transformer_configs = [
        ('T-30M (current)', dict(n_layer=6, n_head=6, n_embd=384, use_rope=True, use_swiglu=True)),
        ('T-60M', dict(n_layer=8, n_head=8, n_embd=512, use_rope=True, use_swiglu=True)),
        ('T-125M', dict(n_layer=12, n_head=12, n_embd=768, use_rope=True, use_swiglu=True)),
        ('T-250M', dict(n_layer=24, n_head=16, n_embd=1024, use_rope=True, use_swiglu=True)),
        ('T-350M', dict(n_layer=24, n_head=16, n_embd=1280, use_rope=True, use_swiglu=True)),
    ]

    for micro_batch in [16, 8, 4]:
        print(f'\n{"="*80}')
        print(f'  TRANSFORMER (D_modern style, RoPE+SwiGLU) — MICRO_BATCH={micro_batch}')
        print(f'{"="*80}')
        for name, kwargs in transformer_configs:
            try:
                peak, params = profile_model(
                    name, lambda kw=kwargs: TransformerV2(**kw), micro_batch, device)
                if peak > 31:
                    print(f'    ^ EXCEEDS 32GB')
                    break
            except torch.cuda.OutOfMemoryError:
                print(f'  {name:40s} | MB={micro_batch:2d} | OOM')
                torch.cuda.empty_cache()
                gc.collect()
                break

    # Mamba3 configs
    from hybrid_model import DiffuMambaH

    mamba_configs = [
        ('M3-25M (current)', dict(n_embd=384, n_head=6, n_mamba=6, n_attn=0, attn_positions=set(), mamba_version=3, d_state=32)),
        ('M3-50M', dict(n_embd=512, n_head=8, n_mamba=6, n_attn=0, attn_positions=set(), mamba_version=3, d_state=32)),
        ('M3-100M', dict(n_embd=768, n_head=6, n_mamba=6, n_attn=0, attn_positions=set(), mamba_version=3, d_state=32)),
        ('M3-100M-8L', dict(n_embd=768, n_head=6, n_mamba=8, n_attn=0, attn_positions=set(), mamba_version=3, d_state=32)),
        ('M3-100M-d16', dict(n_embd=768, n_head=6, n_mamba=6, n_attn=0, attn_positions=set(), mamba_version=3, d_state=16)),
        ('M3-200M', dict(n_embd=1024, n_head=8, n_mamba=6, n_attn=0, attn_positions=set(), mamba_version=3, d_state=32)),
    ]

    for micro_batch in [16, 8, 4]:
        print(f'\n{"="*80}')
        print(f'  MAMBA3 (bidirectional) — MICRO_BATCH={micro_batch}')
        print(f'{"="*80}')
        for name, kwargs in mamba_configs:
            try:
                peak, params = profile_model(
                    name, lambda kw=kwargs: DiffuMambaH(**kw), micro_batch, device)
                if peak > 31:
                    print(f'    ^ EXCEEDS 32GB')
            except torch.cuda.OutOfMemoryError:
                print(f'  {name:40s} | MB={micro_batch:2d} | OOM')
                torch.cuda.empty_cache()
                gc.collect()


if __name__ == '__main__':
    main()
