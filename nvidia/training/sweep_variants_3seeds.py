"""Transformer variant probe: 3 seeds x 5K steps each.

Configs:
  A: Baseline (6L x 384d, learned pos, GELU)        — reference
  B: Deep (12L x 320d, n_head=8, learned pos, GELU) — depth hypothesis
  C: U-Net (6L x 384d, skip connections)             — diffusion standard
  D: Modern (6L x 384d, RoPE + SwiGLU)              — modernization
  E: Combo (6L x 384d, RoPE + SwiGLU + U-Net)       — kitchen sink

All use: Muon lr=0.01 + Adam lr=1e-3 (embed), constant LR, Min-SNR gamma=5.
Eval RNG seeded for reproducibility. Data order fixed (seed=42), only init varies.
"""

import sys
import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformer_v2 import TransformerV2
from muon import Muon
from data import TokenDataset
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

MASK_TOKEN = 50257
MICRO_BATCH = 16
GRAD_ACCUM = 8
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
SEQ_LEN = 1024
TOTAL_STEPS = 5000
WARMUP_STEPS = 200
LOG_EVERY = 500

MUON_LR = 0.01
EMBED_LR = 1e-3

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'muon_exp/outputs/transformer_variants'

SEEDS = [42, 1337, 7]


def mdlm_loss(model, x):
    B, T = x.shape
    device = x.device
    t = torch.rand(B, 1, device=device) * 0.95 + 0.05
    k = 5.0
    alpha = 1.0 - torch.exp(-k * t)
    mask = torch.rand(B, T, device=device) < alpha
    mask_tokens = torch.full_like(x, MASK_TOKEN)
    x_t = torch.where(mask, mask_tokens, x)
    logits = model(x_t, causal=False)[..., :MASK_TOKEN]
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction='none'
    ).reshape(B, T)
    MIN_SNR_GAMMA = 5.0
    elbo_weight = k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))
    elbo_weight = elbo_weight.clamp(max=MIN_SNR_GAMMA)
    mask_float = mask.float()
    per_sample = (per_token_loss * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
    return (per_sample * elbo_weight.squeeze(1)).mean()


def eval_mdlm(model, val_x):
    model.eval()
    device = val_x.device
    n_val = val_x.shape[0]
    total_loss = 0.0
    n_points = 0
    k = 5.0
    eval_rng = torch.Generator(device=device)
    eval_rng.manual_seed(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for t_val in torch.linspace(0.05, 0.95, 20):
            t_batch = torch.full((min(MICRO_BATCH, n_val), 1), t_val.item(), device=device)
            batch_losses = []
            for i in range(0, n_val, MICRO_BATCH):
                vx = val_x[i:i + MICRO_BATCH]
                bs = vx.shape[0]
                t = t_batch[:bs]
                alpha = 1.0 - torch.exp(-k * t)
                mask = torch.rand(bs, vx.shape[1], device=device, generator=eval_rng) < alpha
                mask_tokens = torch.full_like(vx, MASK_TOKEN)
                x_t = torch.where(mask, mask_tokens, vx)
                logits = model(x_t, causal=False)[..., :MASK_TOKEN]
                per_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), vx.reshape(-1), reduction='none'
                ).reshape(bs, -1)
                elbo_weight = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=5.0)
                mask_float = mask.float()
                per_sample = (per_token_loss * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
                batch_losses.append((per_sample * elbo_weight.squeeze(1)).mean().item())
            total_loss += sum(batch_losses) / len(batch_losses)
            n_points += 1
    return total_loss / n_points


def run_config(name, model_kwargs, train_tokens, val_x, device, seed):
    # Separate init seed from data seed: model init varies, data order is fixed
    torch.manual_seed(seed)
    model = TransformerV2(**model_kwargs).to(device)

    muon_params, adamw_params = model.param_groups()
    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)
    optimizers = [adam_opt, muon_opt]

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(42)  # fixed data order across all seeds
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN

    t0 = time.time()
    best_val = float('inf')
    results = []

    for step in range(TOTAL_STEPS):
        if step < WARMUP_STEPS:
            frac = (step + 1) / WARMUP_STEPS
            for pg in adam_opt.param_groups:
                pg['lr'] = EMBED_LR * frac
            for pg in muon_opt.param_groups:
                pg['lr'] = MUON_LR * frac

        model.train()
        total_loss = 0.0
        for opt in optimizers:
            opt.zero_grad()
        for _ in range(GRAD_ACCUM):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = mdlm_loss(model, x) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(adamw_params, 1.0)
        for opt in optimizers:
            opt.step()

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            val_loss = eval_mdlm(model, val_x)
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            if val_loss < best_val:
                best_val = val_loss
            print(f'  [{name} s{seed}] step {step:5d} | val {val_loss:.4f} | '
                  f'{tps/1e3:.0f}K tok/s | {elapsed:.0f}s', flush=True)
            results.append({'step': step, 'val': val_loss, 'elapsed': elapsed})

    final_val = results[-1]['val']
    del model, optimizers
    torch.cuda.empty_cache()
    return final_val, results


def main():
    device = 'cuda'
    print('Loading data...')
    all_tokens = np.load(DATA_PATH).astype(np.int64)
    train_tokens = all_tokens[:1_000_000_000]
    val_tokens = all_tokens[1_000_000_000:]
    n_val = min(512, len(val_tokens) // (SEQ_LEN + 1))
    val_x = torch.from_numpy(val_tokens[:n_val * SEQ_LEN].reshape(n_val, SEQ_LEN).copy()).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = {
        'A_baseline': dict(
            n_layer=6, n_head=6, n_embd=384,
        ),
        'B_deep': dict(
            n_layer=12, n_head=8, n_embd=320,
        ),
        'C_unet': dict(
            n_layer=6, n_head=6, n_embd=384,
            use_unet=True,
        ),
        'D_modern': dict(
            n_layer=6, n_head=6, n_embd=384,
            use_rope=True, use_swiglu=True,
        ),
        'E_combo': dict(
            n_layer=6, n_head=6, n_embd=384,
            use_rope=True, use_swiglu=True, use_unet=True,
        ),
    }

    # Print param counts
    print('\nConfig param counts:')
    for name, kwargs in configs.items():
        torch.manual_seed(42)
        m = TransformerV2(**kwargs)
        muon_p, adam_p = m.param_groups()
        print(f'  {name:15s}: {m.param_count()/1e6:.1f}M total, '
              f'{sum(p.numel() for p in muon_p)/1e6:.1f}M muon, '
              f'{sum(p.numel() for p in adam_p)/1e6:.1f}M adam')
        del m

    # all_results[config_name][seed_idx] = (final_val, history)
    all_results = {name: {'vals': [], 'runs': []} for name in configs}

    # Iterate seeds first, configs second — get one data point per variant early
    for si, seed in enumerate(SEEDS):
        print(f'\n{"="*60}')
        print(f'  SEED {seed} ({si+1}/{len(SEEDS)}) — all configs')
        print(f'{"="*60}')

        for name, kwargs in configs.items():
            final_val, results = run_config(name, kwargs, train_tokens, val_x, device, seed)
            all_results[name]['vals'].append(final_val)
            all_results[name]['runs'].append({'seed': seed, 'final_val': final_val, 'history': results})

        # Print standings after each seed
        print(f'\n  --- Standings after seed {seed} ---')
        for name in configs:
            vals = all_results[name]['vals']
            mean_val = np.mean(vals)
            if len(vals) > 1:
                std_val = np.std(vals)
                print(f'  {name:15s} | {mean_val:.4f} +/- {std_val:.4f} (n={len(vals)})')
            else:
                print(f'  {name:15s} | {mean_val:.4f} (n=1)')

    # Final summary
    print(f'\n{"="*60}')
    print('TRANSFORMER VARIANT PROBE (3 seeds x 5K steps)')
    print(f'{"="*60}')
    print(f'{"Config":15s} | {"Mean":>8s} | {"Std":>6s} | {"Seeds":>30s}')
    print('-' * 70)
    for name in configs:
        vals = all_results[name]['vals']
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        seeds_str = ', '.join(f'{v:.4f}' for v in vals)
        print(f'{name:15s} | {mean_val:8.4f} | {std_val:6.4f} | {seeds_str}')
        all_results[name]['mean'] = float(mean_val)
        all_results[name]['std'] = float(std_val)

    with open(os.path.join(OUTPUT_DIR, 'variant_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f'\nSaved to {OUTPUT_DIR}/variant_results.json')


if __name__ == '__main__':
    main()
