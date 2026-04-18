"""Transformer LR sweep with constant schedule.

The Mamba3 convergence run used Adam lr=1e-3 with constant LR.
The transformer baseline used Muon+AdamW with constant LR, but the
AdamW LR (1.5e-4) was never swept. The Muon LR (0.01) was also fixed.

This sweep tests whether the transformer can close the 0.46-nat gap
with better hyperparameters. Tests:
  A: Adam lr=1e-3 constant (same recipe as Mamba3)
  B: Adam lr=3e-4 constant
  C: Muon lr=0.01 + Adam lr=1e-3 constant (higher embedding LR)

All use the same architecture as the transformer baseline:
GPT2 with vocab_size=50258, n_layer=6, n_head=6, n_embd=384 (30.3M params).

5K steps each (~0.66 epoch), compare val loss to Mamba3's val at same step count.
Mamba3 reference at 5K steps: val ~6.57 (Min-SNR gamma=5).
"""

import sys
import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gpt2 import GPT2
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

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'muon_exp/outputs/transformer_lr_sweep'


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
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for t_val in torch.linspace(0.05, 0.95, 20):
            t_batch = torch.full((min(MICRO_BATCH, n_val), 1), t_val.item(), device=device)
            batch_losses = []
            for i in range(0, n_val, MICRO_BATCH):
                vx = val_x[i:i + MICRO_BATCH]
                bs = vx.shape[0]
                t = t_batch[:bs]
                alpha = 1.0 - torch.exp(-k * t)
                mask = torch.rand(bs, vx.shape[1], device=device) < alpha
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


def run_config(name, model, optimizers, train_tokens, val_x, device, seed=42):
    print(f'\n{"="*60}')
    print(f'  {name} ({model.param_count()/1e6:.1f}M params)')
    print(f'{"="*60}')

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN

    t0 = time.time()
    best_val = float('inf')
    results = []

    for step in range(TOTAL_STEPS):
        # Warmup
        if step < WARMUP_STEPS:
            frac = (step + 1) / WARMUP_STEPS
            for opt in optimizers:
                for pg in opt.param_groups:
                    if '_base_lr' not in pg:
                        pg['_base_lr'] = pg['lr']
                    pg['lr'] = pg['_base_lr'] * frac

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            val_loss = eval_mdlm(model, val_x)
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            if val_loss < best_val:
                best_val = val_loss
            print(f'  step {step:5d} | val {val_loss:.4f} | best {best_val:.4f} | '
                  f'{tps/1e3:.0f}K tok/s | {elapsed:.0f}s', flush=True)
            results.append({'step': step, 'val': val_loss, 'best': best_val,
                            'train_loss': total_loss, 'elapsed': elapsed})

    del model, optimizers
    torch.cuda.empty_cache()
    return results


def main():
    device = 'cuda'
    print('Loading data...')
    all_tokens = np.load(DATA_PATH).astype(np.int64)
    train_tokens = all_tokens[:1_000_000_000]
    val_tokens = all_tokens[1_000_000_000:]
    n_val = min(512, len(val_tokens) // (SEQ_LEN + 1))
    val_x = torch.from_numpy(val_tokens[:n_val * SEQ_LEN].reshape(n_val, SEQ_LEN).copy()).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def make_config_a():
        """Adam lr=1e-3 constant (same recipe as Mamba3)."""
        torch.manual_seed(42)
        model = GPT2(vocab_size=50258, n_layer=6, n_head=6, n_embd=384).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return model, [opt]

    def make_config_b():
        """Adam lr=3e-4 constant."""
        torch.manual_seed(42)
        model = GPT2(vocab_size=50258, n_layer=6, n_head=6, n_embd=384).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        return model, [opt]

    def make_config_c():
        """Muon lr=0.01 + Adam lr=1e-3 (higher embedding LR than baseline's 1.5e-4)."""
        torch.manual_seed(42)
        model = GPT2(vocab_size=50258, n_layer=6, n_head=6, n_embd=384).to(device)
        muon_params, adamw_params, _, _ = model.param_groups()
        adamw_opt = torch.optim.Adam(
            [{'params': adamw_params, 'lr': 1e-3, 'betas': (0.9, 0.999)}])
        muon_opt = Muon([{'params': muon_params}], lr=0.01, momentum=0.95, ns_steps=5)
        return model, [adamw_opt, muon_opt]

    configs = [
        ('A_adam_1e3', make_config_a),
        ('B_adam_3e4', make_config_b),
        ('C_muon_adam1e3', make_config_c),
    ]

    all_results = {}
    for name, make_fn in configs:
        model, optimizers = make_fn()
        all_results[name] = run_config(name, model, optimizers, train_tokens, val_x, device)

    # Summary
    print(f'\n{"="*60}')
    print('TRANSFORMER LR SWEEP SUMMARY (5K steps)')
    print(f'{"="*60}')
    print(f'  Mamba3 reference at 5K steps: val ~6.57 (Adam 1e-3, gamma=5)')
    print(f'  Transformer baseline at 5K steps: val ~6.40 (Muon + AdamW 1.5e-4)')
    print()
    for name, results in all_results.items():
        final = results[-1]
        print(f'  {name:25s} | val={final["val"]:.4f} | best={final["best"]:.4f}')

    with open(os.path.join(OUTPUT_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {OUTPUT_DIR}/sweep_results.json')


if __name__ == '__main__':
    main()
