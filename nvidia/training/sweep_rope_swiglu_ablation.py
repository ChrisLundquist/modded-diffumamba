"""Ablation: isolate RoPE vs SwiGLU contribution in D_modern.

D_modern (RoPE+SwiGLU) = 5.943 at 5K steps (3-seed mean).
Baseline (learned_pos+GELU) = 6.192 at 5K steps.
Gap = 0.249 nats. Where does it come from?

Configs (single seed, 5K steps):
  F: RoPE-only (RoPE + GELU MLP)
  G: SwiGLU-only (learned pos + SwiGLU)
  Reference: A baseline = 6.194, D modern = 5.952 (seed 42)
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
SEED = 42

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'muon_exp/outputs/d_modern_ablation'


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


def run_config(name, model_kwargs, train_tokens, val_x, device):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')

    torch.manual_seed(SEED)
    model = TransformerV2(**model_kwargs).to(device)
    print(f'  Params: {model.param_count()/1e6:.1f}M')

    muon_params, adamw_params = model.param_groups()
    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)
    optimizers = [adam_opt, muon_opt]

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(42)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN

    t0 = time.time()
    best_val = float('inf')

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
            print(f'  [{name}] step {step:5d} | val {val_loss:.4f} | '
                  f'{tps/1e3:.0f}K tok/s | {elapsed:.0f}s', flush=True)

    final_val = val_loss
    del model, optimizers
    torch.cuda.empty_cache()
    return final_val


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
        'F_rope_only': dict(
            n_layer=6, n_head=6, n_embd=384,
            use_rope=True, use_swiglu=False,
        ),
        'G_swiglu_only': dict(
            n_layer=6, n_head=6, n_embd=384,
            use_rope=False, use_swiglu=True,
        ),
    }

    results = {}
    for name, kwargs in configs.items():
        results[name] = run_config(name, kwargs, train_tokens, val_x, device)

    # Summary with references
    print(f'\n{"="*60}')
    print('D_MODERN ABLATION (seed 42, 5K steps)')
    print(f'{"="*60}')
    print(f'  A baseline (ref):    6.1936  (learned_pos + GELU)')
    print(f'  F RoPE-only:         {results["F_rope_only"]:.4f}  (RoPE + GELU)')
    print(f'  G SwiGLU-only:       {results["G_swiglu_only"]:.4f}  (learned_pos + SwiGLU)')
    print(f'  D modern (ref):      5.9518  (RoPE + SwiGLU)')
    print()

    rope_effect = 6.1936 - results['F_rope_only']
    swiglu_effect = 6.1936 - results['G_swiglu_only']
    combined = 6.1936 - 5.9518
    interaction = combined - rope_effect - swiglu_effect
    print(f'  RoPE contribution:   {rope_effect:+.4f} nats')
    print(f'  SwiGLU contribution: {swiglu_effect:+.4f} nats')
    print(f'  Combined (D):        {combined:+.4f} nats')
    print(f'  Interaction:         {interaction:+.4f} nats')

    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {OUTPUT_DIR}/ablation_results.json')


if __name__ == '__main__':
    main()
