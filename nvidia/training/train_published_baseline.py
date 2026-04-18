"""Faithful reproduction of published MDLM recipe at our scale.

Matches Sahoo et al. (NeurIPS 2024 + scaling paper Feb 2026):
  - RoPE + GELU + 6-way AdaLN (DiT backbone)
  - AdamW (beta1=0.9, beta2=0.95, wd=0.1, lr=4e-4)
  - Cosine LR schedule
  - Gradient clipping 1.0 on ALL params

Then: same model but with SwiGLU instead of GELU to isolate SwiGLU contribution.

This anchors our numbers to the literature. Without this, a reviewer will ask
"how do you know your improvements aren't just fixing a broken baseline?"
"""

import sys
import os
import time
import json
import math
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformer_v2 import TransformerV2
from data import TokenDataset
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

MASK_TOKEN = 50257
MICRO_BATCH = 16
GRAD_ACCUM = 8
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
SEQ_LEN = 1024
TOTAL_STEPS = 56000  # ~epoch 7.3
WARMUP_STEPS = 200
LOG_EVERY = 2000
CHECKPOINT_EVERY = 5000

# Published recipe
LR = 4e-4
BETA1 = 0.9
BETA2 = 0.95
WEIGHT_DECAY = 0.1

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_BASE = 'muon_exp/outputs'


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def mdlm_loss(model, x, use_adaln=False):
    B, T = x.shape
    device = x.device
    t = torch.rand(B, 1, device=device) * 0.95 + 0.05
    k = 5.0
    alpha = 1.0 - torch.exp(-k * t)
    mask = torch.rand(B, T, device=device) < alpha
    mask_tokens = torch.full_like(x, MASK_TOKEN)
    x_t = torch.where(mask, mask_tokens, x)
    if use_adaln:
        logits = model(x_t, causal=False, t=t.squeeze(1))[..., :MASK_TOKEN]
    else:
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


def eval_mdlm(model, val_x, use_adaln=False):
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
                if use_adaln:
                    logits = model(x_t, causal=False, t=t.squeeze(1))[..., :MASK_TOKEN]
                else:
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


def run_config(name, model_kwargs, train_tokens, val_x, device, output_dir):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')

    use_adaln = model_kwargs.get('use_adaln', False)

    torch.manual_seed(42)
    model = TransformerV2(**model_kwargs).to(device)
    print(f'  Params: {model.param_count()/1e6:.1f}M')
    print(f'  Recipe: AdamW lr={LR}, beta2={BETA2}, wd={WEIGHT_DECAY}, cosine LR')
    print(f'  AdaLN: {use_adaln}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)

    n_val = min(512, len(val_x) // SEQ_LEN)
    # val_x already prepared by caller

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(42)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    best_val = float('inf')
    best_step = 0

    print(f'\n  Training: {TOTAL_STEPS} steps\n')

    for step in range(TOTAL_STEPS):
        # Cosine LR with warmup
        lr = cosine_lr(step, WARMUP_STEPS, TOTAL_STEPS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = mdlm_loss(model, x, use_adaln=use_adaln) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        # Clip ALL params (published recipe)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            val_loss = eval_mdlm(model, val_x, use_adaln=use_adaln)
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            epoch = (step + 1) * tokens_per_step / len(train_tokens)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
            print(f'  step {step:6d} | val {val_loss:.4f} | best {best_val:.4f} | '
                  f'lr {lr:.2e} | {tps/1e3:.0f}K tok/s | ep {epoch:.2f} | '
                  f'{elapsed:.0f}s', flush=True)

        if (step + 1) % CHECKPOINT_EVERY == 0 or step == TOTAL_STEPS - 1:
            ckpt_path = os.path.join(output_dir, f'checkpoint_{step+1}.pt')
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss if step % LOG_EVERY == 0 else best_val,
            }, ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

    print(f'\n  Done. Best val: {best_val:.4f} at step {best_step} '
          f'(ep {(best_step+1)*tokens_per_step/len(train_tokens):.2f})')
    print(f'  D_modern (our best): 5.272 at epoch 7.34')

    del model, optimizer
    torch.cuda.empty_cache()
    return best_val, best_step


def main():
    device = 'cuda'
    print('Loading data...')
    all_tokens = np.load(DATA_PATH).astype(np.int64)
    train_tokens = all_tokens[:1_000_000_000]
    val_tokens = all_tokens[1_000_000_000:]
    n_val = min(512, len(val_tokens) // (SEQ_LEN + 1))
    val_x = torch.from_numpy(val_tokens[:n_val * SEQ_LEN].reshape(n_val, SEQ_LEN).copy()).to(device)
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN
    total_epochs = TOTAL_STEPS * tokens_per_step / len(train_tokens)
    print(f'Plan: {TOTAL_STEPS} steps = {total_epochs:.1f} epochs')

    results = {}

    # Config 1: Published MDLM baseline (RoPE + GELU + AdaLN + AdamW cosine)
    val1, step1 = run_config(
        'published_baseline',
        dict(n_layer=6, n_head=6, n_embd=384, use_rope=True, use_adaln=True),
        train_tokens, val_x, device,
        os.path.join(OUTPUT_BASE, 'published_baseline'))
    results['published_baseline'] = {'best_val': val1, 'best_step': step1}

    # Config 2: Published baseline + SwiGLU (isolate SwiGLU contribution)
    val2, step2 = run_config(
        'published_swiglu',
        dict(n_layer=6, n_head=6, n_embd=384, use_rope=True, use_swiglu=True, use_adaln=True),
        train_tokens, val_x, device,
        os.path.join(OUTPUT_BASE, 'published_swiglu'))
    results['published_swiglu'] = {'best_val': val2, 'best_step': step2}

    # Summary
    print(f'\n{"="*60}')
    print('PUBLISHED BASELINE COMPARISON')
    print(f'{"="*60}')
    print(f'  Published (RoPE+GELU+AdaLN+AdamW):   {results["published_baseline"]["best_val"]:.4f}')
    print(f'  Published + SwiGLU:                    {results["published_swiglu"]["best_val"]:.4f}')
    print(f'  SwiGLU improvement:                    {results["published_baseline"]["best_val"] - results["published_swiglu"]["best_val"]:+.4f} nats')
    print(f'  D_modern (Muon+RoPE+SwiGLU, no AdaLN): 5.272')

    with open(os.path.join(OUTPUT_BASE, 'published_baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results.')


if __name__ == '__main__':
    main()
