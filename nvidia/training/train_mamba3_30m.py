"""Converge Mamba3 MDLM to epoch 7 with best recipe.

Recipe: Adam lr=1e-3, constant LR, Min-SNR gamma=5, no AdaLN.
Target: match or beat transformer's val 5.88 at epoch 7.3.
"""

import sys
import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hybrid_model import DiffuMambaH
from data import TokenDataset
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

MASK_TOKEN = 50257
MICRO_BATCH = 16
GRAD_ACCUM = 8
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
SEQ_LEN = 1024
TOTAL_STEPS = 56000  # ~epoch 7.3 (matches D_modern comparison)
WARMUP_STEPS = 200
LOG_EVERY = 2000
CHECKPOINT_EVERY = 5000

LR = 1e-3
RESUME_CKPT = None  # set to checkpoint path to resume

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'outputs/mamba3_converge'


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


def eval_mdlm(model, val_x, n_eval_timesteps=20):
    model.eval()
    device = val_x.device
    n_val = val_x.shape[0]
    total_loss = 0.0
    n_points = 0
    k = 5.0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for t_val in torch.linspace(0.05, 0.95, n_eval_timesteps):
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


def main():
    device = 'cuda'
    print('Loading data...')
    all_tokens = np.load(DATA_PATH).astype(np.int64)
    train_tokens = all_tokens[:1_000_000_000]
    val_tokens = all_tokens[1_000_000_000:]
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN
    total_epochs = TOTAL_STEPS * tokens_per_step / len(train_tokens)
    print(f'Plan: {TOTAL_STEPS} steps = {total_epochs:.1f} epochs')
    print(f'Recipe: Adam lr={LR}, constant LR, Min-SNR gamma=5, Mamba3 SISO')

    torch.manual_seed(42)
    # d_state=32 to reduce VRAM (d_state=64 spills 3.4GB to system RAM via PCIe)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    print(f'Params: {model.param_count()/1e6:.1f}M')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)

    start_step = 0
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        print(f'Resuming from {RESUME_CKPT}...')
        ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        print(f'  Resumed at step {start_step}, val {ckpt.get("val_loss", "?"):.4f}')
        del ckpt
        torch.cuda.empty_cache()

    n_val = min(512, len(val_tokens) // (SEQ_LEN + 1))
    val_x = torch.from_numpy(val_tokens[:n_val * SEQ_LEN].reshape(n_val, SEQ_LEN).copy()).to(device)

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(42)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    best_val = float('inf')
    best_step = 0

    print(f'\nTraining: Mamba3 + Adam lr={LR} constant, steps {start_step}→{TOTAL_STEPS}\n')

    for step in range(start_step, TOTAL_STEPS):
        # Warmup only
        if step < WARMUP_STEPS:
            frac = (step + 1) / WARMUP_STEPS
            for g in optimizer.param_groups: g['lr'] = LR * frac

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
                loss = mdlm_loss(model, x) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            val_loss = eval_mdlm(model, val_x)
            elapsed = time.time() - t0
            tps = (step + 1 - start_step) * tokens_per_step / elapsed if elapsed > 0 else 0
            epoch = (step + 1) * tokens_per_step / len(train_tokens)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
            print(f'  step {step:6d} | val {val_loss:.4f} | best {best_val:.4f} | '
                  f'{tps/1e3:.0f}K tok/s | ep {epoch:.2f} | {elapsed:.0f}s', flush=True)

        if (step + 1) % CHECKPOINT_EVERY == 0 or step == TOTAL_STEPS - 1:
            ckpt_path = os.path.join(OUTPUT_DIR, f'checkpoint_{step+1}.pt')
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss if step % LOG_EVERY == 0 else best_val,
            }, ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

    print(f'\nDone. Best val: {best_val:.4f} at step {best_step} '
          f'(ep {(best_step+1)*tokens_per_step/len(train_tokens):.2f})')
    print(f'Transformer baseline: 5.88 at epoch 7.3')
    print(f'Total time: {(time.time()-t0)/3600:.1f} hours')


if __name__ == '__main__':
    main()
