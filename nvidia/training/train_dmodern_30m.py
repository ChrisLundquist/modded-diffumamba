"""Converge D_modern transformer (RoPE + SwiGLU) on MDLM.

Probe result: 5.943 +/- 0.007 at 5K steps (3 seeds), 0.249 nats below baseline.
Expected convergence: ~5.25 at epoch 7.3 (extrapolating 0.25-nat constant offset).

Recipe: Muon lr=0.01 (attention/MLP/SwiGLU) + Adam lr=1e-3 (embeddings/norms), constant LR.
"""

import sys
import os
import time
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
TOTAL_STEPS = 56000  # ~epoch 7.3
WARMUP_STEPS = 200
LOG_EVERY = 2000
CHECKPOINT_EVERY = 5000

MUON_LR = 0.01
EMBED_LR = 1e-3

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'muon_exp/outputs/transformer_converge_v3'


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
    print(f'Recipe: D_modern (RoPE+SwiGLU) + Muon lr={MUON_LR} + Adam lr={EMBED_LR}')

    torch.manual_seed(42)
    model = TransformerV2(
        vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
        use_rope=True, use_swiglu=True,
    ).to(device)
    print(f'Params: {model.param_count()/1e6:.1f}M')

    muon_params, adamw_params = model.param_groups()
    print(f'Muon params: {sum(p.numel() for p in muon_params)/1e6:.1f}M')
    print(f'Adam params: {sum(p.numel() for p in adamw_params)/1e6:.1f}M')

    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)
    optimizers = [adam_opt, muon_opt]

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

    print(f'\nTraining: D_modern + Muon lr={MUON_LR} + Adam lr={EMBED_LR}, '
          f'{TOTAL_STEPS} steps\n')

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
                'adam_state_dict': adam_opt.state_dict(),
                'muon_state_dict': muon_opt.state_dict(),
                'val_loss': val_loss if step % LOG_EVERY == 0 else best_val,
            }, ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

    print(f'\nDone. Best val: {best_val:.4f} at step {best_step} '
          f'(ep {(best_step+1)*tokens_per_step/len(train_tokens):.2f})')
    print(f'Baseline transformer (v2): 5.495 at step 44K')
    print(f'Mamba3: 5.666 at step 56K')
    print(f'Total time: {(time.time()-t0)/3600:.1f} hours')


if __name__ == '__main__':
    main()
