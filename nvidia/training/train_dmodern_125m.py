"""GPT-2 small scale (125M) D_modern on 10B FineWeb-Edu tokens.

Single-epoch training. Our best recipe:
  - RoPE + SwiGLU (D_modern architecture)
  - Muon lr=0.02 + Adam lr=1e-3 (embeddings), constant LR
  - Min-SNR gamma=1.5 (best for Muon from config match experiments)
  - Gradient clipping 1.0 on Adam params

Architecture: 12L x 768d, n_head=12 (GPT-2 small config with RoPE+SwiGLU)
VRAM: ~13GB at MICRO_BATCH=16 (measured by profiler)

10B tokens / (128 * 1024) tokens_per_step = ~76K steps for 1 epoch.
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
WARMUP_STEPS = 1500  # bumped from 500 per feedback
LOG_EVERY = 2000
CHECKPOINT_EVERY = 10000

# WSD schedule: warmup → stable → decay (last 15% to 10% of peak)
DECAY_FRAC = 0.15
DECAY_MIN_FRAC = 0.1

MUON_LR = 0.01  # 0.02 diverged at 125M, halved
EMBED_LR = 1e-3
MIN_SNR_GAMMA = 1.5

N_LAYER = 12
N_EMBD = 768
N_HEAD = 12
VOCAB_SIZE = 50258

DATA_PATH = '/home/clundquist/muon_data/fineweb_10B.npy'
OUTPUT_DIR = 'muon_exp/outputs/125m_10b_dmodern'


def wsd_lr(step, total_steps, peak_lr, warmup=WARMUP_STEPS,
           decay_frac=DECAY_FRAC, decay_min_frac=DECAY_MIN_FRAC):
    """Warmup-Stable-Decay schedule.
    Linear warmup → constant at peak_lr → linear decay to decay_min_frac*peak_lr.
    """
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    decay_start = int(total_steps * (1 - decay_frac))
    if step < decay_start:
        return peak_lr
    # Linear decay from peak_lr to decay_min_frac*peak_lr
    decay_progress = (step - decay_start) / max(1, total_steps - decay_start)
    return peak_lr * (1 - decay_progress * (1 - decay_min_frac))


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
                elbo_weight = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=MIN_SNR_GAMMA)
                mask_float = mask.float()
                per_sample = (per_token_loss * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
                batch_losses.append((per_sample * elbo_weight.squeeze(1)).mean().item())
            total_loss += sum(batch_losses) / len(batch_losses)
            n_points += 1
    return total_loss / n_points


def main():
    device = 'cuda'
    print('Loading data (mmap, no full-RAM copy)...')
    all_tokens = np.load(DATA_PATH, mmap_mode='r')
    n_total = len(all_tokens)
    # Use last 500M tokens for val, rest for train
    val_size = 500_000_000
    # Slices of mmap stay mmap'd — zero RAM until accessed
    train_tokens = all_tokens[:n_total - val_size]
    val_tokens = all_tokens[n_total - val_size:]
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN

    # Train for ~1 epoch
    TOTAL_STEPS = len(train_tokens) // tokens_per_step
    total_epochs = TOTAL_STEPS * tokens_per_step / len(train_tokens)
    print(f'Train: {len(train_tokens)/1e9:.1f}B tokens, Val: {len(val_tokens)/1e6:.0f}M tokens')
    print(f'Plan: {TOTAL_STEPS} steps = {total_epochs:.2f} epochs')
    print(f'Recipe: D_modern (RoPE+SwiGLU+QK-norm) + Muon lr={MUON_LR} + Adam lr={EMBED_LR}')
    print(f'Schedule: WSD (warmup {WARMUP_STEPS}, stable, decay last {int(DECAY_FRAC*100)}% to {int(DECAY_MIN_FRAC*100)}%)')
    print(f'Min-SNR gamma={MIN_SNR_GAMMA}')

    torch.manual_seed(42)
    model = TransformerV2(
        vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        use_rope=True, use_swiglu=True, use_qk_norm=True,
    ).to(device)
    print(f'Params: {model.param_count()/1e6:.1f}M')

    muon_params, adamw_params = model.param_groups()
    print(f'Muon params: {sum(p.numel() for p in muon_params)/1e6:.1f}M')
    print(f'Adam params: {sum(p.numel() for p in adamw_params)/1e6:.1f}M')

    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)
    optimizers = [adam_opt, muon_opt]

    n_val = min(1024, len(val_tokens) // SEQ_LEN)
    val_x = torch.from_numpy(
        val_tokens[:n_val * SEQ_LEN].astype(np.int64).reshape(n_val, SEQ_LEN)
    ).to(device)
    print(f'Val sequences: {n_val}')

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(42)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    best_val = float('inf')
    best_step = 0

    print(f'\nTraining: 125M D_modern, {TOTAL_STEPS} steps\n')

    for step in range(TOTAL_STEPS):
        # WSD schedule for both optimizers
        muon_lr = wsd_lr(step, TOTAL_STEPS, MUON_LR)
        embed_lr = wsd_lr(step, TOTAL_STEPS, EMBED_LR)
        for pg in adam_opt.param_groups:
            pg['lr'] = embed_lr
        for pg in muon_opt.param_groups:
            pg['lr'] = muon_lr

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
    print(f'30M D_modern on 1B (7 epochs): 5.272')
    print(f'Total time: {(time.time()-t0)/3600:.1f} hours')


if __name__ == '__main__':
    main()
