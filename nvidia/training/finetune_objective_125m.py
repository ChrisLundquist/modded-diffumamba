"""Unified objective-ablation fine-tune for 125M D_modern.

Fine-tunes from a baseline checkpoint (default: step 40k) for N more steps with
one of several objective variants — all measured on the same generation harness.

Variants:
  baseline       — vanilla Min-SNR γ=1.5 (control; should reproduce trajectory)
  papl           — Peng 2025 PAPL self-planner reweight (α, τ as flags)
  gamma_decay    — Min-SNR γ linearly decayed γ_start → γ_end across this run
  t_curriculum   — t-sampling interpolated Uniform → Beta(2,2) across this run
  papl_t_curr    — combine PAPL + t-curriculum (synergy probe)

This complements finetune_papl_125m.py (which is hard-coded to PAPL) by giving
us a single entry point for Exps 1–3 of experiments_plan.md, all sharing the
same base checkpoint, eval seed, dataloader seed, and sampler.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformer_v2 import TransformerV2  # noqa: E402
from muon import Muon  # noqa: E402
from data import TokenDataset  # noqa: E402

torch.set_float32_matmul_precision('high')

MASK_TOKEN = 50257
MICRO_BATCH = 16
GRAD_ACCUM = 8
SEQ_LEN = 1024
LOG_EVERY = 500
CHECKPOINT_EVERY = 5000

MUON_LR = 0.01
EMBED_LR = 1e-3

N_LAYER = 12
N_EMBD = 768
N_HEAD = 12
VOCAB_SIZE = 50258

DATA_PATH = '/home/clundquist/muon_data/fineweb_10B.npy'
DEFAULT_RESUME = '/mnt/d/code/gpt-slide/muon_exp/outputs/125m_10b_dmodern/checkpoint_40000.pt'


def sample_t(B, device, step, total_steps, mode):
    """Sample noise level t in (0, 1).

    'uniform' — t ~ U(0.05, 0.95)
    't_curriculum' — interpolate from Uniform → Beta(2,2) (concentrated around 0.5)
                     mixing weight = step / total_steps.
    """
    t_uniform = torch.rand(B, 1, device=device) * 0.95 + 0.05
    if mode == 'uniform':
        return t_uniform
    if mode == 't_curriculum':
        beta = torch.distributions.Beta(2.0, 2.0)
        t_beta = beta.sample((B, 1)).to(device).clamp(0.05, 0.95)
        w = min(1.0, step / max(1, total_steps))
        return (1 - w) * t_uniform + w * t_beta
    raise ValueError(f'Unknown t-sampling mode: {mode}')


def gamma_at_step(step, total_steps, mode, gamma_start, gamma_end):
    """Min-SNR γ schedule. 'constant' returns gamma_start; 'gamma_decay' interpolates."""
    if mode == 'constant':
        return gamma_start
    if mode == 'gamma_decay':
        w = min(1.0, step / max(1, total_steps))
        return gamma_start * (1 - w) + gamma_end * w
    raise ValueError(f'Unknown gamma mode: {mode}')


def objective_loss(model, x, *, gamma, papl_alpha, papl_tau, t):
    """Compute MDLM loss with optional PAPL reweighting.

    gamma > 0 applies Min-SNR clamp; <=0 disables (raw ELBO weight).
    papl_alpha = 0 disables PAPL (recovers vanilla MDLM).
    """
    B, T = x.shape
    device = x.device
    k = 5.0
    alpha_t = 1.0 - torch.exp(-k * t)
    mask = torch.rand(B, T, device=device) < alpha_t
    mask_tokens = torch.full_like(x, MASK_TOKEN)
    x_t = torch.where(mask, mask_tokens, x)
    logits = model(x_t, causal=False)[..., :MASK_TOKEN]
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction='none'
    ).reshape(B, T)

    if papl_alpha > 0:
        with torch.no_grad():
            logprobs = F.log_softmax(logits, dim=-1)
            max_logprob = logprobs.max(dim=-1).values
            scores = max_logprob / papl_tau
            scores = scores.masked_fill(~mask, -1e4)
            w = F.softmax(scores, dim=1) * mask.float()
        papl_weight = 1.0 + papl_alpha * w
    else:
        papl_weight = torch.ones_like(per_token_loss)

    mask_float = mask.float()
    weighted = per_token_loss * mask_float * papl_weight
    per_sample = weighted.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

    elbo_weight = k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))
    if gamma > 0:
        elbo_weight = elbo_weight.clamp(max=gamma)
    return (per_sample * elbo_weight.squeeze(1)).mean()


@torch.no_grad()
def eval_std_minsnr(model, val_x, gamma_eval=1.5):
    """Standard Min-SNR γ eval (always γ=1.5 for trajectory comparability)."""
    model.eval()
    device = val_x.device
    n_val = val_x.shape[0]
    total_loss = 0.0
    n_points = 0
    k = 5.0
    eval_rng = torch.Generator(device=device); eval_rng.manual_seed(0)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for t_val in torch.linspace(0.05, 0.95, 20):
            batch_losses = []
            for i in range(0, n_val, MICRO_BATCH):
                vx = val_x[i:i + MICRO_BATCH]
                bs = vx.shape[0]
                t = torch.full((bs, 1), t_val.item(), device=device)
                alpha_t = 1.0 - torch.exp(-k * t)
                mask = torch.rand(bs, vx.shape[1], device=device, generator=eval_rng) < alpha_t
                mask_tokens = torch.full_like(vx, MASK_TOKEN)
                x_t = torch.where(mask, mask_tokens, vx)
                logits = model(x_t, causal=False)[..., :MASK_TOKEN]
                per_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), vx.reshape(-1), reduction='none'
                ).reshape(bs, -1)
                ew = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=gamma_eval)
                mf = mask.float()
                ps = (per_token_loss * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                batch_losses.append((ps * ew.squeeze(1)).mean().item())
            total_loss += sum(batch_losses) / len(batch_losses)
            n_points += 1
    return total_loss / n_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objective', required=True,
                    choices=['baseline', 'papl', 'gamma_decay', 't_curriculum', 'papl_t_curr'])
    ap.add_argument('--extra-steps', type=int, default=10000)
    ap.add_argument('--resume', type=str, default=DEFAULT_RESUME)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--gamma-start', type=float, default=1.5)
    ap.add_argument('--gamma-end', type=float, default=0.5)
    ap.add_argument('--papl-alpha', type=float, default=1.0)
    ap.add_argument('--papl-tau', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42, help='Affects dataloader shuffle only')
    args = ap.parse_args()

    # Map objective → (gamma_mode, t_mode, papl_alpha)
    if args.objective == 'baseline':
        gamma_mode, t_mode, papl_a = 'constant', 'uniform', 0.0
    elif args.objective == 'papl':
        gamma_mode, t_mode, papl_a = 'constant', 'uniform', args.papl_alpha
    elif args.objective == 'gamma_decay':
        gamma_mode, t_mode, papl_a = 'gamma_decay', 'uniform', 0.0
    elif args.objective == 't_curriculum':
        gamma_mode, t_mode, papl_a = 'constant', 't_curriculum', 0.0
    elif args.objective == 'papl_t_curr':
        gamma_mode, t_mode, papl_a = 'constant', 't_curriculum', args.papl_alpha
    else:
        raise ValueError(args.objective)

    device = 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Objective: {args.objective}  (gamma_mode={gamma_mode}, t_mode={t_mode}, '
          f'papl_alpha={papl_a}, gamma_start={args.gamma_start}, gamma_end={args.gamma_end})')
    print(f'Resume from: {args.resume}')
    print(f'Output dir: {args.output_dir}')

    print('Loading data (mmap)...')
    all_tokens = np.load(DATA_PATH, mmap_mode='r')
    n_total = len(all_tokens)
    val_size = 500_000_000
    train_tokens = all_tokens[:n_total - val_size]
    val_tokens = all_tokens[n_total - val_size:]

    torch.manual_seed(args.seed)
    model = TransformerV2(vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD,
                          n_embd=N_EMBD, use_rope=True, use_swiglu=True,
                          use_qk_norm=True).to(device)
    print(f'Params: {model.param_count()/1e6:.1f}M')

    muon_params, adamw_params = model.param_groups()
    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    adam_opt.load_state_dict(ckpt['adam_state_dict'])
    muon_opt.load_state_dict(ckpt['muon_state_dict'])
    start_step = ckpt['step']
    baseline_val = ckpt.get('val_loss', None)
    print(f'  resumed at step {start_step}, ckpt val_loss {baseline_val}')
    del ckpt
    torch.cuda.empty_cache()

    n_val = min(1024, len(val_tokens) // SEQ_LEN)
    val_x = torch.from_numpy(
        val_tokens[:n_val * SEQ_LEN].astype(np.int64).reshape(n_val, SEQ_LEN)
    ).to(device)

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(args.seed * 1000 + 7)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)

    vloss0 = eval_std_minsnr(model, val_x, gamma_eval=args.gamma_start)
    print(f'Re-verified baseline val (std γ={args.gamma_start}): {vloss0:.4f}')

    # Constant LR for the fine-tune window
    for pg in adam_opt.param_groups:
        pg['lr'] = EMBED_LR
    for pg in muon_opt.param_groups:
        pg['lr'] = MUON_LR

    best_val = vloss0
    t0 = time.time()
    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

    for local_step in range(args.extra_steps):
        step = start_step + local_step
        gamma = gamma_at_step(local_step, args.extra_steps, gamma_mode,
                              args.gamma_start, args.gamma_end)

        model.train()
        for opt in [adam_opt, muon_opt]:
            opt.zero_grad()
        total_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)
            B = x.shape[0]
            t = sample_t(B, device, local_step, args.extra_steps, t_mode)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = objective_loss(model, x, gamma=gamma, papl_alpha=papl_a,
                                      papl_tau=args.papl_tau, t=t) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(adamw_params, 1.0)
        for opt in [adam_opt, muon_opt]:
            opt.step()

        if local_step % LOG_EVERY == 0 or local_step == args.extra_steps - 1:
            vloss = eval_std_minsnr(model, val_x, gamma_eval=args.gamma_start)
            elapsed = time.time() - t0
            tps = (local_step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            if vloss < best_val:
                best_val = vloss
            print(f'  step {step:6d} (+{local_step+1:5d}) | γ={gamma:.3f} | '
                  f'train {total_loss:.4f} | val(γ={args.gamma_start}) {vloss:.4f} | '
                  f'best {best_val:.4f} | {tps/1e3:.0f}K tok/s | {elapsed:.0f}s', flush=True)

        if (local_step + 1) % CHECKPOINT_EVERY == 0 or local_step == args.extra_steps - 1:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_{step+1}.pt')
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'adam_state_dict': adam_opt.state_dict(),
                'muon_state_dict': muon_opt.state_dict(),
                'val_loss': vloss if local_step % LOG_EVERY == 0 else best_val,
                'objective': args.objective,
                'gamma_mode': gamma_mode, 't_mode': t_mode,
                'papl_alpha': papl_a, 'papl_tau': args.papl_tau,
                'gamma_start': args.gamma_start, 'gamma_end': args.gamma_end,
                'resumed_from': args.resume,
                'seed': args.seed,
            }, ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

    print(f'\nDone. Best val (std γ={args.gamma_start}): {best_val:.4f}  '
          f'(baseline at start: {vloss0:.4f})')
    print(f'Total time: {(time.time()-t0)/3600:.2f} hours')


if __name__ == '__main__':
    main()
