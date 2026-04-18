"""Match the other agent's Muon config exactly on our NVIDIA hardware.

Their config (where Muon works):
  - Muon targets: in_proj ONLY (exclude out_proj)
  - Muon lr=0.02, momentum=0.95, ns_steps=5, wd=0.01
  - Adam lr=3e-4 for non-Muon params
  - Min-SNR gamma=1.5
  - bf16

Our previous Muon probe used: in_proj+out_proj, lr=0.01, wd=0, Adam lr=1.5e-4, gamma=5.
Three differences: param routing, LR, gamma. Test each to isolate the cause.

Configs (5K steps, seed 42):
  1. Adam lr=1e-3, gamma=1.5 (reference from our probe: ~6.36)
  2. Our Muon config (in+out_proj, lr=0.01, Adam 1.5e-4, gamma=1.5): ~6.49
  3. Their Muon config (in_proj only, lr=0.02, Adam 3e-4, gamma=1.5, wd=0.01)
  4. Halfway: our routing + their LRs (in+out_proj, lr=0.02, Adam 3e-4, gamma=1.5)
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hybrid_model import DiffuMambaH
from muon import Muon, newton_schulz_orthogonalize
from data import TokenDataset
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

MASK_TOKEN = 50257
MICRO_BATCH = 16
GRAD_ACCUM = 8
SEQ_LEN = 1024
TOTAL_STEPS = 5000
WARMUP_STEPS = 200
LOG_EVERY = 500
GAMMA = 1.5  # their gamma, not ours

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'muon_exp/outputs/muon_config_match'


class MuonVS(torch.optim.Optimizer):
    """Muon with Variance Scaling (from other agent's implementation).

    Key differences from standard Muon:
    - Tracks per-element variance of gradient vs momentum
    - Normalizes by sqrt(variance) BEFORE Newton-Schulz
    - Uses canonical lerp-based Nesterov momentum
    - Bias correction for early steps
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self):
        self._step += 1
        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                if g.dim() != 2:
                    # Fallback: SGD+momentum for non-2D
                    state = self.state[p]
                    if 'buf' not in state:
                        state['buf'] = torch.zeros_like(g)
                    state['buf'].mul_(beta).add_(g, alpha=1 - beta)
                    p.add_(state['buf'], alpha=-lr)
                    continue

                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                    state['var_buf'] = torch.zeros_like(g)

                buf = state['buf']
                var_buf = state['var_buf']

                # Variance update BEFORE momentum update
                var_buf.mul_(beta).addcmul_(buf - g, buf - g, value=beta * (1 - beta))

                # Momentum update (canonical lerp form)
                buf.lerp_(g, 1 - beta)

                # Bias correction
                bc = 1 - beta ** self._step
                M_hat = buf / bc
                Gamma_hat = var_buf / bc

                # Nesterov lookahead
                M_tilde = g + (beta / (1 - beta)) * M_hat

                # Variance normalization BEFORE NS
                update = M_tilde / (Gamma_hat.clamp(min=0).sqrt() + 1e-6)

                # Newton-Schulz orthogonalization
                if ns_steps > 0:
                    update = newton_schulz_orthogonalize(update, ns_steps)
                    update = update * max(1, g.size(-2) / g.size(-1)) ** 0.5

                wd = group.get('weight_decay', 0)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)


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
    elbo_weight = elbo_weight.clamp(max=GAMMA)
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
                elbo_weight = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=GAMMA)
                mask_float = mask.float()
                per_sample = (per_token_loss * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
                batch_losses.append((per_sample * elbo_weight.squeeze(1)).mean().item())
            total_loss += sum(batch_losses) / len(batch_losses)
            n_points += 1
    return total_loss / n_points


def get_muon_params_inproj_only(model):
    """Route only in_proj to Muon (exclude out_proj, matching other agent)."""
    muon_params, adam_params = [], []
    muon_names, adam_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Only in_proj goes to Muon
        if 'in_proj' in name and p.dim() == 2:
            muon_params.append(p)
            muon_names.append(name)
        else:
            adam_params.append(p)
            adam_names.append(name)
    return muon_params, adam_params, muon_names, adam_names


def run_config(name, model, optimizers, train_tokens, val_x, device):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(42)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)
    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

    t0 = time.time()
    best_val = float('inf')

    for step in range(TOTAL_STEPS):
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
            print(f'  [{name}] step {step:5d} | val {val_loss:.4f} | '
                  f'{tps/1e3:.0f}K tok/s | {elapsed:.0f}s', flush=True)

    final = val_loss
    del model, optimizers
    torch.cuda.empty_cache()
    return final


def main():
    device = 'cuda'
    print('Loading data...')
    all_tokens = np.load(DATA_PATH).astype(np.int64)
    train_tokens = all_tokens[:1_000_000_000]
    val_tokens = all_tokens[1_000_000_000:]
    n_val = min(512, len(val_tokens) // (SEQ_LEN + 1))
    val_x = torch.from_numpy(val_tokens[:n_val * SEQ_LEN].reshape(n_val, SEQ_LEN).copy()).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configs 1-3 completed before restart — hardcode results
    results = {
        'adam_ref': 3.7035,
        'our_muon': 3.7388,
        'their_muon': 3.6927,
    }
    print('Resuming from restart. Previous results:')
    for name, val in results.items():
        print(f'  {name}: {val:.4f}')

    # Config 4: Hybrid (our routing + their LRs)
    torch.manual_seed(42)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    muon_p, adam_p, mn, an = model.param_groups()
    adam_opt = torch.optim.Adam([{'params': adam_p, 'lr': 3e-4, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_p}], lr=0.02, momentum=0.95, ns_steps=5)
    results['hybrid'] = run_config('Hybrid (in+out, lr=0.02, Adam 3e-4)', model,
                                    [adam_opt, muon_opt], train_tokens, val_x, device)

    # Config 5: Muon-VS with their routing (in_proj only, lr=0.02, Adam 3e-4)
    torch.manual_seed(42)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    muon_p, adam_p, mn, an = get_muon_params_inproj_only(model)
    print(f'  VS routing: {len(mn)} Muon tensors, {sum(p.numel() for p in muon_p)/1e6:.1f}M')
    adam_opt = torch.optim.Adam([{'params': adam_p, 'lr': 3e-4, 'betas': (0.9, 0.999)}])
    muon_opt = MuonVS([{'params': muon_p}], lr=0.02, momentum=0.95, ns_steps=5)
    results['muon_vs'] = run_config('Muon-VS (in only, lr=0.02)', model,
                                     [adam_opt, muon_opt], train_tokens, val_x, device)

    # Summary
    print(f'\n{"="*60}')
    print(f'MUON CONFIG MATCH (gamma={GAMMA}, 5K steps, seed 42)')
    print(f'{"="*60}')
    for name, val in results.items():
        gap = val - results['adam_ref']
        print(f'  {name:40s}: {val:.4f} ({gap:+.4f} vs Adam)')


if __name__ == '__main__':
    main()
