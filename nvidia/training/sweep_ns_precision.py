"""Validate Muon NS precision hypothesis.

Investigation found: Muon's Newton-Schulz uses bf16 (muon.py line 21),
which causes numerical instability on small singular values from Mamba3 gradients.
Hypothesis: fp32 NS fixes the issue and makes Muon competitive with Adam on NVIDIA.

Test: 4 configs on Mamba3, 5K steps, seed 42.
  1. Adam lr=1e-3 (reference, our current best)
  2. Muon bf16 NS (current, broken on NVIDIA?)
  3. Muon fp32 NS (proposed fix)
  4. Muon fp16 NS (intermediate — more mantissa bits than bf16, no subnormal flush)
"""

import sys
import os
import time
import copy
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
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
SEQ_LEN = 1024
TOTAL_STEPS = 5000
WARMUP_STEPS = 200
LOG_EVERY = 500

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
OUTPUT_DIR = 'muon_exp/outputs/muon_fp32_test'


# Patched NS with configurable precision
def newton_schulz_typed(G, ns_steps=5, dtype=torch.float32):
    """Newton-Schulz with explicit dtype control."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonTyped(Muon):
    """Muon with configurable NS precision."""
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, ns_dtype=torch.float32):
        super().__init__(params, lr=lr, momentum=momentum, ns_steps=ns_steps)
        self.ns_dtype = ns_dtype

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() != 2:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    p.add_(buf, alpha=-lr)
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf_prev = buf.clone()
                buf.mul_(momentum).add_(g)
                g_nesterov = buf + momentum * (buf - buf_prev)

                if ns_steps > 0:
                    update = newton_schulz_typed(g_nesterov, ns_steps, dtype=self.ns_dtype)
                    update = update * max(1, g.size(-2) / g.size(-1)) ** 0.5
                else:
                    update = g_nesterov

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
    tokens_per_step = EFFECTIVE_BATCH * SEQ_LEN

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

    results = {}

    # Config 1: Adam (reference)
    torch.manual_seed(42)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    results['adam'] = run_config('Adam lr=1e-3', model, [opt], train_tokens, val_x, device)

    # Config 2: Muon bf16 NS (current)
    torch.manual_seed(42)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    muon_p, adam_p, _, _ = model.param_groups()
    adam_opt = torch.optim.AdamW([{'params': adam_p, 'lr': 1.5e-4, 'betas': (0.9, 0.95)}])
    muon_opt = MuonTyped([{'params': muon_p}], lr=0.01, momentum=0.95, ns_steps=5,
                          ns_dtype=torch.bfloat16)
    results['muon_bf16'] = run_config('Muon bf16 NS', model, [adam_opt, muon_opt],
                                       train_tokens, val_x, device)

    # Config 3: Muon fp16 NS (intermediate)
    torch.manual_seed(42)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    muon_p, adam_p, _, _ = model.param_groups()
    adam_opt = torch.optim.AdamW([{'params': adam_p, 'lr': 1.5e-4, 'betas': (0.9, 0.95)}])
    muon_opt = MuonTyped([{'params': muon_p}], lr=0.01, momentum=0.95, ns_steps=5,
                          ns_dtype=torch.float16)
    results['muon_fp16'] = run_config('Muon fp16 NS', model, [adam_opt, muon_opt],
                                       train_tokens, val_x, device)

    # Config 4: Muon fp32 NS (full fix)
    torch.manual_seed(42)
    model = DiffuMambaH(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                        attn_positions=set(), mamba_version=3, d_state=32).to(device)
    muon_p, adam_p, _, _ = model.param_groups()
    adam_opt = torch.optim.AdamW([{'params': adam_p, 'lr': 1.5e-4, 'betas': (0.9, 0.95)}])
    muon_opt = MuonTyped([{'params': muon_p}], lr=0.01, momentum=0.95, ns_steps=5,
                          ns_dtype=torch.float32)
    results['muon_fp32'] = run_config('Muon fp32 NS', model, [adam_opt, muon_opt],
                                       train_tokens, val_x, device)

    # Summary
    print(f'\n{"="*60}')
    print('MUON NS PRECISION TEST (Mamba3, 5K steps, seed 42)')
    print(f'{"="*60}')
    for name, val in results.items():
        print(f'  {name:15s}: {val:.4f}')
    print(f'\n  bf16→fp16 delta: {results["muon_bf16"] - results["muon_fp16"]:+.4f} nats')
    print(f'  bf16→fp32 delta: {results["muon_bf16"] - results["muon_fp32"]:+.4f} nats')
    print(f'  fp16 vs Adam:    {results["muon_fp16"] - results["adam"]:+.4f} nats')
    print(f'  fp32 vs Adam:    {results["muon_fp32"] - results["adam"]:+.4f} nats')


if __name__ == '__main__':
    main()
