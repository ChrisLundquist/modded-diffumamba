"""Standard NLL evaluation for MDLM checkpoints.

Computes per-token ELBO upper bound on NLL using the proper continuous-time
ELBO weight: w(t) = k * (1 - exp(-kt)), where k=5 and alpha(t) = 1-exp(-kt).

Derivation: The MDLM continuous-time ELBO (Sahoo et al. 2024, Eq. 10) is:
  L = integral_0^1 [alpha'(t)/(1-alpha(t))] * E[sum_i 1(xt_i=m) * CE_i] dt / T
For alpha(t)=1-exp(-kt): alpha'(t)/(1-alpha(t)) = k (constant).
The expected number of masked tokens at time t is T*alpha(t).
So per-token ELBO = E_t[k * alpha(t) * mean_CE_over_masked(t)].

The training metric uses k*exp(-kt)/(1-exp(-kt)) clamped at 5 (Min-SNR),
which upweights low-mask timesteps. This is good for training signal but
is NOT an upper bound on NLL.

Uses 1000 timesteps for tight Monte Carlo estimate.
Reports nats/token and bits/token.
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
from transformer_v2 import TransformerV2
from hybrid_model import DiffuMambaH

MASK_TOKEN = 50257
SEQ_LEN = 1024
EVAL_BATCH = 64
K = 5.0
N_TIMESTEPS = 1000

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'

CHECKPOINTS = {
    'd_modern_125m_10b': {
        'path': 'muon_exp/outputs/125m_10b_dmodern/checkpoint_72479.pt',
        'model_cls': 'TransformerV2',
        'model_kwargs': dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                             use_rope=True, use_swiglu=True, use_qk_norm=True),
    },
    'd_modern_30m_1b': {
        'path': 'muon_exp/outputs/transformer_converge_v3/checkpoint_56000.pt',
        'model_cls': 'TransformerV2',
        'model_kwargs': dict(vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
                             use_rope=True, use_swiglu=True),
    },
    'mamba3_30m_1b': {
        'path': 'muon_exp/outputs/mamba3_converge/checkpoint_56000.pt',
        'model_cls': 'DiffuMambaH',
        'model_kwargs': dict(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                             attn_positions=set(), mamba_version=3, d_state=32),
    },
}


def eval_elbo(model, val_x, device):
    """Compute proper ELBO, Min-SNR weighted, and 1/t ELBO metrics.

    Returns dict with:
        elbo_nats: proper ELBO per token (upper bound on NLL)
        minsnr_nats: Min-SNR gamma=5 weighted metric (matches training eval)
        inv_t_nats: 1/t ELBO weighted metric (other agent's eval metric)
        per_timestep: diagnostics at key timesteps
    """
    model.eval()
    n_val = val_x.shape[0]
    timesteps = torch.linspace(0.01, 0.99, N_TIMESTEPS)

    elbo_terms = []
    minsnr_terms = []
    inv_t_terms = []
    diagnostics = []

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for ti, t_val in enumerate(timesteps):
            t = t_val.item()
            alpha = 1.0 - np.exp(-K * t)

            # Proper ELBO weight: k * alpha(t)
            w_elbo = K * alpha
            # Min-SNR weight: clamp(k*exp(-kt)/(1-exp(-kt)), 5)
            w_minsnr = min(K * np.exp(-K * t) / alpha, 5.0)
            # 1/t ELBO weight (other agent's metric)
            w_inv_t = 1.0 / max(t, 0.01)

            batch_ces = []
            for i in range(0, n_val, EVAL_BATCH):
                vx = val_x[i:i + EVAL_BATCH]
                bs = vx.shape[0]

                mask = torch.rand(bs, SEQ_LEN, device=device) < alpha
                mask_tokens = torch.full_like(vx, MASK_TOKEN)
                x_t = torch.where(mask, mask_tokens, vx)
                logits = model(x_t, causal=False)[..., :MASK_TOKEN]

                per_token_ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    vx.reshape(-1), reduction='none'
                ).reshape(bs, SEQ_LEN)

                mask_float = mask.float()
                n_masked = mask_float.sum(dim=1).clamp(min=1.0)
                per_sample_ce = (per_token_ce * mask_float).sum(dim=1) / n_masked
                batch_ces.append(per_sample_ce.mean().item())

            mean_ce = np.mean(batch_ces)
            elbo_terms.append(w_elbo * mean_ce)
            minsnr_terms.append(w_minsnr * mean_ce)
            inv_t_terms.append(w_inv_t * mean_ce)
            diagnostics.append((t, alpha, mean_ce, w_elbo * mean_ce, w_minsnr * mean_ce))

            if (ti + 1) % 200 == 0:
                running_elbo = np.mean(elbo_terms)
                print(f'    {ti+1}/{N_TIMESTEPS} | running ELBO {running_elbo:.4f}', flush=True)

    elbo_nats = np.mean(elbo_terms)
    minsnr_nats = np.mean(minsnr_terms)
    inv_t_nats = np.mean(inv_t_terms)

    # Per-timestep diagnostics at key points
    key_t = [0.1, 0.3, 0.5, 0.7, 0.9]
    key_indices = [int(t * N_TIMESTEPS / 1.0) for t in key_t]
    key_diagnostics = {}
    for idx in key_indices:
        if idx < len(diagnostics):
            t, alpha, ce, _, _ = diagnostics[min(idx, len(diagnostics) - 1)]
            key_diagnostics[f't={t:.2f}'] = {'alpha': alpha, 'ce': ce}

    return {
        'elbo_nats': float(elbo_nats),
        'elbo_bits': float(elbo_nats / np.log(2)),
        'minsnr_nats': float(minsnr_nats),
        'minsnr_bits': float(minsnr_nats / np.log(2)),
        'inv_t_nats': float(inv_t_nats),
        'inv_t_bits': float(inv_t_nats / np.log(2)),
        'diagnostics': key_diagnostics,
    }


def main():
    device = 'cuda'
    torch.set_float32_matmul_precision('high')

    print('Loading validation data...')
    all_tokens = np.load(DATA_PATH).astype(np.int64)
    val_tokens = all_tokens[1_000_000_000:]
    n_val = min(512, len(val_tokens) // SEQ_LEN)
    val_x = torch.from_numpy(
        val_tokens[:n_val * SEQ_LEN].reshape(n_val, SEQ_LEN).copy()
    ).to(device)
    print(f'{n_val} val sequences ({n_val * SEQ_LEN / 1e6:.1f}M tokens)')

    results = {}

    for name, cfg in CHECKPOINTS.items():
        print(f'\n{"="*60}')
        print(f'  {name}')
        print(f'{"="*60}')
        print(f'  Loading {cfg["path"]}...')

        if cfg['model_cls'] == 'GPT2':
            model = GPT2(**cfg['model_kwargs']).to(device)
        elif cfg['model_cls'] == 'TransformerV2':
            model = TransformerV2(**cfg['model_kwargs']).to(device)
        else:
            model = DiffuMambaH(**cfg['model_kwargs']).to(device)

        ckpt = torch.load(cfg['path'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        step = ckpt['step']
        print(f'  Step {step}, {model.param_count()/1e6:.1f}M params')
        del ckpt
        torch.cuda.empty_cache()

        t0 = time.time()
        r = eval_elbo(model, val_x, device)
        elapsed = time.time() - t0

        results[name] = r
        print(f'\n  ELBO (proper):  {r["elbo_nats"]:.4f} nats/tok  ({r["elbo_bits"]:.4f} bits/tok)')
        print(f'  MinSNR (train): {r["minsnr_nats"]:.4f} nats/tok  ({r["minsnr_bits"]:.4f} bits/tok)')
        print(f'  1/t ELBO:       {r["inv_t_nats"]:.4f} nats/tok  ({r["inv_t_bits"]:.4f} bits/tok)')
        print(f'  Time: {elapsed:.0f}s')
        print(f'  Per-timestep CE:')
        for tk, tv in r['diagnostics'].items():
            print(f'    {tk}: alpha={tv["alpha"]:.3f}, CE={tv["ce"]:.4f}')

        del model
        torch.cuda.empty_cache()

    # Comparison
    names = list(results.keys())
    if len(names) >= 2:
        print(f'\n{"="*60}')
        print(f'  COMPARISON (proper ELBO)')
        print(f'{"="*60}')
        sorted_names = sorted(names, key=lambda n: results[n]['elbo_nats'])
        best = sorted_names[0]
        for n in sorted_names:
            gap = results[n]['elbo_nats'] - results[best]['elbo_nats']
            print(f'  {n:25s}: {results[n]["elbo_nats"]:.4f} nats  ({gap:+.4f} vs best)')
        print(f'\n  MinSNR metric:')
        best_minsnr = sorted(names, key=lambda n: results[n]['minsnr_nats'])[0]
        for n in sorted_names:
            gap = results[n]['minsnr_nats'] - results[best_minsnr]['minsnr_nats']
            print(f'  {n:25s}: {results[n]["minsnr_nats"]:.4f} nats  ({gap:+.4f} vs best)')
        print(f'\n  1/t ELBO (other agent metric):')
        best_inv_t = sorted(names, key=lambda n: results[n]['inv_t_nats'])[0]
        for n in sorted_names:
            gap = results[n]['inv_t_nats'] - results[best_inv_t]['inv_t_nats']
            print(f'  {n:25s}: {results[n]["inv_t_nats"]:.4f} nats  ({gap:+.4f} vs best)')

    # Save
    output_path = 'muon_exp/outputs/standard_nll_eval.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
