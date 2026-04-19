"""PAPL training-vs-inference alignment diagnostics.

Three diagnostics from the reviewer's priority list:

1. Planner-sampler rank correlation. PAPL's training planner ranks masked
   positions by log p(x_0^i | x_t) (the gt-token logprob). The inference
   sampler ranks by max_j p(j | x_t) (max-over-vocab confidence). Spearman
   correlation between these orderings is the most direct measure of "is
   PAPL aligned to the actual sampler?" Rising = good.

2. T-sweep gap on gen probe. Peng's Fig 3(b) shows PAPL's gains are larger
   at small sampling budgets. We probe rep_4/distinct_4 at T ∈ {32,64,128}
   on a held-out prompt set and report the gap between PAPL@N and std@N.
   Gap widening at small T = PAPL-specific benefit.

3. Calibration (ECE) under the mask distribution. The dangerous failure mode
   for self-distillation losses is confidence collapse: model becomes
   overconfident, planner amplifies, vicious cycle. Bin masked positions by
   max-prob, measure empirical accuracy. Compare PAPL ckpt to baseline.

Usage:
    python papl_diagnostics.py --models d_modern_125m_40k d_modern_125m_papl_45k
"""

import argparse
import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..', 'src'))
from adapters.mdlm_adapter import MDLMAdapter, MODEL_SPECS  # noqa: E402
from samplers.mdlm_topk import demask_topk_prefix  # noqa: E402
from metrics.diversity import distinct_n  # noqa: E402
from metrics.repetition import rep_n  # noqa: E402

DATA_PATH = '/home/clundquist/muon_data/fineweb_10B.npy'
PROMPTS_PATH = os.path.join(THIS_DIR, 'prompts', 'fineweb_edu_held.pt')
SEQ_LEN = 1024
MASK_TOKEN = 50257


def load_val_x(device, n_val=256):
    all_tokens = np.load(DATA_PATH, mmap_mode='r')
    val_tokens = all_tokens[len(all_tokens) - 500_000_000:]
    val_x = torch.from_numpy(
        val_tokens[:n_val * SEQ_LEN].astype(np.int64).reshape(n_val, SEQ_LEN)
    ).to(device)
    return val_x


# ----- Diagnostic 1: planner-sampler rank correlation ------------------------

def _spearman_rho(a, b):
    """Spearman rank correlation between two 1D float arrays."""
    if len(a) < 2:
        return float('nan')
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    n = len(a)
    diffs = ra - rb
    return 1.0 - 6.0 * (diffs ** 2).sum() / (n * (n * n - 1))


@torch.no_grad()
def planner_sampler_correlation(model, val_x, device, n_batches=16, batch_size=16):
    """For each val batch: random t, mask, forward, compute Spearman between
    gt-logprob (training planner) and max-logprob (inference sampler) across
    masked positions. Average per-row rho across all rows.
    """
    model.eval()
    k = 5.0
    rng = torch.Generator(device=device); rng.manual_seed(0)
    rhos = []
    n_done = 0
    for i in range(0, val_x.shape[0], batch_size):
        if n_done >= n_batches:
            break
        vx = val_x[i:i + batch_size]
        bs = vx.shape[0]
        t = torch.rand(bs, 1, device=device, generator=rng) * 0.95 + 0.05
        alpha_t = 1.0 - torch.exp(-k * t)
        mask = torch.rand(bs, vx.shape[1], device=device, generator=rng) < alpha_t
        x_t = torch.where(mask, torch.full_like(vx, MASK_TOKEN), vx)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x_t, causal=False)[..., :MASK_TOKEN]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        gt_lp = logprobs.gather(-1, vx.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        max_lp = logprobs.max(dim=-1).values.cpu().numpy()
        mask_np = mask.cpu().numpy()
        for b in range(bs):
            pos = np.where(mask_np[b])[0]
            if len(pos) >= 4:
                rhos.append(_spearman_rho(gt_lp[b, pos], max_lp[b, pos]))
        n_done += 1
    arr = np.array(rhos)
    return {
        'planner_sampler_rho_mean': float(arr.mean()),
        'planner_sampler_rho_std': float(arr.std()),
        'n_rows': len(arr),
    }


# ----- Diagnostic 2: T-sweep gen probe ---------------------------------------

@torch.no_grad()
def t_sweep_genprobe(model, prompts, device, t_values=(32, 64, 128),
                     batch_size=16, top_k=50):
    prefix = prompts['prefix_ids']
    cont_len = int(prompts['cont_len'])
    prefix_len = int(prompts['prefix_len'])
    out = {}
    for T in t_values:
        samples = []
        for b in range(0, prefix.shape[0], batch_size):
            chunk = prefix[b:b + batch_size]
            s = demask_topk_prefix(model, chunk, cont_len, top_k=top_k,
                                   temperature=1.0, n_steps=T, device=device)
            samples.append(s.cpu())
        full = torch.cat(samples, dim=0)
        completions = [row[prefix_len:].tolist() for row in full]
        out[f'T={T}'] = {
            'rep_4': rep_n(completions, 4),
            'rep_8': rep_n(completions, 8),
            'distinct_4': distinct_n(completions, 4),
        }
    return out


# ----- Diagnostic 3: ECE under mask distribution -----------------------------

@torch.no_grad()
def ece_under_mask(model, val_x, device, n_batches=16, batch_size=16, n_bins=10):
    """Bin masked-position max-prob; measure top-1 accuracy per bin; compute ECE."""
    model.eval()
    k = 5.0
    rng = torch.Generator(device=device); rng.manual_seed(1)
    confidences = []
    correctnesses = []
    n_done = 0
    for i in range(0, val_x.shape[0], batch_size):
        if n_done >= n_batches:
            break
        vx = val_x[i:i + batch_size]
        bs = vx.shape[0]
        t = torch.rand(bs, 1, device=device, generator=rng) * 0.95 + 0.05
        alpha_t = 1.0 - torch.exp(-k * t)
        mask = torch.rand(bs, vx.shape[1], device=device, generator=rng) < alpha_t
        x_t = torch.where(mask, torch.full_like(vx, MASK_TOKEN), vx)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x_t, causal=False)[..., :MASK_TOKEN]
        probs = F.softmax(logits.float(), dim=-1)
        max_p, top_t = probs.max(dim=-1)
        m = mask.cpu().numpy()
        mp = max_p.cpu().numpy()
        tt = top_t.cpu().numpy()
        gt = vx.cpu().numpy()
        for b in range(bs):
            pos = np.where(m[b])[0]
            confidences.extend(mp[b, pos].tolist())
            correctnesses.extend((tt[b, pos] == gt[b, pos]).astype(int).tolist())
        n_done += 1

    confidences = np.array(confidences)
    correctnesses = np.array(correctnesses)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if in_bin.sum() > 0:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(correctnesses[in_bin].mean())
            frac = float(in_bin.sum() / len(confidences))
            ece += frac * abs(avg_conf - avg_acc)
            bin_stats.append({'low': float(bins[i]), 'high': float(bins[i+1]),
                              'count': int(in_bin.sum()), 'avg_conf': avg_conf,
                              'avg_acc': avg_acc})
    return {
        'ece': float(ece),
        'overall_top1_acc': float(correctnesses.mean()),
        'overall_avg_conf': float(confidences.mean()),
        'over_under_gap': float(confidences.mean() - correctnesses.mean()),
        'n_positions': int(len(confidences)),
        'bins': bin_stats,
    }


# ----- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True,
                    help=f'Model spec names. Available: {list(MODEL_SPECS)}')
    ap.add_argument('--n-val', type=int, default=256, help='Val sequences for rho/ECE')
    ap.add_argument('--n-prompts', type=int, default=100, help='Prompts for T-sweep')
    ap.add_argument('--out', type=str,
                    default=os.path.join(THIS_DIR, 'papl_diagnostics.jsonl'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    print(f'Loading val (n_val={args.n_val})...')
    val_x = load_val_x(device, n_val=args.n_val)
    prompts_full = torch.load(PROMPTS_PATH, weights_only=False)
    prompts = {
        **prompts_full,
        'prefix_ids': prompts_full['prefix_ids'][:args.n_prompts],
        'continuation_ids': prompts_full['continuation_ids'][:args.n_prompts],
    }

    rows = []
    for name in args.models:
        print(f'\n=== {name} ===')
        adapter = MDLMAdapter.from_spec(name, device=device)
        m = adapter.model

        rho = planner_sampler_correlation(m, val_x, device)
        print(f'  planner-sampler ρ:  mean={rho["planner_sampler_rho_mean"]:+.4f} '
              f'std={rho["planner_sampler_rho_std"]:.4f}  n={rho["n_rows"]}')

        ece = ece_under_mask(m, val_x, device)
        print(f'  ECE:                {ece["ece"]:.4f}  '
              f'top1_acc={ece["overall_top1_acc"]:.4f}  '
              f'avg_conf={ece["overall_avg_conf"]:.4f}  '
              f'gap={ece["over_under_gap"]:+.4f}')

        ts = t_sweep_genprobe(m, prompts, device)
        for T_str, vals in ts.items():
            print(f'  T-sweep {T_str:>6}:    rep4={vals["rep_4"]:.4f}  '
                  f'rep8={vals["rep_8"]:.4f}  dist4={vals["distinct_4"]:.4f}')

        rows.append({
            'model': name,
            'rho': rho,
            'ece': ece,
            't_sweep': ts,
        })
        adapter.unload()

    with open(args.out, 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
