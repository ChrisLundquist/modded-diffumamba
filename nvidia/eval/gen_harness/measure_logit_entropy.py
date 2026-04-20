"""Measure mean entropy of softmax(logits) at masked positions on existing
30M transformer checkpoints.

Tests the first half of our mechanism hypothesis: transformer-MDLM produces
sharp output distributions at masked positions, and these get sharper over
training (consistent with the rep_4-rises-over-training observation we
documented at 125M). Mamba-MDLM's flatter distributions would have higher
entropy here — but we don't have a Mamba ckpt locally to compare; we instead
look at within-transformer trajectory + cross-config (vanilla vs PAPL vs
inverse-PAPL).

Reference scales:
  log(50257) = 10.83 nats (uniform over full vocab)
  log(50)    = 3.91  nats (uniform over top-50)
  Real text per-token entropy: ~5-7 nats (rough rule of thumb for English)

Sharp output = low entropy + high max_prob → confidence-based sampling
attractor (the rep pathology mechanism).
"""

import os, sys, glob, re, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/clundquist/modded-diffumamba/nvidia/src')
from transformer_v2 import TransformerV2

DATA = '/home/clundquist/muon_data/fineweb_1B.npy'
SEQ_LEN = 1024
B = 16
N_VAL = 64
MASK_TOKEN = 50257
OUT_PATH = '/home/clundquist/modded-diffumamba/nvidia/eval/gen_harness/logit_entropy.jsonl'

device = 'cuda'
torch.set_float32_matmul_precision('high')

print('Loading val data (mmap)...')
tokens = np.load(DATA, mmap_mode='r')
val = tokens[1_000_000_000:]
val_x = torch.from_numpy(
    val[:N_VAL * SEQ_LEN].astype(np.int64).reshape(N_VAL, SEQ_LEN)
).to(device)


def measure(ckpt_path, label, n_layer=6, n_embd=384, n_head=6, use_qk_norm=False):
    model = TransformerV2(vocab_size=50258, n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, use_rope=True, use_swiglu=True,
                          use_qk_norm=use_qk_norm).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    del ckpt

    rng = torch.Generator(device=device); rng.manual_seed(0)
    k = 5.0
    entropies = []
    top1_probs = []
    for i in range(0, N_VAL, B):
        vx = val_x[i:i + B]
        bs = vx.shape[0]
        t = torch.rand(bs, 1, device=device, generator=rng) * 0.95 + 0.05
        alpha_t = 1.0 - torch.exp(-k * t)
        mask = torch.rand(bs, vx.shape[1], device=device, generator=rng) < alpha_t
        x_t = torch.where(mask, torch.full_like(vx, MASK_TOKEN), vx)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x_t, causal=False)[..., :MASK_TOKEN]
        log_p = F.log_softmax(logits.float(), dim=-1)
        probs = log_p.exp()
        H = -(probs * log_p).sum(dim=-1)
        max_p = probs.max(dim=-1).values
        entropies.append(H[mask].cpu())
        top1_probs.append(max_p[mask].cpu())

    H_all = torch.cat(entropies)
    top1_all = torch.cat(top1_probs)
    H_mean, H_std = float(H_all.mean()), float(H_all.std())
    p_mean = float(top1_all.mean())
    n = len(H_all)
    print(f'  {label:38s}  H={H_mean:.3f}±{H_std:.3f}  max_p={p_mean:.4f}  n_pos={n}')
    del model
    torch.cuda.empty_cache()
    return {'label': label, 'ckpt': ckpt_path, 'entropy_mean': H_mean,
            'entropy_std': H_std, 'max_prob_mean': p_mean, 'n_masked_positions': n}


print(f'\nReference: log(50257)={np.log(50257):.3f} nats uniform; '
      f'log(50)={np.log(50):.3f} nats top-50-uniform')
print(f'\n{"label":38s}  entropy (nats)        max_prob  n_pos')

results = []

# Vanilla 30M trajectory across 5k → 56k steps
print('\n--- vanilla converge_v3 trajectory ---')
ckpt_dir = '/mnt/d/code/gpt-slide/muon_exp/outputs/transformer_converge_v3'
ckpts = sorted(glob.glob(f'{ckpt_dir}/checkpoint_*.pt'),
               key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
for cp in ckpts:
    step = int(re.search(r'(\d+)', os.path.basename(cp)).group(1))
    results.append(measure(cp, f'converge_v3_step{step}'))

# Phase-1 + phase-2 30M scratch endpoints (5k each)
print('\n--- 30M scratch endpoints (5k steps) ---')
extras = [
    ('vanilla_s42_5k', '30m_vanilla_scratch_s42/checkpoint_5000.pt'),
    ('vanilla_s43_5k', '30m_vanilla_scratch_s43/checkpoint_5000.pt'),
    ('papl_tau03_s42_5k', '30m_papl_scratch_tau03/checkpoint_5000.pt'),
    ('papl_tau03_s43_5k', '30m_papl_scratch_tau03_s43/checkpoint_5000.pt'),
    ('papl_tau01_s42_5k', '30m_papl_scratch_tau01/checkpoint_5000.pt'),
    ('papl_tau003_s42_5k', '30m_papl_scratch_tau003/checkpoint_5000.pt'),
    ('invpapl_tau03_s42_5k', '30m_invpapl_scratch_tau03/checkpoint_5000.pt'),
    ('invpapl_tau03_s43_5k', '30m_invpapl_scratch_tau03_s43/checkpoint_5000.pt'),
]
root = '/mnt/d/code/gpt-slide/muon_exp/outputs'
for label, rel in extras:
    p = os.path.join(root, rel)
    if os.path.exists(p):
        results.append(measure(p, label))

# Also the 125M (different kwargs)
print('\n--- 125M D_modern ---')
m125 = [
    ('125m_step40k', '125m_10b_dmodern/checkpoint_40000.pt'),
    ('125m_step50k', '125m_10b_dmodern/checkpoint_50000.pt'),
]
for label, rel in m125:
    p = os.path.join(root, rel)
    if os.path.exists(p):
        results.append(measure(p, label, n_layer=12, n_head=12, n_embd=768,
                               use_qk_norm=True))

with open(OUT_PATH, 'a') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f'\nWrote {len(results)} rows to {OUT_PATH}')

# Summary
print('\n--- summary (sorted by entropy ascending = most peaked first) ---')
for r in sorted(results, key=lambda x: x['entropy_mean']):
    print(f'  {r["label"]:38s}  H={r["entropy_mean"]:.3f}  max_p={r["max_prob_mean"]:.4f}')
