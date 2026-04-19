"""Stand-alone PAPL-decomposed val eval — backfill on existing checkpoints.

Loads any saved 125M D_modern checkpoint and reports both:
  - uniform-MDLM-NLL (Min-SNR γ=1.5 weighted, standard MDLM ELBO)
  - planner-weighted-NLL (PAPL self-planner reweight at α, τ, same γ clamp)

Both numbers are computed from the SAME masking pattern in one forward pass,
so they're directly comparable. Used to backfill the train/val decomposition
for checkpoints that were trained before the live decomposition logging.

Usage:
    python eval_papl_decomp.py --ckpt /path/to/checkpoint.pt [--alpha 1.0 --tau 1.0]
    python eval_papl_decomp.py --trajectory  # all 125m_10b_dmodern + papl checkpoints
"""

import argparse
import glob
import os
import re
import sys
import json

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..', 'src'))
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..', 'training'))
from transformer_v2 import TransformerV2  # noqa: E402

# Import the decomp eval directly from the training script
from finetune_papl_125m import eval_mdlm_decomp  # noqa: E402

import numpy as np

DATA_PATH = '/home/clundquist/muon_data/fineweb_10B.npy'
SEQ_LEN = 1024
N_VAL = 1024


def load_val_x(device):
    all_tokens = np.load(DATA_PATH, mmap_mode='r')
    val_tokens = all_tokens[len(all_tokens) - 500_000_000:]
    val_x = torch.from_numpy(
        val_tokens[:N_VAL * SEQ_LEN].astype(np.int64).reshape(N_VAL, SEQ_LEN)
    ).to(device)
    return val_x


def eval_one(ckpt_path, val_x, device, alpha, tau):
    model = TransformerV2(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                          use_rope=True, use_swiglu=True, use_qk_norm=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    step = ckpt.get('step', None)
    train_objective = ckpt.get('objective', 'unknown')
    del ckpt
    torch.cuda.empty_cache()
    decomp = eval_mdlm_decomp(model, val_x, alpha=alpha, tau=tau)
    del model
    torch.cuda.empty_cache()
    return {
        'ckpt': ckpt_path,
        'step': step,
        'train_objective': train_objective,
        'alpha': alpha, 'tau': tau,
        **decomp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=None,
                    help='Single checkpoint path. If unset, --trajectory.')
    ap.add_argument('--trajectory', action='store_true',
                    help='Eval all 125m_10b_dmodern checkpoints + 125m_papl_finetune.')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--out', type=str, default=os.path.join(THIS_DIR, 'papl_decomp.jsonl'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    if args.ckpt:
        ckpts = [args.ckpt]
    elif args.trajectory:
        roots = [
            '/mnt/d/code/gpt-slide/muon_exp/outputs/125m_10b_dmodern',
            '/mnt/d/code/gpt-slide/muon_exp/outputs/125m_papl_finetune',
        ]
        ckpts = []
        for r in roots:
            ckpts.extend(sorted(glob.glob(os.path.join(r, 'checkpoint_*.pt')),
                                key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1))))
    else:
        raise SystemExit('Pass --ckpt or --trajectory')

    print(f'Loading val (mmap, {N_VAL}×{SEQ_LEN} tokens)...')
    val_x = load_val_x(device)

    rows = []
    for cp in ckpts:
        print(f'\n--- {cp} ---')
        r = eval_one(cp, val_x, device, args.alpha, args.tau)
        print(f'  step={r["step"]}  train_obj={r["train_objective"]}  '
              f'uniform={r["uniform_nll_minsnr"]:.4f}  '
              f'papl_w={r["planner_w_nll_minsnr"]:.4f}  '
              f'gap={r["uniform_nll_minsnr"] - r["planner_w_nll_minsnr"]:+.4f}')
        rows.append(r)

    with open(args.out, 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nAppended {len(rows)} rows to {args.out}')

    print('\nSummary (uniform vs planner-weighted, gap = uniform - planner_w):')
    print(f'  {"step":>8} {"train_obj":>20} {"uniform":>12} {"papl_w":>12} {"gap":>12}')
    for r in rows:
        print(f'  {str(r["step"]):>8} {str(r["train_objective"]):>20} '
              f'{r["uniform_nll_minsnr"]:12.4f} {r["planner_w_nll_minsnr"]:12.4f} '
              f'{r["uniform_nll_minsnr"] - r["planner_w_nll_minsnr"]:+12.4f}')


if __name__ == '__main__':
    main()
