"""Checkpoint-trajectory harness for a single model family.

Runs the full metric bundle at each checkpoint step of a training run.
Tests: does the model keep improving on *generation* metrics as ELBO drops,
or do they plateau while training loss still improves?

Outputs: nvidia/eval/gen_harness/trajectory_<model>.jsonl

Usage:
    python sweep_trajectory.py --model d_modern_125m
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'src'))
sys.path.insert(0, SRC_DIR)

from transformer_v2 import TransformerV2      # noqa: E402
from hybrid_model import DiffuMambaH           # noqa: E402
from samplers.mdlm_topk import demask_topk_prefix  # noqa: E402
from adapters.mdlm_adapter import MODEL_SPECS, CHECKPOINT_ROOT  # noqa: E402
from metrics.teacher_nll import teacher_nll, load_teacher  # noqa: E402
from metrics.diversity import distinct_n, self_bleu_4, zipf_slope, uniq_token_ratio  # noqa: E402
from metrics.repetition import rep_n, seq_rep_n, top_word_share  # noqa: E402

PROMPTS_PATH = os.path.join(THIS_DIR, 'prompts', 'fineweb_edu_held.pt')


def build_model(spec, device):
    if spec['family'] == 'transformer_v2':
        return TransformerV2(**spec['kwargs']).to(device)
    if spec['family'] == 'diffumamba_h':
        return DiffuMambaH(**spec['kwargs']).to(device)
    raise ValueError(spec['family'])


def compute_metrics(completion_lists):
    return {
        'distinct_1': distinct_n(completion_lists, 1),
        'distinct_4': distinct_n(completion_lists, 4),
        'self_bleu_4': self_bleu_4(completion_lists),
        'zipf_slope': zipf_slope(completion_lists),
        'rep_4': rep_n(completion_lists, 4),
        'top10_share': top_word_share(completion_lists, 10),
        'uniq_token_ratio': uniq_token_ratio(completion_lists),
    }


@torch.no_grad()
def run_one(model, prefix, cont_len, prefix_len, top_k, temperature, n_steps,
            batch_size, teachers, device):
    """teachers: dict[name -> model]. NLL reported per teacher."""
    N = prefix.shape[0]
    all_samples = []
    t0 = time.time()
    for b in range(0, N, batch_size):
        chunk = prefix[b:b + batch_size]
        out = demask_topk_prefix(model, chunk, cont_len, top_k=top_k,
                                 temperature=temperature, n_steps=n_steps,
                                 device=device)
        all_samples.append(out.cpu())
    full = torch.cat(all_samples, dim=0)
    gen_time = time.time() - t0

    completion_lists = [row[prefix_len:].tolist() for row in full]
    out = {**compute_metrics(completion_lists), 'gen_seconds': gen_time}
    for tname, tmodel in teachers.items():
        nll = teacher_nll(full, prefix_len=prefix_len, teacher=tmodel, device=device)
        key = 'teacher_nll' if tname == 'gpt2' else f'teacher_nll_{tname}'
        out[f'{key}_mean'] = float(nll.mean())
        out[f'{key}_std'] = float(nll.std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='d_modern_125m',
                    help=f'Model spec name. Available: {list(MODEL_SPECS)}')
    ap.add_argument('--ckpt-dir', type=str, default=None,
                    help='Override checkpoint directory. Default: parent of spec ckpt.')
    ap.add_argument('--ckpt-glob', type=str, default='checkpoint_*.pt')
    ap.add_argument('--n-prompts', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--n-steps', type=int, default=64)
    ap.add_argument('--teachers', nargs='+',
                    default=['gpt2',
                             '/home/clundquist/muon_data/hf_models/rhysjones_gpt2-124M'],
                    help='Teacher paths/IDs for NLL scoring (matches harness.py default).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    spec = MODEL_SPECS[args.model]
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.dirname(os.path.join(CHECKPOINT_ROOT, spec['ckpt']))
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)),
                   key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
    if not ckpts:
        raise RuntimeError(f'No checkpoints in {args.ckpt_dir}/{args.ckpt_glob}')
    print(f'Found {len(ckpts)} checkpoints:')
    for p in ckpts:
        print(f'  {p}')

    prompts = torch.load(PROMPTS_PATH, weights_only=False)
    N = args.n_prompts
    prefix = prompts['prefix_ids'][:N]
    real_cont = prompts['continuation_ids'][:N]
    prefix_len = int(prompts['prefix_len'])
    cont_len = int(prompts['cont_len'])

    print(f'\nPrompts: {N}  prefix_len={prefix_len}  cont_len={cont_len}')
    print(f'Sampler: top_k={args.top_k} temp={args.temperature} n_steps={args.n_steps}')

    teachers = {}
    for tpath in args.teachers:
        tname = os.path.basename(tpath.rstrip('/')) or tpath
        if tpath == 'gpt2':
            tname = 'gpt2'
        print(f'Loading teacher: {tname} ({tpath})')
        teachers[tname] = load_teacher(tpath, device=device)

    # Real ceiling
    real_full = torch.cat([prefix, real_cont], dim=1)
    real_comp = [row[prefix_len:].tolist() for row in real_full]
    real_row = {
        'model': 'real_fineweb_edu', 'step': None,
        **compute_metrics(real_comp),
    }
    for tname, tmodel in teachers.items():
        rn = teacher_nll(real_full, prefix_len=prefix_len, teacher=tmodel, device=device)
        key = 'teacher_nll' if tname == 'gpt2' else f'teacher_nll_{tname}'
        real_row[f'{key}_mean'] = float(rn.mean())
        real_row[f'{key}_std'] = float(rn.std())
    print(f'\nREAL ceiling: nll(gpt2)={real_row["teacher_nll_mean"]:.3f} '
          f'dist4={real_row["distinct_4"]:.3f} rep4={real_row["rep_4"]:.3f}')

    model = build_model(spec, device)
    rows = [real_row]
    for path in ckpts:
        step = int(re.search(r'(\d+)', os.path.basename(path)).group(1))
        print(f'\n--- step {step} ({path}) ---')
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        val_loss = ckpt.get('val_loss', None)
        del ckpt
        torch.cuda.empty_cache()

        r = run_one(model, prefix, cont_len, prefix_len,
                    args.top_k, args.temperature, args.n_steps,
                    args.batch_size, teachers, device)
        r['model'] = args.model
        r['step'] = step
        r['val_loss'] = val_loss
        rows.append(r)
        nll_keys = [k for k in r if k.endswith('_mean') and 'teacher_nll' in k]
        nll_str = '  '.join(f'{k.replace("teacher_nll_","").replace("_mean","")}={r[k]:.3f}'
                            for k in nll_keys)
        print(f'  {nll_str}  dist4={r["distinct_4"]:.3f}  rep4={r["rep_4"]:.3f}  '
              f'uniq={r["uniq_token_ratio"]:.3f}  '
              f'val_loss={val_loss if val_loss is not None else "-"}  '
              f'gen={r["gen_seconds"]:.1f}s')

    out_path = os.path.join(THIS_DIR, f'trajectory_{args.model}.jsonl')
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # Summary
    print('\n\nTrajectory summary:')
    hdr = ['step', 'val_loss', 'teacher_nll', 'distinct_4', 'rep_4', 'uniq_tok']
    print('  ' + ' | '.join(f'{h:>12s}' for h in hdr))
    for r in rows:
        step = r['step'] if r['step'] is not None else 'real'
        vl = f'{r.get("val_loss"):.4f}' if isinstance(r.get('val_loss'), float) else '-'
        print(f'  {str(step):>12s} | {vl:>12s} | '
              f'{r["teacher_nll_mean"]:12.4f} | '
              f'{r["distinct_4"]:12.4f} | '
              f'{r["rep_4"]:12.4f} | '
              f'{r["uniq_token_ratio"]:12.4f}')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
