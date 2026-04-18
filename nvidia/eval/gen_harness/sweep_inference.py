"""Inference-budget sweep on a single MDLM checkpoint.

Does more demasking / tighter top-k / lower temperature recover any diversity
without killing teacher NLL? Loads the model once, iterates over configs.

Grid:
  n_steps  in {16, 32, 64 (default), 128}      # 128 = 1 token/step, max conditioning
  top_k    in {10, 50 (default), 200}
  temperature in {0.9, 1.0 (default)}

Output: nvidia/eval/gen_harness/sweep_inference.jsonl  (one row per config)
"""

import argparse
import itertools
import json
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from adapters.mdlm_adapter import MDLMAdapter, MODEL_SPECS
from metrics.teacher_nll import teacher_nll, load_teacher
from metrics.diversity import distinct_n, self_bleu_4, zipf_slope, uniq_token_ratio
from metrics.repetition import rep_n, seq_rep_n, top_word_share

PROMPTS_PATH = os.path.join(THIS_DIR, 'prompts', 'fineweb_edu_held.pt')
OUT_PATH = os.path.join(THIS_DIR, 'sweep_inference.jsonl')


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='d_modern_125m',
                    help=f'Model spec. Available: {list(MODEL_SPECS)}')
    ap.add_argument('--n-prompts', type=int, default=100,
                    help='Prompt count for the sweep (100 balances speed/noise).')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--n-steps', nargs='+', type=int, default=[16, 32, 64, 128])
    ap.add_argument('--top-k', nargs='+', type=int, default=[10, 50, 200])
    ap.add_argument('--temperature', nargs='+', type=float, default=[0.9, 1.0])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    prompts = torch.load(PROMPTS_PATH, weights_only=False)
    N = args.n_prompts
    prefix = prompts['prefix_ids'][:N]
    real_cont = prompts['continuation_ids'][:N]
    prefix_len = int(prompts['prefix_len'])
    cont_len = int(prompts['cont_len'])

    print(f'Sweep: model={args.model}  n_prompts={N}  prefix_len={prefix_len}  cont_len={cont_len}')
    print(f'Grid: n_steps={args.n_steps}  top_k={args.top_k}  temp={args.temperature} '
          f'({len(args.n_steps)*len(args.top_k)*len(args.temperature)} configs)')

    teacher = load_teacher('gpt2', device=device)

    # Real ceiling (same teacher, same prompts) — compute once for reference
    real_full = torch.cat([prefix, real_cont], dim=1)
    real_nll = teacher_nll(real_full, prefix_len=prefix_len, teacher=teacher, device=device)
    real_metrics = compute_metrics([row[prefix_len:].tolist() for row in real_full])
    real_row = {
        'model': 'real_fineweb_edu', 'sampler': 'n/a',
        'n_prompts': N, 'teacher_nll_mean': float(real_nll.mean()),
        'teacher_nll_std': float(real_nll.std()),
        **real_metrics,
    }
    print(f'\nREAL ceiling: teacher_nll={real_row["teacher_nll_mean"]:.4f}  '
          f'distinct_4={real_row["distinct_4"]:.4f}  rep_4={real_row["rep_4"]:.4f}')

    print(f'\nLoading {args.model}...')
    adapter = MDLMAdapter.from_spec(args.model, device=device)

    rows = [real_row]
    grid = list(itertools.product(args.n_steps, args.top_k, args.temperature))

    for i, (n_steps, top_k, temp) in enumerate(grid, 1):
        t0 = time.time()
        all_samples = []
        for b in range(0, N, args.batch_size):
            chunk = prefix[b:b + args.batch_size]
            out = adapter.generate(chunk, cont_len=cont_len, top_k=top_k,
                                   temperature=temp, n_steps=n_steps)
            all_samples.append(out.cpu())
        full = torch.cat(all_samples, dim=0)
        gen_time = time.time() - t0

        completion_lists = [row[prefix_len:].tolist() for row in full]
        metrics = compute_metrics(completion_lists)
        nll = teacher_nll(full, prefix_len=prefix_len, teacher=teacher, device=device)

        row = {
            'model': args.model,
            'sampler': f'topk{top_k}_temp{temp}_steps{n_steps}',
            'n_steps': n_steps, 'top_k': top_k, 'temperature': temp,
            'n_prompts': N, 'gen_seconds': gen_time,
            'teacher_nll_mean': float(nll.mean()),
            'teacher_nll_std': float(nll.std()),
            **metrics,
        }
        rows.append(row)
        delta_fluency = row['teacher_nll_mean'] - real_row['teacher_nll_mean']
        delta_diversity = real_row['distinct_4'] - row['distinct_4']
        print(f'[{i:2d}/{len(grid)}] steps={n_steps:3d} k={top_k:3d} T={temp} | '
              f'NLL {row["teacher_nll_mean"]:.3f} ({delta_fluency:+.3f}) | '
              f'dist4 {row["distinct_4"]:.3f} ({-delta_diversity:+.3f}) | '
              f'rep4 {row["rep_4"]:.3f} | '
              f'{gen_time:.1f}s')

    with open(OUT_PATH, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'\nWrote {len(rows)} rows to {OUT_PATH}')

    # Print a compact summary table sorted by teacher_nll
    print('\nSummary (sorted by teacher_nll_mean):')
    hdr = ['sampler', 'teacher_nll', 'distinct_4', 'rep_4', 'uniq_tok', 'gen_s']
    print('  ' + ' | '.join(f'{h:>22s}' for h in hdr))
    for r in sorted(rows, key=lambda x: x['teacher_nll_mean']):
        sampler = r['sampler']
        print(f'  {sampler:>22s} | '
              f'{r["teacher_nll_mean"]:22.4f} | '
              f'{r["distinct_4"]:22.4f} | '
              f'{r["rep_4"]:22.4f} | '
              f'{r["uniq_token_ratio"]:22.4f} | '
              f'{r.get("gen_seconds", 0):22.1f}')


if __name__ == '__main__':
    main()
