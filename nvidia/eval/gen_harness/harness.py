"""Generation eval harness entry point.

Usage:
  python harness.py --models d_modern_125m --dry-run        # 5 prompts, no teacher
  python harness.py --models d_modern_125m d_modern_30m mamba3_30m  # full run
  python harness.py --models real                            # teacher ceiling only

Appends one JSON line per (model, sampler_config) to eval/gen_harness/leaderboard.jsonl.
"""

import argparse
import json
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from adapters.mdlm_adapter import MDLMAdapter, MODEL_SPECS
from adapters.ar_adapter import ARAdapter, AR_SPECS
from metrics.teacher_nll import teacher_nll, load_teacher
from metrics.diversity import distinct_n, self_bleu_4, zipf_slope, uniq_token_ratio
from metrics.repetition import rep_n, seq_rep_n, top_word_share

PROMPTS_PATH = os.path.join(THIS_DIR, 'prompts', 'fineweb_edu_held.pt')
LEADERBOARD = os.path.join(THIS_DIR, 'leaderboard.jsonl')


def as_completion_lists(sample_ids, prefix_len):
    """[B, T] tensor -> list of list[int] (completion tokens only)."""
    return [row[prefix_len:].tolist() for row in sample_ids.cpu()]


def compute_metrics(completion_lists):
    return {
        'distinct_1': distinct_n(completion_lists, 1),
        'distinct_4': distinct_n(completion_lists, 4),
        'self_bleu_4': self_bleu_4(completion_lists),
        'zipf_slope': zipf_slope(completion_lists),
        'rep_4': rep_n(completion_lists, 4),
        'seq_rep_4': seq_rep_n(completion_lists, 4),
        'top10_share': top_word_share(completion_lists, 10),
        'uniq_token_ratio': uniq_token_ratio(completion_lists),
    }


def run_real_baseline(prompts, device, skip_teacher=False, teachers=None):
    """Score the real FineWeb-Edu continuations as the teacher-NLL ceiling."""
    prefix = prompts['prefix_ids']
    cont = prompts['continuation_ids']
    full = torch.cat([prefix, cont], dim=1)
    completion_lists = as_completion_lists(full, prefix.shape[1])
    result = {
        'model': 'real_fineweb_edu',
        'sampler': 'n/a',
        'n_samples': len(completion_lists),
    }
    result.update(compute_metrics(completion_lists))
    if not skip_teacher:
        for tname, tmodel in (teachers or {'gpt2': None}).items():
            nll = teacher_nll(full, prefix_len=prefix.shape[1], teacher=tmodel,
                              teacher_name=tname, device=device)
            key = 'teacher_nll' if tname == 'gpt2' else f'teacher_nll_{tname}'
            result[f'{key}_mean'] = float(nll.mean())
            result[f'{key}_std'] = float(nll.std())
    return result


def run_model(name, prompts, device, batch_size, top_k, temperature, n_steps,
              skip_teacher=False, teachers=None, sampler='topk'):
    prefix = prompts['prefix_ids']
    cont_len = int(prompts['cont_len'])
    prefix_len = int(prompts['prefix_len'])
    N = prefix.shape[0]

    if name in AR_SPECS:
        adapter = ARAdapter.from_spec(name, device=device)
        sampler_arg = {}  # AR ignores sampler choice
    else:
        adapter = MDLMAdapter.from_spec(name, device=device)
        sampler_arg = {'sampler': sampler}
    all_samples = []
    t0 = time.time()
    for i in range(0, N, batch_size):
        chunk = prefix[i:i + batch_size]
        out = adapter.generate(chunk, cont_len=cont_len, top_k=top_k,
                               temperature=temperature, n_steps=n_steps,
                               **sampler_arg)
        all_samples.append(out.cpu())
        if i == 0:
            print(f'  [{name}] first batch done in {time.time() - t0:.1f}s')
    full = torch.cat(all_samples, dim=0)
    gen_time = time.time() - t0
    print(f'  [{name}] generated {N} in {gen_time:.1f}s ({N / gen_time:.2f}/s)')

    completion_lists = as_completion_lists(full, prefix_len)
    sampler_label = sampler if name not in AR_SPECS else 'ar_topk'
    result = {
        'model': name,
        'sampler': f'{sampler_label}_k{top_k}_T{temperature}_steps{n_steps or "default"}',
        'n_samples': N,
        'gen_seconds': gen_time,
    }
    result.update(compute_metrics(completion_lists))

    if not skip_teacher:
        for tname, tmodel in (teachers or {'gpt2': None}).items():
            nll = teacher_nll(full, prefix_len=prefix_len, teacher=tmodel,
                              teacher_name=tname, device=device)
            key = 'teacher_nll' if tname == 'gpt2' else f'teacher_nll_{tname}'
            result[f'{key}_mean'] = float(nll.mean())
            result[f'{key}_std'] = float(nll.std())

    adapter.unload()
    return result, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['d_modern_125m'],
                    help='Model specs to run. Use "real" for teacher ceiling. '
                         f'Available: {list(MODEL_SPECS)}')
    ap.add_argument('--dry-run', action='store_true',
                    help='Use first 5 prompts only, skip teacher.')
    ap.add_argument('--n-prompts', type=int, default=None,
                    help='Cap prompts (default: all).')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--n-steps', type=int, default=None,
                    help='Demasking steps (topk default cont_len//2; maskgit default 12).')
    ap.add_argument('--sampler', choices=['topk', 'maskgit'], default='topk',
                    help='MDLM sampler (AR models always use ancestral top-k).')
    ap.add_argument('--skip-teacher', action='store_true')
    ap.add_argument('--no-append', action='store_true',
                    help='Do not append to leaderboard.jsonl (just print).')
    ap.add_argument('--save-samples', type=str, default=None,
                    help='Optional path to save raw samples as .pt')
    ap.add_argument('--teachers', nargs='+',
                    default=['gpt2',
                             '/home/clundquist/muon_data/hf_models/rhysjones_gpt2-124M'],
                    help='Teacher model paths/IDs for NLL scoring. First one reported '
                         'as "teacher_nll"; rest as "teacher_nll_<basename>".')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    prompts = torch.load(PROMPTS_PATH, weights_only=False)
    n = args.n_prompts or (5 if args.dry_run else prompts['prefix_ids'].shape[0])
    prompts = {
        **prompts,
        'prefix_ids': prompts['prefix_ids'][:n],
        'continuation_ids': prompts['continuation_ids'][:n],
    }
    print(f'Harness: {n} prompts, prefix_len={prompts["prefix_len"]}, cont_len={prompts["cont_len"]}')

    skip_teacher = args.skip_teacher or args.dry_run

    teachers = {}
    if not skip_teacher:
        for tpath in args.teachers:
            tname = os.path.basename(tpath.rstrip('/')) or tpath
            # normalize "gpt2" so it keeps the short name
            if tpath == 'gpt2':
                tname = 'gpt2'
            print(f'Warming up teacher: {tname} ({tpath})')
            teachers[tname] = load_teacher(tpath, device=device)

    saved_samples = {}
    results = []
    for name in args.models:
        print(f'\n=== {name} ===')
        if name == 'real':
            r = run_real_baseline(prompts, device, skip_teacher=skip_teacher,
                                  teachers=teachers)
        else:
            r, samples = run_model(
                name, prompts, device,
                batch_size=args.batch_size, top_k=args.top_k,
                temperature=args.temperature, n_steps=args.n_steps,
                skip_teacher=skip_teacher, teachers=teachers,
                sampler=args.sampler,
            )
            saved_samples[name] = samples
        results.append(r)
        for k, v in r.items():
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')

    print('\n=== leaderboard ===')
    hdr = ['model', 'teacher_nll_mean', 'distinct_4', 'rep_4', 'top10_share', 'uniq_token_ratio']
    print(' | '.join(f'{h:>18s}' for h in hdr))
    for r in results:
        row = []
        for h in hdr:
            v = r.get(h, '—')
            if isinstance(v, float):
                row.append(f'{v:18.4f}')
            else:
                row.append(f'{str(v):>18s}')
        print(' | '.join(row))

    if not args.no_append and not args.dry_run:
        with open(LEADERBOARD, 'a') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f'\nAppended {len(results)} rows to {LEADERBOARD}')

    if args.save_samples:
        torch.save(saved_samples, args.save_samples)
        print(f'Saved raw samples to {args.save_samples}')


if __name__ == '__main__':
    main()
