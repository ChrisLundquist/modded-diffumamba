"""Aggregate leaderboard.jsonl into a markdown table.

Usage:
    python summarize_leaderboard.py [--source leaderboard.jsonl]
                                    [--filter MODEL_SUBSTR]
                                    [--cols col1,col2,...]

Reads each JSON line and prints one row per (model, sampler) combination.
Default columns: model, sampler, n_samples, teacher_nll_mean,
teacher_nll_rhysjones_gpt2-124M_mean, distinct_4, rep_4, uniq_token_ratio.
"""

import argparse
import json
import os

DEFAULT_COLS = [
    'model', 'sampler', 'n_samples',
    'teacher_nll_mean', 'teacher_nll_rhysjones_gpt2-124M_mean',
    'distinct_4', 'rep_4', 'uniq_token_ratio',
]
SHORT = {
    'teacher_nll_mean': 'NLL_gpt2',
    'teacher_nll_rhysjones_gpt2-124M_mean': 'NLL_rhys',
    'distinct_4': 'dist4',
    'rep_4': 'rep4',
    'uniq_token_ratio': 'uniq_tok',
    'n_samples': 'N',
    'model': 'model',
    'sampler': 'sampler',
}


def fmt(v):
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=os.path.join(os.path.dirname(__file__),
                                                      'leaderboard.jsonl'))
    ap.add_argument('--filter', default=None,
                    help='Substring filter on model name')
    ap.add_argument('--cols', default=None,
                    help='Comma-separated column names')
    ap.add_argument('--latest', action='store_true',
                    help='Keep only the most recent entry per (model, sampler)')
    args = ap.parse_args()

    cols = args.cols.split(',') if args.cols else DEFAULT_COLS
    rows = []
    with open(args.source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.filter and args.filter not in r.get('model', ''):
                continue
            rows.append(r)

    if args.latest:
        seen = {}
        for r in rows:
            key = (r.get('model'), r.get('sampler'))
            seen[key] = r
        rows = list(seen.values())

    headers = [SHORT.get(c, c) for c in cols]
    widths = [max(len(h), 10) for h in headers]
    for r in rows:
        for i, c in enumerate(cols):
            widths[i] = max(widths[i], len(fmt(r.get(c, '—'))))

    sep = '|'.join('-' * (w + 2) for w in widths)
    line = '|'.join(f' {headers[i]:<{widths[i]}} ' for i in range(len(headers)))
    print(line)
    print(sep)
    for r in rows:
        line = '|'.join(f' {fmt(r.get(c, "—")):<{widths[i]}} ' for i, c in enumerate(cols))
        print(line)


if __name__ == '__main__':
    main()
