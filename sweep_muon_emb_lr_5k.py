"""Embedding-LR sweep under Muon routing (exp 3): 4 LRs x 3 seeds at 5k steps.

Follow-up to sweep_muon_tokemb_5k.py. The tok_emb-in-Muon A/B was run with
muon_emb_lr == muon_lr (shared group at 0.01). The embedding has two
properties that suggest its optimal LR may differ from block weights:
  - Sparse gradients (only rows for in-batch tokens get signal)
  - Very rectangular (vocab_size x d_model = 50304 x 384 = 19.3M params)

Muon's Newton-Schulz orthogonalization couples rows through the full
gradient matrix, so a tuned emb-group LR could push the tok_emb win further.

Arms (all with --muon_tok_emb):
  lr=0.003  — lower (sparse-grad intuition)
  lr=0.01   — control (matches block LR; current finding)
  lr=0.02   — higher
  lr=0.03   — higher still

The control cell is re-run within the sweep (same seeds) for clean paired
comparison; we could also reuse sweep_muon_tokemb_5k.py's muon_tok_emb_* cell
but paired within-sweep is cleaner.

12 runs total, ~3 hours.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"


def run_one(name, argv_args):
    import gc
    import torch

    saved_argv = sys.argv
    sys.argv = ["train.py"] + argv_args
    t0 = time.perf_counter()
    try:
        from train import parse_args, train
        args = parse_args()
        val_loss = train(args)
        elapsed = time.perf_counter() - t0
        status = "OK"
        error = ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        val_loss = float("inf")
        status = "FAIL"
        error = str(e)
    finally:
        sys.argv = saved_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "name": name,
        "val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "status": status,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    result_path = RESULTS_DIR / f"emblr_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj", "--muon_tok_emb",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
    ]

    emb_lrs = [0.003, 0.01, 0.02, 0.03]
    conditions = {
        f"emblr{lr:g}".replace(".", "p"): list(common) + ["--muon_emb_lr", str(lr)]
        for lr in emb_lrs
    }
    # names: emblr0p003, emblr0p01, emblr0p02, emblr0p03

    seeds = [42, 137, 2024]
    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")
        for config_name, base_args in conditions.items():
            run_idx += 1
            name = f"{config_name}_s{seed}"
            args = base_args + ["--seed", str(seed)]
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {name}")
            print(f"{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            record["emb_lr"] = float(base_args[base_args.index("--muon_emb_lr") + 1])
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    import statistics

    print(f"\n{'='*60}")
    print(f"EMB-LR SWEEP RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append((r["emb_lr"], r["val_loss"]))

    print(f"\n  {'Config':<14s} {'emb_lr':>8s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")
    for c in conditions.keys():
        entries = configs.get(c, [])
        vals = [v for _, v in entries]
        lr = entries[0][0] if entries else None
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<14s} {lr:>8g} {m:>8.4f} {s:>8.4f}  [{vals_str}]")

    # Paired deltas vs control (emblr=0.01) per seed
    control_key = "emblr0p01"
    print(f"\n  Paired deltas per seed (<arm> - {control_key}):")
    header = f"  {'Seed':>6s}"
    for lr in emb_lrs:
        k = f"emblr{lr:g}".replace(".", "p")
        header += f" {k:>12s}"
    print(header)

    # Per-seed table of val_losses
    for seed in seeds:
        row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
        line = f"  {seed:>6d}"
        for lr in emb_lrs:
            k = f"emblr{lr:g}".replace(".", "p")
            line += f" {row.get(k, float('nan')):>12.4f}"
        print(line)

    # Paired deltas table
    print(f"\n  Deltas vs {control_key} (negative = better):")
    for lr in emb_lrs:
        k = f"emblr{lr:g}".replace(".", "p")
        if k == control_key:
            continue
        deltas = []
        for seed in seeds:
            row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
            if k in row and control_key in row:
                deltas.append(row[k] - row[control_key])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            n = len(deltas)
            t_stat = m / (s / (n ** 0.5)) if s > 0 else float('inf')
            sig = "**SIG**" if abs(t_stat) > 2.92 else ""
            print(f"  {k}:  mean {m:+.4f} ± {s:.4f}  t={t_stat:+.2f}  n={n}  {sig}")

    summary_path = RESULTS_DIR / "emb_lr_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
