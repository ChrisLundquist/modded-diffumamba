"""Extend the muon_emb_lr sweep to find the top of the curve.

Exp3 (sweep_muon_emb_lr_5k.py) ran muon_emb_lr in {0.003, 0.01, 0.02, 0.03}
and found MONOTONIC improvement up to 0.03 with no plateau (val_loss
5.38 -> 5.19 -> 5.12 -> 5.09). This extension adds {0.05, 0.07, 0.10} to
locate the peak before the subsequent ablation uses "Muon at its best".

Same 3 paired seeds (42, 137, 2024) as exp3 so results combine cleanly.
All else matches exp3 and the current best recipe.

9 runs total, ~2 hours. If 0.10 diverges, we've found the ceiling.
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
        status, error = "OK", ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        val_loss = float("inf")
        status, error = "FAIL", str(e)
    finally:
        sys.argv = saved_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Bf16 training can produce NaN without raising; guard so paired stats aren't poisoned.
    import math
    if status == "OK" and (math.isnan(val_loss) or val_loss > 15.0):
        status, error = "FAIL", f"diverged (val_loss={val_loss})"
        val_loss = float("inf")

    record = {"name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
              "status": status, "error": error, "timestamp": datetime.now().isoformat()}
    # Re-use emblr_ prefix so analysis pools with exp3.
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

    emb_lrs = [0.05, 0.07, 0.10]
    # Naming matches exp3 convention: emblr0p05, emblr0p07, emblr0p1
    conditions = {
        f"emblr{lr:g}".replace(".", "p"): list(common) + ["--muon_emb_lr", str(lr)]
        for lr in emb_lrs
    }

    seeds = [42, 137, 2024]
    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    for seed in seeds:
        print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")
        for config_name, base_args in conditions.items():
            run_idx += 1
            name = f"{config_name}_s{seed}"
            args = base_args + ["--seed", str(seed)]
            print(f"\n{'='*60}\n[{run_idx}/{total_runs}] {name}\n{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            record["emb_lr"] = float(base_args[base_args.index("--muon_emb_lr") + 1])
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    import statistics
    print(f"\n{'='*60}")
    print(f"EMB-LR EXTENSION RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append((r["emb_lr"], r["val_loss"]))

    for c in conditions.keys():
        entries = configs.get(c, [])
        vals = [v for _, v in entries]
        lr = entries[0][0] if entries else None
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vs = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<14s} lr={lr:>6g}  mean={m:.4f} std={s:.4f}  [{vs}]")

    summary_path = RESULTS_DIR / "emb_lr_ext_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
