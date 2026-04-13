"""2×2 factorial: {Adam, Muon} × {gamma=1.5, gamma=5} with 3 seeds each.

Isolates optimizer effect from gamma effect with proper replication.
12 runs total, ~30 min at Mamba3 Triton speeds.
"""
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"


def run_one(name, argv_args):
    """Run a single training experiment in-process."""
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
        "argv": argv_args,
    }
    result_path = RESULTS_DIR / f"2x2_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "1000", "--val_every", "50", "--log_every", "250",
        "--warmup_steps", "50", "--no_time_cond", "--lr_schedule", "cosine",
    ]

    seeds = [42, 137, 2024]

    grid = {
        "muon_g1.5": common + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        ],
        "muon_g5": common + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "5",
        ],
        "adam_g1.5": common + [
            "--optimizer", "adam", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        ],
        "adam_g5": common + [
            "--optimizer", "adam", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "5",
        ],
    }

    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(grid) * len(seeds)

    for config_name, base_args in grid.items():
        for seed in seeds:
            run_idx += 1
            name = f"{config_name}_s{seed}"
            args = base_args + ["--seed", str(seed)]
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {name}")
            print(f"{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    # Analysis
    print(f"\n{'='*60}")
    print(f"2×2 FACTORIAL RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    import statistics

    # Group by config
    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    print(f"\n  {'Config':<15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}  Seeds")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
    for c in ["muon_g1.5", "muon_g5", "adam_g1.5", "adam_g5"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            print(f"  {c:<15s} {m:>8.4f} {s:>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}  {vals}")
        elif vals:
            print(f"  {c:<15s} {vals[0]:>8.4f}     n/a     n/a     n/a  {vals}")

    # Factorial analysis
    print(f"\n  Main effects:")
    muon_vals = configs.get("muon_g1.5", []) + configs.get("muon_g5", [])
    adam_vals = configs.get("adam_g1.5", []) + configs.get("adam_g5", [])
    g15_vals = configs.get("muon_g1.5", []) + configs.get("adam_g1.5", [])
    g5_vals = configs.get("muon_g5", []) + configs.get("adam_g5", [])

    if muon_vals and adam_vals:
        muon_m = statistics.mean(muon_vals)
        adam_m = statistics.mean(adam_vals)
        print(f"    Optimizer: Muon={muon_m:.4f}, Adam={adam_m:.4f}, "
              f"delta={adam_m - muon_m:+.4f} ({'Muon wins' if muon_m < adam_m else 'Adam wins'})")
    if g15_vals and g5_vals:
        g15_m = statistics.mean(g15_vals)
        g5_m = statistics.mean(g5_vals)
        print(f"    Gamma:     1.5={g15_m:.4f},  5={g5_m:.4f}, "
              f"delta={g5_m - g15_m:+.4f} ({'g1.5 wins' if g15_m < g5_m else 'g5 wins'})")

    # Interaction
    print(f"\n  Interaction (Muon advantage by gamma):")
    for gamma, g_name in [("g1.5", "gamma=1.5"), ("g5", "gamma=5")]:
        muon = configs.get(f"muon_{gamma}", [])
        adam = configs.get(f"adam_{gamma}", [])
        if muon and adam:
            delta = statistics.mean(adam) - statistics.mean(muon)
            print(f"    {g_name}: Adam-Muon = {delta:+.4f} "
                  f"({'Muon better' if delta > 0 else 'Adam better'})")

    summary_path = RESULTS_DIR / "2x2_factorial_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
