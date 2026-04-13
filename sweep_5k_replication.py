"""5k-step replication: {Adam, Muon} × {gamma=1.5, gamma=5} × 3 seeds.

Paired-seed design: same init for all 4 conditions per seed.
12 runs at 5000 steps each, ~2 hours total at Mamba3 Triton speeds.
"""
import sys
import json
import time
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
    }
    result_path = RESULTS_DIR / f"5k_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--no_time_cond", "--lr_schedule", "cosine",
    ]

    conditions = {
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

    seeds = [42, 137, 2024]

    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    # Run paired: all 4 conditions for seed 0, then all 4 for seed 1, etc.
    # This way each seed block is self-contained for paired analysis.
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
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    # Analysis
    import statistics

    print(f"\n{'='*60}")
    print(f"5K-STEP REPLICATION RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Per-condition stats
    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    print(f"\n  {'Config':<15s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*15} {'-'*8} {'-'*8}  {'-'*30}")
    for c in ["muon_g1.5", "muon_g5", "adam_g1.5", "adam_g5"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<15s} {m:>8.4f} {s:>8.4f}  [{vals_str}]")

    # Main effects
    print(f"\n  Main effects (mean ± std):")
    muon_all = configs.get("muon_g1.5", []) + configs.get("muon_g5", [])
    adam_all = configs.get("adam_g1.5", []) + configs.get("adam_g5", [])
    g15_all = configs.get("muon_g1.5", []) + configs.get("adam_g1.5", [])
    g5_all = configs.get("muon_g5", []) + configs.get("adam_g5", [])

    if len(muon_all) >= 2 and len(adam_all) >= 2:
        print(f"    Optimizer: Muon={statistics.mean(muon_all):.4f}±{statistics.stdev(muon_all):.4f}, "
              f"Adam={statistics.mean(adam_all):.4f}±{statistics.stdev(adam_all):.4f}")
    if len(g15_all) >= 2 and len(g5_all) >= 2:
        print(f"    Gamma:     1.5={statistics.mean(g15_all):.4f}±{statistics.stdev(g15_all):.4f}, "
              f"5={statistics.mean(g5_all):.4f}±{statistics.stdev(g5_all):.4f}")

    # Paired differences per seed
    print(f"\n  Paired differences (within each seed):")
    print(f"  {'Seed':>6s} {'muon_g1.5':>10s} {'muon_g5':>10s} {'adam_g1.5':>10s} {'adam_g5':>10s} {'Muon adv':>10s}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    muon_advantages = []
    for seed in seeds:
        row = {}
        for r in results:
            if r["seed"] == seed:
                row[r["config"]] = r["val_loss"]
        if len(row) == 4:
            muon_avg = (row["muon_g1.5"] + row["muon_g5"]) / 2
            adam_avg = (row["adam_g1.5"] + row["adam_g5"]) / 2
            adv = adam_avg - muon_avg
            muon_advantages.append(adv)
            print(f"  {seed:>6d} {row.get('muon_g1.5', 0):>10.4f} {row.get('muon_g5', 0):>10.4f} "
                  f"{row.get('adam_g1.5', 0):>10.4f} {row.get('adam_g5', 0):>10.4f} {adv:>+10.4f}")

    if len(muon_advantages) >= 2:
        m = statistics.mean(muon_advantages)
        s = statistics.stdev(muon_advantages)
        # Simple t-test: t = mean / (std / sqrt(n))
        n = len(muon_advantages)
        t_stat = m / (s / (n ** 0.5)) if s > 0 else float('inf')
        print(f"\n  Muon advantage: {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, n={n})")
        print(f"  {'SIGNIFICANT (t>2.9, p<0.05 two-tailed for df=2)' if abs(t_stat) > 2.92 else 'NOT SIGNIFICANT at p<0.05'}")

    summary_path = RESULTS_DIR / "5k_replication_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
