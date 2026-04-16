"""Final validation: new best config at 10k steps, 3 seeds.

New best: FineWeb-Edu + Muon-VS + out_proj + lr=0.01 + gamma=1.5
Compared against old best: FineWeb + Muon-VS + out_proj + lr=0.02 + gamma=1.5
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import statistics

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def run_one(name, argv_args):
    import gc, torch
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
        import traceback
        elapsed = time.perf_counter() - t0
        val_loss = float("inf")
        status = "FAIL"
        error = traceback.format_exc()
    finally:
        sys.argv = saved_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
        "status": status, "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    result_path = RESULTS_DIR / f"final10k_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "10000", "--val_every", "500", "--log_every", "500",
        "--warmup_steps", "400", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--save_best",
    ]

    seeds = [42, 137, 2024]

    conditions = {
        # NEW BEST: FineWeb-Edu + lr=0.01
        "new_best": common + [
            "--data_dir", "data/fineweb-edu-10B",
            "--muon_lr", "0.01",
        ],
        # OLD BEST: FineWeb + lr=0.02 (baseline from previous 10k run)
        "old_best": common + [
            "--data_dir", "data/fineweb10B",
            "--muon_lr", "0.02",
        ],
    }

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
            ckpt = str(CKPT_DIR / f"final10k_{name}.pt")
            args = base_args + ["--seed", str(seed), "--save_path", ckpt]
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {name}")
            print(f"{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}")
    print(f"FINAL 10K VALIDATION (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    print(f"\n  {'Config':<12s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*12} {'-'*8} {'-'*8}  {'-'*35}")
    for c in ["new_best", "old_best"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<12s} {m:>8.4f} {s:>8.4f}  [{vals_str}]")

    # Paired t-test
    if "new_best" in configs and "old_best" in configs:
        deltas = []
        for seed in seeds:
            new_r = next((r for r in results if r["config"] == "new_best" and r["seed"] == seed), None)
            old_r = next((r for r in results if r["config"] == "old_best" and r["seed"] == seed), None)
            if new_r and old_r:
                deltas.append(new_r["val_loss"] - old_r["val_loss"])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            t_stat = m / (s / len(deltas)**0.5) if s > 0 else float('inf')
            sig = "SIG" if abs(t_stat) > 2.92 else "n.s."
            direction = "better" if m < 0 else "worse"
            print(f"\n  Paired delta (new vs old): {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, {sig}, new is {direction})")

    summary_path = RESULTS_DIR / "final_10k_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
