"""Phase 2: Validate gamma=1.5 finding at 5000 steps + Adam comparison.

Follows evaluation plan Path B: new gamma beat flat.
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

    result_path = RESULTS_DIR / f"val_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    common_1k = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "1000", "--val_every", "50", "--log_every", "100",
        "--warmup_steps", "50", "--no_time_cond", "--lr_schedule", "cosine",
    ]

    common_5k = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "250",
        "--warmup_steps", "200", "--no_time_cond", "--lr_schedule", "cosine",
    ]

    experiments = [
        # 1) Adam baselines at 1000 steps (for relative comparison)
        ("adam_minsnr_1k", common_1k + [
            "--optimizer", "adam", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "5",
        ]),
        ("adam_gamma1.5_1k", common_1k + [
            "--optimizer", "adam", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        ]),

        # 2) 5000-step convergence: muon_gamma1.5 vs adam_minsnr
        ("muon_gamma1.5_5k", common_5k + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        ]),
        ("adam_minsnr_5k", common_5k + [
            "--optimizer", "adam", "--adam_lr", "3e-4",
            "--loss_weight", "minsnr", "--minsnr_gamma", "5",
        ]),

        # 3) 5000-step: muon_flat for reference (was previous best)
        ("muon_flat_5k", common_5k + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
            "--loss_weight", "flat",
        ]),
    ]

    results = []
    total_t0 = time.perf_counter()

    for name, args in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name} ({len(results)+1}/{len(experiments)})")
        print(f"{'='*60}")
        record = run_one(name, args)
        results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Group by step count
    for steps in ["1k", "5k"]:
        group = [r for r in results if steps in r["name"]]
        if group:
            print(f"\n  {steps} steps:")
            group.sort(key=lambda r: r["val_loss"])
            for i, r in enumerate(group):
                marker = " <-- BEST" if i == 0 else ""
                print(f"    {r['name']:>25s}: val_loss={r['val_loss']:.4f}, "
                      f"time={r['elapsed_seconds']:.0f}s{marker}")

    summary_path = RESULTS_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
