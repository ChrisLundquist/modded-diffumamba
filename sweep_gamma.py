"""Gamma sweep: find Muon-optimal loss weighting for masked diffusion.

Sweeps minsnr_gamma across [flat, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] with
Muon optimizer on quokka config (31.5M), 1000 steps each.

Based on evaluation ranking: highest-value experiment per GPU-hour.
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
        # Reload parse_args each time to pick up new argv
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

    result_path = RESULTS_DIR / f"gamma_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    base = [
        "--config", "quokka",
        "--batch_size", "8",
        "--max_steps", "1000",
        "--val_every", "50",
        "--log_every", "50",
        "--warmup_steps", "50",
        "--no_time_cond",
        "--lr_schedule", "cosine",
        "--optimizer", "muon",
        "--muon_lr", "0.02",
        "--adam_lr", "3e-4",
    ]

    experiments = [
        ("muon_flat",       base + ["--loss_weight", "flat"]),
        ("muon_gamma1.0",   base + ["--loss_weight", "minsnr", "--minsnr_gamma", "1.0"]),
        ("muon_gamma1.5",   base + ["--loss_weight", "minsnr", "--minsnr_gamma", "1.5"]),
        ("muon_gamma2.0",   base + ["--loss_weight", "minsnr", "--minsnr_gamma", "2.0"]),
        ("muon_gamma3.0",   base + ["--loss_weight", "minsnr", "--minsnr_gamma", "3.0"]),
        ("muon_gamma5.0",   base + ["--loss_weight", "minsnr", "--minsnr_gamma", "5.0"]),
        ("muon_gamma10.0",  base + ["--loss_weight", "minsnr", "--minsnr_gamma", "10.0"]),
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
    print(f"GAMMA SWEEP SUMMARY (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    results.sort(key=lambda r: r["val_loss"])
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {r['name']:>20s}: val_loss={r['val_loss']:.4f}, "
              f"time={r['elapsed_seconds']:.0f}s{marker}")

    # Save summary
    summary_path = RESULTS_DIR / "gamma_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
