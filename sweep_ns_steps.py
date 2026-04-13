"""NS steps probe: does Newton-Schulz iteration count interact with loss weighting?

Tests whether rougher orthogonalization (fewer NS steps) is more tolerant
of gradient scale variance from non-flat loss weights.
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
    result_path = RESULTS_DIR / f"ns_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    base = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "1000", "--val_every", "50", "--log_every", "100",
        "--warmup_steps", "50", "--no_time_cond", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
    ]

    experiments = [
        # NS steps × loss weight grid (cherry-picked per evaluation)
        ("ns3_gamma1.5",  base + ["--ns_steps", "3", "--loss_weight", "minsnr", "--minsnr_gamma", "1.5"]),
        ("ns7_gamma1.5",  base + ["--ns_steps", "7", "--loss_weight", "minsnr", "--minsnr_gamma", "1.5"]),
        ("ns3_gamma5",    base + ["--ns_steps", "3", "--loss_weight", "minsnr", "--minsnr_gamma", "5"]),
        ("ns7_gamma5",    base + ["--ns_steps", "7", "--loss_weight", "minsnr", "--minsnr_gamma", "5"]),
        ("ns3_flat",      base + ["--ns_steps", "3", "--loss_weight", "flat"]),
        ("ns7_flat",      base + ["--ns_steps", "7", "--loss_weight", "flat"]),
    ]

    results = []
    total_t0 = time.perf_counter()
    for name, args in experiments:
        print(f"\n{'='*60}")
        print(f"NS PROBE: {name} ({len(results)+1}/{len(experiments)})")
        print(f"{'='*60}")
        record = run_one(name, args)
        results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}")
    print(f"NS STEPS PROBE SUMMARY (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Reference values from gamma sweep (ns=5 default)
    print("\n  Reference (ns=5, from gamma sweep):")
    print(f"    {'gamma1.5':>15s}: val_loss=6.3921")
    print(f"    {'gamma5':>15s}: val_loss=7.2603")
    print(f"    {'flat':>15s}: val_loss=7.5980")

    print("\n  NS steps probe results:")
    for lw in ["gamma1.5", "gamma5", "flat"]:
        group = [r for r in results if lw in r["name"]]
        group.sort(key=lambda r: r["val_loss"])
        for r in group:
            ns = r["name"].split("_")[0]  # ns3 or ns7
            print(f"    {r['name']:>15s}: val_loss={r['val_loss']:.4f} ({ns})")

    summary_path = RESULTS_DIR / "ns_steps_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
