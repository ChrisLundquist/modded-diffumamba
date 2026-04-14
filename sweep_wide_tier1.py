"""Tier 1 wide screen: architecture variants at 1k steps.

Screens hybrid attention, weight tying, merge strategies, time conditioning.
All use Muon+minsnr(gamma=1.5), quokka config, seed=42 for consistency.
Winners validated at 5k steps with 3 seeds separately.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"


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
    result_path = RESULTS_DIR / f"wide_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    base = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "1000", "--val_every", "50", "--log_every", "250",
        "--warmup_steps", "50", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--seed", "42",
    ]

    # All experiments use --no_time_cond except the time conditioning test
    base_notc = base + ["--no_time_cond"]

    experiments = [
        # === Controls ===
        ("baseline",              base_notc),

        # === A1: Hybrid attention ratios ===
        ("hybrid25_first",        base_notc + ["--attn_layers", "0"]),
        ("hybrid25_last",         base_notc + ["--attn_layers", "3"]),
        ("hybrid50",              base_notc + ["--attn_layers", "1,3"]),

        # === A2: Weight tying (Caduceus-style) ===
        ("tied",                  base_notc + ["--tie_weights"]),

        # === A3: Merge strategy ===
        ("mul_merge",             base_notc + ["--merge", "mul"]),
        ("gate_merge",            base_notc + ["--merge", "gate"]),

        # === C4: Time conditioning re-test ===
        ("time_cond_on",          base),  # no --no_time_cond

        # === Combinations ===
        ("hybrid25_first_tied",   base_notc + ["--attn_layers", "0", "--tie_weights"]),
        ("hybrid25_first_timeon", base + ["--attn_layers", "0"]),
    ]

    results = []
    total_t0 = time.perf_counter()

    for name, args in experiments:
        run_idx = len(results) + 1
        print(f"\n{'='*60}")
        print(f"[{run_idx}/{len(experiments)}] SCREEN: {name}")
        print(f"{'='*60}")
        record = run_one(name, args)
        results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}")
    print(f"TIER 1 WIDE SCREEN (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    results.sort(key=lambda r: r["val_loss"])
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        delta = r["val_loss"] - results[0]["val_loss"]
        print(f"  {r['name']:>25s}: val_loss={r['val_loss']:.4f} "
              f"(+{delta:.4f}){marker}")

    summary_path = RESULTS_DIR / "wide_tier1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
