"""Width vs depth at ~30M params: 5k steps, 3 seeds.

Tests whether deeper-narrower models outperform our shallow-wide quokka.
Uses Muon-VS + out_proj (our best optimizer config).

Note: not perfectly iso-param due to embedding table (vocab×d_model).
Embedding fraction decreases with narrower models, giving more capacity
to Mamba blocks — which is part of the hypothesis.
"""
import sys
import json
import time
import copy
from pathlib import Path
from datetime import datetime

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
    result_path = RESULTS_DIR / f"dw_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    common = [
        "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.02", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--save_best",
    ]

    seeds = [42, 137, 2024]

    # Depth scaling at fixed d_model=384 (following DiffuMamba's approach)
    # Research says: don't go below 384d, just add depth. Embedding stays
    # fixed at 19.3M, more params go to Mamba blocks.
    conditions = {
        "4L_384d":  common + ["--config", "quokka"],                                         # 31.5M (baseline)
        "6L_384d":  common + ["--config", "quokka", "--n_layers", "6"],                      # ~37.6M
        "8L_384d":  common + ["--config", "quokka", "--n_layers", "8"],                      # ~43.7M
        "8L_320d":  common + ["--config", "quokka", "--n_layers", "8", "--d_model", "320"],  # ~35.0M (iso-param)
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
            ckpt = str(CKPT_DIR / f"dw_{name}.pt")
            args = base_args + ["--seed", str(seed), "--save_path", ckpt]
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {name}")
            print(f"{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    import statistics

    print(f"\n{'='*60}")
    print(f"WIDTH vs DEPTH RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    baseline_key = "4L_384d"
    print(f"\n  {'Config':<14s} {'Mean':>8s} {'Std':>8s} {'vs 4L':>8s}  Seeds")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")
    for c in ["4L_384d", "6L_384d", "8L_384d", "8L_320d"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            bm = statistics.mean(configs.get(baseline_key, [m]))
            delta = m - bm
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<14s} {m:>8.4f} {s:>8.4f} {delta:>+8.4f}  [{vals_str}]")

    # Paired analysis vs baseline
    print(f"\n  Paired differences vs {baseline_key}:")
    for c in ["6L_384d", "8L_384d", "8L_320d"]:
        deltas = []
        for seed in seeds:
            base_r = next((r for r in results if r["config"] == baseline_key and r["seed"] == seed), None)
            test_r = next((r for r in results if r["config"] == c and r["seed"] == seed), None)
            if base_r and test_r and base_r["val_loss"] != float("inf") and test_r["val_loss"] != float("inf"):
                deltas.append(test_r["val_loss"] - base_r["val_loss"])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            t_stat = m / (s / len(deltas)**0.5) if s > 0 else float('inf')
            sig = "SIG" if abs(t_stat) > 2.92 else "n.s."
            print(f"    {c:<14s}: {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, {sig})")

    summary_path = RESULTS_DIR / "depth_width_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
