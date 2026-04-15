"""5k-step comparison: out_proj routing + GELU vs SwiGLU.

Tests:
1. Muon-VS with out_proj excluded (current) vs included (5090 finding)
2. SwiGLU (current) vs GELU (DiffuMamba style) at expansion=2
3 seeds each, paired design.
"""
import sys
import json
import time
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
    result_path = RESULTS_DIR / f"opg_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
        "--muon_variant", "vs",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--save_best",
    ]

    seeds = [42, 137, 2024]

    conditions = {
        # Baseline: Muon-VS, SwiGLU, out_proj excluded (current best)
        "vs_swiglu": common + ["--no_time_cond"],
        # out_proj included (5090 finding)
        "vs_swiglu_outproj": common + ["--no_time_cond", "--muon_out_proj"],
        # GELU at expansion=2 (DiffuMamba style, fewer params)
        "vs_gelu": common + ["--no_time_cond", "--mlp_type", "gelu"],
        # GELU + out_proj (full DiffuMamba-like config)
        "vs_gelu_outproj": common + ["--no_time_cond", "--mlp_type", "gelu", "--muon_out_proj"],
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
            ckpt = str(CKPT_DIR / f"opg_{name}.pt")
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
    print(f"OUT_PROJ + GELU RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    baseline_key = "vs_swiglu"
    print(f"\n  {'Config':<22s} {'Mean':>8s} {'Std':>8s} {'vs base':>8s}  Seeds")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")
    for c in ["vs_swiglu", "vs_swiglu_outproj", "vs_gelu", "vs_gelu_outproj"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            bm = statistics.mean(configs.get(baseline_key, [m]))
            delta = m - bm
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<22s} {m:>8.4f} {s:>8.4f} {delta:>+8.4f}  [{vals_str}]")

    # Paired analysis
    print(f"\n  Paired differences vs {baseline_key}:")
    for c in ["vs_swiglu_outproj", "vs_gelu", "vs_gelu_outproj"]:
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
            print(f"    {c:<22s}: {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, {sig})")

    summary_path = RESULTS_DIR / "outproj_gelu_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
