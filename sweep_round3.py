"""Round 3: Learning & loss optimization sweep.

Block A: FineWeb-Edu vs FineWeb (5k, 3 seeds)
Block B: Muon-VS LR screen (5k, 1 seed) — {0.01, 0.02, 0.04, 0.06, 0.08}
Block C: Validate top 2 LRs (5k, 3 seeds)
Block D: Loss weighting × VS — gamma={1.5, 5, ELBO} (5k, 3 seeds, reuse gamma=1.5)
Block E: WD screen (5k, 1 seed) — {0, 0.01, 0.1}

All use Muon-VS + out_proj, cosine schedule.
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
    result_path = RESULTS_DIR / f"r3_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def run_block(block_name, experiments, seeds):
    results = []
    t0 = time.perf_counter()
    total = len(experiments) * len(seeds)
    idx = 0
    for seed in seeds:
        for config_name, base_args in experiments.items():
            idx += 1
            name = f"{config_name}_s{seed}"
            args = base_args + ["--seed", str(seed)]
            print(f"\n{'='*60}")
            print(f"[{block_name} {idx}/{total}] {name}")
            print(f"{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            results.append(record)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def summarize(results, baseline_key=None):
    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    print(f"\n  {'Config':<20s} {'Mean':>8s} {'Std':>8s} {'vs base':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    sorted_configs = sorted(configs.items(), key=lambda kv: statistics.mean(kv[1]))
    for c, vals in sorted_configs:
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) >= 2 else 0
        bm = statistics.mean(configs[baseline_key]) if baseline_key and baseline_key in configs else m
        delta = m - bm
        n = len(vals)
        print(f"  {c:<20s} {m:>8.4f} {s:>8.4f} {delta:>+8.4f}  (n={n})")
    return configs


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    base_5k = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.02", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
    ]

    total_t0 = time.perf_counter()

    # ================================================================
    # BLOCK A: FineWeb-Edu vs FineWeb (5k, 3 seeds)
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# BLOCK A: FineWeb-Edu vs FineWeb")
    print(f"{'#'*60}")

    block_a = {
        "fineweb": base_5k + ["--data_dir", "data/fineweb10B"],
        "fineweb_edu": base_5k + ["--data_dir", "data/fineweb-edu-10B"],
    }
    results_a, elapsed_a = run_block("A", block_a, [42, 137, 2024])

    print(f"\n  BLOCK A RESULTS ({elapsed_a/60:.1f} min):")
    configs_a = summarize(results_a, "fineweb")

    # Pick winner for subsequent blocks
    fw_mean = statistics.mean(configs_a.get("fineweb", [99]))
    edu_mean = statistics.mean(configs_a.get("fineweb_edu", [99]))
    best_data = "data/fineweb-edu-10B" if edu_mean < fw_mean else "data/fineweb10B"
    best_data_name = "fineweb_edu" if edu_mean < fw_mean else "fineweb"
    print(f"\n  Winner: {best_data_name} → using for subsequent blocks")

    # ================================================================
    # BLOCK B: LR Screen (5k, 1 seed)
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# BLOCK B: Muon-VS LR Screen")
    print(f"{'#'*60}")

    block_b = {}
    for lr in [0.01, 0.02, 0.04, 0.06, 0.08]:
        lr_str = f"lr{lr:.2f}".replace(".", "p")
        block_b[lr_str] = [
            "--config", "quokka", "--batch_size", "8",
            "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
            "--warmup_steps", "200", "--lr_schedule", "cosine",
            "--optimizer", "muon", "--muon_variant", "vs",
            "--muon_lr", str(lr), "--adam_lr", "3e-4",
            "--muon_out_proj",
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
            "--data_dir", best_data,
        ]
    results_b, elapsed_b = run_block("B", block_b, [42])

    print(f"\n  BLOCK B RESULTS ({elapsed_b/60:.1f} min):")
    configs_b = summarize(results_b, "lr0p02")

    # Pick top 2 LRs
    lr_ranked = sorted(configs_b.items(), key=lambda kv: statistics.mean(kv[1]))
    top2_lrs = [kv[0] for kv in lr_ranked[:2]]
    print(f"\n  Top 2 LRs: {top2_lrs}")

    # ================================================================
    # BLOCK C: Validate top 2 LRs (5k, 3 seeds)
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# BLOCK C: Validate Top LRs")
    print(f"{'#'*60}")

    block_c = {}
    for lr_key in top2_lrs:
        # Extract LR from key like "lr0p04" → 0.04
        lr_val = lr_key.replace("lr", "").replace("p", ".")
        block_c[lr_key] = [
            "--config", "quokka", "--batch_size", "8",
            "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
            "--warmup_steps", "200", "--lr_schedule", "cosine",
            "--optimizer", "muon", "--muon_variant", "vs",
            "--muon_lr", lr_val, "--adam_lr", "3e-4",
            "--muon_out_proj",
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
            "--data_dir", best_data,
        ]
    # Only need seeds 137 and 2024 (seed 42 already done in Block B)
    results_c, elapsed_c = run_block("C", block_c, [137, 2024])

    # Merge with Block B seed 42 results
    for r in results_b:
        if r["config"] in top2_lrs:
            results_c.append(r)

    print(f"\n  BLOCK C RESULTS ({elapsed_c/60:.1f} min):")
    configs_c = summarize(results_c)

    # Pick winning LR
    lr_winner = sorted(configs_c.items(), key=lambda kv: statistics.mean(kv[1]))[0][0]
    lr_winner_val = lr_winner.replace("lr", "").replace("p", ".")
    print(f"\n  Winner: {lr_winner} (lr={lr_winner_val})")

    # ================================================================
    # BLOCK D: Loss weighting × VS (5k, 3 seeds)
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# BLOCK D: Loss Weighting × VS")
    print(f"{'#'*60}")

    block_d_base = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", lr_winner_val, "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--data_dir", best_data,
    ]

    block_d = {
        "gamma1p5": block_d_base + ["--loss_weight", "minsnr", "--minsnr_gamma", "1.5"],
        "gamma5":   block_d_base + ["--loss_weight", "minsnr", "--minsnr_gamma", "5"],
        "elbo":     block_d_base + ["--loss_weight", "elbo"],
    }
    results_d, elapsed_d = run_block("D", block_d, [42, 137, 2024])

    print(f"\n  BLOCK D RESULTS ({elapsed_d/60:.1f} min):")
    summarize(results_d, "gamma1p5")

    # ================================================================
    # BLOCK E: Weight Decay Screen (5k, 1 seed)
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# BLOCK E: Weight Decay Screen")
    print(f"{'#'*60}")

    block_e = {}
    for wd in [0.0, 0.01, 0.1]:
        wd_str = f"wd{wd:.2f}".replace(".", "p")
        block_e[wd_str] = [
            "--config", "quokka", "--batch_size", "8",
            "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
            "--warmup_steps", "200", "--lr_schedule", "cosine",
            "--optimizer", "muon", "--muon_variant", "vs",
            "--muon_lr", lr_winner_val, "--adam_lr", "3e-4",
            "--muon_out_proj", "--muon_wd", str(wd), "--adam_wd", str(wd),
            "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
            "--data_dir", best_data,
        ]
    results_e, elapsed_e = run_block("E", block_e, [42])

    print(f"\n  BLOCK E RESULTS ({elapsed_e/60:.1f} min):")
    summarize(results_e, "wd0p01")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_elapsed = time.perf_counter() - total_t0
    all_results = results_a + results_b + results_c + results_d + results_e

    print(f"\n{'='*60}")
    print(f"ROUND 3 COMPLETE (total: {total_elapsed/60:.1f} min, {len(all_results)} runs)")
    print(f"{'='*60}")
    print(f"  Best data: {best_data_name}")
    print(f"  Best LR: {lr_winner_val}")

    summary_path = RESULTS_DIR / "round3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
