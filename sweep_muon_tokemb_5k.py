"""tok_emb-in-Muon A/B: 2 arms x 3 seeds at 5k steps.

Geometric-analysis follow-up (see results/geometry/REPORT.md). The only
weight in our best model showing classic Huh-2021 simplicity-bias
signature (stable rank falling, sigma_max growing) is the Adam-routed
tok_emb. Muon-routed blocks are spectrally static. Question: is that
bias costing val_loss, or is the modded-nanogpt "don't route embeddings
through Muon" lore correct?

Arm A (baseline): current best recipe — tok_emb in Adam group.
Arm B (test):     add --muon_tok_emb — tok_emb (+ tied lm_head) in Muon-VS group.

Only one variable changes between A and B. No HP tuning on the Muon
embedding group (same muon_lr=0.01 as blocks); if B loses here the lore
is called correct at this LR. If B wins, a follow-up LR sweep on the
embedding group is motivated.

6 runs total, ~1.5-2 hours at Mamba3 Triton speeds.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"


def run_one(name, argv_args):
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
    result_path = RESULTS_DIR / f"tokemb_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Current best recipe from HANDOFF.md (10k, 3 seeds, val_loss 5.07)
    # Shortened to 5k for this A/B. FineWeb-Edu as the training data.
    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
    ]

    conditions = {
        "baseline": list(common),
        "muon_tok_emb": list(common) + ["--muon_tok_emb"],
    }

    seeds = [42, 137, 2024]
    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    # Paired design: all arms per seed before moving to next seed.
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

    import statistics

    print(f"\n{'='*60}")
    print(f"tok_emb-Muon A/B RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append(r["val_loss"])

    print(f"\n  {'Config':<18s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*18} {'-'*8} {'-'*8}  {'-'*30}")
    for c in ["baseline", "muon_tok_emb"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<18s} {m:>8.4f} {s:>8.4f}  [{vals_str}]")

    # Paired deltas: muon_tok_emb - baseline per seed (negative = Muon-routed embed wins)
    print(f"\n  Paired deltas per seed (muon_tok_emb - baseline):")
    print(f"  {'Seed':>6s} {'baseline':>10s} {'muon_tok_emb':>14s} {'delta':>10s}")
    print(f"  {'-'*6} {'-'*10} {'-'*14} {'-'*10}")

    deltas = []
    for seed in seeds:
        row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
        if "baseline" in row and "muon_tok_emb" in row:
            d = row["muon_tok_emb"] - row["baseline"]
            deltas.append(d)
            print(f"  {seed:>6d} {row['baseline']:>10.4f} {row['muon_tok_emb']:>14.4f} {d:>+10.4f}")

    if len(deltas) >= 2:
        m = statistics.mean(deltas)
        s = statistics.stdev(deltas)
        n = len(deltas)
        t_stat = m / (s / (n ** 0.5)) if s > 0 else float('inf')
        print(f"\n  Mean delta: {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, n={n})")
        if t_stat < -2.92:
            verdict = "muon_tok_emb WINS (p<0.05)"
        elif t_stat > 2.92:
            verdict = "baseline WINS — lore is correct (p<0.05)"
        else:
            verdict = "INCONCLUSIVE at p<0.05 (need more seeds or LR tuning)"
        print(f"  Verdict: {verdict}")

    summary_path = RESULTS_DIR / "tokemb_muon_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
