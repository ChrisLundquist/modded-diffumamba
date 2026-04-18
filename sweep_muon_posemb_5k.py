"""pos_emb-in-Muon A/B (exp 2): 2 arms x 3 seeds at 5k steps.

Follow-up to sweep_muon_tokemb_5k.py. Figure 2 of results/geometry/REPORT.md
shows pos_emb (Adam-routed) also loses entropy over training, though less
sharply than tok_emb. Does moving it to Muon add further val_loss gains,
or is the tok_emb win already capturing the bulk of the embedding-routing
effect?

Arm A (baseline): --muon_tok_emb only
Arm B (test):     --muon_tok_emb --muon_pos_emb

Note: pos_emb is small (max_seq_len x d_model = 1024 x 384 = 0.39M params)
so the expected effect size is smaller than tok_emb.

6 runs total, ~1.5 hours.
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
    result_path = RESULTS_DIR / f"posemb_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Baseline = new best recipe (tok_emb already in Muon)
    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj", "--muon_tok_emb",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
    ]

    conditions = {
        "tokemb_only": list(common),
        "tokemb_and_posemb": list(common) + ["--muon_pos_emb"],
    }

    seeds = [42, 137, 2024]
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
    print(f"pos_emb-Muon A/B RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append(r["val_loss"])

    print(f"\n  {'Config':<22s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*22} {'-'*8} {'-'*8}  {'-'*30}")
    for c in ["tokemb_only", "tokemb_and_posemb"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<22s} {m:>8.4f} {s:>8.4f}  [{vals_str}]")

    print(f"\n  Paired deltas per seed (tokemb_and_posemb - tokemb_only):")
    print(f"  {'Seed':>6s} {'tok only':>10s} {'tok+pos':>10s} {'delta':>10s}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    deltas = []
    for seed in seeds:
        row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
        if "tokemb_only" in row and "tokemb_and_posemb" in row:
            d = row["tokemb_and_posemb"] - row["tokemb_only"]
            deltas.append(d)
            print(f"  {seed:>6d} {row['tokemb_only']:>10.4f} {row['tokemb_and_posemb']:>10.4f} {d:>+10.4f}")

    if len(deltas) >= 2:
        m = statistics.mean(deltas)
        s = statistics.stdev(deltas)
        n = len(deltas)
        t_stat = m / (s / (n ** 0.5)) if s > 0 else float('inf')
        print(f"\n  Mean delta: {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, n={n})")
        if t_stat < -2.92:
            verdict = "+pos_emb WINS (p<0.05)"
        elif t_stat > 2.92:
            verdict = "+pos_emb LOSES (p<0.05) — keep pos_emb in Adam"
        else:
            verdict = "INCONCLUSIVE at p<0.05"
        print(f"  Verdict: {verdict}")

    summary_path = RESULTS_DIR / "posemb_muon_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
