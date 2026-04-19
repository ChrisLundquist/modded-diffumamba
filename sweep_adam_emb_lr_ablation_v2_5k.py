"""Revised Adam-emb-LR vs Muon-emb ablation: 5 arms x 3 seeds at 5k.

Replaces sweep_adam_emb_lr_ablation_5k.py. The v1 used Muon Arm C at
emb_lr=0.01, but exp3 + its extension showed the Muon peak is higher.
This v2 reads the best-observed emb_lr from results/emblr_*.json files
(pooled across exp3 + extension sweep) and uses that for Arm C, so the
comparison is Adam-at-its-best vs Muon-at-its-best.

Arms:
  A:  Adam tok_emb @ 3e-4                    (current baseline)
  B1: Adam tok_emb @ 1e-3                    (matches nvidia finding)
  B2: Adam tok_emb @ 3e-3                    (pushes further)
  B3: Adam tok_emb @ 1e-2                    (matched to Muon scale; may diverge)
  C:  Muon tok_emb @ <winning emb_lr>        (dynamically selected)

Decision rule: if C beats max(B1, B2, B3) significantly, geometry wins.
If max Adam arm >= C, it's an LR story.

pos_emb stays in main Adam group at 3e-4 in every arm.

15 runs total, ~3.3 hours.
"""
import sys
import json
import time
import glob
import re
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def pick_best_muon_emb_lr():
    """Pool all emblr_*.json results and pick the LR with lowest mean val_loss."""
    by_lr = defaultdict(list)
    for path in glob.glob(str(RESULTS_DIR / "emblr_*.json")):
        # Filenames are emblr_<LR_pat>_s<seed>.json, e.g. emblr_emblr0p03_s42.json
        name = Path(path).stem  # e.g. emblr_emblr0p03_s42
        m = re.search(r"emblr0p(\d+)", name)
        if not m:
            continue
        lr = float("0." + m.group(1).rstrip("_"))
        with open(path) as f:
            rec = json.load(f)
        if rec.get("status") != "OK":
            continue
        if rec["val_loss"] == float("inf"):
            continue
        by_lr[lr].append(rec["val_loss"])

    if not by_lr:
        raise RuntimeError("No emblr_*.json results found; run exp3 first.")

    means = {lr: statistics.mean(vs) for lr, vs in by_lr.items()}
    best_lr, best_mean = min(means.items(), key=lambda x: x[1])
    print(f"  emb_lr curve so far:")
    for lr in sorted(means.keys()):
        print(f"    lr={lr:>6g}  n={len(by_lr[lr])}  mean={means[lr]:.4f}")
    print(f"  -> Arm C uses winning Muon emb_lr={best_lr}")
    return best_lr


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
        status, error = "OK", ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        val_loss = float("inf")
        status, error = "FAIL", str(e)
    finally:
        sys.argv = saved_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    import math
    if status == "OK" and (math.isnan(val_loss) or val_loss > 15.0):
        status, error = "FAIL", f"diverged (val_loss={val_loss})"
        val_loss = float("inf")
    record = {"name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
              "status": status, "error": error, "timestamp": datetime.now().isoformat()}
    result_path = RESULTS_DIR / f"ablation2_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    winning_muon_lr = pick_best_muon_emb_lr()

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
        "A_baseline":      list(common),
        "B1_adam_emb_1e3": list(common) + ["--adam_emb_lr", "1e-3"],
        "B2_adam_emb_3e3": list(common) + ["--adam_emb_lr", "3e-3"],
        "B3_adam_emb_1e2": list(common) + ["--adam_emb_lr", "1e-2"],
        "C_muon_tok_emb":  list(common) + ["--muon_tok_emb",
                                            "--muon_emb_lr", str(winning_muon_lr)],
    }

    seeds = [42, 137, 2024]
    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    for seed in seeds:
        print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")
        for config_name, base_args in conditions.items():
            run_idx += 1
            name = f"{config_name}_s{seed}"
            args = base_args + ["--seed", str(seed)]
            print(f"\n{'='*60}\n[{run_idx}/{total_runs}] {name}\n{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}")
    print(f"ABLATION v2 RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"  Arm C used muon_emb_lr={winning_muon_lr}")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append(r["val_loss"])

    print(f"\n  {'Arm':<20s} {'Mean':>8s} {'Std':>8s}  Seeds")
    for c in ["A_baseline", "B1_adam_emb_1e3", "B2_adam_emb_3e3",
              "B3_adam_emb_1e2", "C_muon_tok_emb"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vs = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<20s} {m:>8.4f} {s:>8.4f}  [{vs}]")

    # Per-seed table
    print(f"\n  Per-seed val_loss:")
    header = f"  {'Seed':>6s}"
    for c in conditions.keys():
        header += f" {c:>18s}"
    print(header)
    for seed in seeds:
        row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
        line = f"  {seed:>6d}"
        for c in conditions.keys():
            v = row.get(c, float("nan"))
            line += f" {v:>18.4f}"
        print(line)

    # Paired deltas vs baseline
    print(f"\n  Paired deltas vs A_baseline (negative = arm better):")
    for arm in ["B1_adam_emb_1e3", "B2_adam_emb_3e3", "B3_adam_emb_1e2", "C_muon_tok_emb"]:
        deltas = []
        for seed in seeds:
            row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
            if arm in row and "A_baseline" in row:
                deltas.append(row[arm] - row["A_baseline"])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            n = len(deltas)
            t_stat = m / (s / (n ** 0.5)) if s > 0 else float("inf")
            sig = "**SIG**" if abs(t_stat) > 2.92 else ""
            print(f"  {arm}: {m:+.4f} ± {s:.4f}  t={t_stat:+.2f}  n={n}  {sig}")

    # Key disambiguation
    print(f"\n  Key disambiguation: C (Muon) vs best Adam arm")
    best_adam_arm, best_adam_mean = None, float("inf")
    for arm in ["B1_adam_emb_1e3", "B2_adam_emb_3e3", "B3_adam_emb_1e2"]:
        vals = configs.get(arm, [])
        if vals:
            m = statistics.mean(vals)
            if m < best_adam_mean:
                best_adam_mean = m
                best_adam_arm = arm
    c_vals = configs.get("C_muon_tok_emb", [])
    if best_adam_arm and len(c_vals) >= 2:
        c_mean = statistics.mean(c_vals)
        print(f"  Best Adam arm: {best_adam_arm} @ {best_adam_mean:.4f}")
        print(f"  C_muon_tok_emb @ {c_mean:.4f}")
        deltas = []
        for seed in seeds:
            row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
            if "C_muon_tok_emb" in row and best_adam_arm in row:
                deltas.append(row["C_muon_tok_emb"] - row[best_adam_arm])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            n = len(deltas)
            t_stat = m / (s / (n ** 0.5)) if s > 0 else float("inf")
            if t_stat < -2.92:
                verdict = "GEOMETRY WINS (Muon > best Adam, p<0.05)"
            elif t_stat > 2.92:
                verdict = "LR STORY (best Adam >= Muon, p<0.05)"
            else:
                verdict = "INCONCLUSIVE (Muon ~ best Adam)"
            print(f"  Paired (C - {best_adam_arm}): {m:+.4f} ± {s:.4f}, t={t_stat:+.2f}, n={n}")
            print(f"  Verdict: {verdict}")

    summary_path = RESULTS_DIR / "adam_emb_lr_ablation_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"winning_muon_emb_lr": winning_muon_lr, "results": results}, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
