"""Adam-emb-LR vs Muon-emb ablation (disambiguation): 4 arms x 3 seeds at 5k steps.

The tok_emb-in-Muon A/B (see sweep_muon_tokemb_5k.py, results/tokemb_*.json)
found a -0.115 nat win at 5k. But toggling --muon_tok_emb bundles THREE
changes: (a) effective per-step update on 19.3M embed params jumps ~380x,
(b) Newton-Schulz orthogonalization, (c) different momentum/WD semantics.

Null hypothesis to refute: "Adam tok_emb at lr=3e-4 is simply undertrained,
and any sufficiently large LR on that matrix closes the gap."

Per nvidia/HANDOFF_nvidia.md finding #9, raising Adam embed LR 1.5e-4 -> 1e-3
was worth 0.20 nats at 30M scale. Our 0.115 is comfortably in that range,
so the null is not ruled out. This sweep tests it directly.

Arms (all 5k, quokka, FineWeb-Edu, seeds 42/137/2024):
  A:  tok_emb in Adam @ 3e-4                    (current baseline, 2 groups)
  B1: tok_emb in Adam @ 1e-3                    (matches nvidia finding)
  B2: tok_emb in Adam @ 3e-3                    (pushes further)
  C:  tok_emb in Muon @ 0.01                    (current claim, 2 groups)

Decision rule:
  - If C > max(B1, B2) significantly -> geometry wins, Muon-tok_emb is real.
  - If max(B1, B2) >= C -> it's an Adam-LR story; the modded-nanogpt
    "Muon only for hidden weights" lore is correct, just underappreciated.
  - If B2 diverges (val_loss blows up) -> we've located the Adam-LR ceiling;
    use min(C, B1) comparison.

Only tok_emb routing changes between arms; pos_emb stays in the main Adam
group at lr=3e-4 in every arm, by design (see train.py `is_adam_emb_candidate`).

12 runs total, ~2.5 hours.
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
    result_path = RESULTS_DIR / f"ablation_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

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
        "C_muon_tok_emb":  list(common) + ["--muon_tok_emb"],
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
    print(f"ADAM-EMB-LR ABLATION RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append(r["val_loss"])

    print(f"\n  {'Arm':<20s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*20} {'-'*8} {'-'*8}  {'-'*30}")
    for c in ["A_baseline", "B1_adam_emb_1e3", "B2_adam_emb_3e3", "C_muon_tok_emb"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<20s} {m:>8.4f} {s:>8.4f}  [{vals_str}]")

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

    # Paired deltas vs A baseline
    print(f"\n  Paired deltas vs A_baseline (negative = arm better):")
    for arm in ["B1_adam_emb_1e3", "B2_adam_emb_3e3", "C_muon_tok_emb"]:
        deltas = []
        for seed in seeds:
            row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
            if arm in row and "A_baseline" in row:
                deltas.append(row[arm] - row["A_baseline"])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            n = len(deltas)
            t_stat = m / (s / (n ** 0.5)) if s > 0 else float('inf')
            sig = "**SIG**" if abs(t_stat) > 2.92 else ""
            print(f"  {arm}: mean {m:+.4f} ± {s:.4f}  t={t_stat:+.2f}  n={n}  {sig}")

    # Key comparison: C vs max(B1, B2)
    print(f"\n  Key disambiguation: C (Muon) vs best Adam arm")
    best_adam_arm = None
    best_adam_mean = float("inf")
    for arm in ["B1_adam_emb_1e3", "B2_adam_emb_3e3"]:
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
        print(f"  Muon - best Adam: {c_mean - best_adam_mean:+.4f}")

        # Paired C vs best Adam
        deltas = []
        for seed in seeds:
            row = {r["config"]: r["val_loss"] for r in results if r["seed"] == seed}
            if "C_muon_tok_emb" in row and best_adam_arm in row:
                deltas.append(row["C_muon_tok_emb"] - row[best_adam_arm])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            n = len(deltas)
            t_stat = m / (s / (n ** 0.5)) if s > 0 else float('inf')
            if t_stat < -2.92:
                verdict = "GEOMETRY WINS (Muon > best Adam, p<0.05)"
            elif t_stat > 2.92:
                verdict = "LR STORY (best Adam >= Muon, p<0.05)"
            else:
                verdict = "INCONCLUSIVE at p<0.05 (Muon ~ best Adam)"
            print(f"  Paired delta (C - {best_adam_arm}): {m:+.4f} ± {s:.4f}, t={t_stat:+.2f}, n={n}")
            print(f"  Verdict: {verdict}")

    summary_path = RESULTS_DIR / "adam_emb_lr_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
