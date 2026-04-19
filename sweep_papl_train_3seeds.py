"""PAPL training port to Mamba3: does PAPL drop planner_w_nll on our stack?

Context: nvidia agent fine-tuned their 125M transformer with PAPL and saw
rep_4 drop 0.153 -> 0.128 over a 5k fine-tune. Does the same objective
improve val_loss / rep_4 on our Mamba3 + FineWeb-Edu stack?

Design: 2 arms x 3 seeds = 6 runs at 5k steps (from scratch). Both arms
use --val_decomp and --gen_probe with matched flags so seed pairing is
preserved (baseline for this comparison is RE-RUN with flags, not the
pre-flag emblr0p1 runs from overnight).

  baseline: our best recipe (Muon tok_emb @ 0.10)
  papl:     baseline + --papl_train --papl_alpha 1.0 --papl_tau 0.3

Expected signature of successful PAPL (Peng Fig 3b, nvidia finding):
  - planner_w_nll_masked drops meaningfully under PAPL
  - uniform_nll_masked barely moves (PAPL deallocates capacity from
    positions the planner rarely visits — uniform reflects this)
  - rep_4 decreases; distinct_4 increases (generation less repetitive)

Expected runtime: ~80 min.
"""
import sys
import json
import time
import math
import statistics
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
    if status == "OK" and (math.isnan(val_loss) or val_loss > 15.0):
        status, error = "FAIL", f"diverged (val_loss={val_loss})"
        val_loss = float("inf")
    record = {"name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
              "status": status, "error": error, "timestamp": datetime.now().isoformat()}
    result_path = RESULTS_DIR / f"papl_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Both arms use val_decomp + gen_probe so we can measure the PAPL
    # signature (planner_w drop) and gen-quality (rep_4/distinct_4).
    # Flags are matched across arms so seed-paired RNG consumption is
    # identical between baseline and PAPL runs.
    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj", "--muon_tok_emb", "--muon_emb_lr", "0.10",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
        "--val_decomp",
        "--gen_probe", "--gen_probe_every", "4",
    ]

    seeds = [42, 137, 2024]
    conditions = {
        "baseline": common,
        # τ=0.1 per nvidia agent's preliminary finding (still sweeping; previously
        # they thought 0.3; now converging to 0.1). Sharper planner peaks → more
        # localized reweight on high-gt-logprob positions. If their final answer
        # is different, we rerun.
        "papl": common + ["--papl_train", "--papl_alpha", "1.0", "--papl_tau", "0.1"],
    }

    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    for seed in seeds:
        print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")
        for name, base_args in conditions.items():
            run_idx += 1
            run_name = f"{name}_s{seed}"
            args = base_args + ["--seed", str(seed)]
            print(f"\n{'='*60}\n[{run_idx}/{total_runs}] {run_name}\n{'='*60}")
            rec = run_one(run_name, args)
            rec["seed"] = seed
            rec["config"] = name
            results.append(rec)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}")
    print(f"PAPL TRAINING VS BASELINE (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    by_config = {}
    for r in results:
        if r["status"] == "OK":
            by_config.setdefault(r["config"], {})[r["seed"]] = r["val_loss"]

    baseline = by_config.get("baseline", {})
    papl = by_config.get("papl", {})

    print(f"\n  Val loss (uniform MDLM, PAPL reweight disabled for val):")
    print(f"  {'Seed':>6s} {'baseline':>12s} {'papl':>12s} {'delta':>12s}")
    deltas = []
    for s in seeds:
        b, p = baseline.get(s, float("nan")), papl.get(s, float("nan"))
        d = p - b if not (math.isnan(b) or math.isnan(p)) else float("nan")
        if not math.isnan(d):
            deltas.append(d)
        print(f"  {s:>6d} {b:>12.4f} {p:>12.4f} {d:>+12.4f}")

    if len(deltas) >= 2:
        m = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        n = len(deltas)
        t = m / (sd / n**0.5) if sd > 0 else float("inf")
        sig = "**SIG**" if abs(t) > 2.92 else ""
        print(f"\n  Mean delta (papl - baseline): {m:+.4f} ± {sd:.4f}  t={t:+.2f}  n={n}  {sig}")
        if m < -0.02:
            print(f"  PAPL IMPROVES val_loss on Mamba3. Check training log for "
                  f"planner_w drop + rep_4 trajectory.")
        elif m > 0.02:
            print(f"  PAPL HURTS val_loss on Mamba3 — expected for uniform_nll under "
                  f"PAPL, but if papl_w didn't drop either, the objective didn't work.")
        else:
            print(f"  PAPL NEUTRAL on val_loss. Decomp + gen metrics in log determine "
                  f"whether the reweight did anything useful.")

    summary_path = RESULTS_DIR / "papl_train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
