"""Adam emb_lr follow-up: 3e-3 and 1e-2, 3 seeds each at 5k.

Extends sweep_adam_emb_1e3_3seeds.py. Earlier mechanism check showed
Adam@1e-3 closed 65% of the Muon-vs-Adam@3e-4 gap (residual 0.097 nats).
This sweep tests whether Adam@3e-3 or 1e-2 (still tok_emb-only via
--adam_emb_lr, so global Adam stays at 3e-4) closes MORE of that residual.

If Adam@3e-3 or 1e-2 matches or beats Muon@0.10 (mean 5.030), the
residual-geometry claim collapses into a full LR story.
If they plateau/regress/diverge, the 0.097 nat geometry signal is real.

Decision budget for an Adam-tok_emb-only LR ceiling (nvidia finding #10
only applies to GLOBAL Adam@1e-2 — here we raise only the tok_emb group,
so divergence through attention/MLP doesn't apply).

Arms: {3e-3, 1e-2} x seeds {42, 137, 2024} = 6 runs.
val_decomp + gen_probe enabled so we also collect PAPL-style uniform_nll
and rep_4/distinct_4 diagnostics for these Adam arms.
Total: ~80 min.
"""
import sys
import json
import time
import glob
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
    result_path = RESULTS_DIR / f"adamhi_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def load_paired(pattern, seeds, label):
    out = {}
    for s in seeds:
        cand = pattern.format(seed=s)
        matches = glob.glob(cand)
        if not matches:
            print(f"  [warn] {label} seed {s}: no file at {cand}")
            continue
        with open(matches[0]) as f:
            rec = json.load(f)
        if rec.get("status") != "OK":
            print(f"  [warn] {label} seed {s}: status={rec.get('status')}")
            continue
        v = rec["val_loss"]
        if math.isnan(v) or v == float("inf"):
            print(f"  [warn] {label} seed {s}: val_loss={v} (rejected)")
            continue
        out[s] = v
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # NOTE: intentionally NO --val_decomp and NO --gen_probe here.
    # Both advance the global CUDA RNG (compute_loss_decomp does its own
    # forward with a fresh t/mask; model.sample() calls torch.rand_like).
    # The existing paired baselines (tokemb_baseline, adam1e3, emblr_emblr0p1)
    # were recorded BEFORE these flags existed, so enabling them here would
    # desynchronize the mask/timestep RNG in the new runs relative to the
    # baselines starting at the first val step, breaking paired-seed pairing.
    # We'll collect decomp / gen metrics for these arms in a separate
    # follow-up once we have a matching flag-enabled baseline batch.
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

    seeds = [42, 137, 2024]
    conditions = {
        "adam3e3": common + ["--adam_emb_lr", "3e-3"],
        "adam1e2": common + ["--adam_emb_lr", "1e-2"],
    }

    new_results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    total_runs = len(conditions) * len(seeds)

    # Paired order: for each seed, run both Adam LRs before next seed.
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
            new_results.append(rec)

    total_elapsed = time.perf_counter() - total_t0

    # Pool with existing paired data.
    A3e4 = load_paired(str(RESULTS_DIR / "tokemb_baseline_s{seed}.json"), seeds, "Adam@3e-4")
    A1e3 = load_paired(str(RESULTS_DIR / "adam1e3_s{seed}.json"), seeds, "Adam@1e-3")
    A3e3 = {r["seed"]: r["val_loss"] for r in new_results if r["config"] == "adam3e3" and r["status"] == "OK"}
    A1e2 = {r["seed"]: r["val_loss"] for r in new_results if r["config"] == "adam1e2" and r["status"] == "OK"}
    MuonC = load_paired(str(RESULTS_DIR / "emblr_emblr0p1_s{seed}.json"), seeds, "Muon@0.10")

    print(f"\n{'='*60}")
    print(f"ADAM EMB-LR LADDER (new runtime: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    print(f"\n  Per-seed val_loss:")
    print(f"  {'Seed':>6s} {'Adam 3e-4':>12s} {'Adam 1e-3':>12s} {'Adam 3e-3':>12s} {'Adam 1e-2':>12s} {'Muon 0.10':>12s}")
    for s in seeds:
        row = [A3e4.get(s, float("nan")), A1e3.get(s, float("nan")),
               A3e3.get(s, float("nan")), A1e2.get(s, float("nan")),
               MuonC.get(s, float("nan"))]
        print(f"  {s:>6d} " + " ".join(f"{v:>12.4f}" for v in row))

    print(f"\n  Means:")
    for label, d in [("Adam@3e-4", A3e4), ("Adam@1e-3", A1e3),
                      ("Adam@3e-3", A3e3), ("Adam@1e-2", A1e2),
                      ("Muon@0.10", MuonC)]:
        vs = [v for v in d.values() if not (math.isnan(v) or v == float("inf"))]
        if len(vs) >= 2:
            m = statistics.mean(vs)
            sd = statistics.stdev(vs)
            print(f"  {label:<12s}  mean={m:.4f}  std={sd:.4f}  n={len(vs)}")

    def paired(lhs, rhs, tag):
        deltas = []
        for s in seeds:
            if s in lhs and s in rhs:
                deltas.append(lhs[s] - rhs[s])
        if len(deltas) < 2:
            print(f"  {tag}: insufficient paired data")
            return
        m = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        n = len(deltas)
        t = m / (sd / n**0.5) if sd > 0 else float("inf")
        sig = "**SIG**" if abs(t) > 2.92 else ""
        print(f"  {tag}: mean={m:+.4f} std={sd:.4f} t={t:+.2f} n={n} {sig}")

    print(f"\n  Paired deltas (vs each other):")
    paired(A3e3, A1e3, "3e-3 vs 1e-3 (does Adam keep improving?)")
    paired(A1e2, A3e3, "1e-2 vs 3e-3 (diminishing/regression?)")
    paired(A3e3, MuonC, "Adam@3e-3 vs Muon@0.10 (geometry-closer?)")
    paired(A1e2, MuonC, "Adam@1e-2 vs Muon@0.10 (full LR story?)")

    # Best-Adam vs Muon verdict
    best_adam_key, best_adam_mean = None, float("inf")
    for lbl, d in [("3e-4", A3e4), ("1e-3", A1e3), ("3e-3", A3e3), ("1e-2", A1e2)]:
        vs = [v for v in d.values() if not (math.isnan(v) or v == float("inf"))]
        if vs:
            m = statistics.mean(vs)
            if m < best_adam_mean:
                best_adam_mean = m
                best_adam_key = lbl
    if best_adam_key and MuonC:
        mc = statistics.mean([v for v in MuonC.values() if not (math.isnan(v) or v == float("inf"))])
        print(f"\n  VERDICT:")
        print(f"    Best Adam arm: Adam@{best_adam_key} -> {best_adam_mean:.4f}")
        print(f"    Muon@0.10 -> {mc:.4f}")
        print(f"    Muon minus best Adam: {mc - best_adam_mean:+.4f}")
        if mc - best_adam_mean > -0.03:
            print(f"    LR story wins: Adam at tuned LR matches or beats Muon.")
        elif mc - best_adam_mean < -0.05:
            print(f"    Residual geometry survives even vs best Adam LR.")
        else:
            print(f"    Borderline.")

    summary_path = RESULTS_DIR / "adam_emb_higher_lr_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "new_runs": new_results,
            "A3e4": A3e4, "A1e3": A1e3,
            "A3e3": A3e3, "A1e2": A1e2,
            "MuonC": MuonC, "seeds": seeds,
        }, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
