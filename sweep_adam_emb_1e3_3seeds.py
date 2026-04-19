"""Lean 3-arm mechanism check: Adam-embed-LR vs Muon-embed.

Goal: answer the ONE load-bearing question the reviewer raised —
does Adam @ tok_emb_lr=1e-3 close the gap to Muon @ tok_emb_lr=0.10?

We already have paired-seed data for two of the three arms:
  A (Adam @ 3e-4): results/tokemb_baseline_s{42,137,2024}.json
  C (Muon @ 0.10): results/emblr_emblr0p1_s{42,137,2024}.json

This sweep ONLY runs Arm B (Adam @ 1e-3 via --adam_emb_lr), 3 seeds,
and then pools all three arms into the paired analysis.

3 runs, ~40 min.

Notes on code-path parity with the existing Arm A data:
  - tokemb_baseline runs were pre-refactor (single Adam group).
  - Post-refactor baseline (no flags) path: tok_emb routed through the
    same main Adam group at adam_lr=3e-4, bit-identical updates per-param
    (state is keyed by Tensor id, not group). No re-baseline needed; if the
    result is ambiguous we'll re-run Arm A as a sanity check.

Decision rule (post-run):
  - If Adam@1e-3 mean val_loss is within 0.03 nats of Muon@0.10 (noise
    floor) across all 3 paired deltas -> "LR story" confirmed;
    skip exp-b (scaling sweep) as the geometry claim is dead.
  - If Muon@0.10 still beats Adam@1e-3 by >= 0.05 nats paired ->
    some residual geometry benefit exists; consider Adam@3e-3/1e-2
    follow-ups or proceed to exp-b.
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
    result_path = RESULTS_DIR / f"adam1e3_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def load_paired(pattern, seeds, label):
    """Load existing per-seed val_losses from a glob pattern.
    Returns {seed: val_loss} dict, warning about any missing/failed entries."""
    out = {}
    for s in seeds:
        cand = pattern.format(seed=s)
        matches = glob.glob(cand)
        if not matches:
            print(f"  [warn] {label} seed {s}: no file matching {cand}")
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

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
        "--adam_emb_lr", "1e-3",
    ]

    seeds = [42, 137, 2024]
    new_results = []
    total_t0 = time.perf_counter()

    for i, seed in enumerate(seeds, start=1):
        name = f"s{seed}"
        args = common + ["--seed", str(seed)]
        print(f"\n{'='*60}\n[{i}/3] adam1e3_s{seed}\n{'='*60}")
        record = run_one(name, args)
        record["seed"] = seed
        new_results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    # Pool with existing A (tokemb_baseline) and C (emblr_emblr0p1) data.
    A = load_paired(str(RESULTS_DIR / "tokemb_baseline_s{seed}.json"), seeds, "A(Adam@3e-4)")
    B = {r["seed"]: r["val_loss"] for r in new_results if r["status"] == "OK"}
    C = load_paired(str(RESULTS_DIR / "emblr_emblr0p1_s{seed}.json"), seeds, "C(Muon@0.10)")

    print(f"\n{'='*60}")
    print(f"3-ARM MECHANISM CHECK (new runtime: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    print(f"\n  Per-seed val_loss:")
    print(f"  {'Seed':>6s} {'A:Adam@3e-4':>14s} {'B:Adam@1e-3':>14s} {'C:Muon@0.10':>14s}")
    for s in seeds:
        a = A.get(s, float("nan"))
        b = B.get(s, float("nan"))
        c = C.get(s, float("nan"))
        print(f"  {s:>6d} {a:>14.4f} {b:>14.4f} {c:>14.4f}")

    print(f"\n  Means:")
    for label, d in [("A Adam@3e-4", A), ("B Adam@1e-3", B), ("C Muon@0.10", C)]:
        vs = list(d.values())
        if len(vs) >= 2:
            m = statistics.mean(vs)
            s = statistics.stdev(vs)
            print(f"  {label:<15s}  mean={m:.4f}  std={s:.4f}  n={len(vs)}")

    # Paired deltas
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

    print(f"\n  Paired deltas:")
    paired(B, A, "B - A (Adam@1e-3 vs Adam@3e-4; replicates nvidia finding #9)")
    paired(C, A, "C - A (Muon@0.10 vs Adam@3e-4; our original claim)")
    paired(C, B, "C - B (Muon@0.10 vs Adam@1e-3; the load-bearing question)")

    # Verdict
    if len(B) == 3 and len(C) == 3:
        deltas_cb = [C[s] - B[s] for s in seeds if s in C and s in B]
        m = statistics.mean(deltas_cb)
        print(f"\n  VERDICT on geometry-vs-LR (n=3 paired; interpret with caution):")
        if abs(m) <= 0.03:
            print(f"    |delta| <= 0.03 nats: consistent with an LR STORY.")
            print(f"    Geometry effect at this scale is <= 0.03 nats (one std of our noise).")
            print(f"    Recommend skip exp-b; consider writing up as 'tok_emb wants higher LR'.")
        elif m < -0.05:
            print(f"    Muon beats Adam@1e-3 by {abs(m):.3f} nats -> residual geometry signal.")
            print(f"    Consider running Adam@3e-3 / 1e-2 before committing to exp-b.")
        elif m > 0.05:
            print(f"    Adam@1e-3 beats Muon@0.10 by {m:.3f} nats -> LR story favors Adam.")
            print(f"    Modded-nanogpt lore is correct, just underappreciated at our scale.")
        else:
            print(f"    Borderline (delta {m:+.3f}). Consider adding Adam@3e-3 arm to clarify.")

    summary_path = RESULTS_DIR / "adam1e3_mechanism_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "new_runs": new_results,
            "existing_A_seeds": A,
            "existing_C_seeds": C,
            "seeds": seeds,
        }, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
