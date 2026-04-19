"""Scaling sweep: does Muon-tok_emb's win grow or shrink with model size?

User hypothesis: the -0.1 to -0.2 nat advantage of Muon-on-tok_emb at
5k on quokka might be an artifact of quokka being embedding-dominant
(tok_emb = 61% of params). At larger scales the embedding fraction
falls: 29% at 10L x 640d (111.7M), ~4% at 1B. If the advantage is
real-in-geometry it should persist; if it's an LR/embedding-fraction
artifact it will shrink.

Arms per scale:
  Adam baseline: muon blocks + Adam tok_emb @ 3e-4  (current default)
  Muon best:     muon blocks + Muon tok_emb @ <winning emb_lr>

Winning emb_lr is pooled from emblr_*.json (exp3 + extension sweep) at
startup, same logic as sweep_adam_emb_lr_ablation_v2_5k.py.

Scales:
  tiny    (8.4M,   d=128, 4L,  seq=256, bs=8)    - fast
  quokka  (31.5M,  d=384, 4L,  seq=1024, bs=8)   - our headline
  10L640d (111.7M, d=640, 10L, seq=1024, bs=4)   - largest that fits

3 scales x 2 arms x 3 seeds = 18 runs.
ETA: tiny ~4 min, quokka ~13 min, 10L640d ~30-40 min per run.
Total: ~5 hours.
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
    by_lr = defaultdict(list)
    for path in glob.glob(str(RESULTS_DIR / "emblr_*.json")):
        name = Path(path).stem
        m = re.search(r"emblr0p(\d+)", name)
        if not m:
            continue
        lr = float("0." + m.group(1).rstrip("_"))
        with open(path) as f:
            rec = json.load(f)
        if rec.get("status") != "OK" or rec["val_loss"] == float("inf"):
            continue
        by_lr[lr].append(rec["val_loss"])
    if not by_lr:
        raise RuntimeError("No emblr_*.json results found; run exp3 first.")
    means = {lr: statistics.mean(vs) for lr, vs in by_lr.items()}
    best_lr = min(means, key=means.get)
    print(f"  Pooled emb_lr curve:")
    for lr in sorted(means):
        print(f"    lr={lr:>6g}  n={len(by_lr[lr])}  mean={means[lr]:.4f}")
    print(f"  -> Muon arm uses emb_lr={best_lr}")
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
    result_path = RESULTS_DIR / f"scaling_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    winning_muon_lr = pick_best_muon_emb_lr()

    # Common args for all scales (training hyperparameters shared).
    shared = [
        "--max_steps", "5000", "--val_every", "500", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
        # Decomp + gen probes enabled on all arms so seed pairing within
        # scaling is preserved and we collect comparable numbers at each
        # scale. NOT paired with overnight runs — scaling is standalone.
        "--val_decomp",
        "--gen_probe", "--gen_probe_every", "2",
        # End-of-training big probe on the best-val checkpoint.
        "--save_best",
        "--gen_probe_final",
        "--gen_probe_final_samples", "64",
        "--gen_probe_final_seq_len", "128",
        "--gen_probe_final_steps", "128",
    ]

    # Per-scale config + batch settings.
    scales = {
        "tiny":    ["--config", "tiny",   "--batch_size", "8"],
        "quokka":  ["--config", "quokka", "--batch_size", "8"],
        "10L640d": ["--config", "quokka", "--n_layers", "10",
                    "--d_model", "640",   "--batch_size", "4"],
    }

    # Per-scale emb_lr caps for the large scales (NaN safety). At 10L x 640d,
    # clamp Muon emb_lr to <=0.05. Adam embed lr=1e-2 on a wider tok_emb may
    # also need care if it diverges, but nvidia's GLOBAL Adam@1e-2 divergence
    # is attention/MLP, not the embedding group; keep the Adam-tuned emb_lr
    # at 1e-2 unless we observe blowup (the run_one NaN guard catches it).
    MAX_LR_FOR_LARGE_SCALES = 0.05
    per_scale_muon_lr = {}
    per_scale_adam_emb_lr = {}  # for the adam_tuned arm
    for s in scales:
        lr = winning_muon_lr
        if s in ("10L640d",) and lr > MAX_LR_FOR_LARGE_SCALES:
            print(f"  Clamping {s} muon_emb_lr {lr} -> {MAX_LR_FOR_LARGE_SCALES} (NaN safety)")
            lr = MAX_LR_FOR_LARGE_SCALES
        per_scale_muon_lr[s] = lr
        per_scale_adam_emb_lr[s] = 1e-2  # from our own A sweep: tok_emb-only @ 1e-2 ties Muon@0.10

    seeds = [42, 137, 2024]
    results = []
    total_t0 = time.perf_counter()
    run_idx = 0
    # arms are built per-scale inside the loop; arm count is constant = 3
    # (adam_baseline, adam_tuned, muon_tok_emb).
    total_runs = len(scales) * 3 * len(seeds)

    # Order: for each scale, interleave arms per seed (paired). Two arms per
    # scale: the mechanism verdict at quokka already settled that Adam@3e-4
    # is simply undertrained, so the "default Adam" arm is an uninteresting
    # reference. The research question is whether Adam@1e-2 ≈ Muon@0.10 holds
    # at scale, so we compare those two directly.
    CKPT_DIR = Path(__file__).parent / "checkpoints"
    CKPT_DIR.mkdir(exist_ok=True)
    for scale_name, scale_args in scales.items():
        muon_lr_here = per_scale_muon_lr[scale_name]
        adam_emb_lr_here = per_scale_adam_emb_lr[scale_name]
        arms = {
            "adam_tuned":    ["--adam_emb_lr", str(adam_emb_lr_here)],
            "muon_tok_emb":  ["--muon_tok_emb", "--muon_emb_lr", str(muon_lr_here)],
        }
        print(f"\n{'='*60}\n# SCALE: {scale_name} "
              f"(muon emb_lr={muon_lr_here}, adam_tuned emb_lr={adam_emb_lr_here})\n{'='*60}")
        for seed in seeds:
            print(f"\n  -- seed {seed} --")
            for arm_name, arm_args in arms.items():
                run_idx += 1
                name = f"{scale_name}_{arm_name}_s{seed}"
                # Per-run save path so --save_best + --gen_probe_final fire on each.
                save_path = str(CKPT_DIR / f"scaling_{name}.pt")
                full_args = (scale_args + shared + arm_args +
                             ["--seed", str(seed), "--save_path", save_path])
                print(f"\n[{run_idx}/{total_runs}] {name}")
                record = run_one(name, full_args)
                record["scale"] = scale_name
                record["arm"] = arm_name
                record["seed"] = seed
                if arm_name == "muon_tok_emb":
                    record["emb_lr"] = muon_lr_here
                elif arm_name == "adam_tuned":
                    record["emb_lr"] = adam_emb_lr_here
                else:
                    record["emb_lr"] = None
                results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}")
    print(f"SCALING SWEEP RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"  Muon arm used emb_lr={winning_muon_lr}")
    print(f"{'='*60}")

    # Per-scale summary + paired deltas
    for scale_name in scales:
        print(f"\n  ## Scale: {scale_name} ##")
        by_arm = defaultdict(list)
        for r in results:
            if r["scale"] == scale_name:
                by_arm[r["arm"]].append((r["seed"], r["val_loss"]))
        for arm in ["adam_baseline", "adam_tuned", "muon_tok_emb"]:
            entries = by_arm.get(arm, [])
            vals = [v for _, v in entries]
            if len(vals) >= 2:
                m = statistics.mean(vals)
                s = statistics.stdev(vals)
                vs = ", ".join(f"{v:.4f}" for v in vals)
                print(f"    {arm:<18s} mean={m:.4f} std={s:.4f}  [{vs}]")

        def paired(lhs_arm, rhs_arm, tag):
            deltas = []
            for seed in seeds:
                a = [v for s, v in by_arm[rhs_arm] if s == seed]
                b = [v for s, v in by_arm[lhs_arm] if s == seed]
                if a and b:
                    deltas.append(b[0] - a[0])
            if len(deltas) >= 2:
                md = statistics.mean(deltas)
                sd = statistics.stdev(deltas)
                n = len(deltas)
                t_stat = md / (sd / (n ** 0.5)) if sd > 0 else float("inf")
                # df=2 p=0.05 two-sided = 4.303; we report direction consistency
                # rather than literal p-value.
                dir_note = ("3/3 agree" if n == 3 and all((d < 0) == (md < 0) for d in deltas)
                            else f"{n} seeds")
                print(f"    {tag}: {md:+.4f} ± {sd:.4f}  t={t_stat:+.2f}  ({dir_note})")

        paired("muon_tok_emb", "adam_baseline", "muon - adam_baseline (original claim)")
        paired("adam_tuned",   "adam_baseline", "adam_tuned - adam_baseline (LR story, nvidia #9)")
        paired("muon_tok_emb", "adam_tuned",    "muon - adam_tuned (residual geometry @ scale)")

    summary_path = RESULTS_DIR / "scaling_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"winning_muon_emb_lr": winning_muon_lr, "results": results}, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
