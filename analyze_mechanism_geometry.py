"""Geometric analysis of the 5 mechanism checkpoints captured by
sweep_mechanism_checkpoints.py.

Answers: at matched val_loss (Adam@1e-2 and Muon@0.10 are essentially tied),
are their weight SPECTRA also matched? Or does Muon leave a fingerprint
beyond what val_loss captures?

Metrics (from analyze_weight_geometry.py): stable_rank_norm, svd_entropy,
sigma_max, condition number, per-layer-type pooled distributions.

Emits JSON per checkpoint plus a pooled comparison suitable for a quick
plot. Fast: ~7 min total (SVDs on CPU per the earlier REPORT.md).
"""
from __future__ import annotations
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Import the analyzer verbatim so metrics match results/geometry/REPORT.md.
from analyze_weight_geometry import analyze_checkpoint

CKPT_DIR = Path(__file__).parent / "checkpoints"
OUT_DIR = Path(__file__).parent / "results" / "mechanism_geometry"

ARMS = [
    ("A_3e4",  False, "Adam tok_emb @ 3e-4 (baseline)"),
    ("A_1e3",  False, "Adam tok_emb @ 1e-3"),
    ("A_3e3",  False, "Adam tok_emb @ 3e-3"),
    ("A_1e2",  False, "Adam tok_emb @ 1e-2 (ties Muon@0.10 in val)"),
    ("M_0p10", True,  "Muon tok_emb @ 0.10"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for arm, _muon_out_proj_unused, desc in ARMS:
        ckpt = CKPT_DIR / f"mechanism_{arm}_s42.pt"
        if not ckpt.exists():
            print(f"  MISSING {ckpt}")
            continue
        print(f"\nanalyzing {arm}  ({desc})")
        res = analyze_checkpoint(ckpt, muon_out_proj=True, slice_in_proj=True)
        res["arm"] = arm
        res["arm_desc"] = desc
        all_runs.append(res)
        with open(OUT_DIR / f"{arm}.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(OUT_DIR / "all.json", "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)

    # Pooled comparison: mean / stable_rank_norm / svd_entropy / sigma_max per arm,
    # broken down by Muon-routed vs Adam-routed tensors (inferred from the arm).
    print("\n" + "=" * 70)
    print("POOLED SPECTRA PER ARM (across all 2D weights)")
    print("=" * 70)
    print(f"  {'arm':<8s} {'layer_type':<24s} {'n':>4s} "
          f"{'sr_norm':>10s} {'entropy':>10s} {'sigma_max':>10s}")
    print("  " + "-" * 70)
    for r in all_runs:
        arm = r["arm"]
        # Group layers by simple type key
        buckets = defaultdict(list)
        for lname, lm in r["layers"].items():
            if "tok_emb" in lname:
                key = "tok_emb"
            elif "pos_emb" in lname:
                key = "pos_emb"
            elif "in_proj" in lname and "z" in lname.split("/")[-1]:
                key = "in_proj.z"
            elif "in_proj" in lname and "x" in lname.split("/")[-1]:
                key = "in_proj.x"
            elif "mlp.w1" in lname:
                key = "mlp.w1"
            elif "mlp.w2" in lname:
                key = "mlp.w2"
            elif "mlp.w3" in lname:
                key = "mlp.w3"
            elif "out_proj" in lname:
                key = "mamba_out_proj"
            else:
                continue
            buckets[key].append(lm)
        for key in sorted(buckets):
            lms = buckets[key]
            sr = statistics.mean(x["stable_rank_norm"] for x in lms)
            en = statistics.mean(x["svd_entropy"] for x in lms)
            sm = statistics.mean(x["sigma_max"] for x in lms)
            print(f"  {arm:<8s} {key:<24s} {len(lms):>4d} "
                  f"{sr:>10.4f} {en:>10.4f} {sm:>10.4f}")
    # Headline comparison: tok_emb spectral stats across all 5 arms.
    print(f"\n  tok_emb spectrum across arms:")
    print(f"  {'arm':<8s} {'sr_norm':>10s} {'entropy':>10s} {'sigma_max':>10s}  {'desc'}")
    for r in all_runs:
        tok = next((lm for ln, lm in r["layers"].items() if "tok_emb" in ln), None)
        if tok is None:
            continue
        print(f"  {r['arm']:<8s} {tok['stable_rank_norm']:>10.4f} "
              f"{tok['svd_entropy']:>10.4f} {tok['sigma_max']:>10.4f}  {r['arm_desc']}")

    print(f"\nWrote per-arm JSONs and pooled all.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
