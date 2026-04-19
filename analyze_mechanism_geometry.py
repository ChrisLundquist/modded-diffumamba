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

# (arm_name, is_muon_tok_emb_arm, description)
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
    for arm, is_muon_tok_emb, desc in ARMS:
        ckpt = CKPT_DIR / f"mechanism_{arm}_s42.pt"
        if not ckpt.exists():
            print(f"  MISSING {ckpt}")
            continue
        print(f"\nanalyzing {arm}  ({desc})")
        # muon_out_proj flag only affects the 'routing' label in analyze_checkpoint,
        # not any metric. None of the mechanism arms route out_proj through Muon
        # (out_proj is NOT in the --muon_out_proj path for any of them — this sweep
        # doesn't enable that flag). So pass False to get accurate routing labels.
        res = analyze_checkpoint(ckpt, muon_out_proj=False, slice_in_proj=True)
        res["arm"] = arm
        res["arm_desc"] = desc
        all_runs.append(res)
        with open(OUT_DIR / f"{arm}.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(OUT_DIR / "all.json", "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)

    # Pooled comparison: mean / stable_rank_norm / svd_entropy / sigma_max per arm,
    # broken down by layer-type key derived from the matrix name.
    # analyze_checkpoint returns results under res["matrices"], with each entry
    # having {"layer_type", "routing", "metrics": {...}}.
    print("\n" + "=" * 70)
    print("POOLED SPECTRA PER ARM (across all 2D weights)")
    print("=" * 70)
    print(f"  {'arm':<8s} {'layer_type':<24s} {'n':>4s} "
          f"{'sr_norm':>10s} {'entropy':>10s} {'sigma_max':>10s}")
    print("  " + "-" * 70)
    for r in all_runs:
        arm = r["arm"]
        buckets = defaultdict(list)
        for lname, lentry in r["matrices"].items():
            metrics = lentry.get("metrics", {})
            if not metrics or "stable_rank_norm" not in metrics:
                continue
            if "tok_emb" in lname:
                key = "tok_emb"
            elif "pos_emb" in lname:
                key = "pos_emb"
            elif "in_proj" in lname and lname.endswith("::z"):
                key = "in_proj.z"
            elif "in_proj" in lname and lname.endswith("::x"):
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
            buckets[key].append(metrics)
        for key in sorted(buckets):
            ms = buckets[key]
            sr = statistics.mean(x["stable_rank_norm"] for x in ms)
            en = statistics.mean(x["svd_entropy"] for x in ms)
            sm = statistics.mean(x["sigma_max"] for x in ms)
            print(f"  {arm:<8s} {key:<24s} {len(ms):>4d} "
                  f"{sr:>10.4f} {en:>10.4f} {sm:>10.4f}")

    # Headline: tok_emb spectral stats across all 5 arms.
    print(f"\n  tok_emb spectrum across arms:")
    print(f"  {'arm':<8s} {'sr_norm':>10s} {'entropy':>10s} {'sigma_max':>10s}  {'desc'}")
    for r in all_runs:
        tok_entry = next((lentry for lname, lentry in r["matrices"].items()
                          if "tok_emb" in lname), None)
        if tok_entry is None or "metrics" not in tok_entry:
            continue
        m = tok_entry["metrics"]
        print(f"  {r['arm']:<8s} {m['stable_rank_norm']:>10.4f} "
              f"{m['svd_entropy']:>10.4f} {m['sigma_max']:>10.4f}  {r['arm_desc']}")

    print(f"\nWrote per-arm JSONs and pooled all.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
