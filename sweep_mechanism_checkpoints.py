"""Capture checkpoints at 5 key Adam-LR-vs-Muon arms for geometric analysis.

After the mechanism sweep showed Adam@1e-2 on tok_emb is essentially tied
with Muon@0.10 in val_loss, the sharper question becomes: are their
WEIGHT SPECTRA also the same? If Adam at tuned per-group LR produces
the same stable rank / SVD entropy / sigma_max as Muon's orthogonalized
updates, the "residual geometry" story is fully dead. If Muon still
produces a measurably flatter spectrum at matched val_loss, geometry
matters beyond LR.

This sweep captures 5 checkpoints at seed 42 (single-seed is enough for
spectral metrics per results/geometry/REPORT.md: seed variance is 5-10x
smaller than optimizer variance).

All flags matched (no --val_decomp, no --gen_probe; pure training) so
the training trajectory is apples-to-apples.

Arms:
  A_3e4  : Adam tok_emb @ 3e-4      (baseline)
  A_1e3  : Adam tok_emb @ 1e-3
  A_3e3  : Adam tok_emb @ 3e-3
  A_1e2  : Adam tok_emb @ 1e-2       (essentially ties Muon@0.10 in val)
  M_0p10 : Muon tok_emb @ 0.10       (current best)

5 runs x seed 42 x 5k steps = ~65 min. --save_best writes best-val
checkpoint to checkpoints/mechanism_<arm>_s42.pt for later analysis.
"""
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR = Path(__file__).parent / "checkpoints"


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
    ckpt = CKPT_DIR / f"mechanism_{name}_s42.pt"
    record = {"name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
              "status": status, "error": error, "ckpt": str(ckpt),
              "timestamp": datetime.now().isoformat()}
    result_path = RESULTS_DIR / f"ckpt_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, ckpt={ckpt.name}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "5000", "--val_every", "250", "--log_every", "500",
        "--warmup_steps", "200", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
        "--seed", "42",
        "--save_best",
        # End-of-training large gen probe so each arm's rep_4/distinct_4
        # is resolved past the per-val n=16 noise floor for direct
        # comparison across the LR ladder + Muon arm.
        "--gen_probe_final",
        "--gen_probe_final_samples", "64",
        "--gen_probe_final_seq_len", "128",
        "--gen_probe_final_steps", "128",
    ]

    # Each arm gets its own save_path and its own routing flag.
    arms = [
        ("A_3e4",  []),
        ("A_1e3",  ["--adam_emb_lr", "1e-3"]),
        ("A_3e3",  ["--adam_emb_lr", "3e-3"]),
        ("A_1e2",  ["--adam_emb_lr", "1e-2"]),
        ("M_0p10", ["--muon_tok_emb", "--muon_emb_lr", "0.10"]),
    ]

    total_t0 = time.perf_counter()
    for i, (name, extra) in enumerate(arms, start=1):
        ckpt_path = str(CKPT_DIR / f"mechanism_{name}_s42.pt")
        run_args = common + ["--save_path", ckpt_path] + extra
        print(f"\n{'='*60}\n[{i}/{len(arms)}] {name} -> {ckpt_path}\n{'='*60}")
        run_one(name, run_args)

    print(f"\n{'='*60}")
    print(f"CHECKPOINT CAPTURE DONE (total: {(time.perf_counter() - total_t0)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Checkpoints in {CKPT_DIR}:")
    for name, _ in arms:
        p = CKPT_DIR / f"mechanism_{name}_s42.pt"
        if p.exists():
            print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  {p.name}  MISSING")


if __name__ == "__main__":
    main()
