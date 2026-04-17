"""Train the largest model that fits in 16GB: 12L×640d (130.8M params).

Starts with 10k steps as a promising-check. If val_loss < 5.07 (current
quokka best), we have a sign the full stack transfers and can continue.

Uses our best config:
- Muon-VS + out_proj, lr=0.01
- FineWeb-Edu, gamma=1.5
- Saves checkpoints every 10k steps
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def run(max_steps, run_name, warmup=None):
    import gc, torch

    if warmup is None:
        # Scale warmup with steps (4% of total, capped at 800)
        warmup = min(800, max(200, max_steps // 25))

    saved_argv = sys.argv
    sys.argv = [
        "train.py",
        "--config", "quokka",
        # Override config for 12L×640d
        "--n_layers", "12",
        "--d_model", "640",
        "--batch_size", "8",
        "--max_steps", str(max_steps),
        "--val_every", "500",
        "--log_every", "500",
        "--warmup_steps", str(warmup),
        "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.01", "--adam_lr", "3e-4",
        "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--data_dir", "data/fineweb-edu-10B",
        "--save_best", "--save_every", "10000",
        "--save_path", str(CKPT_DIR / f"{run_name}.pt"),
        "--seed", "42",
    ]

    t0 = time.perf_counter()
    try:
        from train import parse_args, train
        args = parse_args()
        # Also override d_state, headdim, cond_dim for 640d (matching small/base scale)
        val_loss = train(args)
        elapsed = time.perf_counter() - t0
        status = "OK"
        error = ""
    except Exception as e:
        import traceback
        elapsed = time.perf_counter() - t0
        val_loss = float("inf")
        status = "FAIL"
        error = traceback.format_exc()
    finally:
        sys.argv = saved_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "run": run_name, "max_steps": max_steps,
        "val_loss": val_loss, "elapsed_seconds": elapsed,
        "status": status, "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / f"large_{run_name}.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[{status}] {run_name}: val_loss={val_loss:.4f}, time={elapsed/60:.1f} min")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    # Phase 1: 10k "promising" check (~42 min)
    # Gate: val_loss < 5.20 means the config transfers well enough to continue
    print(f"{'#'*60}")
    print(f"# Phase 1: 12L×640d at 10k steps (gate: val_loss < 5.20)")
    print(f"{'#'*60}")
    r1 = run(10000, "12L640d_10k")

    if r1["val_loss"] < 5.20 and r1["status"] == "OK":
        print(f"\n✓ Phase 1 promising (val_loss={r1['val_loss']:.4f}). Continuing to 50k.")

        # Phase 2: 50k (~3.5 hrs) — get into the data-efficient regime
        print(f"\n{'#'*60}")
        print(f"# Phase 2: 12L×640d at 50k steps (gate: val_loss < 4.80)")
        print(f"{'#'*60}")
        r2 = run(50000, "12L640d_50k")

        if r2["val_loss"] < 4.80 and r2["status"] == "OK":
            print(f"\n✓ Phase 2 still improving. Continuing to 100k.")

            # Phase 3: 100k (~7 hrs) — sample quality emerges
            print(f"\n{'#'*60}")
            print(f"# Phase 3: 12L×640d at 100k steps (overnight run)")
            print(f"{'#'*60}")
            r3 = run(100000, "12L640d_100k")
            print(f"\nPhase 3 done: val_loss={r3['val_loss']:.4f}")
        else:
            print(f"\n✗ Phase 2 below gate or failed (val_loss={r2['val_loss']:.4f}). Stopping.")
    else:
        print(f"\n✗ Phase 1 below gate or failed (val_loss={r1['val_loss']:.4f}). Stopping.")
        print(f"  Reference: quokka 31.5M got 5.07 at 10k. 12L×640d should do better if well-tuned.")


if __name__ == "__main__":
    main()
