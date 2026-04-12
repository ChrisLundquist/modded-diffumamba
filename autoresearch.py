"""
autoresearch.py — Automated experiment runner (Karpathy autoresearch style)

Runs train.py with different configurations and tracks results.
Each experiment runs for a fixed step budget, and we compare final
validation loss to find the best setup.

Usage:
    python autoresearch.py --mode compare_optimizers
    python autoresearch.py --mode sweep --budget_steps 500
    python autoresearch.py --mode single --config small --optimizer muon
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

TRAIN_SCRIPT = Path(__file__).parent / "train.py"
RESULTS_DIR = Path(__file__).parent / "results"


def run_experiment(name: str, args: dict, budget_steps: int = None) -> dict:
    """Run a single training experiment and return results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    if budget_steps and "max_steps" not in args:
        cmd.extend(["--max_steps", str(budget_steps)])

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    # Parse output for final val loss
    val_loss = float("inf")
    for line in result.stdout.split("\n"):
        if "best val_loss=" in line:
            try:
                val_loss = float(line.split("best val_loss=")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "val_loss" in line and "|" in line:
            try:
                parts = line.split("val_loss")[1].strip()
                val_loss = float(parts.split()[0].rstrip("*"))
            except (ValueError, IndexError):
                pass

    record = {
        "name": name,
        "args": args,
        "val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        "returncode": result.returncode,
    }

    # Save result
    result_path = RESULTS_DIR / f"{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)

    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"\n[{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")

    return record


def compare_optimizers(args):
    """The core experiment: Muon vs Adam on masked diffusion LM."""
    budget = args.budget_steps
    config = args.config
    batch_size = args.batch_size

    base_args = {
        "config": config,
        "batch_size": batch_size,
        "max_steps": budget,
        "val_every": max(50, budget // 20),
        "log_every": max(10, budget // 100),
        "warmup_steps": min(200, budget // 10),
    }

    experiments = [
        ("adam_baseline", {**base_args, "optimizer": "adam", "adam_lr": 3e-4}),
        ("adam_lr1e3", {**base_args, "optimizer": "adam", "adam_lr": 1e-3}),
        ("muon_default", {**base_args, "optimizer": "muon",
                          "muon_lr": 0.02, "adam_lr": 3e-4}),
        ("muon_lr005", {**base_args, "optimizer": "muon",
                        "muon_lr": 0.005, "adam_lr": 3e-4}),
        ("muon_lr01", {**base_args, "optimizer": "muon",
                       "muon_lr": 0.01, "adam_lr": 3e-4}),
    ]

    results = []
    for name, exp_args in experiments:
        record = run_experiment(name, exp_args)
        results.append(record)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY: Muon vs Adam for Masked Diffusion LM")
    print("=" * 60)
    results.sort(key=lambda r: r["val_loss"])
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {r['name']:>20s}: val_loss={r['val_loss']:.4f}, "
              f"time={r['elapsed_seconds']:.0f}s{marker}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")

    return results


def sweep(args):
    """Hyperparameter sweep across key dimensions."""
    budget = args.budget_steps
    config = args.config

    sweep_grid = {
        "optimizer": ["muon", "adam"],
        "muon_lr": [0.005, 0.01, 0.02, 0.05],
        "adam_lr": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64],
    }

    base_args = {
        "config": config,
        "max_steps": budget,
        "val_every": max(50, budget // 20),
        "log_every": max(10, budget // 100),
        "warmup_steps": min(200, budget // 10),
    }

    results = []
    i = 0
    for opt in sweep_grid["optimizer"]:
        lrs = sweep_grid["muon_lr"] if opt == "muon" else sweep_grid["adam_lr"]
        for lr in lrs:
            for bs in sweep_grid["batch_size"]:
                exp_args = {**base_args, "optimizer": opt, "batch_size": bs}
                if opt == "muon":
                    exp_args["muon_lr"] = lr
                else:
                    exp_args["adam_lr"] = lr

                name = f"sweep_{i:03d}_{opt}_lr{lr}_bs{bs}"
                record = run_experiment(name, exp_args)
                results.append(record)
                i += 1

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    results.sort(key=lambda r: r["val_loss"])
    for r in results[:10]:
        print(f"  {r['name']:>40s}: val_loss={r['val_loss']:.4f}")

    summary_path = RESULTS_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def single(args):
    """Run a single experiment."""
    exp_args = {
        "config": args.config,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "max_steps": args.budget_steps,
        "val_every": max(50, args.budget_steps // 20),
        "log_every": 10,
    }
    if args.optimizer == "muon":
        exp_args["muon_lr"] = args.muon_lr
    else:
        exp_args["adam_lr"] = args.adam_lr

    return run_experiment(f"single_{args.config}_{args.optimizer}", exp_args)


def parse_args():
    p = argparse.ArgumentParser(description="DiffuMamba3 Autoresearch")

    p.add_argument("--mode", type=str, default="compare_optimizers",
                   choices=["compare_optimizers", "sweep", "single"])
    p.add_argument("--config", type=str, default="small")
    p.add_argument("--optimizer", type=str, default="muon")
    p.add_argument("--budget_steps", type=int, default=1000,
                   help="Training steps per experiment")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--adam_lr", type=float, default=3e-4)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.mode == "compare_optimizers":
        compare_optimizers(args)
    elif args.mode == "sweep":
        sweep(args)
    elif args.mode == "single":
        single(args)
