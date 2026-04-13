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

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"


def run_experiment(name: str, args: dict, budget_steps: int = None) -> dict:
    """Run a single training experiment and return results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    train_argv = ['train.py']
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                train_argv.append(f"--{k}")
        else:
            train_argv.extend([f"--{k}", str(v)])

    if budget_steps and "max_steps" not in args:
        train_argv.extend(["--max_steps", str(budget_steps)])

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  ARGS: {' '.join(train_argv[1:])}")
    print(f"{'='*60}\n")

    # Run in-process to avoid ROCm HIP fork-inheritance bugs.
    # Import train lazily so autoresearch.py stays lightweight.
    import gc, torch
    saved_argv = sys.argv
    sys.argv = train_argv
    t0 = time.perf_counter()
    try:
        from train import parse_args as _parse, train as _train
        _args = _parse()
        val_loss = _train(_args)
        elapsed = time.perf_counter() - t0
        returncode = 0
        stderr_tail = ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        val_loss = float("inf")
        returncode = 1
        stderr_tail = str(e)
    finally:
        sys.argv = saved_argv
        # Force cleanup of model/optimizer GPU memory between experiments
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "name": name,
        "args": args,
        "val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "stdout_tail": "",
        "stderr_tail": stderr_tail,
        "returncode": returncode,
    }

    # Save result
    result_path = RESULTS_DIR / f"{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)

    status = "OK" if returncode == 0 else "FAIL"
    print(f"\n[{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")

    if returncode != 0:
        print(f"  ERROR: {stderr_tail[-500:]}")

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


def opt_x_lossweight(args):
    """Core experiment: optimizer × loss_weight interaction.

    2×3 grid: {muon, adam} × {elbo, flat, minsnr}, all with --no_time_cond.
    Tests whether Muon's Newton-Schulz orthogonalization is sensitive to the
    gradient scale distribution imposed by different MDLM loss weightings.
    """
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
        "no_time_cond": True,
        "lr_schedule": "cosine",
    }
    if args.max_data_tokens:
        base_args["max_data_tokens"] = args.max_data_tokens

    grid = []
    for opt in ["muon", "adam"]:
        for lw in ["elbo", "flat", "minsnr"]:
            exp_args = {**base_args, "optimizer": opt, "loss_weight": lw}
            if opt == "muon":
                exp_args["muon_lr"] = args.muon_lr
                exp_args["adam_lr"] = args.adam_lr
            else:
                exp_args["adam_lr"] = args.adam_lr
            grid.append((f"{opt}_{lw}", exp_args))

    results = []
    for name, exp_args in grid:
        record = run_experiment(name, exp_args)
        results.append(record)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY: Optimizer × Loss Weight (no time cond)")
    print("=" * 60)
    results.sort(key=lambda r: r["val_loss"])
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {r['name']:>20s}: val_loss={r['val_loss']:.4f}, "
              f"time={r['elapsed_seconds']:.0f}s{marker}")

    # Show Muon vs Adam gap per loss weight
    print("\n  Muon advantage by loss weight:")
    for lw in ["elbo", "flat", "minsnr"]:
        muon_r = next((r for r in results if r["name"] == f"muon_{lw}"), None)
        adam_r = next((r for r in results if r["name"] == f"adam_{lw}"), None)
        if muon_r and adam_r:
            delta = adam_r["val_loss"] - muon_r["val_loss"]
            sign = "+" if delta > 0 else ""
            print(f"    {lw:>8s}: adam={adam_r['val_loss']:.4f}, "
                  f"muon={muon_r['val_loss']:.4f}, "
                  f"delta={sign}{delta:.4f} nats")

    summary_path = RESULTS_DIR / "opt_x_lossweight.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    return results


def minsnr_gamma_sweep(args):
    """Sweep Min-SNR gamma with the winning optimizer from opt_x_lossweight."""
    budget = args.budget_steps
    config = args.config

    base_args = {
        "config": config,
        "batch_size": args.batch_size,
        "max_steps": budget,
        "val_every": max(50, budget // 20),
        "log_every": max(10, budget // 100),
        "warmup_steps": min(200, budget // 10),
        "no_time_cond": True,
        "optimizer": args.optimizer,
        "loss_weight": "minsnr",
        "lr_schedule": "cosine",
    }
    if args.max_data_tokens:
        base_args["max_data_tokens"] = args.max_data_tokens
    if args.optimizer == "muon":
        base_args["muon_lr"] = args.muon_lr
        base_args["adam_lr"] = args.adam_lr
    else:
        base_args["adam_lr"] = args.adam_lr

    results = []
    for gamma in [1.0, 3.0, 5.0, 10.0]:
        exp_args = {**base_args, "minsnr_gamma": gamma}
        record = run_experiment(f"minsnr_g{gamma:.0f}_{args.optimizer}", exp_args)
        results.append(record)

    print("\n" + "=" * 60)
    print(f"MIN-SNR GAMMA SWEEP ({args.optimizer})")
    print("=" * 60)
    results.sort(key=lambda r: r["val_loss"])
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {r['name']:>30s}: val_loss={r['val_loss']:.4f}{marker}")

    summary_path = RESULTS_DIR / "minsnr_gamma_sweep.json"
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
                   choices=["compare_optimizers", "sweep", "single",
                            "opt_x_lossweight", "minsnr_gamma"])
    p.add_argument("--config", type=str, default="small")
    p.add_argument("--optimizer", type=str, default="muon")
    p.add_argument("--budget_steps", type=int, default=1000,
                   help="Training steps per experiment")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--adam_lr", type=float, default=3e-4)
    p.add_argument("--max_data_tokens", type=int, default=None,
                   help="Limit training tokens (faster iteration)")

    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    modes = {
        "compare_optimizers": compare_optimizers,
        "sweep": sweep,
        "opt_x_lossweight": opt_x_lossweight,
        "minsnr_gamma": minsnr_gamma_sweep,
        "single": single,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
