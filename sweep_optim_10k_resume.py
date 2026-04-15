"""Resume 10k optimizer sweep from where it crashed.

Checks which results already exist and only runs the missing ones.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def run_one(name, argv_args):
    import gc, torch
    saved_argv = sys.argv
    sys.argv = ["train.py"] + argv_args
    t0 = time.perf_counter()
    try:
        from train import parse_args, train
        args = parse_args()
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
        "name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
        "status": status, "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    result_path = RESULTS_DIR / f"opt10k_{name}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "10000", "--val_every", "500", "--log_every", "500",
        "--warmup_steps", "400", "--lr_schedule", "cosine",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
        "--save_best",
    ]

    seeds = [42, 137, 2024]

    conditions = {
        "muon": common + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
        ],
        "mousse": common + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
            "--muon_variant", "mousse",
        ],
        "muon_vs": common + [
            "--optimizer", "muon", "--muon_lr", "0.02", "--adam_lr", "3e-4",
            "--muon_variant", "vs",
        ],
        "adam": common + [
            "--optimizer", "adam", "--adam_lr", "3e-4",
        ],
    }

    # Check what's already done
    existing = set()
    for f in RESULTS_DIR.glob("opt10k_*.json"):
        try:
            d = json.load(open(f))
            if d["val_loss"] != float("inf"):
                existing.add(d["name"])
        except:
            pass
    print(f"Already completed: {sorted(existing)}")

    results = []
    total_t0 = time.perf_counter()
    remaining = []

    for seed in seeds:
        for config_name, base_args in conditions.items():
            name = f"{config_name}_s{seed}"
            if name in existing:
                # Load existing result
                d = json.load(open(RESULTS_DIR / f"opt10k_{name}.json"))
                d["seed"] = seed
                d["config"] = config_name
                results.append(d)
                continue
            remaining.append((name, config_name, seed, base_args))

    print(f"Remaining: {len(remaining)} runs")

    for name, config_name, seed, base_args in remaining:
        ckpt = str(CKPT_DIR / f"opt10k_{name}.pt")
        args = base_args + ["--seed", str(seed), "--save_path", ckpt]
        print(f"\n{'='*60}")
        print(f"[{len(results)+1}/12] {name}")
        print(f"{'='*60}")
        record = run_one(name, args)
        record["seed"] = seed
        record["config"] = config_name
        results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    import statistics

    print(f"\n{'='*60}")
    print(f"10K OPTIMIZER VARIANT RESULTS (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    configs = {}
    for r in results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r["val_loss"])

    print(f"\n  {'Config':<12s} {'Mean':>8s} {'Std':>8s}  Seeds")
    print(f"  {'-'*12} {'-'*8} {'-'*8}  {'-'*35}")
    for c in ["muon", "mousse", "muon_vs", "adam"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m = statistics.mean(vals)
            s = statistics.stdev(vals)
            vs_muon = ""
            if c != "muon" and "muon" in configs:
                bm = statistics.mean(configs["muon"])
                delta = m - bm
                vs_muon = f"  vs muon: {delta:+.4f}"
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            print(f"  {c:<12s} {m:>8.4f} {s:>8.4f}  [{vals_str}]{vs_muon}")

    # Paired analysis vs base Muon
    print(f"\n  Paired differences vs base Muon:")
    for c in ["mousse", "muon_vs", "adam"]:
        deltas = []
        for seed in seeds:
            base_r = next((r for r in results if r["config"] == "muon" and r["seed"] == seed), None)
            test_r = next((r for r in results if r["config"] == c and r["seed"] == seed), None)
            if base_r and test_r and base_r["val_loss"] != float("inf") and test_r["val_loss"] != float("inf"):
                deltas.append(test_r["val_loss"] - base_r["val_loss"])
        if len(deltas) >= 2:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas)
            t_stat = m / (s / len(deltas)**0.5) if s > 0 else float('inf')
            sig = "SIG" if abs(t_stat) > 2.92 else "n.s."
            direction = "better" if m < 0 else "worse"
            print(f"    {c:<12s}: {m:+.4f} ± {s:.4f} (t={t_stat:.2f}, {sig}, {direction})")

    ckpts = list(CKPT_DIR.glob("opt10k_*.pt"))
    print(f"\n  Checkpoints saved: {len(ckpts)} in {CKPT_DIR}/")

    summary_path = RESULTS_DIR / "optim_10k_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
