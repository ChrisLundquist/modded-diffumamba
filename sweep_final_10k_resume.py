"""Resume final 10k validation - skips completed runs."""
import sys, json, time
from pathlib import Path
from datetime import datetime
import statistics

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
    record = {"name": name, "val_loss": val_loss, "elapsed_seconds": elapsed,
              "status": status, "error": error, "timestamp": datetime.now().isoformat()}
    with open(RESULTS_DIR / f"final10k_{name}.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [{status}] {name}: val_loss={val_loss:.4f}, time={elapsed:.0f}s")
    return record


def main():
    common = [
        "--config", "quokka", "--batch_size", "8",
        "--max_steps", "10000", "--val_every", "500", "--log_every", "500",
        "--warmup_steps", "400", "--lr_schedule", "cosine",
        "--optimizer", "muon", "--muon_variant", "vs",
        "--adam_lr", "3e-4", "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5", "--save_best",
    ]

    conditions = {
        "new_best": common + ["--data_dir", "data/fineweb-edu-10B", "--muon_lr", "0.01"],
        "old_best": common + ["--data_dir", "data/fineweb10B", "--muon_lr", "0.02"],
    }

    # Find completed runs
    existing = set()
    for f in RESULTS_DIR.glob("final10k_*.json"):
        d = json.load(open(f))
        if d["val_loss"] != float("inf"):
            existing.add(d["name"])
    print(f"Already done: {sorted(existing)}")

    results = []
    for f in RESULTS_DIR.glob("final10k_*.json"):
        d = json.load(open(f))
        if d["val_loss"] != float("inf"):
            for c in conditions:
                if d["name"].startswith(c):
                    d["config"] = c
                    d["seed"] = int(d["name"].split("_s")[-1])
                    break
            results.append(d)

    total_t0 = time.perf_counter()
    for seed in [42, 137, 2024]:
        for config_name, base_args in conditions.items():
            name = f"{config_name}_s{seed}"
            if name in existing:
                continue
            ckpt = str(CKPT_DIR / f"final10k_{name}.pt")
            args = base_args + ["--seed", str(seed), "--save_path", ckpt]
            print(f"\n{'='*60}\n{name}\n{'='*60}")
            record = run_one(name, args)
            record["seed"] = seed
            record["config"] = config_name
            results.append(record)

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'='*60}\nFINAL 10K RESULTS ({total_elapsed/60:.1f} min)\n{'='*60}")
    configs = {}
    for r in results:
        configs.setdefault(r["config"], []).append(r["val_loss"])
    for c in ["new_best", "old_best"]:
        vals = configs.get(c, [])
        if len(vals) >= 2:
            m, s = statistics.mean(vals), statistics.stdev(vals)
            print(f"  {c:<12s} {m:>8.4f} ± {s:.4f}  {vals}")

    # Paired test
    if all(c in configs for c in ["new_best", "old_best"]) and len(configs["new_best"]) >= 2:
        deltas = []
        for seed in [42, 137, 2024]:
            n = next((r for r in results if r["config"] == "new_best" and r.get("seed") == seed), None)
            o = next((r for r in results if r["config"] == "old_best" and r.get("seed") == seed), None)
            if n and o:
                deltas.append(n["val_loss"] - o["val_loss"])
        if len(deltas) >= 2:
            m, s = statistics.mean(deltas), statistics.stdev(deltas)
            t = m / (s / len(deltas)**0.5) if s > 0 else float("inf")
            print(f"\n  Paired delta: {m:+.4f} ± {s:.4f} (t={t:.2f}, n={len(deltas)})")

    with open(RESULTS_DIR / "final_10k_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
