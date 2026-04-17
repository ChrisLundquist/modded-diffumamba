"""Find the largest DiffuMamba3 model that fits in 16GB VRAM.

Tests progressively larger configs with our training setup:
- bf16, batch_size=8, seq_len=1024
- Muon-VS optimizer (1× momentum per 2D param + Adam's 2× for aux params)
- Full forward + backward + optimizer step

Reports peak memory for each config and stops when we OOM.
"""
import sys
import copy
import time
import torch
import gc
import traceback

sys.path.insert(0, '.')
from model import DiffuMamba3, DiffuMamba3Config


def test_config(name, cfg_kwargs, batch_size=8, seq_len=1024, n_warmup=2, n_steps=3):
    """Try to train a config for a few steps. Return peak memory and status."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = "cuda"
    status = "OK"
    err = ""
    peak_mb = 0
    n_params = 0
    step_ms = 0

    try:
        cfg = DiffuMamba3Config(**cfg_kwargs)
        model = DiffuMamba3(cfg).to(device, dtype=torch.bfloat16)
        n_params = sum(p.numel() for p in model.parameters())

        # Build Muon-VS + Adam hybrid (same routing as our best config)
        sys_argv_saved = sys.argv
        sys.argv = [
            "train.py", "--optimizer", "muon", "--muon_variant", "vs",
            "--muon_lr", "0.01", "--adam_lr", "3e-4",
            "--muon_out_proj", "--muon_wd", "0.01", "--adam_wd", "0.01",
            "--adam_beta2", "0.999", "--ns_steps", "5",
        ]
        from train import parse_args, build_optimizer
        args = parse_args()
        sys.argv = sys_argv_saved
        optimizer = build_optimizer(model, args)

        # Training steps
        t0 = time.perf_counter()
        for step in range(n_warmup + n_steps):
            x_0 = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            loss, _ = model.compute_loss(x_0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if step == n_warmup - 1:
                torch.cuda.reset_peak_memory_stats()  # Reset after warmup
                t_bench = time.perf_counter()
        elapsed = time.perf_counter() - t_bench
        step_ms = (elapsed / n_steps) * 1000
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

        del model, optimizer
    except torch.cuda.OutOfMemoryError as e:
        status = "OOM"
        err = "CUDA OOM"
    except Exception as e:
        status = "FAIL"
        err = str(e)[:200]

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "params_M": n_params / 1e6,
        "peak_mb": peak_mb,
        "peak_gb": peak_mb / 1024,
        "step_ms": step_ms,
        "status": status,
        "error": err,
        **cfg_kwargs,
    }


def main():
    # Fixed training settings (our best config)
    batch_size = 8
    seq_len = 1024

    # Test progressively larger configs, keeping d_state=32, headdim=32, expand=2
    # Scale d_model and n_layers together (following DiffuMamba's pattern).
    # Start near our current quokka (31.5M) and grow.
    configs = [
        ("quokka (4L×384d)",    dict(d_model=384, n_layers=4,  d_state=32, headdim=32, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=64, mlp_expansion=2)),
        ("6L×384d",             dict(d_model=384, n_layers=6,  d_state=32, headdim=32, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=64, mlp_expansion=2)),
        ("8L×384d",             dict(d_model=384, n_layers=8,  d_state=32, headdim=32, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=64, mlp_expansion=2)),
        ("8L×512d (small-like)",dict(d_model=512, n_layers=8,  d_state=64, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("12L×512d",            dict(d_model=512, n_layers=12, d_state=64, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("12L×640d",            dict(d_model=640, n_layers=12, d_state=64, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("12L×768d (base-like)",dict(d_model=768, n_layers=12, d_state=128, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("16L×768d",            dict(d_model=768, n_layers=16, d_state=128, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("20L×768d",            dict(d_model=768, n_layers=20, d_state=128, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("24L×768d",            dict(d_model=768, n_layers=24, d_state=128, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
        ("24L×1024d",           dict(d_model=1024, n_layers=24, d_state=128, headdim=64, expand=2, is_mimo=False, max_seq_len=1024, cond_dim=128, mlp_expansion=2)),
    ]

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB)")
    print(f"Settings: batch_size={batch_size}, seq_len={seq_len}, bf16, Muon-VS + out_proj")
    print()

    results = []
    print(f"  {'Config':<25s} {'Params':>8s} {'Peak VRAM':>12s} {'ms/step':>10s}  Status")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*10}  {'-'*6}")

    for name, kwargs in configs:
        result = test_config(name, kwargs, batch_size=batch_size, seq_len=seq_len)
        results.append(result)
        if result["status"] == "OK":
            print(f"  {name:<25s} {result['params_M']:>6.1f}M  {result['peak_gb']:>8.2f} GB "
                  f"{result['step_ms']:>8.0f}ms  OK")
        else:
            print(f"  {name:<25s} {result['params_M']:>6.1f}M  {'—':>8s}     "
                  f"{'—':>8s}    {result['status']} ({result['error'][:50]})")
            if result["status"] == "OOM":
                break  # Stop on first OOM

    # Report the largest that worked
    ok = [r for r in results if r["status"] == "OK"]
    if ok:
        largest = max(ok, key=lambda r: r["params_M"])
        print(f"\n  Largest fitting config: {largest['name']} ({largest['params_M']:.1f}M params)")
        print(f"  Peak VRAM: {largest['peak_gb']:.2f} / {total_vram:.1f} GB "
              f"({largest['peak_gb']/total_vram*100:.0f}% utilization)")
        print(f"  Throughput: {largest['step_ms']:.0f} ms/step = "
              f"{batch_size * seq_len / (largest['step_ms']/1000) / 1000:.1f}k tok/s")

    import json
    from pathlib import Path
    out = Path("results") / "max_model_search.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
