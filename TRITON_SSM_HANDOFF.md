# Project: Fix Mamba-3 Triton Kernels for RDNA4 (AMD RX 9070 XT)

## Summary

The Mamba-3 Triton kernels in `state-spaces/mamba` fail on RDNA4 GPUs with a register
allocation error. The Mamba-2 SSD Triton kernels work fine (935k tok/s). The fix is
adding RDNA4-compatible autotune configs with smaller tile sizes to the Mamba-3 kernels.

## Current State

| Component | Status | Speed |
|-----------|--------|-------|
| Mamba-2 SSD Triton (`ssd_combined.py`) | **WORKS** | 935k tok/s |
| Mamba-3 SISO Triton (`mamba3_siso_combined.py`) | **FAILS** — register allocation | - |
| Mamba-3 MIMO Triton | untested, likely same issue | - |
| PureSSM (our fallback) | works | 1.3k tok/s |

## The Error

```
error: couldn't allocate output register for constraint 'f'
```

This is the LLVM/AMDGCN backend refusing to compile a Triton kernel because it needs
more VGPRs (Vector General Purpose Registers) than RDNA4 provides per wavefront.

RDNA4 (gfx1201) has:
- 32-wide wavefronts (vs 64 on CDNA MI250/MI300X)
- Fewer VGPRs per work-item than CDNA datacenter GPUs
- The Mamba-3 kernel tile sizes were tuned for NVIDIA/CDNA hardware

## The Fix

Add smaller autotune configs to the Mamba-3 Triton kernels. The kernel algorithm stays
the same — only the tile sizes and `num_warps` change.

### Where to change

The Mamba-3 Triton kernels are in:
- `mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
- `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py`
- `mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py`

Look for `@triton.autotune` decorators with `configs=[...]`. Add configs with smaller
`BLOCK_SIZE`, fewer `num_warps`, and/or smaller `num_stages` that fit RDNA4's register
budget.

Example pattern:
```python
@triton.autotune(
    configs=[
        # Existing configs (NVIDIA/CDNA)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        # NEW: RDNA4-compatible (smaller tiles, fewer warps)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=2),
    ],
    key=[...],
)
```

The autotuner will automatically pick the best config for the hardware. NVIDIA GPUs
will still select their large-tile configs. RDNA4 will select the smaller ones.

### Approach

1. Read the existing autotune configs in the Mamba-3 kernels
2. Identify which kernel(s) fail (fwd, bwd, or both)
3. Add smaller tile configs that reduce register pressure
4. Test on RDNA4 — verify the kernel compiles and produces correct output
5. Benchmark to find the fastest config that fits
6. Verify existing configs still work on NVIDIA (or at least don't regress)

### Do NOT

- Spill registers to VRAM. That would silently tank performance. The error is better
  than being 100x slower with no warning.
- Rewrite the kernel algorithm. The math is correct; only the tiling needs to change.
- Touch the Mamba-2 SSD kernels. They already work on RDNA4.

## PR Target

**Repository:** [state-spaces/mamba](https://github.com/state-spaces/mamba)

**Related issues:**
- #65: "Rocm support" (open since 2024-07-19)
- #821: "performance of running mamba_ssm on ROCm is poor"

**Precedent:** PR #377 (merged) updated Triton kernel compatibility. The maintainers
accept Triton kernel improvements.

**PR title suggestion:** "Add RDNA4-compatible autotune configs for Mamba-3 Triton kernels"

## Installation

The pip package `mamba-ssm==2.3.1` does NOT include Mamba-3. Must install from git HEAD:

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"
```

`MAMBA_SKIP_CUDA_BUILD=TRUE` skips the CUDA kernel compilation (which fails on ROCm)
and builds a pure-Python wheel. The Triton kernels JIT-compile at runtime.

## Hardware for Testing

- AMD Radeon RX 9070 XT (gfx1201, RDNA4, 16GB VRAM)
- ROCm 7.2.1, PyTorch 2.12.0.dev20260408+rocm7.2
- Triton 3.7.0 (ROCm build) — verified working for basic kernels, reductions, cumsum
- WSL2 with DXG passthrough (requires LD_PRELOAD rocprofiler stub)

```bash
export LD_PRELOAD=./librocprofiler_stub.so
export HSA_ENABLE_DXG_DETECTION=1
.venv/bin/python your_test.py
```

## Verification

Test against PureSSM for correctness:
```python
from ssm import PureSSM  # reference (slow, correct)
# Compare outputs for same input, should match within bf16 tolerance
```

Benchmark:
```python
# Mamba-2 SSD baseline: 935k tok/s at L=256 on RDNA4
# Target for Mamba-3: similar or better throughput
```
