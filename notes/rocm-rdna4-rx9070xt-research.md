# ROCm Support for AMD Radeon RX 9070 XT (RDNA 4 / Navi 48)

Research compiled: April 2026

---

## 1. ROCm Version Requirements for RDNA 4

| ROCm Version | RDNA 4 Status | Notes |
|---|---|---|
| 6.4.0 | No RDNA 4 support | Released before RX 9070 launch |
| 6.4.1 | First official RDNA 4 support | Released May 2025, right after AMD Computex keynote |
| 6.4.4 | Improved RDNA 4 + Windows PyTorch | Added PyTorch support on Windows for RX 9000 series |
| 7.0 | Stable RDNA 4 support | September 2025; official PyTorch 2.7, TF 2.19.1, JAX 0.6.0 |
| 7.1 | Mature RDNA 4 support | Recommended minimum for AI workloads |
| **7.2 / 7.2.1** | **Current recommended** | Official listing of RX 9070 XT (gfx1201) as supported compute GPU |

**Bottom line:** Use ROCm 7.1+ (ideally 7.2.1) for the RX 9070 XT.

---

## 2. GPU Architecture Identifiers

- **Architecture family:** RDNA 4 (GFX12)
- **LLVM target:** `gfx1201` (Navi 48)
- **Chip ID:** `0x7550`
- **Navi 44 variant:** `gfx1200` (lower-tier RDNA 4, e.g., RX 9060 series)
- **Execution model:** Wave32 (differs from CDNA's Wave64 -- this is the root cause of many compatibility issues)

---

## 3. PyTorch Support

### Official Status

PyTorch officially supports the RX 9070 XT via ROCm. Two installation paths:

**Stable ROCm 6.3 wheels (simplest):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

**ROCm 7.x with PyTorch 2.8+ (recommended for RDNA 4):**
ROCm 7.1+ paired with PyTorch 2.8 nightly/custom builds provides the best RDNA 4 experience.

### Verification
```python
import torch
print(torch.cuda.is_available())        # True (HIP translates CUDA API)
print(torch.cuda.get_device_name(0))    # AMD Radeon RX 9070 XT
```

### Key Configuration
```bash
# Enable runtime kernel tuning for RDNA 4 native FP8 operations
export PYTORCH_TUNABLEOP_ENABLED="1"
export PYTORCH_TUNABLEOP_TUNING_DURATION="short"
```

---

## 4. Flash Attention on RDNA 4

### The Problem

Flash Attention's default Composable Kernel (CK) backend is optimized for CDNA Wave64 execution. RDNA 4 uses Wave32, causing compilation failures: the execution mask register (`exec`) is 32-bit on RDNA 4 but legacy code treats it as 64-bit.

### Solution: Triton Backend

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
pip install flash-attn --no-build-isolation
```

Triton acts as a JIT compiler, generating kernels specifically for `gfx1201` at runtime, bypassing the CK architecture mismatch entirely.

### Alternative: CK GFX12 Branch

A specialized `enable-ck-gfx12` branch of Flash Attention provides high-performance attention kernels tuned for the RDNA 4 instruction set. This approach gives better peak performance than Triton but requires building from a specific branch.

---

## 5. Triton Kernel Support

Triton is the recommended kernel compilation path for RDNA 4:

- Triton automatically compiles kernels down to native WMMA instructions when it detects FP8 data types on gfx1201 hardware
- FP8 support on RDNA4 for vLLM has been enabled through Triton kernel paths
- AITER's C++/ASM kernels do NOT work on RDNA 4 and must be disabled; however, AITER's Triton kernels work once gfx1201 is recognized in the architecture mapping

### Known Limitation

Some libraries' build systems assume NVIDIA/CUDA environments, calling `torch.cuda.get_device_capability()` (undefined on AMD), using `-gencode` flags (nvcc-specific), or including `cuda_runtime.h` without HIP alternatives.

---

## 6. Known Issues and Workarounds

| Issue | Workaround |
|---|---|
| Wave32/Wave64 mismatch causing CK compile failures | Use Triton backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) |
| Card not recognized by older ROCm | Upgrade to ROCm 7.1+ |
| AMDGPU display issues on RDNA 4 | Set kernel param `amdgpu.dcdebugmask=0x10` |
| MES firmware version warnings | Ensure firmware >= 0x82 |
| Secure Boot blocks kernel module | Disable Secure Boot or sign the amdgpu module |
| VRAM memory leaks | Call `torch.cuda.empty_cache()` explicitly |
| GPU not in compatibility matrix | Set `HSA_OVERRIDE_GFX_VERSION` (format: `MAJOR.MINOR.PATCH`) -- generally NOT needed for 9070 XT on ROCm 7.2+ |
| Linux kernel too old for RDNA 4 | Minimum kernel 6.13.5, latest Mesa 25 required |
| Ollama falls back to CPU | Known issue with gfx1201 detection; check for updates |
| ROCm 6.4.4 core dump on basic operations | Upgrade to ROCm 7.x |

### Supported Linux Distributions

- Ubuntu 24.04.4
- Ubuntu 22.04.5
- RHEL 10.1
- RHEL 9.7

---

## 7. RX 9070 XT Hardware Capabilities for ML

| Spec | Value |
|---|---|
| VRAM | 16 GB GDDR6 |
| Memory Bus | 256-bit |
| Architecture | RDNA 4 (gfx1201) |
| FP8 Support | Native (hardware-level) |
| AI Throughput | 2x per CU vs RDNA 3 |

### Model Size Guidelines

| Model Size | Quantization | Feasibility |
|---|---|---|
| 7B | FP16/BF16 | Comfortable fit |
| 13-14B | 8-bit (INT8/FP8) | Fits with careful memory management |
| 24B | 4-6 bit | Maximum practical limit |
| LoRA fine-tuning | 7B FP16 | Well-suited |

### Performance Context

- ~75% of RTX 5080 speed for SDXL image generation
- Exceeds RTX 5080 for SD3.5-Turbo models
- Close to RTX 4070 class for token throughput
- ROCm narrows practical gap to CUDA to ~10-15% in most workloads

---

## 8. Community Training Reports

- LLM inference (Llama 4 8B) works with usable performance
- Image generation (Stable Diffusion, Flux.1) works via ComfyUI with Triton backend
- vLLM inference enabled with FP8 WMMA support through Triton kernel paths
- LoRA fine-tuning of 7B models is practical within 16GB VRAM
- General consensus: "safest consumer bet" for ROCm AI work in 2026

---

## 9. Karpathy's Autoresearch on AMD

### Original Autoresearch Design

- Single GPU, single file (`train.py`), single metric (`val_bpb`)
- 5-minute experiment budget, ~12 experiments/hour, ~100+ overnight
- AI agent autonomously modifies code, trains, evaluates, keeps/discards
- Found ~20 additive improvements in ~700 experiments, 11% efficiency gain

### AMD Compatibility

- Original repo targets NVIDIA (tested on H100)
- **AMD fork exists:** [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch)
- The 5-minute budget approach naturally optimizes for your specific hardware
- Results become platform-dependent, which is actually advantageous for finding optimal configurations for RDNA 4

### NanoGPT on AMD ROCm

AMD provides official documentation for nanoGPT training on ROCm:
- JAX-based implementation with ROCm blog guides
- Docker setup: `rocm/jax-build:rocm6.1.1-jax0.4.30-py3.10.14`
- Mixed precision training optimization guides available
- PyTorch-based nanoGPT works with `--compile=False` flag (torch.compile has compatibility caveats on ROCm)

---

## 10. Recommended Setup Steps

### Step 1: System Preparation

```bash
# Ensure kernel >= 6.13.5 and Mesa >= 25
uname -r
# If needed, upgrade kernel

# Disable Secure Boot if enabled (or sign amdgpu module)
```

### Step 2: Install ROCm 7.2+

```bash
# Follow AMD official docs for your distro (Ubuntu 24.04 recommended)
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

# After install, add user to required groups
sudo usermod -a -G render,video $USER

# Verify
rocminfo | grep gfx
# Should show: gfx1201
clinfo
```

### Step 3: Install PyTorch with ROCm

```bash
# Create virtual environment
python3 -m venv ~/rocm-venv
source ~/rocm-venv/bin/activate

# Install PyTorch (stable ROCm 6.3 wheels)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Or for latest ROCm 7.x support, check PyTorch nightly:
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

### Step 4: Install Flash Attention (Triton backend)

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
pip install flash-attn --no-build-isolation
```

### Step 5: Environment Variables for Training

```bash
# Add to your shell profile or launch script:
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export PYTORCH_TUNABLEOP_ENABLED="1"
export PYTORCH_TUNABLEOP_TUNING_DURATION="short"
```

### Step 6: Docker Alternative (simpler)

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  rocm/pytorch:latest \
  /bin/bash
```

### Step 7: Verify Everything Works

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Quick compute test
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
print(f"Matrix multiply test passed: {y.shape}")
```

---

## 11. Summary Assessment

| Aspect | Rating | Notes |
|---|---|---|
| ROCm official support | Good | Fully supported since ROCm 6.4.1, mature on 7.2+ |
| PyTorch compatibility | Good | Official wheels available, HIP translation layer works |
| Flash Attention | Workable | Requires Triton backend flag, not plug-and-play |
| Triton kernels | Good | JIT compilation for gfx1201 works, FP8 native support |
| Ecosystem maturity | Fair | ~10-15% behind CUDA ecosystem, improving rapidly |
| Autoresearch feasibility | Feasible | AMD fork exists; 5-min budget approach is hardware-agnostic |
| Training small LLMs | Good | 16GB VRAM handles 7B FP16 models for fine-tuning |
| Community support | Growing | Active GitHub discussions, guides, and workarounds available |

**Overall recommendation:** The RX 9070 XT is viable for ML training and research on Linux with ROCm 7.2+. The main friction points are Flash Attention setup (solvable with Triton backend) and occasional build-system assumptions about CUDA. For autoresearch-style experimentation, the 5-minute budget approach naturally discovers what works best on your specific hardware, making it an excellent fit for exploring RDNA 4 capabilities.
