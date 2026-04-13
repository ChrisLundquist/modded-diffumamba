# DiffuMamba3: Masked Diffusion LM with Mamba-3 + Muon

## Project Goal

modded-nanogpt-style implementation of a masked diffusion language model
using a bidirectional Mamba-3 backbone and Muon optimizer, targeting AMD RX 9070 XT
(RDNA 4 / ROCm). Uses Karpathy's autoresearch methodology to find the fastest
training configuration.

## Repo Structure

All code lives at root level (modded-nanogpt style):
- `train.py` — training loop, optimizer (Muon+AdamW), data loading, LR schedule
- `model.py` — DiffuMamba3 architecture, MDLM diffusion, sampling
- `ssm.py` — PureSSM: pure-PyTorch selective scan fallback (chunked parallel scan)
- `autoresearch.py` — automated experiment runner (compare optimizers, HP sweep)
- `sweep_*.py` — targeted experiment scripts (gamma sweep, validation, NS probe)
- `data/get_data.py` — download pre-tokenized FineWeb-10B shards from HF Hub
- `data/get_data.sh` — curl-based alternative (no Python deps for download)
- `data/tokenize.py` — tokenize raw parquet files into .bin shards

## Data Format

Pre-tokenized `.bin` shards (modded-nanogpt format):
- Header: 256 int32s (magic=20240520, version=1, num_tokens)
- Body: uint16 tokens (GPT-2 tokenizer)
- Source: kjj0/fineweb10B-gpt2 on HuggingFace Hub
- Default: 10 train shards (1B tokens) + 1 val shard (100M tokens)

## Architecture

- **Backbone:** Bidirectional Mamba-3 (forward + backward scan, additive merge)
- **Diffusion:** MDLM absorbing-state masked diffusion (Sahoo et al. 2024)
- **Optimizer:** Muon for 2D hidden weights + AdamW for embeddings/biases/norms
- **Conditioning:** AdaLN (zero-initialized, from DiT) on noise timestep

## SSM Backend (auto-detected at import)

model.py probes backends at import time with small forward passes:
- **Mamba3 Triton** (preferred): ~105k tok/s bf16, ~58k tok/s fwd+bwd+optim
- **Mamba3 MIMO**: broken on RDNA4 (tilelang `_NestedLoopCheckVisitor` bug)
- **Mamba2 Triton**: broken on RDNA4 (missing `causal_conv1d` CUDA kernel)
- **PureSSM** (fallback): ~5k tok/s, pure PyTorch, works everywhere

Current status: Mamba3 non-MIMO works. MIMO configs silently fall back to non-MIMO.

## Key Findings

**Muon works for masked diffusion LMs.** This is novel — Muon fails catastrophically
for image diffusion (arXiv 2512.12386) but succeeds here because MDLM uses cross-entropy
over masked tokens, not continuous noise prediction.

**Best configuration (validated at 5000 steps, quokka 31.5M):**
- Muon lr=0.02 + AdamW lr=3e-4 (auxiliary)
- Min-SNR gamma=1.5 loss weighting (Muon-optimal, see below)
- Cosine LR schedule, no time conditioning
- val_loss=5.52 vs Adam+minsnr baseline 5.95 (0.43 nat advantage)

**Muon-optimal loss weighting: gamma=1.5.** Standard Min-SNR (gamma=5) conflicts with
Muon's gradient orthogonalization. Flat weighting avoids the conflict but wastes
information. Gamma=1.5 is the sweet spot — mild enough to not corrupt Muon's NS
iteration, strong enough to bias toward informative timesteps.

## Hardware Target

AMD RX 9070 XT (16GB VRAM, RDNA 4, gfx1201)
- ROCm 7.2+ / PyTorch 2.12+ with ROCm
- WSL2: requires `LD_PRELOAD=./librocprofiler_stub.so` (see stub_rocprof.c)
- **Do NOT set** `PYTORCH_TUNABLEOP_ENABLED=1` or
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — both crash on RDNA4

## Running

```bash
python data/get_data.py                          # download data (1B tokens)
python train.py --config quokka --optimizer muon \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --no_time_cond --max_steps 5000                # train (best config)
python autoresearch.py --mode sweep              # HP sweep
```

## Style

nanogpt-clean: minimal dependencies, flat file layout, readable code.
