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

**Muon beats Adam for masked diffusion LMs (t=40, p<0.001).** Novel — Muon fails for
image diffusion (arXiv 2512.12386) but succeeds here because MDLM uses cross-entropy.
Advantage: +0.34 nats, validated with 3 paired seeds at 5k steps.

**Best configuration (validated at 10000 steps, 3 seeds, quokka 31.5M):**
- Muon-VS (variance-scaled) muon_lr=0.01 + AdamW lr=3e-4 (auxiliary)
- out_proj included in Muon routing (--muon_out_proj)
- Min-SNR gamma=1.5 (gamma barely matters — 1.5 vs 5 is ~0.015 nats)
- Cosine LR schedule, SwiGLU MLP
- All-Mamba, additive merge (hybrid attention hurts at this scale)
- **FineWeb-Edu** data (beats plain FineWeb by 0.07 nats)
- val_loss = 5.07 ± 0.08 vs Adam 5.71 ± 0.03 (0.64 nat advantage)

**Improvement stack from Adam baseline (10k, 3 seeds):**
- Adam: 5.711
- + Muon: 5.362 (-0.35)
- + Variance scaling (VS): 5.323 (-0.04)
- + out_proj in Muon: 5.266 (-0.06)
- + FineWeb-Edu + lr=0.01: **5.069** (-0.20)

**Optimizer ranking (10k steps, 3 seeds, FineWeb):**
- Mousse: 5.300 (best raw loss, 2.4x wall-clock)
- Muon-VS + out_proj: 5.266 (best practical)
- Muon-VS: 5.323
- Muon: 5.362
- Adam: 5.711

**Architecture: all-Mamba wins.** Hybrid attention (+0.06, sig), gated merge (+0.24, sig),
and weight tying (+0.54, screen) all hurt. SwiGLU beats GELU (+0.08).
Depth doesn't help at iso-params (8L×320d ≈ 4L×384d). out_proj in Muon helps (-0.06).

**Loss weighting: gamma=1.5 ≈ gamma=5, ELBO still bad** — Muon-VS does NOT
decouple optimizer from loss weighting. ELBO (1/t) is +0.18 nats worse even with VS.

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
