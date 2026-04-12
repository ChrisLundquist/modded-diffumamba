# modded-diffumamba

A [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)-style research repo for **masked diffusion language models** with a **bidirectional Mamba-3** backbone.

The goal: use [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology to systematically find the fastest training configuration for discrete diffusion LMs on consumer AMD hardware.

## What is this?

**Masked diffusion LMs** (MDLM) train by progressively masking tokens and learning to unmask them — like BERT meets diffusion. Recent work shows they can match autoregressive LMs at scale ([LLaDA](https://arxiv.org/abs/2502.09992), 8B params) while enabling parallel generation.

**DiffuMamba** ([Singh et al. 2025](https://arxiv.org/abs/2511.15927)) replaced the Transformer backbone with bidirectional Mamba-2, getting 8.2x throughput with no quality loss. We extend this to **Mamba-3** ([Gu et al. 2026](https://arxiv.org/abs/2603.15569)) — complex-valued states, MIMO formulation, half the state size.

**Muon** ([Jordan et al.](https://kellerjordan.github.io/posts/muon/)) is an orthogonalized momentum optimizer that gave 1.35x speedup in the nanogpt speedrun. It's never been tested on masked diffusion LMs. Our hypothesis: since MDLM training loss is cross-entropy over masked positions (structurally identical to MLM), Muon should transfer from LM pretraining.

## Quick Start

```bash
# Install deps
pip install torch numpy tiktoken huggingface_hub

# Download pre-tokenized FineWeb-10B (1B tokens, ~2GB)
python data/get_data.py          # uses huggingface_hub
# OR
bash data/get_data.sh            # uses curl, no Python deps

# Train (defaults: small config, Muon+AdamW, ELBO weighting)
python train.py --config small --max_steps 5000

# Best config found so far (Min-SNR + Muon + compile)
python train.py --config quokka --optimizer muon --loss_weight minsnr --compile

# Autoresearch: sweep Muon vs Adam
python autoresearch.py --mode compare_optimizers --budget_steps 500
```

## Architecture

```
Input tokens --> Embed + PosEmbed --> [BiMamba3Block x N] --> AdaLN --> LM Head --> Logits
                                          |
                                     sigma(t) via AdaLN
```

Each **BiMamba3Block** (following DiffuMamba):
- AdaLN conditioning on noise timestep (zero-initialized, from [DiT](https://arxiv.org/abs/2212.09748))
- Forward Mamba-3 scan + backward Mamba-3 scan, additive merge
- SwiGLU MLP (2x expansion) with gated residual

**Diffusion** ([MDLM](https://arxiv.org/abs/2406.07524)):
- Absorbing-state masked diffusion: tokens -> [MASK] at rate governed by t
- Log-linear noise schedule with antithetic sampling
- SUBS parameterization: kill mask logit, force unmasked positions to copy
- Sampling: iterative unmasking from fully masked sequence

**SSM backend**: tries `mamba_ssm` (Triton kernels) first, falls back to `PureSSM` — a pure PyTorch selective scan that runs on any device including AMD ROCm without custom kernels.

## Data

Pre-tokenized FineWeb-10B `.bin` shards from [kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2) — the same data used by modded-nanogpt. Each shard is ~200MB (100M GPT-2 tokens as uint16).

To tokenize your own data from raw parquet files:
```bash
python data/tokenize.py --src data/my_parquets/ --dst data/my_tokens/ --tokens 1B
```

## Model Configs

| Config | Params | d_model | layers | seq_len | Use case |
|--------|--------|---------|--------|---------|----------|
| tiny   | 8.4M   | 128     | 4      | 256     | Quick HP sweep |
| quokka | 35.9M  | 384     | 4      | 1024    | 1B-token scale ([Quokka](https://arxiv.org/abs/2510.03280)-optimal) |
| small  | 84.2M  | 512     | 8      | 512     | Medium experiments |
| base   | 231M   | 768     | 12     | 1024    | Full scale |
| large  | 350M   | 1024    | 24     | 1024    | Large-scale |

## Optimizer

**MuonAdamW** — hybrid optimizer following the [Muon](https://kellerjordan.github.io/posts/muon/) recipe:
- **Muon** (Newton-Schulz orthogonalization) for 2D weight matrices in Mamba blocks
- **AdamW** for embeddings, biases, layernorms, AdaLN modulation, output projection
- Parameter assignment is automatic based on tensor shape and layer name

## Key Findings

From autoresearch sweeps (see `HANDOFF.md` for full details and confidence levels):

| Finding | Confidence | Detail |
|---------|-----------|--------|
| **Min-SNR gamma=5** beats ELBO weighting | HIGH | 2.2% improvement, matches [Hang et al. ICCV 2023](https://arxiv.org/abs/2303.09556) |
| Flat ELBO + Muon + compile is fast | MEDIUM | Better val_loss in 3x less wall time vs Adam baseline |
| Time conditioning helps | MEDIUM | +1.3% vs no conditioning |
| High LR (3e-3) wins short runs | LOW | 500 steps biases toward fast LR; needs validation at scale |

## Files

```
train.py            # training loop, data loading, Muon+AdamW optimizer
model.py            # DiffuMamba3: BiMamba3 blocks + MDLM diffusion + sampling
ssm.py              # PureSSM: pure-PyTorch selective scan (no custom kernels)
autoresearch.py     # automated experiment runner (compare, sweep, single)
test_muon.py        # correctness tests for Muon + Newton-Schulz
data/
  get_data.py       # download pre-tokenized .bin shards from HF Hub
  get_data.sh       # curl-based alternative
  tokenize.py       # stream-tokenize parquet files -> .bin shards
ref/                # reference implementations (muon, modded-nanogpt)
notes/              # research notes, paper summaries
setup_rocm.sh       # AMD ROCm environment setup
HANDOFF.md          # training recipe and experimental findings
```

## Hardware

Developed on **AMD RX 9070 XT** (RDNA 4, 16GB VRAM, gfx1201) via ROCm on WSL2.
Should work on any PyTorch-supported GPU (CUDA or ROCm). See `setup_rocm.sh` for AMD-specific setup.

WSL2 ROCm note: requires `LD_PRELOAD=./librocprofiler_stub.so` to work around a [rocprofiler-register crash](https://github.com/ROCm/rocprofiler-register/issues) in the DXG passthrough environment. See `stub_rocprof.c` for the workaround source.

## Inspiration and Prior Art

This project combines ideas from several lines of work:

- **[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** (Keller Jordan) — The template: single-file training scripts, FineWeb-10B data pipeline, Muon optimizer. We adapt the repo structure, data format, and optimizer implementation.
- **[autoresearch](https://github.com/karpathy/autoresearch)** (Andrej Karpathy) — The methodology: AI agents autonomously run experiments, analyze results, and iterate on training configurations.
- **[MDLM](https://arxiv.org/abs/2406.07524)** (Sahoo et al. NeurIPS 2024) — The diffusion framework: continuous-time ELBO for absorbing-state masked diffusion, SUBS parameterization, log-linear noise schedule.
- **[DiffuMamba](https://arxiv.org/abs/2511.15927)** (Singh et al. 2025) — The architecture pattern: bidirectional Mamba backbone for masked diffusion LMs, additive merge of forward/backward scans, 8.2x throughput over Transformer.
- **[Mamba](https://arxiv.org/abs/2312.00752)** / **[Mamba-2](https://arxiv.org/abs/2405.21060)** / **[Mamba-3](https://arxiv.org/abs/2603.15569)** (Gu, Dao et al.) — The SSM backbone: selective state spaces with input-dependent gating, evolving from Mamba-1's sequential scan through Mamba-2's structured state space duality to Mamba-3's complex-valued MIMO formulation.
- **[Muon](https://kellerjordan.github.io/posts/muon/)** (Keller Jordan et al.) — The optimizer: Newton-Schulz orthogonalization of gradient momentum, 1.35x speedup on nanogpt. Applied here to masked diffusion for the first time.
- **[LLaDA](https://arxiv.org/abs/2502.09992)** (Nie et al. 2025) — Proof that masked diffusion scales: 8B params competitive with LLaMA3 on reasoning benchmarks.
- **[Min-SNR](https://arxiv.org/abs/2303.09556)** (Hang et al. ICCV 2023) — Loss weighting: clamp the ELBO weight at gamma=5 to reduce gradient conflict across timesteps. Our most reliable finding.
- **[Quokka](https://arxiv.org/abs/2510.03280)** (Ni et al. 2025) — Scaling laws for masked diffusion LMs: optimal model size given a fixed token budget.

## License

MIT
