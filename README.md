# modded-diffumamba

A [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)-style research repo for **masked diffusion language models** with a **bidirectional Mamba-3** backbone.

The goal: use [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology to systematically find the fastest training configuration for discrete diffusion LMs on consumer AMD hardware.

> **Parallel NVIDIA (RTX 5090) experiments**: see [`nvidia/`](nvidia/) for transformer baselines (D_modern = RoPE+SwiGLU+QK-norm), published MDLM reproduction, 3-metric ELBO eval, and cross-agent replication of Muon-config sensitivity. Key cross-validated findings: (1) Muon-for-Mamba3 is config-sensitive, not hardware-specific (gamma≤2, in_proj routing) — (2) NS precision (bf16/fp16/fp32) does not matter — (3) **ELBO improvements do not translate to generation coherence at 125M/10B**. Start with [`nvidia/CLAUDE.md`](nvidia/CLAUDE.md).

## What is this?

**Masked diffusion LMs** (MDLM) train by progressively masking tokens and learning to unmask them — like BERT meets diffusion. Recent work shows they can match autoregressive LMs at scale ([LLaDA](https://arxiv.org/abs/2502.09992), 8B params) while enabling parallel generation.

**DiffuMamba** ([Singh et al. 2025](https://arxiv.org/abs/2511.15927)) replaced the Transformer backbone with bidirectional Mamba-2, getting 8.2x throughput with no quality loss. We extend this to **Mamba-3** ([Gu et al. 2026](https://arxiv.org/abs/2603.15569)) — complex-valued states, MIMO formulation, half the state size.

**Muon** ([Jordan et al.](https://kellerjordan.github.io/posts/muon/)) is an orthogonalized momentum optimizer that gave 1.35x speedup in the nanogpt speedrun. It's never been tested on masked diffusion LMs. Our hypothesis: since MDLM training loss is cross-entropy over masked positions (structurally identical to MLM), Muon should transfer from LM pretraining.

## Quick Start

```bash
# Install deps
pip install torch numpy tiktoken huggingface_hub transformers

# Data: pick one (both ~2GB, 1B tokens, GPT-2 tokenized .bin shards)
python data/get_data.py          # FineWeb-10B (kjj0/fineweb10B-gpt2)
# FineWeb-Edu-10B shards: see data/fineweb-edu-10B/README.md for curl cmd

# TRAIN best quokka config (31.5M, ~40 min on RX 9070 XT)
python train.py --config quokka \
  --optimizer muon --muon_variant vs --muon_lr 0.01 --adam_lr 3e-4 \
  --muon_out_proj \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --data_dir data/fineweb-edu-10B \
  --batch_size 8 --max_steps 10000 --save_best

# TRAIN largest model (10L×640d, 111.7M, 16GB VRAM max)
python train_large.py            # staged: 10k→50k→100k with gating

# EVALUATE with gen-PPL under GPT-2 small (authoritative quality metric)
python eval_gen_ppl.py           # loads GPT-2, samples from checkpoints, scores

# SAMPLE with sweep of sampler configs
python sample_topp.py            # temperature × top-p × top-k grid

# AUTORESEARCH sweeps
python sweep_round3.py           # data × LR × loss × WD (Round 3)
python autoresearch.py --mode compare_optimizers --budget_steps 500
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | DiffuMamba3 architecture, MDLM diffusion, sampling (with top-p/top-k) |
| `train.py` | Training loop, MuonAdamW with Muon-VS/Mousse variants, data loading |
| `ssm.py` | PureSSM: pure-PyTorch SSM fallback (slower but device-agnostic) |
| `eval_gen_ppl.py` | **Evaluation** — sample + GPT-2 scoring + diagnostics |
| `sample_*.py` | Sampling scripts (checkpoint comparison, top-p sweep) |
| `train_large.py` | Staged training for the 111M config with gating & periodic ckpts |
| `sweep_*.py` | Various experiment sweeps (gamma, LR, 2×2 factorials, etc.) |
| `proposals/` | Research proposals & evaluations from subagent reviews |
| `HANDOFF.md` | **Recipe & findings** — read this for current best config |
| `CLAUDE.md` | Project overview & key findings |

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

**SSM backend** (auto-detected at import via probe forward passes):
- **Mamba3 Triton** (preferred): ~58k tok/s bf16 fwd+bwd+optim on RX 9070 XT
- Mamba3 MIMO: broken on RDNA4 — configs silently fall back to non-MIMO
- **PureSSM** (fallback): ~5k tok/s, pure PyTorch chunked parallel scan, works on any device

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

All findings validated at 5000 steps with 3 paired seeds and t-tests (see `HANDOFF.md`):

**Current best generative model: 10L×640d @ 30k steps = gen-PPL 54.3** (under GPT-2 small
with top-k=50), beating the MDLM paper's reported 82 at 169M/1M-steps. At 111M params /
~250M tokens trained, we're competitive with MDLMs at ~100× less compute.

**Current best quokka (31M): val_loss = 5.07 vs Adam 5.71 → 0.64 nat advantage**

**Sampler bugs fixed (2026-04-17):** bf16 Gumbel truncation and mis-placed temperature
caused mode-collapse-like generations before the fix. See `model.py` sample() method.

| Finding | Confidence | Detail |
|---------|-----------|--------|
| **Muon beats Adam** | HIGH (t=40-66, p<0.001) | +0.35 nats baseline → +0.64 with full stack. Novel — Muon fails for image diffusion. |
| **FineWeb-Edu val_loss better, gen-PPL worse** | CONFLICTED | -0.20 nats val but +13 gen-PPL under GPT-2. GPT-2 is general-web-biased. |
| **Muon-VS beats base Muon** | HIGH (t=-5.8, p<0.01) | -0.039 nats, parameter-free, same wall-clock. |
| **Mousse beats Muon** | HIGH (t=-11.6, p<0.001) | -0.062 nats, but 2.4x wall-clock overhead. |
| **out_proj in Muon helps** | HIGH (t=-37.8, p<0.001) | -0.061 nats. Confirmed independently on NVIDIA 5090. |
| **lr=0.01 beats lr=0.02 for Muon-VS** | HIGH | LR monotonically worse as it rises (0.01→0.08). |
| **ELBO still bad under Muon-VS** | HIGH | +0.18 nats vs gamma=1.5. VS does NOT decouple optimizer from loss weighting. |
| **SwiGLU beats GELU** | MEDIUM (n.s.) | +0.077 nats for GELU at same expansion. DiffuMamba uses GELU. |
| **All-Mamba beats hybrid attn** | HIGH (t=3.7, p<0.05) | 25% attention hurts by 0.06 nats at 31.5M. |
| **Additive merge is best** | HIGH (t=7.5, p<0.01) | Gated merge +0.24 worse. Multiplicative also worse. |
| Depth doesn't help at iso-params | HIGH (n.s.) | 8L×320d ≈ 4L×384d at ~32M. Width matters more. |
| Gamma 1.5 ≈ gamma 5 | HIGH | ~0.015-0.025 nat difference. Either works. |
| Mamba3 Triton on RDNA4 | HIGH | 58k tok/s. MIMO broken (tilelang), Mamba2 broken (causal_conv1d). |

**Why Muon works here but [fails for image diffusion](https://arxiv.org/abs/2512.12386):**
MDLM uses cross-entropy over masked tokens — structurally identical to MLM where Muon
excels. Image diffusion uses continuous noise prediction with implicit 1/SNR weighting
that creates gradient scale conflicts with Muon's NS orthogonalization.

## Files

```
train.py            # training loop, data loading, Muon+AdamW optimizer
model.py            # DiffuMamba3: BiMamba3 blocks + MDLM diffusion + sampling
ssm.py              # PureSSM: pure-PyTorch chunked parallel scan (no custom kernels)
autoresearch.py     # automated experiment runner (compare, sweep, single)
sweep_gamma.py      # gamma sweep: find Muon-optimal loss weighting
sweep_validation.py # convergence validation at 5000 steps
sweep_ns_steps.py   # Newton-Schulz iteration count probe
test_muon.py        # correctness tests for Muon + Newton-Schulz
data/
  get_data.py       # download pre-tokenized .bin shards from HF Hub
  get_data.sh       # curl-based alternative
  tokenize.py       # stream-tokenize parquet files -> .bin shards
results/            # experiment results (JSON + markdown logs)
proposals/          # experiment proposals and evaluation
notes/              # research notes, paper summaries
HANDOFF.md          # training recipe and experimental findings
```

## Hardware

Developed on **AMD RX 9070 XT** (RDNA 4, 16GB VRAM, gfx1201) via ROCm 7.2 on WSL2.
Should work on any PyTorch-supported GPU (CUDA or ROCm).

RDNA 4 notes:
- WSL2 requires `LD_PRELOAD=./librocprofiler_stub.so` (see `stub_rocprof.c`)
- **Do NOT** set `PYTORCH_TUNABLEOP_ENABLED=1` or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — both crash with `hipErrorInvalidValue` on RDNA4
- bf16 is critical: 7.5x speedup over fp32 (train.py enables it automatically on CUDA)

## Inspiration and Prior Art

This project combines ideas from several lines of work:

- **[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** (Keller Jordan) — The template: single-file training scripts, FineWeb-10B data pipeline, Muon optimizer. We adapt the repo structure, data format, and optimizer implementation.
- **[autoresearch](https://github.com/karpathy/autoresearch)** (Andrej Karpathy) — The methodology: AI agents autonomously run experiments, analyze results, and iterate on training configurations.
- **[MDLM](https://arxiv.org/abs/2406.07524)** (Sahoo et al. NeurIPS 2024) — The diffusion framework: continuous-time ELBO for absorbing-state masked diffusion, SUBS parameterization, log-linear noise schedule.
- **[DiffuMamba](https://arxiv.org/abs/2511.15927)** (Singh et al. 2025) — The architecture pattern: bidirectional Mamba backbone for masked diffusion LMs, additive merge of forward/backward scans, 8.2x throughput over Transformer.
- **[Mamba](https://arxiv.org/abs/2312.00752)** / **[Mamba-2](https://arxiv.org/abs/2405.21060)** / **[Mamba-3](https://arxiv.org/abs/2603.15569)** (Gu, Dao et al.) — The SSM backbone: selective state spaces with input-dependent gating, evolving from Mamba-1's sequential scan through Mamba-2's structured state space duality to Mamba-3's complex-valued MIMO formulation.
- **[Muon](https://kellerjordan.github.io/posts/muon/)** (Keller Jordan et al.) — The optimizer: Newton-Schulz orthogonalization of gradient momentum, 1.35x speedup on nanogpt. Applied here to masked diffusion for the first time.
- **[LLaDA](https://arxiv.org/abs/2502.09992)** (Nie et al. 2025) — Proof that masked diffusion scales: 8B params competitive with LLaMA3 on reasoning benchmarks.
- **[Min-SNR](https://arxiv.org/abs/2303.09556)** (Hang et al. ICCV 2023) — Loss weighting: clamp the ELBO weight to reduce gradient conflict across timesteps. Standard gamma=5 is optimal for Adam; we find **gamma=1.5 is Muon-optimal**.
- **[EGD](https://arxiv.org/abs/2510.04930)** (Pasand & Dohmatob, ICLR 2026) — Egalitarian gradient descent theory: explains why Muon (approximate EGD) is sensitive to loss weight scale variance.
- **[Quokka](https://arxiv.org/abs/2510.03280)** (Ni et al. 2025) — Scaling laws for masked diffusion LMs: optimal model size given a fixed token budget.

## License

MIT
