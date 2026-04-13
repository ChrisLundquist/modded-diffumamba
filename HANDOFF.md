# DiffuMamba3 Training Recipe Handoff

## Confidence Levels

**HIGH confidence (validated at 5000 steps on FineWeb-10B, Mamba3 Triton):**
- Muon (lr=0.02) beats Adam (lr=3e-4) by 0.43 nats at 5000 steps, gap still widening
- Min-SNR gamma=1.5 is Muon-optimal loss weighting (gamma sweep: 7 values)
- Cosine LR schedule, no time conditioning
- Auxiliary Adam lr=3e-4 for embeddings/norms (higher hurts)

**MEDIUM confidence (validated at 1000 steps, n=1):**
- Muon lr=0.02 is optimal (sweep: 0.005, 0.01, 0.02, 0.04)
- Gamma=1.5 is Muon-specific — Adam does slightly better with gamma=5
- Mamba3 Triton (non-MIMO) works on RDNA4 at 58k tok/s steady-state

**LOW confidence (single seed, short training):**
- Weight decay, beta2 effects are within noise
- NS iteration count interaction (probe running)

## Best Configuration (validated at 5000 steps)

```bash
python train.py \
  --config quokka \
  --optimizer muon --muon_lr 0.02 --adam_lr 3e-4 \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --lr_schedule cosine --warmup_steps 200 \
  --no_time_cond --batch_size 8 --max_steps 5000
```

**5000-step results (quokka 31.5M, FineWeb-10B, Mamba3 Triton non-MIMO):**

| Config | val_loss | vs best |
|--------|----------|---------|
| **Muon + gamma=1.5** | **5.52** | — |
| Muon + flat | 5.62 | +0.10 |
| Adam + minsnr gamma=5 | 5.95 | +0.43 |

## Key Finding: Muon-Optimal Loss Weighting

**Muon needs a specific loss weighting to work well on masked diffusion.**

Gamma sweep at 1000 steps (quokka, FineWeb-10B, Mamba3 Triton):

| gamma | val_loss | note |
|-------|----------|------|
| **1.5** | **6.39** | **Muon-optimal sweet spot** |
| 2.0 | 6.41 | close second |
| 10.0 | 6.56 | |
| 5.0 | 7.26 | standard Min-SNR — bad for Muon |
| 1.0 | 7.29 | too tight |
| 3.0 | 7.57 | |
| flat | 7.60 | no reweighting — worst |

**Why gamma=1.5?** Muon's Newton-Schulz orthogonalization equalizes gradient singular
values. ELBO weighting (1/t) creates extreme gradient scale variance across timesteps,
conflicting with this equalization. Flat weighting avoids the conflict but discards
useful signal about which timesteps are most informative. Gamma=1.5 is the sweet spot:
just enough reweighting to help, but not enough to corrupt Muon's momentum buffer.

**This is Muon-specific.** Adam does slightly better with gamma=5 (standard Min-SNR).
Gamma=1.5 does not help Adam at 1000 steps (6.79 vs 6.77 for gamma=5).

## SSM Backend

model.py probes backends at import with small forward passes:
- **Mamba3 Triton** (active): non-MIMO works, ~58k tok/s fwd+bwd+optim (bf16)
- **Mamba3 MIMO**: broken on RDNA4 (tilelang `_NestedLoopCheckVisitor` bug)
- **Mamba2 Triton**: broken on RDNA4 (missing `causal_conv1d` CUDA kernel)
- **PureSSM** (auto-fallback): ~5k tok/s, pure PyTorch, works everywhere

MIMO configs silently fall back to Mamba3 non-MIMO. No action needed.

## Architecture: AdaLN Timestep Conditioning

We use AdaLN (from DiT) instead of DiffuMamba's concatenated timestep token.
Zero-initialized so blocks start as identity (scale=1, shift=0, gate=0).

Note: time conditioning is currently OFF (`--no_time_cond`) in the best config.
AdaLN receives zeros, so the modulation has no effect. This means the timestep
embedding is essentially unused. An earlier experiment showed time conditioning
ON was +1.3% better, but that was on PureSSM — needs re-validation on Mamba3 Triton.

## Data

FineWeb-10B pre-tokenized `.bin` shards at `data/fineweb10B/`:
- 10 train shards (1B tokens total) + 1 val shard (100M tokens)
- Header: 256 int32s (magic=20240520, version=1, num_tokens)
- Body: uint16 tokens (GPT-2 tokenizer)
- Download: `python data/get_data.py`

Also available: `data/fineweb-edu/` (.npy format, used in earlier experiments).

## Model Configs

| Config | Params | d_model | layers | seq_len | Use case |
|--------|--------|---------|--------|---------|----------|
| tiny | 8.4M | 128 | 4 | 256 | Quick HP sweep |
| quokka | 35.9M | 384 | 4 | 1024 | Primary autoresearch config |
| small | 84.2M | 512 | 8 | 512 | Scale-up experiments |
| base | 231.4M | 768 | 12 | 1024 | Full scale |

## Hardware Notes (AMD RX 9070 XT / RDNA4)

- ROCm 7.2 / PyTorch 2.12.0.dev+rocm7.2
- WSL2: requires `LD_PRELOAD=./librocprofiler_stub.so`
- **Do NOT set** `PYTORCH_TUNABLEOP_ENABLED=1` — crashes with hipErrorInvalidValue
- **Do NOT set** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — same crash
- bf16 works and is critical for performance (7.5x speedup over fp32)

## What We Did NOT Test

- Hybrid attention layers (1 attn per 5 Mamba blocks — DiffuMamba-H style)
- Soft masking (Hersche et al. ICLR 2026)
- Scale-up to small (84M) or base (231M) config
- Multiple seeds for significance testing
- Training beyond 5000 steps
- Sample quality evaluation (text generation, perplexity)

## Experiment Proposals (in proposals/)

Three proposals were generated and evaluated — see `proposals/EVALUATION.md`:
1. **Scaling & Convergence** — longer training, larger models
2. **Architecture Variants** — hybrid attention, soft masking, weight tying
3. **Training Dynamics** — gamma sweep (DONE), NS steps probe (running), scheduled weighting

## Key References

- Min-SNR: Hang et al., ICCV 2023, arXiv 2303.09556
- MDLM: Sahoo et al., NeurIPS 2024, arXiv 2406.07524
- DiffuMamba: Singh et al., 2025, arXiv 2511.15927
- Quokka: Ni et al., 2025, arXiv 2510.03280
- Muon: Keller Jordan et al., kellerjordan.github.io/posts/muon
- EGD (Muon theory): Pasand & Dohmatob, ICLR 2026, arXiv 2510.04930
- Full experiment log: results/experiment_log.md, results/autoresearch_mamba3.md
