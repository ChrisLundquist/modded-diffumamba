# DiffuMamba3 Training Recipe Handoff

## Confidence Levels

**HIGH confidence (validated at 5000 steps, 3 seeds, paired t-tests):**
- Muon beats Adam by 0.34 ± 0.01 nats (t=40, p<0.001)
- All-Mamba architecture beats hybrid Mamba-attention (+0.06, t=3.7, p<0.05)
- Additive merge beats gated merge (+0.24, t=7.5, p<0.01)
- Gamma 1.5 vs 5 difference is negligible (~0.025 nats, n.s.)

**MEDIUM confidence (validated at 5000 steps, 3 seeds, not significant):**
- Time conditioning ON may be marginally better (-0.013 nats, t=-2.0, p~0.09)
- Muon lr=0.02 is optimal (sweep: 0.005, 0.01, 0.02, 0.04, n=1)
- Mamba3 Triton (non-MIMO) works on RDNA4 at 58k tok/s steady-state

**LOW confidence (single seed, short training):**
- Weight decay, beta2 effects are within noise
- 1k-step n=1 rankings are unreliable (gamma sweep showed >1 nat effects that vanished at 5k)

## Best Configuration (validated at 5000 steps)

```bash
python train.py \
  --config quokka \
  --optimizer muon --muon_lr 0.02 --adam_lr 3e-4 \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --lr_schedule cosine --warmup_steps 200 \
  --batch_size 8 --max_steps 5000 --save_best
```

## Key Finding 1: Muon Beats Adam (definitive)

**2×2 factorial, 3 paired seeds, 5000 steps:**

| Config | Mean ± Std | Seeds |
|--------|-----------|-------|
| **Muon + gamma=1.5** | **5.528 ± 0.060** | [5.485, 5.596, 5.502] |
| **Muon + gamma=5** | **5.553 ± 0.058** | [5.517, 5.620, 5.522] |
| Adam + gamma=1.5 | 5.867 ± 0.069 | [5.810, 5.943, 5.847] |
| Adam + gamma=5 | 5.894 ± 0.070 | [5.838, 5.972, 5.872] |

**Muon advantage: +0.34 ± 0.01 nats (t=40, p<0.001).** Consistent across all seeds.
Gamma effect is negligible (~0.025 nats) — either 1.5 or 5 works.

An earlier 1k-step gamma sweep showed >1 nat differences between gamma values, but this
was n=1 noise that vanished at 5k steps with proper seeding. **Lesson: never trust
single-seed 1k-step rankings.**

## Key Finding 2: Architecture (all Mamba wins)

**4 configs, 3 paired seeds, 5000 steps:**

| Config | Mean ± Std | vs baseline | Sig? |
|--------|-----------|-------------|------|
| time_cond ON | 5.521 ± 0.057 | -0.013 | no (t=-2.0, p~0.09) |
| **baseline** (all Mamba) | **5.535 ± 0.047** | — | — |
| hybrid 25% attn | 5.595 ± 0.056 | +0.061 | **yes** (t=3.7) |
| gated merge | 5.772 ± 0.063 | +0.238 | **yes** (t=7.5) |

- **Hybrid attention hurts** at this scale (31.5M, 4 layers). DiffuMamba-H found it
  helps at 1.3B — the benefit likely requires more scale/depth.
- **Additive merge is best.** Gated merge is significantly worse. Multiplicative was
  also worse in 1k screens.
- **Time conditioning ON is marginally better** (p~0.09). Not significant at 3 seeds
  but consistent. DiffuMamba uses it. We recommend keeping it on.
- **Weight tying (Caduceus-style) was worse** in 1k screens (+0.54). Not validated at 5k.

## SSM Backend

model.py probes backends at import with small forward passes:
- **Mamba3 Triton** (active): non-MIMO works, ~58k tok/s fwd+bwd+optim (bf16)
- **Mamba3 MIMO**: broken on RDNA4 (tilelang `_NestedLoopCheckVisitor` bug)
- **Mamba2 Triton**: broken on RDNA4 (missing `causal_conv1d` CUDA kernel)
- **PureSSM** (auto-fallback): ~5k tok/s, pure PyTorch, works everywhere

MIMO configs silently fall back to Mamba3 non-MIMO. No action needed.

## Architecture

- **Backbone:** All-Mamba bidirectional (forward + backward scan, additive merge)
- **Conditioning:** AdaLN (from DiT) on noise timestep, zero-initialized
- **Time conditioning:** ON (marginally better than OFF, matching DiffuMamba)
- **Hybrid attention:** Tested, hurts at 31.5M scale — may help at larger scale per DiffuMamba-H
- **CLI:** `--attn_layers`, `--tie_weights`, `--merge` for architecture experiments

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

## What We Tested (Architecture Screen, 1k steps)

| Direction | Result | Note |
|-----------|--------|------|
| Hybrid 25% attn (pos 0) | +1.22 nats | Hurts badly at 1k, +0.06 at 5k |
| Hybrid 25% attn (pos 3) | +1.20 nats | Similar |
| Hybrid 50% attn | +1.25 nats | Worse |
| Weight tying (Caduceus) | +0.54 nats | Hurts |
| Multiplicative merge | +0.26 nats | Slightly worse |
| Gated merge | +0.20 nats* | *Broken init in 1k screen; +0.24 at 5k |
| Time cond ON | +0.23 nats | Hurts at 1k, -0.01 at 5k (reverses!) |

**Lesson:** 1k-step screens are directionally useful but magnitude/sign can flip at 5k.

## What We Have NOT Tested

- Scale-up to small (84M) or base (231M) config
- Training beyond 5000 steps
- Soft masking (Hersche et al. ICLR 2026)
- Optimizer variants (Mousse, Muon-VS, AdaMuon)
- Sample quality evaluation (text generation, perplexity)
- Width vs depth tradeoffs at fixed param count
- Noise schedule alternatives

## Experiment Proposals (in proposals/)

See `proposals/EVALUATION.md` and `proposals/round2_EVALUATION.md` for ranked plans.
See `proposals/wide_exploration_eval.md` for literature-backed direction analysis.

## Key References

- Min-SNR: Hang et al., ICCV 2023, arXiv 2303.09556
- MDLM: Sahoo et al., NeurIPS 2024, arXiv 2406.07524
- DiffuMamba: Singh et al., 2025, arXiv 2511.15927
- Quokka: Ni et al., 2025, arXiv 2510.03280
- Muon: Keller Jordan et al., kellerjordan.github.io/posts/muon
- EGD (Muon theory): Pasand & Dohmatob, ICLR 2026, arXiv 2510.04930
- Full experiment log: results/experiment_log.md, results/autoresearch_mamba3.md
