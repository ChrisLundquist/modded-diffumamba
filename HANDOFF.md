# DiffuMamba3 Training Recipe Handoff

## Entry Points

```bash
# REPRO current best (quokka 31M, ~40 min)
python train.py --config quokka --optimizer muon --muon_variant vs \
  --muon_lr 0.01 --adam_lr 3e-4 --muon_out_proj --muon_tok_emb \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --data_dir data/fineweb-edu-10B \
  --batch_size 8 --max_steps 10000 --save_best --save_path ckpt.pt

# REPRO best generative (10L×640d 111M, staged 10k/50k/100k)
python train_large.py            # saves to checkpoints/10L640d_{phase}.pt

# EVALUATE — gen-PPL under GPT-2 small + diagnostics
python eval_gen_ppl.py           # our real north-star metric

# DATA — see data/fineweb-edu-10B/README.md
```

## Confidence Levels

**HIGH confidence (validated at 5k/10k steps, 3 seeds, paired t-tests):**
- Muon beats Adam by 0.35+ nats (t=40-66, p<0.001) at both 5k and 10k
  — also confirmed on gen-PPL (Muon 68 vs Adam 76 at quokka 10k)
- Muon-VS beats base Muon (-0.04, t=-5.8) — parameter-free, same wall-clock
- Mousse beats Muon (-0.06, t=-11.6) — but 2.4x wall-clock overhead
- out_proj in Muon routing helps (-0.06, t=-37.8) — confirmed on NVIDIA 5090 too
- **tok_emb (+ tied lm_head) in Muon routing helps at 5k (-0.115, 3/3 seeds agree).**
  Motivated by results/geometry/REPORT.md Fig 1-2: tok_emb is the ONLY matrix showing
  Huh-2021 simplicity-bias decay under Adam. The *empirical win* is solid; the
  *causal story* is NOT yet: the flag bundles three changes (effective LR ~380x
  higher, Newton-Schulz orthogonalization, different momentum/WD semantics). The
  Adam-tok_emb-undertrained null hypothesis is unrefuted — nvidia finding #9
  showed raising Adam embed LR 1.5e-4->1e-3 was worth 0.20 nats at similar scale,
  which is in the same ballpark as our 0.115. CLI: `--muon_tok_emb`. Same
  muon_lr=0.01 as blocks, no throughput hit. **Promotion to high-confidence
  requires the adam_emb_lr 3-arm A/B/C** (sweep_adam_emb_lr_ablation_5k.py).
  Also: 10k validation pending, gen-quality impact untested.
- All-Mamba beats hybrid Mamba-attention (+0.06, t=3.7) at 31.5M scale
- Additive merge beats gated merge (+0.24, t=7.5)
- lr=0.01 beats lr=0.02 for Muon-VS (LR monotonically worse as it increases)
- ELBO still bad under Muon-VS (+0.18 nats vs gamma=1.5) — VS does NOT decouple optimizer from loss weighting

**CONFLICTED (val_loss vs gen-PPL disagree):**
- FineWeb-Edu beats FineWeb on val_loss (-0.20, t=-3.4) but is WORSE on
  gen-PPL under GPT-2 (81 vs 68 at quokka 10k). Likely a GPT-2 reference
  bias (trained on general web, scores formal Edu-style text as unusual).
  We don't know yet which metric to trust for downstream quality.

**MEDIUM confidence:**
- Time conditioning ON may be marginally better (-0.013, t=-2.0, p~0.09)
- Gamma 1.5 vs 5 is negligible (~0.015-0.025 nats)

**LOW confidence:**
- Weight decay effects invisible at sub-epoch training
- Depth vs width: no significant benefit at iso-params (8L×320d ≈ 4L×384d)
- 1k-step n=1 rankings are unreliable

## Best Configuration (10k-validated is WITHOUT `--muon_tok_emb`)

```bash
# 10k-validated best (val_loss = 5.07 ± 0.08)
python train.py \
  --config quokka \
  --optimizer muon --muon_variant vs --muon_lr 0.01 --adam_lr 3e-4 \
  --muon_out_proj \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --lr_schedule cosine --warmup_steps 400 \
  --data_dir data/fineweb-edu-10B \
  --batch_size 8 --max_steps 10000 --save_best

# Newer best at 5k (pending 10k validation) — add --muon_tok_emb
# Expected 5k val_loss: 5.19 ± 0.05 (vs 5.31 without)
python train.py \
  --config quokka \
  --optimizer muon --muon_variant vs --muon_lr 0.01 --adam_lr 3e-4 \
  --muon_out_proj --muon_tok_emb \
  --loss_weight minsnr --minsnr_gamma 1.5 \
  --lr_schedule cosine --warmup_steps 400 \
  --data_dir data/fineweb-edu-10B \
  --batch_size 8 --max_steps 10000 --save_best
```

**Previous best (10k, 3 seeds): val_loss = 5.07 ± 0.08** (seeds: 4.976, 5.120, 5.111)
vs Adam baseline 5.71 ± 0.03 → **0.64 nat advantage**

## Best Generative Model (10L×640d scale-up)

**111.7M params, FineWeb-Edu, 50k steps bs=4, our full stack** — evaluated
with gen-PPL under GPT-2 small (top-k=50 sampling, 16 samples × 128 tokens):

| Checkpoint | gen-PPL | Unigram H | Notes |
|------------|---------|-----------|-------|
| **10L×640d @ 30k steps** | **54.3** | 5.24 | **best, beats MDLM paper's 82** |
| 10L×640d @ 50k (best val) | 57.0 | 5.30 | |
| quokka old_best (FineWeb) | 68.3 | 5.15 | |
| quokka Adam baseline | 75.6 | 5.11 | |
| quokka new_best (FineWeb-Edu) | 81.3 | 5.32 | worse on gen-PPL (see above) |

**Reference anchors:** MDLM 169M/1M-steps = 82, LLaDA 1B = 60-80, GPT-2 self = 30.
Our 111M @ 250M tokens beating MDLM 169M @ 30B tokens is the biggest headline.
Peak at 30k, slight regression by 50k — early stopping matters.

## Sampling (CRITICAL — required for coherent output)

Two sampler bugs in our code before 2026-04-17 caused mode collapse
despite valid training. Both fixed in model.py; all prior checkpoints
are recoverable with the fixed sampler. Use:

```bash
python eval_gen_ppl.py  # uses fixed sampler with top-k=50
```

Bugs fixed:
- bf16 Gumbel-max truncated noise (Zheng et al. 2024); cast to fp32
- Temperature was applied to full transition distribution (including
  mask-retention prob); now applied to token distribution pre-mixing

Improvement stack from Adam baseline (10k, 3 seeds):

| Improvement | Nats gained | Val loss |
|-------------|-------------|----------|
| Adam baseline | — | 5.711 |
| + Muon (base) | -0.349 | 5.362 |
| + Variance scaling (VS) | -0.039 | 5.323 |
| + out_proj in Muon routing | -0.057 | 5.266 |
| + FineWeb-Edu + lr=0.01 | -0.197 | **5.069** |
| **10k total vs Adam** | **-0.642** | **5.069** |
| + tok_emb in Muon routing (5k only) | -0.115* | **~5.19 @ 5k** |

*Measured at 5k: baseline 5.307 → +tok_emb 5.192, 3/3 paired seeds agreed
(deltas -0.1032, -0.1236, -0.1176). Note: with n=3 / df=2 the reported
"t=-18.94" is misleading precision; treat as "direction consistent across
all 3 seeds with tight spread," not a literal p-value. Expected 10k
val_loss if gain scales linearly: ~4.95 — but 10k validation not yet run
AND the Adam-embed-LR confound is not yet controlled (see caveats below).

out_proj in Muon confirmed independently on NVIDIA 5090 by another agent.
Higher new_best variance (std 0.08 vs old 0.02) — seed 42 was lucky at 4.976;
seeds 137/2024 were closer to 5.11. Still significant at p<0.05.

**Caveat 1 (Adam-embed-LR confound, CRITICAL):** `--muon_tok_emb` bundles
three changes when toggled: (a) per-step update magnitude on the 19.3M
embedding params jumps ~380x (Adam 3e-4 vs Muon 0.01 with rectangular
scaling factor sqrt(50304/384)=11.4); (b) Newton-Schulz orthogonalization;
(c) momentum + weight-decay semantics. The -0.115 nats could be any of
these. Per nvidia/HANDOFF_nvidia.md finding #9, just raising Adam embed
LR from 1.5e-4 to 1e-3 (no Muon) was worth 0.20 nats at 30M scale - so
"Adam tok_emb at 3e-4 is undertrained" is a live alternative explanation.
REPORT.md flagged this same confound (lines 226-229, 297-301). Resolution
requires the 3-arm A/B/C: {baseline, Adam_emb_lr=1e-3, Muon_emb}. Sweep
script: sweep_adam_emb_lr_ablation_5k.py, flag: `--adam_emb_lr`.

**Caveat 2 (ELBO vs generation):** nvidia/HANDOFF_nvidia.md shows ELBO and
generation quality decouple sharply at 125M/10B scale. All improvements in
this table are val_loss (ELBO). The tok_emb finding has a plausible but
untested generation-quality upside: the geometric REPORT shows tok_emb
under Adam loses 20% of its stable rank and gains 40% of sigma_max over
10k->50k steps, temporally correlated with the 30k gen-PPL peak. Muon
routing eliminates that decay. Whether it also fixes the post-30k gen
regression is the obvious follow-up - requires training a 10L x 640d run
with `--muon_tok_emb` and running gen-PPL trajectory.

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

Also available:
- `data/fineweb-edu-10B/` — FineWeb-Edu .bin shards (from karpathy/fineweb-edu-100B-gpt2-token-shards)
- `data/fineweb-edu/` — FineWeb-Edu .npy format (from earlier experiments)

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

## Depth vs Width (5k steps, 3 seeds)

| Config | Params | Mean ± Std | vs 4L×384d |
|--------|--------|-----------|------------|
| 8L×384d | 43.1M | 5.384 ± 0.024 | -0.018 (n.s.) |
| 4L×384d | 31.5M | 5.402 ± 0.055 | — |
| 6L×384d | 37.3M | 5.423 ± 0.018 | +0.021 (n.s.) |
| 8L×320d | 32.9M | 5.436 ± 0.023 | +0.034 (n.s.) |

**Finding:** Depth doesn't help at iso-params. Narrowing to 320d hurts more than 8 layers
helps. 6L is a dead zone (worse than both 4L and 8L). More depth only helps with more
params. The current quokka (4L×384d) is well-tuned for ~31M.

## What We Have NOT Tested

- Scale-up to small (84M) or base (231M) config
- Training beyond 10000 steps
- Soft masking (Hersche et al. ICLR 2026)
- Sample quality evaluation (text generation, perplexity)
- FineWeb-Edu vs FineWeb comparison
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
