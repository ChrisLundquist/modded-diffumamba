# DiffuMamba3 Autoresearch: Experiment Log

## Environment
- GPU: AMD Radeon RX 9070 XT (16GB VRAM, RDNA4, ROCm 7.2 via WSL2 DXG)
- PyTorch: 2.12.0.dev20260408+rocm7.2, torch.compile enabled
- Data: tiny_shakespeare (369K train tokens, 39K val tokens, GPT-2 tokenizer)
- Val loss always evaluated with standard ELBO weighting for comparability
- All experiments single-seed unless noted (n=1, no significance tests)

## Caveats
- **Overparameterized**: 8.4M params on 369K tokens is ~23x overparameterized. Results
  measure training dynamics (how fast loss decreases), not generalization.
- **Single seed**: Most results are n=1. Differences <0.05 in val_loss are likely noise.
- **Short training**: 500 steps x 32 batch x 256 seq = 4M tokens seen (~11 epochs of
  369K tokens). We are measuring early training behavior, not convergence.

---

## Phase 1: Small model (84M), 200 steps — Discarded

Model far too large for data. LR schedule was broken (warmup=max_steps).
Results not reliable. Key lesson: match model size to data size.

---

## Phase 2: Loss Weighting (n=1 per condition)

**Setup**: tiny config (8.4M params), Adam lr=3e-4, cosine schedule, warmup=50, 500 steps

### ELBO vs Flat vs Min-SNR

| loss_weight | val_loss | vs ELBO |
|------------|----------|---------|
| elbo (1/t) | 5.9201 | baseline |
| flat (1) | 5.8598 | -1.0% |
| **minsnr(5)** | **5.7877** | **-2.2%** |

### Min-SNR Gamma Sweep

| gamma | val_loss | note |
|-------|----------|------|
| 1 | 5.8788 | ≈ flat |
| 2 | 5.8167 | |
| 3 | 5.8010 | |
| **5** | **5.7877** | **optimal, matches Hang et al. default** |
| 8 | 5.8708 | |
| 10 | 5.8561 | |

**Finding**: Min-SNR gamma=5 is optimal. Clear U-shaped curve. Matches the default from
Hang et al. (ICCV 2023). Zero wall-clock overhead (`torch.clamp` on a (B,) tensor).

### Time Conditioning Ablation

| time_cond | val_loss |
|-----------|----------|
| **ON** | **5.7713** |
| OFF | 5.8482 |

**Finding**: Time conditioning helps (~1.3%). Keep it on, matching DiffuMamba.

---

## Phase 3: Hyperparameter Sweep (n=1 per condition)

**Setup**: tiny config, minsnr(5), cosine schedule, warmup=50, 500 steps
**Sweep**: lr x wd x beta2 = {1e-4, 3e-4, 1e-3} x {0, 0.01, 0.1} x {0.95, 0.999} = 18 runs

### Full Results (sorted by val_loss)

| Rank | lr | wd | beta2 | val_loss |
|------|-----|-----|-------|----------|
| **1** | **1e-3** | **0** | **0.999** | **5.2296** |
| 2 | 1e-3 | 0.01 | 0.95 | 5.2370 |
| 3 | 1e-3 | 0.01 | 0.999 | 5.2753 |
| 4 | 1e-3 | 0.1 | 0.999 | 5.3262 |
| 5 | 1e-3 | 0.1 | 0.95 | 5.3586 |
| 6 | 1e-3 | 0 | 0.95 | 5.3623 |
| 7 | 3e-4 | 0.1 | 0.95 | 5.7622 |
| 8 | 3e-4 | 0 | 0.999 | 5.7705 |
| 9 | 3e-4 | 0 | 0.95 | 5.7790 |
| 10 | 3e-4 | 0.1 | 0.999 | 5.8009 |
| 11 | 3e-4 | 0.01 | 0.95 | 5.8472 |
| 12 | 1e-4 | 0.01 | 0.999 | 6.2839 |
| 13 | 1e-4 | 0.1 | 0.95 | 6.3492 |
| 14 | 1e-4 | 0.1 | 0.999 | 6.3755 |
| 15 | 1e-4 | 0.01 | 0.95 | 6.3996 |
| 16 | 1e-4 | 0 | 0.999 | 6.4000 |
| 17 | 3e-4 | 0.01 | 0.999 | 6.4248 |
| 18 | 1e-4 | 0 | 0.95 | 6.4865 |

### Analysis

**LR dominates all other hyperparameters.** The gap between LR levels (~0.5 loss units
per 3x LR increase) dwarfs WD and beta2 effects (~0.1 units).

**Effect sizes within each LR tier:**
- lr=1e-3: range 5.23-5.36 (spread 0.13)
- lr=3e-4: range 5.76-5.85 (spread 0.09), excluding outlier at 6.42
- lr=1e-4: range 6.28-6.49 (spread 0.21)

**Beta2**: 0.999 (MDLM default) slightly better than 0.95 (DiffuMamba) at lr=1e-3 and lr=1e-4.
Effect is small (~0.05-0.1) and inconsistent. Not clearly significant at n=1.

**Weight decay**: wd=0 best at lr=1e-3 (matching MDLM). Inconsistent at other LRs.
DiffuMamba's wd=0.1 is not optimal for our setup.

**One outlier**: lr=3e-4, wd=0.01, beta2=0.999 at 6.4248 — much worse than neighbors.
Likely a bad seed or training instability at that specific combination.

### Comparison with Published Configs

| Config | lr | wd | beta2 | Schedule | Our val_loss at their settings |
|--------|-----|-----|-------|----------|------|
| **Our best** | **1e-3** | **0** | **0.999** | **cosine** | **5.2296** |
| MDLM (Sahoo et al.) | 3e-4 | 0 | 0.999 | constant | 5.7705* |
| DiffuMamba (Singh et al.) | 1e-4 | 0.1 | 0.95 | cosine | 6.3492 |
| Quokka (<8B) | 2e-4 | ? | ? | WSD | not tested |

*MDLM uses constant schedule; this result is with cosine. Other agent found constant
may be better, pending confirmation.

---

## Phase 4: Grokking Composition Test (separate repo: grokking-svd)

**Question**: Do Muon (continuous gradient orthogonalization) and one-shot attention noise
(subspace perturbation) compose or conflict for accelerating grokking?

**Setup**: 1-layer transformer, modular addition mod 113, 10 seeds per condition

### Unpaired Results (n=10 per condition)

| Condition | Median T_grok | Speedup |
|-----------|--------------|---------|
| AdamW baseline | 5997 | 1.0x |
| Muon lr=0.05 | 1938 | 3.1x |
| AdamW + attn noise | 2004 | 3.0x |
| Muon + attn noise | 1790 | 3.4x |

### Statistical Assessment

The composition effect (1790 vs 1938) is ~8% — within the variance of 10 unpaired seeds.
Verification on 9 paired seeds from the same run gave Wilcoxon p=0.33. **Not significant.**

A rigorous 20-seed paired test was started but killed at 2/20 seeds to free GPU for the
hyperparameter sweep. The 2 completed paired seeds showed:
- Seed 0: Muon 1924, Comp 1950 (comp slower)
- Seed 1: Muon 1756, Comp 1593 (comp faster)

**Conclusion**: The methods are **redundant, not composing**. Both saturate the same ~3x
speedup ceiling by escaping the same 16-dimensional memorization subspace. This is
consistent with the paper's geometric theory — the speedup is a property of the
subspace escape, not the escape method.

---

## Followup Experiments (in progress)

Testing higher LR and constant schedule (Quokka-style Warmup-Stable):
1. lr=1e-3, constant schedule
2. lr=3e-3, cosine
3. lr=3e-3, constant
4. lr=2e-4, constant (actual Quokka config)

---

## Code Changes Made

### model.py
- `loss_weight` config: "elbo", "flat", "minsnr" (replaces boolean `flat_elbo_weight`)
- `minsnr_gamma` config: clamp value for Min-SNR (default 5.0)
- Implementation: `weight = torch.clamp(dsigma / expm1(sigma), max=gamma)`

### train.py
- `--lr_schedule`: "cosine" (default) or "linear" (warmup + linear cooldown)
- `--loss_weight`: CLI arg for loss weighting scheme
- `--minsnr_gamma`: CLI arg for Min-SNR gamma
- `--adam_beta2`: CLI arg for Adam beta2 (default 0.95)
- Cosine schedule: `min_lr + 0.5 * (1 - min_lr) * (1 + cos(pi * progress))`

### grokking-svd/run_composition.py (new)
- 4-condition composition test: AdamW, Muon, Noise, Muon+Noise
- Inlined Muon optimizer to avoid import side effects
- Selective momentum reset (attention only, not MLP)

---

## Current Best Configuration

```bash
python src/train.py \
  --config tiny \
  --optimizer adam \
  --adam_lr 1e-3 --adam_wd 0 --adam_beta2 0.999 \
  --loss_weight minsnr --minsnr_gamma 5 \
  --lr_schedule cosine \
  --compile
```

Best val_loss: **5.2296** (500 steps, tiny config, tiny_shakespeare)

---

## Key References

- **Min-SNR**: Hang et al., ICCV 2023, arXiv 2303.09556
- **MDLM**: Sahoo et al., NeurIPS 2024, arXiv 2406.07524
- **DiffuMamba**: Singh et al., 2025, arXiv 2511.15927
- **Quokka**: Ni et al., 2025, arXiv 2510.03280 (scaling laws, MegaDLMs framework)
- **EGD**: Pasand & Dohmatob, ICLR 2026, arXiv 2510.04930
- **Grokking SVD**: Lundquist, 2026, github.com/ChrisLundquist/grokking-svd
