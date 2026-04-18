# Training Regime Fixes for val-loss / sample-quality divergence

Advisor: research proposal for DiffuMamba3 (bidirectional Mamba-3 + Muon-VS MDLM).
Target hardware: RX 9070 XT (16 GB), ~58 k tok/s fwd+bwd+opt at 1024 ctx.
Budget assumption: quokka 31.5 M at 5 k steps = ~15–20 min/run; 10 k = ~40 min; 10L x 640 d at 50 k = ~3.5 h.

## TL;DR

The training recipe has at least four compounding issues that jointly drive the
observed divergence between val ELBO and sample coherence:

1. **Mask-only absorbing diffusion is intrinsically prone to "fill-in-the-blank"
   shortcut learning.** At low mask rate the model sees almost the whole
   sentence — it can copy local context (unigram/bigram frequency + SUBS copy) to
   drive loss down without ever building a coherent joint. GIDD (ICML 2025)
   demonstrates this exact pathology: pure mask models reach generative PPL ~904
   while adding 10 % uniform (token-substitution) noise drops PPL to ~387 **with
   slightly higher validation loss**. This is the single best explanation for
   what we see.
2. **Min-SNR γ = 1.5 is *too small* for MDLM + cross-entropy.** γ was tuned at
   5 k steps for val-loss (which is ELBO). At longer training, clipping 1/t at
   1.5 under-weights the low-noise regime — the very timesteps where "this word
   follows that one" lives. Sample coherence regresses even though ELBO keeps
   improving. The gamma sweep memory note confirms γ effect is ~0.025 nats at
   5 k — small enough to be dominated by a sampling-quality regression at 50 k.
3. **Val uses ELBO while training uses γ = 1.5.** The metric we select and
   early-stop on rewards the exact regime we detuned. Any "improvement" we see
   in val may be the model drifting toward the un-weighted behaviour.
4. **Cosine LR on < 1 epoch at bs = 4** bakes whatever patterns were learned
   during warmup, with no opportunity to "re-learn" from LR restart. At 111 M
   params and 50 k steps, model memorises local n-gram behaviour before coherent
   long-range structure emerges.

Of these, #1 is the highest-confidence, highest-leverage fix; #2–#4 are cheap
ablations worth running in parallel.

The proposal below is a **tiered experiment plan**: (A) cheap ~5 k-step
diagnostics at quokka scale (~15 min each); (B) one 50 k rematch at 10L×640d
against the existing 50 k baseline; (C) architectural extensions if A/B hold.

---

## Hypothesis (ranked by expected impact)

H1. **Hybrid mask + uniform noise (GIDD-style) is the root fix.** Mixing a small
  fraction of uniform token substitution into the forward process removes the
  SUBS shortcut (positions are no longer guaranteed correct) and forces the
  model to *verify* context rather than merely infilling. Expected: val ELBO
  slightly higher, generative quality substantially better.

H2. **γ too small.** Raising Min-SNR γ (or switching to flat weighting, or to
  MaskGIT-style weighting) at long schedules re-weights low-noise timesteps
  where coherence is learned. Expected: higher val but better samples.

H3. **Eval-aligned training.** Either (a) train with ELBO 1/t (let val match
  training) or (b) eval with γ = 1.5 (align eval to training). Either removes
  the perverse incentive that lower val = more shortcut.

H4. **Regularisation gap.** No dropout, no stochastic depth, no token masking
  augmentation at 111 M params; implicit regularisation only. Adding modest
  dropout (0.1) + per-token noise may delay shortcut learning.

H5. **Noise schedule.** Log-linear concentrates samples near t = 0 (after
  antithetic). Linear α_t = 1–t, or the Quokka-DLM easy-to-hard curriculum
  (sample low-t first, anneal to high-t), may improve both metric and samples.

---

## Method: four experiment tiers

### Tier A — 5 k-step quokka sweeps, n = 3 seeds each
Budget: ~15 min × 3 seeds × 8 configs ≈ 6 h total wall-clock.

All runs: quokka 31.5 M config (4 L × 384 d), bs = 4, seq = 1024,
FineWeb-Edu, cosine LR, warmup = 200, Muon-VS + out_proj, muon_lr = 0.01,
adam_lr = 3e-4, 3 seeds {42, 137, 2024}, eval uses *both* ELBO **and** γ = 1.5
ELBO and a "gen-PPL proxy": unconditional sample 64×256, score under the model
with γ = 1.5, report mean NLL.

| id   | change vs current best | purpose |
|------|-----------------------|---------|
| A0   | none (control: γ=1.5 log-linear MDLM) | baseline |
| A1   | loss_weight=flat | eliminate Min-SNR weighting confound |
| A2   | minsnr_gamma=5 | "bigger γ" test — re-weight low-noise |
| A3   | minsnr_gamma=∞ (= ELBO 1/t) | pure ELBO, eval-aligned |
| A4   | +10 % uniform noise (GIDD p_u=0.1) | root fix for shortcut |
| A5   | A4 ∧ minsnr_gamma=5 | H1 + H2 stacked |
| A6   | dropout=0.1 on MLP + residual stream | regularisation |
| A7   | easy-to-hard t curriculum (t ~ Beta(1,4) warmup → U[ε,1]) | H5 |

**Primary metric:** mean 4-gram repetition rate on 64 samples (sample_len=256,
num_steps=128, temp=0.8) — this is the actual failure mode we're debugging,
not val loss. **Secondary:** val ELBO, generative NLL under the model itself,
distinct-n.

Gating rule: any config that improves repetition rate by ≥ 30 % vs A0 with
val-ELBO within +0.10 nats promotes to tier B.

### Tier A implementation notes

- Uniform noise (A4): MDLM's forward becomes a two-step process
  `x_t_mask = mask with prob (1-pu)*t; x_t_unif = uniform-resample with prob pu*t`.
  SUBS parameterisation must change: for uniform-corrupted positions the "copy
  the input" shortcut no longer applies, so we must *not* force identity on
  unmasked positions. Use the GIDD "clamped" weighting with w_max=1 as a safe
  starting point. Reference impl at https://github.com/dvruette/gidd.
- Dropout (A6): wrap each BiMamba3Block output with `nn.Dropout(0.1)` and the
  SwiGLU output with `nn.Dropout(0.1)`. Keep during train only — MDLM is
  bidirectional, no attention dropout needed.
- Curriculum (A7): during first 20 % of steps, sample t from Beta(2,5) (more
  mass near 0 = easier); linearly interpolate concentration → uniform by 40 %.
- All runs store a single checkpoint at best val; also sample at step
  5 k regardless, to decouple "checkpoint selection" confound from the
  val/sample divergence question.

### Tier B — 50 k long-run rematch (gated by Tier A)
Budget: ~3.5 h each.

Take the **top-2 configs from Tier A by repetition-rate metric** and re-run at
10L × 640 d 50 k steps (the exact setup that produced the "family, family, family"
collapse) with seeds {42, 137}. Save checkpoints every 10 k. Sample at each
checkpoint to plot **gen-quality vs val-loss trajectories** explicitly — if
H1/H2 are right, divergence should either disappear or flip sign.

Compare against the existing 10L640d_50k checkpoint trajectory
(`samples/samples.json` + `checkpoints/10L640d_50k_step{10,20,30,40,50}k.pt`).

### Tier C — architectural add-ons (gated by Tier B win)
Budget: ~4 h each, single seed first.

- C1. Self-conditioning: feed previous-step p_x0 as extra input channel during
  training (50 % of the time), sampling uses the same. Dream-style. May further
  improve sample coherence at same val.
- C2. Linear noise schedule instead of log-linear (Quokka-DLM finding —
  "linear consistently outperforms the others in both train/val loss and
  MMLU").
- C3. Pre-training from AR weights (Dream 7B-style). Out of scope for GPT-2
  vocab + Mamba, but: initialize Mamba blocks by copying weights from an AR
  run of the same architecture trained for 5 k steps with causal mask.

### Tier D — regime we deliberately avoid

- Gradient accumulation. Arxiv 2507.07101 (Small Batch Size Training) shows it
  is wasteful; our bs = 4 is actually a feature, not a bug. Skip.
- Reinforcement learning (SPG, MDPO, ReMDM training). Budget incompatible at
  our scale.
- Switching to SEDD or uniform-only diffusion. Prior work shows uniform-only
  is worse at scale; GIDD hybrid is the sweet spot.

---

## Minimal code diffs required

1. `model.py` — add `loss_weight="flat"` already exists; add
   `minsnr_gamma=float("inf")` path (treat as ELBO). Add `uniform_noise_p`
   config field; in `q_xt` / `_subs_parameterization`, handle uniform-corrupted
   positions (they should not be forced to copy). Approx 60 LOC.
2. `model.py` — add `dropout` config field; thread through BiMamba3Block and
   SwiGLU. Approx 15 LOC.
3. `model.py` — add `t_sampling` config {"uniform", "beta_2_5", "curriculum"};
   switch in `_sample_t`. Approx 20 LOC.
4. `train.py` — add `--val_loss_weight` flag (default "elbo" preserved); when
   set to "both", log ELBO and γ=1.5 val separately. Approx 10 LOC.
5. `sample_and_categorize.py` — already exists; extend to emit repetition-rate
   and distinct-n summary JSON per-checkpoint for automated gating.
6. New `sweep_regime_5k.py` — Tier A driver. Approx 80 LOC, copy
   `sweep_round3.py` shape.

Everything backward-compatible; existing quokka results unchanged.

## Expected outcomes (probability-weighted)

- **H1 hybrid noise (A4, A5):** ~70 % chance of ≥30 % reduction in 4-gram
  repetition with val within +0.10 nats. GIDD paper shows this exact dissociation.
- **H2 larger γ / H3 eval-aligned (A2, A3):** ~50 % chance of modest (~10–15 %)
  repetition reduction but possibly *worse* val. Mostly useful as diagnostic.
- **H4 dropout (A6):** ~30 % chance of marginal improvement at this scale. Known
  to be more useful multi-epoch; we are sub-epoch. Worth one run.
- **H5 curriculum (A7):** ~25 %. Quokka-DLM paper shows linear schedule
  (C2) > cosine/log-linear but curriculum effect was modest.

Dominant risk: Tier A's cheap metric (4-gram repetition on 64 samples at 5 k
steps) may not correlate with 50 k behaviour. Mitigation: also sample at 1 k
and 5 k within each Tier-A run and check that repetition is already worse
than ELBO-optimal n-gram stats predict; this is the *early* analogue of the
50 k collapse.

Secondary risk: Uniform noise + SUBS rework is the biggest code change. If
we get it subtly wrong, A4/A5 become uninterpretable. Mitigation: unit-test
against GIDD reference impl on a tiny fixed batch before running the sweep.

## Cost summary

| Tier | Runs      | Budget   | Gated?                 |
|------|-----------|----------|------------------------|
| A    | 24 (8×3)  | ~6 h     | no — starting tier     |
| B    | 4  (2×2)  | ~14 h    | yes, on Tier A win     |
| C    | 2–3       | ~12 h    | yes, on Tier B win     |

Total worst case: ~32 h training; if H1 holds at Tier A, Tier C likely
unnecessary → ~20 h. Fits within a weekend.

## Decision point

If Tier A returns **no** config with ≥30 % repetition reduction at comparable
val, the problem is probably deeper than the training regime — likely
Mamba-3's factorised-per-position prediction (the "independent parallel
sampling is incoherent" issue that ReMDM/ADJUST address at inference time) or
an SSM-specific representation failure. At that point the follow-up is
inference-side (ReMDM low-confidence remasking) rather than training-side,
which is a separate proposal.

---

## References consulted

- Sahoo et al., *Simple and Effective Masked Diffusion Language Models*,
  NeurIPS 2024 (arXiv 2406.07524). Establishes SUBS, low-discrepancy sampler,
  log-linear schedule.
- von Rütte et al., *Generalized Interpolating Discrete Diffusion (GIDD)*,
  ICML 2025 (arXiv 2503.04482). **Key paper for this proposal:** hybrid
  mask+uniform (p_u=0.1) cuts generative PPL 904 → 387 with slight val loss
  increase. Matches our symptom exactly.
- Hang et al., *Min-SNR Weighting Strategy*, ICCV 2023 (arXiv 2303.09556).
  Origin of γ clipping; tuned for pixel diffusion, not discrete.
- *Training Optimal Large Diffusion Language Models* (arXiv 2510.03280).
  Recommends linear schedule, bs=256+, weight decay in multi-epoch, confirms
  MaskGIT ≈ ELBO initially but ELBO wins long-run.
- Ye et al., *Dream 7B* (arXiv 2508.15487). AR init + context-adaptive noise
  rescheduling; reference for C3.
- Nie et al., *LLaDA: Large Language Diffusion Models* (arXiv 2502.09992).
  Uses p_mask = 0.15 random per-token masking in gradient updates — a mild
  form of what we'd add in A6.
- *Small Batch Size Training for Language Models* (arXiv 2507.07101).
  Supports staying at bs = 4 rather than accumulating.
- *ReMDM: Remasking Discrete Diffusion Models with Inference-Time Scaling*
  (arXiv 2503.00307). Inference-time fix for the same repetition failure;
  reserved for the fallback proposal.
