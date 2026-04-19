# MDLM Training Objective Experiments — Pre-registered Plan

**Goal.** Resolve whether the ELBO/generation trajectory divergence at quokka
scale is caused by (a) schedule-weight mis-specification, (b) mask-rate
sampling, (c) planner/sampler mismatch, or (d) structural per-position
objective ceiling. Our harness (`nvidia/eval/gen_harness/`) provides the eval.

**Current evidence (2026-04-18).**
- Trajectory of our 125M D_modern on 10B FineWeb-Edu (steps 10k→72k, 100 held-out
  prompts): val_loss monotone 3.38→3.04; teacher_NLL U-shape with minimum at 30k
  (3.22); distinct_4 plateaus at 0.85 after step 40k; rep_4 flatlines 0.13–0.18.
- Real FineWeb-Edu continuations: distinct_4=0.99, rep_4=0.014. We are ~10×
  worse on rep_4.
- Inference sweep (24 configs): monotone Pareto NLL↔diversity, cannot reach
  real-text region by sampling tweaks alone.
- rhysjones 124M AR (same data/scale): distinct_4=0.97, rep_4=0.028 — matches
  real text on diversity while the MDLM doesn't. AR beats MDLM on generation at
  matched scale, but recipe differs (llm.c vs our Muon+WSD).

## Theory

Training loss decomposes as `w(λ) · p(λ)` — weight on noise level × density
over noise levels (VDM++, Kingma & Gao 2023). γ-decay and a mask-rate
curriculum are the same intervention at different leverage points.
Empirically in image diffusion, `p(λ) > w(λ)` in leverage: modifying what you
*sample* beats modifying how you *weight*.

The Peng 2025 PAPL paper adds orthogonal leverage: even with perfect w(λ)/p(λ),
the standard MDLM ELBO is provably mis-specified when the sampler uses a
non-uniform unmasking planner. Correction is a drop-in per-position reweight.

## Papers referenced

All present in `/home/clundquist/modded-diffumamba/papers/`.

### Loss / objective (immediate relevance)
- **PAPL / P-ELBO** (Peng 2025, 2509.23405) — sampler-aware ELBO; running fine-tune now.
- **P2 Path Planning for MDM Sampling** (Peng 2025a, 2502.03540) — PAPL's predecessor; planning + denoising as 2-stage; unifies MaskGIT/RDM/top-k.
- **VDM++** (Kingma & Gao 2023, 2303.00848) — w(λ)·p(λ) decomposition framework.
- **Improved Noise Schedule** (Hang 2024, 2407.03297) — p(λ) > w(λ) leverage; Laplace/Cauchy schedules.
- **Denoising Curriculum** (Kim 2024, 2403.10348) — easy-to-hard t-curriculum.
- **Frequency-Informed MDLM** (Kosmopoulou 2025, 2509.05056) — cosine/bimodal mask schedules.
- **MuLAN: Diffusion Models with Learned Adaptive Noise** (Sahoo 2023, 2312.13236) — learned per-timestep noise.

### Architectural ELBO tightening
- **Block Diffusion (BD3LM)** (Arriola 2025, 2503.09573) — block-AR + within-block diffusion. NELBO variance analysis + clipped noise schedules. **Reviewer top-1 recommendation** for stacking with PAPL.
- **ADLM Anchored Diffusion LM** (2505.18456) — anchor network predicting pivot token; OWT PPL 20.14, first DLM to beat AR on MAUVE w/ remasking. **Reviewer top-3 recommendation** (architectural change, biggest payoff).
- **Loopholing Discrete Diffusion** (Jo 2025, 2510.19304) — deterministic forward of predicted distribution; -55% Gen-PPL vs MDLM.

### Inference-time correctors
- **ReMDM** (Wang 2025, 2503.00307) — remasking diffusion, allows committed tokens to revert. **Reviewer top-2 recommendation** (cheap, stacks with PAPL).
- **Informed Correctors** (Zhao 2024, 2407.21243) — discrete diffusion correctors at inference.

### Planner-side / order policies
- **Any-Order Flexible-Length MDM** (Kim 2025, 2509.01025) — adds insertion to MDM; learns order end-to-end.

## Experiments (pre-registered)

Protocol for all: 40M scale, 3 seeds, 5B tokens effective (except Exp 4).
Trajectory logging every 500 steps: val_loss, teacher_NLL ×2 teachers
(gpt2-small, rhysjones-FWE), rep_4, distinct_4, uniq_token_ratio.

### 1. Baseline (current setup)
- Min-SNR ELBO, γ=1.5 constant
- Uniform t ~ U(0.05, 0.95)
- Purpose: reproduces observed trajectory divergence as reference

### 2. γ-decay
- Min-SNR γ linearly decayed 1.5 → 0.5 over training
- Uniform t
- Leverage point: **weight** function w(λ)
- Predicted effect (modest, per Hang 2025): minor generation gain, minor val_loss loss

### 3. p(t) curriculum
- Min-SNR γ=1.5 constant
- t-sampling: U(0.05, 0.95) at step 0 → Beta(2,2) at end (or Laplace around t=0.5)
- Leverage point: **density** p(λ) — per VDM++ and Hang 2025, this dominates
- Predicted effect (larger than Exp 2): shifts gradient budget toward hard masks

### 4. P-ELBO fine-tune (CHEAP FIRST EXPERIMENT)
- Resume from `checkpoint_40000.pt` (our diversity peak)
- PAPL loss: `L = Σ (1 + α·w_i) · CE_i` with self-planner at τ=1.0, α=1.0
- Constant LR, +10k steps, Min-SNR γ=1.5 preserved
- Leverage point: **per-position reweight** that accounts for non-uniform sampler
- Purpose: minimum-viable diagnostic. If rep_4 stays ≤ 0.13 at effective step
  50k (where std training regresses to 0.16), the objective-sampler mismatch
  hypothesis has signal. If not, the cheap-fix path is dead.
- **This runs first. Results gate Exp 5.**

### 5. P-ELBO from scratch
- Only if Exp 4 shows rep_4 movement
- Full 125M retrain with PAPL loss, α=1.0, τ=1.0
- 10B FWE × 1 epoch, WSD schedule, identical to baseline recipe
- Purpose: confirm that the fix requires full-training integration, not just a
  late-stage correction

## Decision criteria (pre-registered, do not move goalposts)

Effect size must exceed seed variance. Estimated seed noise on rep_4 at
quokka scale ±0.02 (from prior 3-seed ablations).

**Meaningful hit on Exp 4**: rep_4 at PAPL-fine-tuned step 50k ≤ 0.13
(≤ original step 40k level). distinct_4 ≥ 0.87. Either without a catastrophic
collapse on teacher_NLL (within 0.3 nats of step-40k baseline).

**Null result on Exp 4**: rep_4 ≥ 0.15. In that case, skip Exp 5; conclude
the cheap P-ELBO retrofit is insufficient at our scale.

**Meaningful hit across Exps 2–3**: rep_4 < 0.10 at end of training for any
seed. That would support the VDM++ p(λ) > w(λ) conjecture in our setting.

**Null result across all**: per-position denoising objectives structurally
cannot shape joint-sample statistics at this scale. Next direction: SEDD
(score-entropy, sequence-level) or explicit AR-distillation.

## Methodological caveats (from code review, 2026-04-18)

These limitations are baked into the harness as currently built. Report results
with these in mind; do not advertise them away.

1. **rhysjones AR teacher saw the held-out region.** The 124M
   `rhysjones/gpt2-124M-edu-fineweb-10B` was trained on the entire
   FineWeb-Edu 10B corpus, which includes our `[9.5e9 : 9.9e9]` prompt region.
   Its NLL on AR-generated continuations is artificially deflated by
   memorization of the surrounding text distribution. Use it as a *secondary*
   teacher; rely on GPT-2-small (WebText, not FWE) for the unbiased number.
   The 125M MDLM's training overlap is `[:9.5e9]` so it does NOT include
   the prompt region — its rhysjones teacher_nll is fair.

2. **top-k=50 has different operational meaning for AR vs. MDLM.** AR top-k
   filters the next-token distribution conditioned on a *committed* prefix
   (one position at a time). MDLM top-k filters the per-position marginal at
   each unmask step, where surrounding masks are still uncertain. The
   distributions sampled from are not directly comparable in entropy. Same
   nominal `k` is not the same operational filter.

3. **AR uses 2× more inference compute than MDLM** in our harness: 128
   forward passes (one per generated token) vs MDLM's 64 (cont_len/2 demask
   steps with 2 tokens/step). Inference compute is asymmetric in AR's favor.
   Reported `gen_seconds` is also unfair because AR adapter has no KV cache.

4. **PAPL implementation correctness** (fixed 2026-04-18 after code review):
   the planner score uses log-probability of the GROUND-TRUTH token at each
   masked position (`logprobs.gather(-1, x.unsqueeze(-1))`), per Peng 2025
   Algorithm 1, NOT max-confidence-over-vocab. The latter was the buggy
   first implementation that we killed and restarted before any results were
   gathered.

5. **Min-SNR γ=1.5 stacked on PAPL.** The Peng paper's formulation has no
   Min-SNR clamp. We retain γ=1.5 by default to keep the fine-tune comparable
   to the std baseline checkpoints (also trained with γ=1.5). For a
   paper-faithful PAPL ablation, run with `--gamma-start 0` to disable the
   clamp. This is a planned follow-up control, not a current result.

## Compute

- Exp 4 (running): resume + 10k steps × 125M. ~3.7 hours on 5090. Current run.
- Exp 2, 3: 40M × 5B × 3 seeds. ~2 hours per seed. ~6 hours each variant.
- Exp 5: 72k steps × 125M. ~12 hours.

Total if all run: ~36 hours of 5090 time. Exp 4 is a ~10% of that cost and
gates the rest.

## Status

| Exp | Status | Output |
|---|---|---|
| 4. P-ELBO fine-tune (40k → 45k) | done, signal positive | rep_4 0.153 → 0.128 in 5k steps |
| 4b. P-ELBO fine-tune (45k → 50k) | **in progress** | `muon_exp/outputs/125m_papl_finetune/checkpoint_50000.pt` |
| 1. Baseline | already run (72k trajectory) | `trajectory_d_modern_125m.jsonl` |
| 2. γ-decay | not started | — |
| 3. p(t) curriculum | not started | — |
| 5. P-ELBO from scratch | blocked on Exp 4b | — |

## Future work — ranked by reviewer's leverage estimate

These compose with our current PAPL track. Implement only if Exp 4 + diagnostics
confirm the mechanism is real on our setup.

1. **Clipped noise schedule (BD3LM-style)** — tune α_t schedule for minimum NELBO
   variance on a held-out grid; replace our linear `1-exp(-5t)` schedule. Cheap,
   composes cleanly with PAPL. Reviewer's top recommendation.
2. **Remasking corrector at inference (ReMDM / Informed Correctors)** — inference-time
   add-on that lets committed tokens revert under low-confidence signal. Stacks
   with PAPL. Cheap; needs mostly sampler-side code.
3. **ADLM anchor network** — architectural change, requires retraining from scratch
   with an extra "pivot" head. Strongest published numbers (OWT PPL 20.14,
   first MDLM to beat AR on MAUVE with remasking). Largest investment, largest
   potential gain.

Lower-priority items (catalogued, not currently planned):
- Hierarchical / tree-structured vocab (HDLM, TDLM)
- Soft-Masked Diffusion (mask embedding blended w/ top-k predictions)
- Post-training/alignment: VRPO, ELBO-KTO, diffu-GRPO, GDPO, LFPO
- Variance reduction: antithetic t-sampling, length-stratified batches
