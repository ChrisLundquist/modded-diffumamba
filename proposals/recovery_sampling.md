# Checkpoint Recovery for DiffuMamba3: Sampling-Time Fixes for val/quality Divergence

## Title

**Recovering sample quality from an already-trained MDLM: a cheap-first sweep of
top-p/top-k, remasking (ReMDM), confidence-ordered unmasking, classifier-free
guidance, and brief fine-tuning.**

## Problem statement

Our 10L×640d 111.7M model trained to val_loss 4.75 produces strictly worse
samples than a 31.5M quokka trained to val_loss 5.07:

- quokka 10k (val=5.07, T=0.8, 128 steps): ungrammatical but diverse
  fragments ("due to the one of the events of the cause cause...")
- 10L×640d 50k (val=4.75): repetitive collapse ("family, family, family, the
  family..."; "I, I, and I 'toa.'s ..."; the Unicode-replacement character
  appears multiple times per sample).

The ELBO is strictly better, yet the marginals produced by MDLM sampling have
drifted to a regime where the Gumbel-max decoder in `model.sample()` repeatedly
latches onto a small set of tokens. This is the canonical "generative
perplexity collapse" phenomenon that Sahoo et al. observed for MDLM and that
the 2025-2026 literature has now produced several fixes for.

Crucially, **our `sample()` has no top-p, no top-k, no low-confidence
remasking, and no CFG**. It is the simplest possible ancestral sampler with a
temperature knob applied post-caching. The literature is essentially
unanimous that this sampler is inadequate for MDLMs at this scale.

This proposal is therefore CHEAP-FIRST: the first four experiments change
only `model.sample()` and can be run against existing checkpoints in minutes.

## Hypotheses

**H1 (nucleus sampling is the biggest free win).** MDLM's own paper (Sahoo
et al. 2024) reports that nucleus sampling (top-p=0.9) is critical for their
generative-perplexity numbers. Our `q_xs ** (1/T)` with no truncation means
the long tail of the unmasked-token distribution contributes non-trivial
mass; with temperature<1 the model's confident-but-wrong tokens dominate
while the diverse-but-low-probability tail still contributes noise. Expected
effect: large reduction in repetitive collapse, no change in val_loss.

**H2 (Gumbel-max + cache invalidation interacts badly with temperature).**
Our sampler applies temperature as `q_xs ** (1/T)`, which includes the
mask-retention probability — so temperature<1 makes the model more eager
both to keep masks AND to commit confidently. Decoupling temperature from
the mask-retention probability, and applying top-p to only the non-mask
support, should help. Expected effect: additive with H1.

**H3 (ReMDM fixes the irreversibility problem).** Once MDLM commits a token
it cannot revise it. At 128 steps with seq_len=128, any "family" committed
early forces subsequent positions to be conditionally consistent with it,
which the model does by repeating. ReMDM (Wang et al. 2025, ICLR/NeurIPS
2025) adds a remasking step with probability σ_t that lets the model
revisit early errors. Critically, the authors state: *"using σ_t=0 for
training is equivalent to combining the ReMDM sampler with a pre-trained
MDLM model"* — i.e., **ReMDM is plug-and-play on our existing checkpoints**.
Expected effect: bigger than H1 at large sample budgets (T≥1024 steps),
smaller than H1 at 128 steps. 2.23× MAUVE improvement reported on OWT.

**H4 (confidence-ordered unmasking > random Gumbel commit).** The MDLM
ancestral sampler unmasks positions stochastically (`unmask_prob` applies
uniformly across all masked positions). MaskGIT-style decoders instead
unmask the top-k most confident positions per step. This is what LLaDA and
Dream-7B actually use for generation. Ablations in LLaDA show low-confidence
remasking substantially improves over random remasking. Expected effect:
qualitatively fewer locally-inconsistent commits.

**H5 (entropy-bounded adaptive step count, free speedup).** EB-Sampler (Yu
et al. 2025) reports 2-3× fewer NFEs at same quality by unmasking as many
positions as the cumulative entropy budget allows. Not a quality fix per
se, but halves compute on the quality-fix sweeps.

**H6 (brief fine-tune with flat or MaskGIT weighting recovers quality).**
Min-SNR γ=1.5 clips the 1/t weight, concentrating gradient signal on
moderate-to-high noise levels. At 50k steps this may have over-optimized
for "easy" denoising (t∈[0.1,0.5]) at the expense of the late unmasking
steps (t near 0) that actually determine sample quality. A brief
fine-tune (~2k steps) under flat weighting (MaskGIT-style, weight=1 on
all masked positions) or under ELBO (1/t, uncapped) should rebalance
without undoing the learned representation. Expected effect: moderate
(0.02-0.10 nat val change) but meaningful sample recovery.

**H7 (CFG via unconditional-prompt contrast).** Discrete CFG for MDLMs
(A-CFG, ICG) works by running the model twice, once on the real (possibly
prompt-conditioned) sequence and once on a fully-masked sequence, then
extrapolating the log-probs: `log p_guided = log p_cond + w*(log p_cond -
log p_uncond)`. For unconditional generation it still works if
"unconditional" means "everything is mask" — which sharpens toward
high-density regions. Since our model is unconditional, we use TSG
(time-step guidance) or sigma-contrast CFG: compare predictions at
two noise levels.

## Method

All experiments use **10L640d_50k.pt** (the broken one, val=4.75) and
**10L640d_10k.pt** (the best-samples-so-far, val=5.33) as fixed
checkpoints. No retraining for experiments 1-5. Evaluate on 64 samples
per condition, seq_len=128.

### Eval protocol

Quality metrics (cheap proxies, no GPT-eval in the loop):
1. **Repetition rate**: fraction of 4-grams that appear ≥2 times within a
   sample, averaged across samples. Good MDLMs report <0.02, our current
   50k model likely at ~0.3-0.5 given the "family, family, family" mode.
2. **Distinct-n**: distinct 4-grams / total 4-grams, pooled across samples.
3. **Unicode-replacement-char rate**: fraction of samples containing the
   `U+FFFD` or raw byte 0xEF/0xBF/0xBD triple (the "�" that showed up).
4. **Mean token entropy**: entropy of the empirical token distribution
   pooled across samples. Collapse → low entropy.
5. **Val loss on a held-out batch** (unchanged from training-time metric;
   this is the sanity anchor — should NOT move under sampling changes).
6. **(Optional, after sweep) Generative perplexity**: score our samples
   under a small AR LM (e.g., GPT-2 small) — this is what MDLM/ReMDM
   papers report. Costs one forward pass per sample per condition.

Decision rule: a condition "wins" if it reduces repetition rate by ≥30%
AND keeps distinct-4 within 10% of the baseline quokka. Qualitative
inspection of 8 samples per condition is also part of decision (cheap).

### Experiment 1: Temperature × top-p × top-k grid (1 hour, existing ckpt)

**Code change:** add top-p and top-k to `DiffuMamba3.sample()`. Both operate
on `p_x0` (the token distribution) BEFORE mixing with the mask-retention
probability, so the mask-stay probability is unaffected.

```python
def _truncate(p_x0, top_k=None, top_p=None):
    # p_x0: (B, L, V). Operates on vocab dim.
    if top_k is not None:
        topk_vals, _ = p_x0.topk(top_k, dim=-1)
        thresh = topk_vals[..., -1:].expand_as(p_x0)
        p_x0 = torch.where(p_x0 >= thresh, p_x0, torch.zeros_like(p_x0))
    if top_p is not None:
        sorted_p, sorted_idx = p_x0.sort(dim=-1, descending=True)
        cumsum = sorted_p.cumsum(dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True  # always keep top-1
        mask = torch.zeros_like(p_x0).scatter(-1, sorted_idx, keep.float())
        p_x0 = p_x0 * mask
    p_x0 = p_x0 / (p_x0.sum(dim=-1, keepdim=True) + 1e-8)
    return p_x0
```

Apply `_truncate(p_x0, ...)` right after `p_x0 = log_probs.exp()` and
BEFORE `q_xs = p_x0 * unmask_prob`. Also set `q_xs[...,mask_id]` using
the original (untruncated) move_chance ratio, not a powered one.

**Also fix:** apply temperature to `p_x0` only (not to `q_xs` as a whole).
Replace the current `q_xs = q_xs ** (1/T)` with:

```python
if temperature != 1.0:
    p_x0 = (p_x0 ** (1.0 / temperature))
    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)
```

before the `_truncate` call.

**Grid** (64 samples each, on 10L640d_50k):

| T    | top_p | top_k | Notes                              |
|------|-------|-------|------------------------------------|
| 1.0  | —     | —     | baseline (current code, no trunc)  |
| 1.0  | 0.9   | —     | MDLM paper default                 |
| 1.0  | 0.95  | —     | less aggressive                    |
| 0.9  | 0.9   | —     | mild sharpen + nucleus             |
| 0.8  | 0.9   | —     | current T + nucleus                |
| 0.8  | —     | 50    | top-k sanity                       |
| 0.7  | 0.92  | —     | sharper still                      |
| 1.0  | —     | 100   | pure top-k large                   |
| 1.3  | 0.9   | —     | higher T + nucleus (diversity)     |

Cost: 9 configs × 64 samples × 128 steps × ~30 ms/step ≈ 25 minutes.

### Experiment 2: ReMDM remasking on pretrained ckpt (2 hours)

Follow the ReMDM paper's max-capped schedule, applicable without retraining.

**Code change:** extend `sample()` with a remasking path that runs AFTER
the main unmask step but BEFORE cache invalidation:

```python
# After x_new = torch.where(is_masked, sampled, x):
if remask_eta > 0.0 and i < int(num_steps * remask_t_off_frac):
    if i >= int(num_steps * remask_t_on_frac):
        # sigma_t = min(eta_cap, (1 - alpha_s) / alpha_t)
        alpha_t = 1.0 - move_chance_t
        alpha_s = 1.0 - move_chance_s
        sigma_max = torch.clamp((1 - alpha_s) / (alpha_t + 1e-8), max=1.0)
        sigma_t = torch.clamp(torch.tensor(remask_eta), max=sigma_max).item()
        # Remask previously-unmasked tokens with prob sigma_t
        can_remask = (x_new != self.config.mask_token_id)
        do_remask = torch.rand_like(x_new, dtype=torch.float) < sigma_t
        x_new = torch.where(can_remask & do_remask,
                            torch.full_like(x_new, self.config.mask_token_id),
                            x_new)
# IMPORTANT: ReMDM invalidates cache every step where remasking fires
p_x0_cache = None
```

**Conditions** (all with top_p=0.9, T=1.0 based on Exp 1 winner):

| Label        | remask_eta | t_on | t_off | num_steps | Rationale             |
|--------------|-----------|------|-------|-----------|-----------------------|
| remdm-low    | 0.02      | 0.0  | 1.0   | 256       | paper default, full   |
| remdm-mid    | 0.04      | 0.0  | 1.0   | 256       | paper small-T default |
| remdm-loop   | 0.02      | 0.55 | 0.95  | 512       | "loop" strategy       |
| remdm-aggr   | 0.08      | 0.0  | 0.9   | 256       | more aggressive       |
| remdm-budget | 0.04      | 0.0  | 1.0   | 128       | same NFE as baseline  |

Cost: 5 configs × 64 samples × 256 steps × ~30 ms/step ≈ 55 minutes.
`remdm-budget` is critical: it tests whether ReMDM wins at EQUAL compute.

### Experiment 3: Confidence-ordered (MaskGIT-style) unmasking (30 minutes)

Replace the stochastic "each masked position independently decides to
unmask with prob (1 - move_chance_s/move_chance_t)" with deterministic
top-k confidence selection: at step i, unmask exactly `n_unmask(i)`
positions — those with the highest `p_x0.max(-1)` (predicted-token
confidence). Use cosine schedule for n_unmask:
`n_unmask(i) = round(L * (cos(pi/2 * (1 - (i+1)/num_steps))**2 -
                            cos(pi/2 * (1 - i/num_steps))**2))`.

**Code sketch:**
```python
# Replace the "apply temperature + gumbel-argmax + where(is_masked,...)" block with:
sampled_tok = p_x0.argmax(dim=-1)  # or Gumbel-sampled from (truncated) p_x0
conf = p_x0.max(dim=-1).values
masked_conf = conf.masked_fill(~is_masked, -1e9)
# Pick n_unmask positions with highest confidence per batch row
n_to_unmask = cosine_schedule(i, num_steps, seq_len)
_, top_pos = masked_conf.topk(n_to_unmask, dim=-1)
x_new = x.clone()
# scatter in the chosen positions
row_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(-1, n_to_unmask)
x_new[row_idx, top_pos] = sampled_tok[row_idx, top_pos]
```

**Conditions** (64 samples each):
- `maskgit-cosine` (above, T=1.0, top_p=0.9, 128 steps, Gumbel-from-truncated)
- `maskgit-linear` (linear unmask schedule, 128 steps)
- `maskgit-greedy-argmax` (argmax instead of Gumbel, 128 steps)

Cost: ~20 minutes.

### Experiment 4: Entropy-bounded adaptive (EB-Sampler) (30 minutes)

Standalone verification that EB matches baseline at fewer NFE.

Conditions (64 samples each):
- `eb-gamma0.1` (γ=0.1, top_p=0.9, T=1.0, cap at 256 steps)
- `eb-gamma0.5`
- `eb-gamma1.0`

Cost: ~30 minutes. Ship this only if Exp 1-3 fix quality; EB is for
speed, not quality.

### Experiment 5: Sigma-contrast CFG (45 minutes)

No retraining needed: use a second forward pass at sigma_unc = 1.0 (max
noise, everything masked) as the "unconditional" predictor. Reference:
A-CFG, ICG, TSG-style adaptations for MDLM.

**Code:**
```python
log_probs_c = self(x, sigma_cond)  # conditional (normal sampling)
# Unconditional: run on a fully-masked input at max sigma
x_null = torch.full_like(x, self.config.mask_token_id)
sigma_null = self.noise.sigma(torch.ones_like(t_batch) * (1.0 - eps))
log_probs_u = self(x_null, sigma_null if self.config.time_conditioning else sigma_cond*0)
log_probs = log_probs_c + cfg_w * (log_probs_c - log_probs_u)
log_probs = log_probs - log_probs.logsumexp(-1, keepdim=True)
p_x0 = log_probs.exp()
```

**Conditions** (64 samples each, stacked on Exp-1 winner):
- `cfg-0.5`, `cfg-1.0`, `cfg-2.0`, `cfg-4.0`

Cost: ~25 minutes (2x forward passes).

### Experiment 6 (CONDITIONAL, only if 1-5 insufficient): brief fine-tune

Only run if none of Exp 1-5 drops repetition rate by ≥50%. Hypothesis is
that the 50k model's final-decode distribution is broken in a way that
sampling cannot recover.

Start from `10L640d_50k.pt`. Fine-tune for 2000 steps with:

| Label      | loss_weight | extra                         |
|------------|-------------|-------------------------------|
| ft-flat    | maskgit     | uniform weight, no 1/t factor |
| ft-elbo    | elbo        | uncapped 1/t                  |
| ft-lowT    | minsnr      | γ=1.5 (same as before) but lr=3e-5 flat, no warmup — pure "polish" |

`ft-lowT` is the cheapest: just anneal the existing objective. `ft-flat`
and `ft-elbo` shift the gradient signal toward late-denoise timesteps
which govern sample quality.

Exact CLI:
```bash
python train.py --config quokka --n_layers 10 --d_model 640 \
  --resume checkpoints/10L640d_50k.pt \
  --batch_size 8 --max_steps 2000 --val_every 200 \
  --warmup_steps 0 --lr_schedule constant \
  --optimizer muon --muon_variant vs \
  --muon_lr 1e-3 --adam_lr 3e-5 --muon_out_proj \
  --loss_weight maskgit \
  --seed 42 --save_best --ckpt_name 10L640d_50k_ft-flat
```
(Note: `--resume`, `--n_layers`, `--d_model`, `--ckpt_name`, and
`--loss_weight maskgit` may need to be wired up — check `train.py`.
`maskgit` weight is just `dsigma` without `1/expm1(sigma)`.)

Cost: 3 × ~2000 steps ≈ 1.5 hours total.

### Experiment 7 (CONDITIONAL, stretch goal): self-consistency decoding

Generate K=4 samples from the best sampler per prompt-slot, then rank by
mean pairwise 4-gram overlap and keep the median-similarity one. Cheap,
proven to reduce degenerate outliers in AR LMs (Wang et al. 2023).
Re-use Exp-1..5 samples, no new generation needed.

## Experimental matrix summary

| Exp | What                   | Code change | Compute  | Expected outcome                     |
|----|-------------------------|-------------|----------|--------------------------------------|
| 1  | top-p / top-k / T grid  | ~30 lines   | 25 min   | biggest single fix (H1,H2)           |
| 2  | ReMDM remasking         | ~20 lines   | 55 min   | additive win at high NFE (H3)        |
| 3  | MaskGIT confidence order| ~25 lines   | 20 min   | diversity improvement (H4)           |
| 4  | EB-Sampler              | ~15 lines   | 30 min   | free speedup (H5)                    |
| 5  | Sigma-contrast CFG      | ~10 lines   | 25 min   | sharpening, may help or hurt (H7)    |
| 6  | Fine-tune (conditional) | none        | 1.5 hr   | last-resort quality recovery (H6)    |
| 7  | Self-consistency        | ~5 lines    | reuse    | robustness win                       |

**Total unconditional compute: ~2.5 hours on existing checkpoints.**

## Expected outcomes

**Most likely (65%)**: Experiment 1 (top-p=0.9, T in {1.0, 0.9}) recovers
most of the quality gap. The 50k model's val_loss advantage (4.75 vs 5.07)
translates into better samples once the decoder stops letting the long
tail poison the Gumbel argmax. This would make the "val goes down,
samples go bad" finding a SAMPLER bug, not a model bug.

**Second most likely (25%)**: Exp 1 helps but doesn't fully recover (still
more repetitive than quokka). Exp 2 (ReMDM) with `remdm-loop` at 512
steps closes the gap. The interpretation: MDLM irreversibility is
fundamentally fighting us at seq_len=128 / 128 steps, and we need either
more steps or remasking to compensate.

**Third most likely (7%)**: Sampling changes insufficient; the 50k model
genuinely over-committed to high-probability tokens during training.
Exp 6 (`ft-flat` or `ft-elbo`) recovers quality in 2k steps at the cost
of a small val_loss regression. This would be new evidence that
Min-SNR γ=1.5 is quality-suboptimal at scale (a revision of our
gamma finding).

**Unlikely but interesting (3%)**: The 50k degradation is irreducible
(e.g., vocab-frequency imbalance in FineWeb-Edu at 410M tokens seen).
Only retraining from scratch with Exp-6-style weighting helps. This
would be a genuinely negative result that motivates a Round 4 proposal.

## Risk / cost

### Compute risk: LOW
Exps 1-5 fit in ~2.5 hours of wall clock on a single GPU (no training,
just forward passes). Exp 6 adds 1.5 hours. Total worst case: 4 hours.

### Implementation risk: LOW-MODERATE
- Exp 1: trivial, <30 lines of well-understood code.
- Exp 2: ReMDM is described in an ICLR/NeurIPS paper with reference code
  (github.com/kuleshov-group/remdm). Algorithm is ~20 lines. Risk is
  getting the `sigma_max` clamp right — test with σ=0 to confirm it
  reduces to the baseline sampler exactly.
- Exp 3: MaskGIT is a 2022 paper with many reference implementations.
- Exp 5: CFG is a 1-liner extrapolation. The risk is that
  `log_probs_u` on a fully-masked input is degenerate for our SUBS
  parameterization (unmasked positions are forced-copy; all positions
  are masked → all positions go through the model). Should just work.
- Exp 6: requires `--resume` in `train.py`. Check whether `train.py`
  already supports resuming from a bare state_dict `.pt` — if not, ~20
  lines to add.

### Scientific risk: LOW
Negative results on Exp 1 are informative (rules out "it was a sampler
bug"). Negative results on Exp 6 are informative (rules out "it was a
loss-weighting bug"). The only way to waste compute is to run Exp 7
without first showing a winner.

### Generalization risk: N/A
This is a checkpoint-recovery proposal. Findings inform future training
runs (e.g., "train with MaskGIT loss from the start" or "default
sampler should include top-p=0.9") but we are not claiming anything
beyond our own checkpoints.

## Literature context

- **MDLM (Sahoo et al. 2024, NeurIPS):** Our base recipe. Their own
  samples use `top_p=0.9`; we accidentally dropped this.
- **ReMDM (Wang, Schiff, Sahoo, Kuleshov 2025, ICLR/NeurIPS):** The
  primary inspiration for Exp 2. Reports 2.23× MAUVE improvement on
  OpenWebText vs MDLM baseline. Explicitly states the sampler is
  plug-and-play on pretrained MDLMs.
  (arXiv 2503.00307, github.com/kuleshov-group/remdm)
- **EB-Sampler (Yu et al. 2025):** Adaptive step count, 2-3× faster at
  equal quality. (arXiv 2505.24857)
- **Saber (Dong et al. 2025):** Adaptive acceleration + backtracking
  remasking, 251% inference speedup on code. Domain-specific but the
  backtracking idea generalizes. (arXiv 2510.18165)
- **LLaDA (Nie et al. 2025):** Low-confidence remasking substantially
  beats random remasking. Applied to 8B parameter diffusion LLM.
  (arXiv 2502.09992)
- **Dream-7B (2025):** Uses MaskGIT-style confidence-ordered decoding.
- **A-CFG (Wu et al. 2025):** Adaptive classifier-free guidance
  specifically for MDLMs. Dynamically shapes the unconditional input
  by its current uncertainty. (arXiv 2505.20199)
- **No Training, No Problem (Sadat et al. 2025, ICLR):** ICG and TSG —
  CFG variants that need no conditional training. Applicable to our
  unconditional MDLM. (arXiv 2407.02687)
- **Soft-Masked Diffusion (2025, ICLR 2026 sub):** Replaces hard [MASK]
  with a blend of mask embedding + top-k predicted-token embeddings.
  Requires retraining, so flagged for Round 5 not this proposal.
  (arXiv 2510.17206)
- **Generative perplexity collapse in discrete diffusion:** Essentially
  every 2025 MDLM paper notes that nucleus sampling is critical and
  that untruncated sampling collapses. This is the robustly established
  baseline for Exp 1.

## Decision criteria

After Exp 1 (the cheap grid):
- **If any (T, top-p) config reduces repetition rate ≥ 50% AND keeps
  distinct-4 within 10% of quokka baseline:** declare primary win.
  Still run Exp 2-3 to stack additional gains, skip Exp 6.
- **If best Exp-1 config reduces repetition 20-50%:** proceed to
  Exp 2 and Exp 3. Goal is compounding gains to ≥50%.
- **If no Exp-1 config reduces repetition ≥20%:** something deeper is
  wrong. Proceed directly to Exp 6 (fine-tune). Exp 2-5 become
  secondary.

After Exp 2-3:
- **If repetition now ≤ quokka baseline:** we have a new default sampler.
  Update `sample_large.py` and CLAUDE.md. Publish the finding as
  "MDLM sample quality is sampler-dominated, not model-dominated" —
  reasonably novel if the val_loss improvement is preserved.
- **If still substantially worse than quokka:** run Exp 6.

After Exp 6 (if run):
- **If ft-flat or ft-elbo recovers quality with ≤ 0.05 nat val
  regression:** strong evidence that Min-SNR γ=1.5 is
  quality-suboptimal. Opens a Round 4 proposal to re-sweep loss
  weightings with sample-quality metrics (not just val_loss) as the
  objective.
- **If fine-tune does not help:** escalate to full retraining with
  soft-masking (Round 5) or architecture change.

## Implementation notes

1. **Add `sample()` keyword args**: `top_p=None`, `top_k=None`,
   `remask_eta=0.0`, `remask_t_on=0.0`, `remask_t_off=1.0`,
   `order="random"` (or `"confidence"`), `cfg_w=0.0`. All default to
   current behavior so existing callers are unaffected.

2. **Add `sample_recovery.py`** script that loads a checkpoint, runs
   the grid, computes metrics, dumps `samples/recovery_<ckpt>.json`
   with per-condition sample text + metrics. Pattern after
   `sample_large.py`.

3. **Fix the current temperature bug first** (one-line change:
   `q_xs = q_xs ** (1/T)` becomes `p_x0 = p_x0 ** (1/T); p_x0 /=
   p_x0.sum(-1, keepdim=True)` before mixing with `unmask_prob`).
   This is unambiguously correct and makes subsequent experiments
   interpretable.

4. **Fix Unicode byte-fragment issue**: the "�" character indicates
   the model is generating partial UTF-8 byte sequences that don't
   decode. GPT-2 BPE tokens are bytes; this is expected when the model
   is uncertain. Top-p=0.9 should dramatically reduce this because the
   partial-byte tokens typically have <1% probability each but
   collectively hold non-trivial mass. Track this as a metric.
