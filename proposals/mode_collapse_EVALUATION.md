# Mode-Collapse Evaluation: Three Proposals for the val/quality Divergence

**Date:** 2026-04-17
**Evidence under discussion:** 10L x 640d (111.7M) at 50k steps, val_loss=4.75,
produces repetitive collapse ("family, family, family", U+FFFD), whereas
quokka (31.5M) at 10k steps, val_loss=5.07, produces ungrammatical but
diverse fragments.

---

## Executive Summary

All three proposals correctly identify that the symptom (val going down while
samples get worse) has to mean our current measurement apparatus is wrong. They
differ in which layer of the apparatus they blame: `recovery_sampling.md` says
the sampler itself is mis-specified (no top-p, a temperature bug that touches
the mask-retention probability, and no remasking to recover from early bad
commits). `eval_metrics.md` says the metric is a mirage, additionally fingers a
bf16 Gumbel precision bug (Zheng et al. 2409.02908), and proposes a three-tier
metric stack that we should have had from day one. `training_regime.md` blames
the MDLM objective itself for rewarding SUBS fill-in-the-blank shortcut
learning, and proposes a GIDD-style hybrid-noise rematch plus regime fixes.

These are not competing explanations; they are almost certainly all partially
true, and they have a strict order of operations dictated by dependency. If the
sampler is broken we cannot interpret any "sample quality" number from any
training run; if the metric is broken we cannot rank any training regime; only
after both are fixed does it make sense to re-litigate the objective. The
correct plan is therefore: fix the sampler bugs first (30 min, reuses 63
existing checkpoints, answers the "is this a mirage?" question), add an
entropy-gated evaluation stack second (couple hours, makes further runs
trustworthy), and only then spend 6-12 GPU-hours on GIDD-style training
rematches. Any other order wastes compute, because Tier A of the training
proposal would be evaluated with the very metric we already know is
misleading.

---

## Ranked Ordering

### #1 - `recovery_sampling.md` (highest information per compute-minute)

**Why first:** The highest-probability failure mode for the observed symptom
is a sampler pathology, for three reasons. First, `model.py:638-644` applies
temperature to `q_xs` *after* the mask-retention probability is stitched in,
which is mathematically wrong (the temperature knob affects the mask-stay
decision, which is not what temperature should do in an MDLM sampler; Sahoo
et al. separate these). Second, we have no nucleus/top-p truncation at all,
and the MDLM paper itself uses top_p=0.9 to produce the numbers we're
implicitly comparing to. Third, MDLM is irreversible at inference: once
"family" is committed at position 3, every subsequent unmasking is
conditioned on it, and a small early error multiplies. ReMDM is literally
plug-and-play on our existing 63 checkpoints (the paper states
`sigma_t=0` training is equivalent to ReMDM-sampling a vanilla MDLM), so we
get to test a real remedy without training a single new step.

**Cost:** ~2.5h for experiments 1-5 on existing checkpoints, zero new
training. Highest quality-per-minute of the three proposals by a wide margin.

**Risks:** The 30% failure mode is that even with top-p, ReMDM, and
MaskGIT-confidence decoding, the 50k checkpoint is genuinely over-committed
to a low-entropy distribution (H2/H3 of the training proposal). In that case
sampling fixes close maybe half the gap and we need eval-aligned retraining.
This proposal does include a conditional Exp-6 fine-tune path, but the
"definitive" answer still requires the training proposal's Tier B.

**Blind spot:** Does not address the bf16 Gumbel precision issue, which is
a well-documented MDLM-specific bug (Zheng et al. arXiv 2409.02908). That
fix comes from `eval_metrics.md` and *must* be folded in before Exp 1.

### #2 - `eval_metrics.md` (highest leverage over future work)

**Why second:** This is the only proposal that produces a durable fix to
the meta-problem. Every future training run we do is going to need an
evaluation stack that is robust to the val/quality divergence; tier-0 (per-
bucket val loss, flat-weighted val loss, prior-KL against unigram) is
essentially free and catches mode-collapse-in-training before we commit
compute to a 50k run. The save-best gating rule (`ok_health = H_1 >= 5.2
AND distinct_4 >= 0.6`) is exactly the protection against the "longer
training = worse samples" scenario that bit us.

Critically, this proposal *also* identifies the bf16 Gumbel bug in
`model.py:643`, which is a two-line fix that is a prerequisite to
trusting any of the recovery_sampling.md experiments. That fix alone
could resolve 30-50% of the observed collapse.

**Cost:** Tier-0 metrics ~30 LOC, essentially free overhead. Tier-1
generation metrics ~80 LOC, ~2% training overhead. Tier-2 (GPT-2-small
gen-ppl) ~50 LOC, ~5% overhead. One-time precompute of FineWeb-Edu unigram
empirical distribution.

**Risks:** Gen-ppl under a reference LM has its own pathologies (it
rewards a model that matches GPT-2's own biases; FineWeb-Edu is not
WebText). The H_1 guardrail mitigates this but doesn't eliminate it.
Must be treated as a proxy, not a ceiling.

### #3 - `training_regime.md` (highest ceiling, but blocked until 1 and 2 are done)

**Why third:** The GIDD hypothesis (H1) is the single most plausible
*deep* explanation for the pathology, and the evidence from von Rutte
et al. (gen-PPL 904 -> 387 with 10% uniform noise and slight val
increase) is a textbook match for our symptom. If sampling fixes fail,
this is the right follow-up, and it's the only proposal that could
justify a new foundational training recipe.

But: Tier A of this proposal evaluates candidates on "mean 4-gram
repetition rate on 64 samples with temp=0.8, num_steps=128" using the
current sampler. That is precisely the measurement apparatus the other
two proposals are telling us is broken. If we run Tier A today, we will
compare configurations through a distorted lens - an A4 (GIDD) run
whose samples are genuinely better could lose to A0 (baseline) because
the baseline's particular failure mode (low-entropy commits) interacts
with our buggy top-p-less sampler in an unpredictable way. We'd
mis-pick a config and waste the Tier B budget.

**Cost:** Tier A ~6h (24 runs), Tier B ~14h (4 x 50k runs), Tier C
~12h. Not running this now does not delay it much - the prereqs from
1 and 2 are a ~3-hour investment.

**Risks:** The biggest code change in the three proposals (uniform-noise
+ SUBS rework). Worth doing carefully once the eval stack is trusted.
Until then, we would not be able to tell whether a subtle SUBS bug or a
genuine hybrid-noise improvement produced the Tier-A numbers.

---

## Concrete 6-Hour Work Plan (the combined best of all three)

### Phase 0 - Sampler correctness (30 min, zero compute cost)

A single commit that is a prerequisite for everything else. No experiments
run yet; we are simply removing known bugs so subsequent experiments are
interpretable.

1. **Fix bf16 Gumbel precision** (`model.py:643`, 2 lines): cast `q_xs` to
   fp32 for the Gumbel-max step, then cast back. This is the
   Zheng-et-al-2409.02908 fix.
2. **Fix temperature-on-mask-retention** (`model.py:638-640`): apply
   temperature to `p_x0` BEFORE mixing with `unmask_prob`, normalize, then
   build `q_xs`. Do not apply it to the full `q_xs`.
3. **Add sampler kwargs** (`top_p`, `top_k`, `order`, `remask_eta`,
   `remask_t_on`, `remask_t_off`, `cfg_w`) with defaults that preserve
   current behavior. Implementation skeleton is in recovery_sampling.md.

### Phase 1 - The 30-minute gateway experiment (see below, high-info)

### Phase 2 - Full sampler recovery sweep on existing checkpoints (2 h)

Run recovery_sampling.md Exp 1 (temp x top-p x top-k grid) and Exp 2
(ReMDM) on **both** `10L640d_50k.pt` AND `10L640d_10k.pt`. The second
checkpoint is the control: if recovery techniques lift *both* equally,
the problem is 100% sampler. If they lift 10k but 50k lags, the 50k
model has drifted into a regime samplers cannot fully rescue and we
need training-side fixes.

Exit criterion: a (ckpt, sampler_config) pair with repetition_4 <= 0.05
and distinct_4 >= 0.6. If we hit this for 50k, the training regime
proposal drops in priority significantly.

### Phase 3 - Tier-0 evaluation stack into train.py (1 h)

Add the free-cost tier-0 metrics from eval_metrics.md: per-bucket val
loss (buckets on mask-rate t), flat-weighted val loss, and prior-KL vs
FineWeb-Edu unigram. Also precompute `data/unigram_emp_fwe.pt` once.

These go into `validate()` in `train.py` at basically zero wall-clock
cost. Every future run, including the training_regime.md Tier A, gets
these for free. This is the "durable fix" payoff of proposal 2.

### Phase 4 - Re-rank existing 10k 3-seed checkpoints (1 h)

Using the Phase-2 winning sampler AND the Phase-3 tier-0 metrics, score
all 10k-step checkpoints we already have on disk (Adam vs Muon variants,
baselines, etc.). If the ranking changes materially under flat-weighted
val + H_1 + gen-ppl, we immediately learn which prior findings are
suspect (see "Suspect Findings" below).

This is the cheapest possible retroactive check on our own history.

### Phase 5 - Decision branch (1.5 h)

**If Phase 2 recovered 50k quality:** the pathology was sampler-side.
Update `sample_large.py` defaults, update CLAUDE.md, write up "MDLM
sample quality is sampler-dominated at this scale." Then run a small
Tier-A-lite from the training proposal (just A0 + A4 + A5, 6 runs = 1.5h)
as a follow-up experiment, not a crisis.

**If Phase 2 did NOT recover 50k quality:** the pathology is at least
partly training-side. Skip to training_regime.md Tier A, but run it
with the new sampler and new metrics. The 6-hour budget is spent;
schedule Tier A for the next day.

---

## The One Thing to Run FIRST (highest information per minute, next 30 min)

**Action:** implement Phase-0 sampler fixes (bf16 Gumbel + temperature on
`p_x0` only + optional top-p=0.9 kwarg), then regenerate 64 samples from
`10L640d_50k.pt` with `T=1.0, top_p=0.9, num_steps=128`. Compute
repetition_4 and distinct_4 against baseline samples.

**Why this is the highest-info experiment we can run:** Three of the
four leading hypotheses (bf16 Gumbel, temperature stitching, no
nucleus) collapse to a single test. Any of:

- quality recovers -> we've explained the divergence cheaply; most of
  CLAUDE.md's rankings are probably salvageable; GIDD retraining is
  probably unnecessary.
- quality partially recovers -> sampler + training both contribute;
  proceed with the full 6-hour plan.
- quality does not recover -> the 50k model has genuinely drifted;
  training-regime proposal leaps to #1 in priority.

Each outcome drives a different next-three-days plan. No other single
experiment in any of the three proposals has this branching power per
30-minute investment.

**Concrete shell:** after the two-line bf16 fix and the one-line
`p_x0 ** (1/T)` fix plus ~20 lines of top-p, run:

```bash
python sample_large.py --ckpt checkpoints/10L640d_50k.pt \
  --num_samples 64 --num_steps 128 --temperature 1.0 \
  --top_p 0.9 --out samples/phase0_smoke.json
python -c "from sample_and_categorize import repetition_4, distinct_4; ..."
```

Target: repetition_4 < 0.10 (current baseline is probably ~0.3-0.5).

---

## Suspect Findings (treat as unvalidated until re-checked)

Using the new tier-0 metrics plus the fixed sampler, the following
entries in CLAUDE.md / HANDOFF.md / README.md / MEMORY.md are plausibly
affected. Ordered by how much we'd lose if they were wrong.

1. **"Best config at 10k, val=5.07" (lr=0.01 + FineWeb-Edu + Muon-VS +
   out_proj).** Ranked entirely on ELBO val. Under flat-weighted val,
   the 0.07-nat FineWeb-Edu margin is within the range that can flip.
   **Priority: high.**

2. **"Min-SNR gamma=1.5 barely matters (~0.015 nats at 10k)".**
   Already flagged as n=1 noise. But we also never checked gamma at
   50k with sample-quality metrics; GIDD suggests gamma=1.5 is
   *directly* the knob that produces the collapse at long training.
   **Priority: high. Must re-check at >=10k with H_1 / distinct-n.**

3. **"All-Mamba wins, hybrid attention hurts (+0.06 sig)".** Attention
   is known to help long-range diversity. A 0.06-nat val loss to
   attention could easily be a net win under gen-ppl + H_1.
   **Priority: high (flagged explicitly in eval_metrics.md).**

4. **"Depth doesn't help at iso-params (8Lx320 ~= 4Lx384)".** The very
   phenomenon that 10Lx640 over-compresses toward low-entropy samples
   suggests depth + size interacts with val vs quality. Iso-params
   finding may be valid only at very short training. **Priority: medium.**

5. **"Mousse best raw loss at 2.4x wall-clock".** If Mousse at matched
   wall-clock produces lower H_1, framing is misleading. **Priority:
   medium.**

6. **"Muon beats Adam (+0.34 nats, t=40, p<0.001)".** The magnitude
   is so large that the qualitative headline almost certainly holds
   even under a corrected metric. The exact 0.34-nat number may
   shrink. **Priority: low - do not re-run, just footnote.**

7. **"SwiGLU beats GELU (+0.08)".** Small absolute advantage, could
   flip under flat weighting; low stakes. **Priority: low.**

8. **"Weight tying hurts (+0.54)".** Effect size too large to be
   metric-dependent. **Priority: very low.**

Findings that are NOT suspect: Mamba3 Triton backend works, MIMO /
Mamba2 broken on RDNA4, PyTorch/TunableOp crash observations, tok/s
measurements. These are orthogonal to val-loss.

---

## How Should We Measure Training Efficiently If val Is So Misleading?

Val loss is not useless; it is an *inlier* metric - high-variance,
cheap, and monotone-ish within a regime. The right posture is to
demote it from "north star" to "sanity tripwire" and to replace it
with a composite in two tiers. At the free tier (every val step),
log val loss broken into ELBO-weighted + flat-weighted + per-mask-rate
buckets, plus the prior-KL of the model's output distribution against
FineWeb-Edu unigram. Any run where val goes down while prior-KL goes
up is, with very high probability, mode-collapsing during training
rather than improving. At the cheap-generation tier (every
~5x val_every), generate 16 samples, report H_1 unigram entropy
(healthy is >=5.2, our "family family family" regime is far below
that), distinct-4, and repetition-4. Gate save-best on a composite
rule like `(flat_val < best_flat_val) AND (H_1 >= 5.2) AND
(distinct_4 >= 0.6)`, which removes the perverse incentive to save
checkpoints that farm ELBO by collapsing entropy. This turns val
from a misleading single number into a multi-signal vector where
divergence between signals is itself diagnostic; the bf16 Gumbel fix
and top-p=0.9 default sampler are prerequisites that make the
generated-sample signals honest. Total overhead is ~3-5% of training
wall-clock, which is a price we should have been paying all along.
