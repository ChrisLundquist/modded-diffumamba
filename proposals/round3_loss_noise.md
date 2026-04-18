# Round 3: Loss Weighting and Noise Schedule Optimization for Muon-VS

## Title

**Exploiting Muon-VS Variance Normalization to Unlock Aggressive Loss
Weighting and Cosine Noise Schedules for Masked Diffusion LMs**

## Motivation

Our current best config (Muon-VS + out_proj, minsnr gamma=1.5, log-linear
noise, 10k steps) achieves val_loss = 5.27 +/- 0.02. The gamma sweep at 5k
steps showed gamma barely matters (1.5 vs 5 is ~0.025 nats). But that sweep
used base Muon, not Muon-VS. The key insight motivating this round:

**Muon-VS applies per-element variance normalization to the momentum buffer
BEFORE Newton-Schulz orthogonalization** (line 432 in train.py:
`update = M_tilde / (Gamma_hat.clamp(min=0).sqrt() + eps)`). This is
structurally analogous to Adam's second-moment normalization: it decouples
the gradient's magnitude distribution from the spectral structure fed to NS.

This means Muon-VS may be fundamentally more tolerant of aggressive loss
weightings (ELBO's 1/t, large-gamma Min-SNR) that produce high gradient
variance across timesteps. Base Muon feeds raw gradient magnitudes into NS,
so timestep-dependent scale variance corrupts the orthogonalization. Muon-VS
normalizes this away first.

Separately, recent theoretical work (Zhang & Syed, arXiv 2508.04884) proves
the cosine noise schedule is Fisher-Rao optimal for masked discrete diffusion
inference. While MDLM's ELBO is theoretically invariant to schedule functional
form (the integral telescopes), the training objective depends on the schedule
through the sampling distribution of timesteps and the per-timestep gradient
magnitudes. Different schedules concentrate training signal at different noise
levels, interacting with optimizer dynamics.

## Hypotheses

**H1 (Muon-VS x Loss Weight):** Muon-VS's variance normalization makes it
robust to loss weighting choice. Specifically, ELBO (1/t) and higher-gamma
Min-SNR (gamma=5) will perform comparably to gamma=1.5 under Muon-VS, unlike
under base Muon where gamma=1.5 was marginally preferred.

Rationale: Adam is the "variance-adaptive sign update" (Balles & Hennig 2018).
Muon-VS is the "variance-adaptive orthogonal update" (arXiv 2601.14603). Both
normalize per-element gradient variance before computing direction. The
Variance-Adaptive Muon paper shows this normalization accelerates convergence
by 1.36x on LLM pretraining. For diffusion, loss weighting primarily affects
gradient magnitude variance across timesteps -- exactly what VS normalizes.

**H2 (Cosine Noise Schedule):** A cosine masking schedule (move_chance(t) =
1 - cos(pi*t/2)^2, or equivalently alpha(t) = cos(pi*t/2)^2) will improve
training loss under Muon-VS by concentrating more training signal at
intermediate noise levels where the model's predictions matter most.

Rationale: The log-linear schedule (alpha(t) = 1-t) spends equal probability
at all noise levels. The cosine schedule spends more time at intermediate
levels (the derivative is largest near t=0.5), which are where most tokens
transition between masked/unmasked. The Fisher-Rao optimality result (arXiv
2508.04884) shows this is information-theoretically optimal for inference;
it may also improve training by concentrating gradient signal where
denoising is hardest.

**H3 (Noise Eps):** The noise floor eps=1e-3 may not be optimal. At t near
0, the model sees nearly clean inputs and must predict trivially. At t near
1, nearly everything is masked and prediction is impossible. The eps value
controls how much time we spend in both regimes.

**H4 (MaskGIT/Uniform Weighting):** Following the Quokka finding that
MaskGIT loss (uniform weighting, no 1/t factor) "converges faster initially"
but ELBO achieves better final performance -- we can test whether MaskGIT
loss is competitive at our 10k-step horizon where early convergence matters.

## Method

All experiments: quokka config, batch_size=8, 10k steps, val_every=500,
warmup=400, cosine LR, Muon-VS (lr=0.02) + AdamW (lr=3e-4), --muon_out_proj.
Seeds: {42, 137, 2024}. Eval metric: ELBO val loss (always).

**Budget: 15 runs x ~24 min each = ~6 hours. Gated: Phase 1 first (9 runs,
~3.6 hours), then Phase 2 only if Phase 1 shows signal (6 runs, ~2.4 hours).**

### Phase 1: Muon-VS x Loss Weighting (9 runs, ~3.6 hours)

Test 3 loss weightings x 3 seeds under Muon-VS:

| Run   | loss_weight | minsnr_gamma | Notes                    |
|-------|-------------|-------------|--------------------------|
| A1-A3 | minsnr      | 1.5         | Current best (control)   |
| B1-B3 | minsnr      | 5.0         | Higher gamma (less clip)  |
| C1-C3 | elbo        | --          | Unclipped 1/t weighting  |

Config A is the existing best (vs_outproj from sweep_best_10k.py, mean 5.27).
We can reuse those 3 results if seed/config match exactly. If so, only 6 new
runs needed (~2.4 hours).

**Exact CLI for new runs (example B1):**
```bash
python train.py --config quokka --batch_size 8 \
  --max_steps 10000 --val_every 500 --log_every 500 \
  --warmup_steps 400 --lr_schedule cosine \
  --optimizer muon --muon_variant vs \
  --muon_lr 0.02 --adam_lr 3e-4 --muon_out_proj \
  --loss_weight minsnr --minsnr_gamma 5.0 \
  --seed 42 --save_best
```

**For C (ELBO), replace:** `--loss_weight elbo`

**Analysis:** Paired t-test (3 seeds) for each condition vs A. Report mean
+/- std and paired delta with significance. If |delta| < 0.03 nats for all
conditions, loss weighting is confirmed irrelevant under Muon-VS (supports H1).
If any condition beats A by > 0.05 nats (p < 0.05), adopt it.

### Phase 2: Noise Schedule and Eps (6 runs, ~2.4 hours)

Conditional on Phase 1 best weighting. Test cosine noise schedule and
different eps values.

**Requires code changes to model.py:**

```python
class CosineNoise:
    """Cosine noise schedule for masked diffusion.

    alpha(t) = cos(pi*t/2)^2, so move_chance(t) = 1 - cos(pi*t/2)^2 = sin(pi*t/2)^2
    sigma(t) = -log(alpha(t)) = -2*log(cos(pi*t/2))
    dsigma(t) = pi * tan(pi*t/2)
    """
    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = t.clamp(max=1 - self.eps)
        return -2 * torch.log(torch.cos(math.pi * t_clamped / 2).clamp(min=1e-8))

    def dsigma(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = t.clamp(max=1 - self.eps)
        return math.pi * torch.tan(math.pi * t_clamped / 2).clamp(max=1e4)

    def move_chance(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * t / 2) ** 2

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sigma(t), self.dsigma(t)
```

Add `--noise_schedule {loglinear, cosine}` CLI arg. Add `--noise_eps` CLI arg
(currently hardcoded in DiffuMamba3Config).

| Run   | noise_schedule | noise_eps | loss_weight (from Phase 1) |
|-------|---------------|-----------|---------------------------|
| D1-D3 | cosine        | 1e-3      | Phase 1 best               |
| E1-E3 | loglinear     | 1e-4      | Phase 1 best               |

**Rationale for E (eps=1e-4):** Smaller eps means training on slightly
cleaner inputs (t closer to 0) and slightly noisier inputs (t closer to 1).
The MDLM paper uses 1e-3 but does not tune it. Since ELBO is theoretically
schedule-invariant, the effect is entirely through training dynamics.

**Analysis:** Same paired t-test framework vs Phase 1 best. The cosine
schedule changes the distribution of training signal but NOT the ELBO eval
metric, so comparison is fair.

## What We Are NOT Testing (and Why)

1. **Velocity parameterization:** Requires rewriting the entire SUBS
   parameterization and loss computation. High implementation cost, unclear
   benefit for masked (discrete) diffusion where SUBS is standard.

2. **Loss-free importance sampling of timesteps:** Recent work (Elastic-MDM,
   LoMDM) explores learnable timestep sampling, but these require online
   tracking of per-timestep loss variance and add significant complexity.
   The payoff is uncertain at our 10k-step horizon.

3. **Bimodal/frequency-informed weighting:** The BabyLM workshop paper
   (Masked Diffusion LMs with Frequency-Informed Training) showed
   frequency-based token weighting helps, but this is orthogonal to
   timestep weighting and adds per-token complexity.

4. **MaskGIT loss:** Quokka found MaskGIT converges faster but ELBO
   wins at scale. Our 10k-step horizon is "intermediate" -- MaskGIT
   might win here, but testing it requires adding a `--loss_weight maskgit`
   option (weight = dsigma, no 1/expm1). If Phase 1 shows loss weighting
   matters under Muon-VS, this becomes the natural Phase 3 candidate.
   Keeping it out of Phase 1 to stay within budget.

## Expected Outcomes

### Phase 1 Predictions

**Most likely (60%):** All three loss weightings perform within ~0.03 nats of
each other under Muon-VS (confirms H1). This is because Muon-VS's variance
normalization effectively absorbs the per-timestep gradient scale differences
that distinguished loss weightings under base Muon. The practical implication:
loss weighting becomes a free hyperparameter under Muon-VS, and we can use
ELBO (theoretically principled) without worrying about optimizer interaction.

**Second most likely (25%):** ELBO (1/t) underperforms by 0.05-0.10 nats
because its extreme upweighting of low-noise timesteps (t near 0) produces
very high per-element gradient variance that even Muon-VS cannot fully
normalize. This would mean Min-SNR clipping is still needed, just less
aggressively (gamma=5 might match gamma=1.5).

**Unlikely but interesting (15%):** ELBO or gamma=5 BEATS gamma=1.5 by
> 0.05 nats. This would mean base Muon was actively harmed by the ELBO
weighting and Muon-VS rescues it. The practical win: better loss without
needing the gamma hyperparameter at all.

### Phase 2 Predictions

**Cosine schedule (D):** Expected to be neutral-to-slightly-positive
(< 0.02 nats). Theory says the ELBO is schedule-invariant, so any
difference comes from training dynamics (gradient variance, timestep
sampling distribution). The cosine schedule concentrates more samples
at intermediate noise levels, which may help the model learn the
hardest denoising steps faster.

**Smaller eps (E):** Expected to be neutral. The difference between
eps=1e-3 and 1e-4 only affects the extreme ends of the noise range
(t < 0.001 and t > 0.999), which contribute very little to the loss.

## Risk Assessment

### Budget Risk: LOW

Phase 1 is 6 new runs (reusing 3 existing) at ~24 min each = ~2.4 hours.
Phase 2 is conditional and adds ~2.4 hours. Total worst case: ~4.8 hours.
Well within the ~3-hour target if we skip Phase 2, slightly over if we
run both.

### Implementation Risk: LOW

Phase 1 requires zero code changes -- only CLI argument variations.
Phase 2 requires adding CosineNoise class (~20 lines) and two CLI args.
Both are straightforward.

### Statistical Risk: MODERATE

With 3 seeds, we can detect ~0.04 nat differences (assuming std ~0.02
based on existing results: vs_outproj showed 5.29, 5.24, 5.27 across
seeds, std = 0.025). Effects smaller than 0.04 nats will be invisible.
Given that gamma=1.5 vs 5 was only 0.025 nats at 5k steps, we may see
null results. This is still informative: confirming that loss weighting
does not matter under Muon-VS simplifies the hyperparameter space.

### Generalization Risk: LOW

If loss weighting is indeed irrelevant under Muon-VS, this finding
generalizes (the mechanism is optimizer-internal, not model-specific).
If the cosine schedule helps, the magnitude may change at larger scale.

## Decision Criteria

After Phase 1:
- **If all conditions within 0.03 nats:** Loss weighting is irrelevant
  under Muon-VS. Adopt ELBO (simplest, no gamma HP). Proceed to Phase 2.
- **If ELBO or gamma=5 wins by > 0.05 nats:** Adopt winner. Proceed to
  Phase 2 with the new best weighting.
- **If gamma=1.5 still wins by > 0.05 nats:** Muon-VS does NOT decouple
  optimizer from weighting. Skip Phase 2 (noise schedule is unlikely to
  matter if the optimizer is still sensitive to gradient scale). Instead,
  investigate WHY (gradient spectrum analysis from round2 proposal).

After Phase 2:
- **If cosine schedule improves > 0.03 nats:** Adopt cosine as default
  noise schedule. Combine with Phase 1 best weighting for new best config.
- **If eps=1e-4 improves > 0.03 nats:** Adopt lower eps. This is a free
  win with no throughput cost.
- **If neither helps:** Current config (log-linear, eps=1e-3) is confirmed
  as good enough. Focus future rounds on scaling (more layers/steps) or
  architectural changes (e.g., Eso-LM hybrid AR+MDM).

## Implementation Plan

### Sweep Script: sweep_loss_noise_10k.py

Follow the pattern from sweep_best_10k.py and sweep_optim_10k.py:

```python
# Phase 1 conditions
conditions = {
    "vs_gamma1.5": common + [
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.02", "--adam_lr", "3e-4", "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "1.5",
    ],
    "vs_gamma5": common + [
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.02", "--adam_lr", "3e-4", "--muon_out_proj",
        "--loss_weight", "minsnr", "--minsnr_gamma", "5.0",
    ],
    "vs_elbo": common + [
        "--optimizer", "muon", "--muon_variant", "vs",
        "--muon_lr", "0.02", "--adam_lr", "3e-4", "--muon_out_proj",
        "--loss_weight", "elbo",
    ],
}
# Phase 2 (add after Phase 1 analysis)
# "vs_cosine": ... + ["--noise_schedule", "cosine"]
# "vs_eps1e4": ... + ["--noise_eps", "1e-4"]
```

### Code Changes for Phase 2

1. **model.py:** Add `CosineNoise` class (see Method section above).
   Add `noise_schedule: str = "loglinear"` to `DiffuMamba3Config`.
   In `DiffuMamba3.__init__`, dispatch on `config.noise_schedule`.

2. **train.py:** Add `--noise_schedule` and `--noise_eps` CLI args.
   Wire them to config before model creation:
   ```python
   if args.noise_schedule:
       config.noise_schedule = args.noise_schedule
   if args.noise_eps is not None:
       config.noise_eps = args.noise_eps
   ```

Total code changes: ~40 lines for Phase 2.

## Literature Context

- **ELBO schedule invariance (Sahoo et al. 2024, Shi et al. 2024):**
  The MDLM ELBO is invariant to noise schedule functional form -- the
  integral depends only on endpoint SNR values. This means any training
  effect of schedule choice is purely through optimization dynamics
  (gradient distribution, not objective value).

- **Cosine is Fisher-Rao optimal (Zhang & Syed, arXiv 2508.04884):**
  For inference (sampling), the cosine schedule minimizes information
  loss per step under the Fisher-Rao metric. Training benefit is
  conjectured but not proven.

- **Reweighted losses as variational bounds (Shi & Titsias, arXiv
  2511.19664):** Reweighted diffusion losses correspond to cascading
  time-dependent variational lower bounds that are tighter than the
  standard ELBO. This theoretically justifies Min-SNR-type clipping
  for both continuous and masked diffusion.

- **MDMs are secretly time-agnostic (Shi et al. 2024, ICLR 2025):**
  At infinite capacity, the optimal MDM prediction is purely a function
  of observed masks, not the continuous time variable. Weighted loss
  with arbitrary positive weights yields the same optimal solution.
  This means loss weighting affects convergence speed but not the
  optimum -- supporting our hypothesis that Muon-VS may neutralize
  the convergence speed differences.

- **Quokka (Ni et al. 2025, arXiv 2510.03280):** MaskGIT loss
  (uniform weighting) converges faster but diffusion ELBO achieves
  better final performance over 300B tokens. At our 10k-step / ~80M
  token horizon, the faster convergence of uniform weighting might
  dominate.

- **Muon-VS (arXiv 2601.14603):** Variance-scaled Muon normalizes
  each gradient element by its running variance estimate before NS
  orthogonalization, achieving 1.36x speedup on LLM pretraining.
  The normalization step is the key mechanism that may decouple
  optimizer sensitivity from loss weighting.
