# Round 2: Statistical Replication of Optimizer x Loss-Weight Rankings

## Title

Paired-seed replication study: Muon vs Adam x {flat, minsnr} with variance
estimation and significance testing

## Hypothesis

The n=1 rankings from Round 1 (muon_flat > muon_minsnr > adam_minsnr) are
unreliable due to high run-to-run variance. We hypothesize that:

1. Paired-seed evaluation (same init + data order, different optimizer/loss)
   will reduce inter-run variance by 50-80% compared to unpaired designs,
   making the true effect sizes visible with fewer seeds.
2. The true Muon vs Adam gap is 0.1-0.4 nats (from the 5k-step run:
   muon_flat=5.62, adam_minsnr=5.95), detectable at k=5 paired seeds.
3. The gamma reversal (gamma=1.5 won at ns=5 but lost at ns=3,7) is
   primarily seed variance, not a genuine ns x gamma interaction.

## Background: Why This Matters

All prior results are n=1. The alarming inconsistency:

| Config          | Context A (gamma sweep, ns=5) | Context B (ns probe) |
|-----------------|-------------------------------|----------------------|
| gamma=1.5       | 6.39 (best)                   | 7.48-7.57 (worst)   |
| gamma=5         | 7.26 (worst)                  | 6.41-6.46 (best)    |

A >1 nat reversal across nominally identical configs means either (a) the
effect is real and ns x gamma interact strongly, or (b) single-seed variance
at 1000 steps is on the order of 1+ nat. Either way, we cannot rank configs
without replication.

## Literature Support

**Paired Seed Evaluation (Sharma et al., Dec 2025, arXiv:2512.24145):**
Formalizes the common-random-numbers principle for ML. Training baseline and
variant under identical seeds induces positive correlation (r=0.68-0.99 in
their experiments), yielding strict variance reduction on paired deltas.
This is the key insight: we do not need to estimate absolute val_loss
variance -- we only need the *paired delta* variance to be small.

**"When +1% Is Not Enough" (arXiv:2511.19794, Nov 2025):**
Proposes a conservative paired bootstrap protocol: BCa confidence intervals +
sign-flip permutation test, requiring both to pass for significance. At k=3
seeds, the protocol never declares significance for 0.5-2pp gains. At k>=5,
it can detect moderate effects. Key recommendation: always pair by seed, use
bootstrap + permutation, under-claim when in doubt.

**"How Many Random Seeds?" (Colas et al., 2018, arXiv:1806.08295):**
Power analysis for deep RL. Recommends: (1) pilot study with n>=20 to
estimate sigma, (2) power analysis to compute N, (3) run N+buffer seeds.
For effect sizes of ~1 sigma, n=5-10 suffices at alpha=0.05, power=0.8.
For ~0.5 sigma, need n=15-20 unpaired -- but pairing reduces this.

**MDLM (Sahoo et al., NeurIPS 2024):**
Reports standard deviations over 5 seeds for evaluation. The SUBS
parameterization provides Rao-Blackwellized variance reduction in the
ELBO objective itself, so MDLM should have lower per-run variance than
older discrete diffusion models. This works in our favor.

## Method

### Phase 1: Variance Pilot (12 runs, ~30 min)

Estimate raw and paired-delta variance to calibrate the main study.

**Design:** 6 seeds x 2 configs, 1000 steps each.

| Seed | muon_flat | adam_minsnr |
|------|-----------|-------------|
| 42   | run       | run         |
| 137  | run       | run         |
| 256  | run       | run         |
| 512  | run       | run         |
| 777  | run       | run         |
| 1337 | run       | run         |

Each run: quokka config, bs=8, 1000 steps, cosine LR, no_time_cond.
Estimated time: ~2.5 min/run x 12 = 30 min.

**Seeding protocol:** Before each run, set:
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `np.random.seed(seed)`
- Deterministic data loader offset: `loader_offset = seed % num_tokens`

This ensures identical weight init and data order within each seed pair.

**Measurements:**
- Final val_loss (ELBO-weighted, 10 val batches -- as current code does)
- Val_loss at steps {250, 500, 750, 1000} for trajectory analysis
- Compute: mean, std, range across 6 seeds (unpaired)
- Compute: mean_delta, std_delta across 6 paired deltas
- Compute: paired correlation r (expect r > 0.7 if pairing helps)

**Decision gate:** If std_delta < 0.3 * std_unpaired, pairing works and
we proceed with k=5 paired seeds for the main study (adequate power for
~1 sigma paired effects). If std_delta is still large, increase to k=8.

### Phase 2: Main Replication (20-32 runs, ~50-80 min)

Full paired comparison of the 4 configs that matter most:

| Config       | Rationale                                       |
|--------------|-------------------------------------------------|
| muon_flat    | Round 1 winner (1k steps), strong at 5k         |
| muon_minsnr5 | Default gamma, Muon + current best loss weight  |
| adam_minsnr5  | Round 1 Adam winner, strong at 5k              |
| adam_flat    | Control: isolate optimizer effect from loss wt   |

**Design:** k seeds (5-8, calibrated from Phase 1) x 4 configs.
All configs share the same seed for each row (paired design).
Use 1000 steps (the quokka sweet spot for throughput/signal).

**Statistical analysis (per comparison pair):**

1. **Paired deltas:** d_i = val_loss(config_A, seed_i) - val_loss(config_B, seed_i)
2. **Paired t-test:** t = mean(d) / (std(d) / sqrt(k)), df = k-1
3. **BCa bootstrap CI:** 10000 bootstrap resamples of d, bias-corrected percentile CI
4. **Sign-flip permutation test:** 2^k sign permutations, exact p-value
5. **Significance criterion (conservative):** BCa 95% CI excludes 0 AND permutation p < 0.05
6. **Effect size:** report Cohen's d_z = mean(d) / std(d) with CI

**Primary comparisons (Bonferroni-corrected for 3 tests, alpha=0.017):**
- muon_flat vs adam_minsnr5 (the headline result)
- muon_flat vs muon_minsnr5 (loss weight effect within Muon)
- muon_flat vs adam_flat (optimizer effect within flat weighting)

### Phase 3: NS-Steps x Gamma Interaction Probe (16 runs, ~40 min)

Directly test whether the gamma reversal is real or noise.

**Design:** 4 seeds x 2 gamma x 2 ns_steps

| Seed | gamma=1.5, ns=5 | gamma=5, ns=5 | gamma=1.5, ns=3 | gamma=5, ns=3 |
|------|-----------------|---------------|-----------------|---------------|
| 42   | run             | run           | run             | run           |
| 137  | run             | run           | run             | run           |
| 256  | run             | run           | run             | run           |
| 512  | run             | run           | run             | run           |

All muon, minsnr weighting, 1000 steps.

**Analysis:** 2x2 repeated-measures ANOVA on val_loss:
- Main effect of gamma (1.5 vs 5)
- Main effect of ns_steps (3 vs 5)
- Interaction: gamma x ns_steps
- If interaction p > 0.1, conclude the reversal was seed noise.

## Implementation Changes Required

### 1. Add --seed argument to train.py

```python
# In parse_args():
p.add_argument("--seed", type=int, default=None,
               help="Random seed for reproducibility (None = no seeding)")

# At top of train():
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # Note: not setting torch.use_deterministic_algorithms(True)
    # because Mamba3 Triton kernels may not have deterministic paths.
    # Seed control is sufficient for variance reduction via pairing.
```

### 2. Add --mode replicate to autoresearch.py

New mode that runs the paired-seed protocol:
- Takes a list of configs and seeds
- Runs all (config, seed) pairs
- Computes paired statistics automatically
- Outputs a summary table with CIs and p-values

### 3. Analysis script (analyze_replication.py)

Standalone script that reads results/*.json and computes:
- Paired t-tests with Bonferroni correction
- BCa bootstrap confidence intervals (scipy.stats.bootstrap)
- Sign-flip exact permutation p-values
- Summary table and recommendation

## Expected Outcomes

**Optimistic (paired std < 0.15 nats):**
- 5 seeds sufficient for all primary comparisons
- Clear ranking with p < 0.01 for muon_flat vs adam_minsnr
- Total budget: ~45 runs, ~2 hours

**Realistic (paired std ~ 0.2-0.3 nats):**
- 5 seeds detects ~0.3 nat effects (the Muon vs Adam gap)
- May not resolve muon_flat vs muon_minsnr (smaller effect)
- Could need k=8 for the fine-grained comparisons
- Total budget: ~55-65 runs, ~3 hours

**Pessimistic (paired std > 0.4 nats):**
- Pairing provides less reduction than expected
- Need k=10+ seeds, or longer runs (2000 steps) to reduce noise
- Focus on the single most important comparison (muon vs adam)
- Total budget: may need a full day

## Risk and Cost Analysis

**Compute cost:**
- Phase 1: 12 runs x 2.5 min = 30 min (fixed, non-negotiable)
- Phase 2: 20-32 runs x 2.5 min = 50-80 min
- Phase 3: 16 runs x 2.5 min = 40 min
- Total: 2-3 hours (fits easily in a single session)

**Risks:**
1. **Non-determinism despite seeding:** Triton kernels on ROCm may not be
   fully deterministic even with fixed seeds. Mitigation: Phase 1 includes
   a determinism check (run same seed twice, compare val_loss).
2. **VRAM OOM on repeated in-process runs:** GPU memory may leak across
   autoresearch.py runs. Mitigation: force gc.collect() + cuda.empty_cache()
   between runs (already in autoresearch.py), or use subprocess isolation.
3. **1000 steps too short:** Effect sizes may be smaller (harder to detect)
   at 1000 steps than at 5000. Mitigation: Phase 1 captures trajectories at
   250/500/750/1000 to check if signal strengthens with training.
4. **Bonferroni too conservative for k=3 comparisons:** At alpha=0.017 with
   k=5, we can only detect large effects. Mitigation: also report uncorrected
   p-values and effect sizes; the reader can apply their own threshold.

## Success Criteria

The experiment succeeds if:
1. Phase 1 produces a finite variance estimate (not all runs fail/OOM)
2. At least one primary comparison achieves significance at alpha=0.05 (uncorrected)
3. We can state with confidence: "muon_flat is [better/worse/indistinguishable from]
   adam_minsnr at 1000 steps, with effect size d = X +/- Y nats (p = Z)"

Even a null result (no significant differences) is valuable: it means the
optimizer choice does not matter much at this scale, and we should focus
engineering effort elsewhere (architecture, data, longer training).

## Priority and Sequencing

This experiment should run BEFORE any further hyperparameter exploration.
Every subsequent result is uninterpretable without variance estimates.

Sequence:
1. Implement --seed in train.py (5 min)
2. Run Phase 1 pilot (30 min)
3. Analyze Phase 1, decide k for Phase 2 (5 min)
4. Run Phase 2 main replication (50-80 min)
5. Run Phase 3 interaction probe (40 min, can run in parallel if time allows)
6. Write up final analysis with recommendations
