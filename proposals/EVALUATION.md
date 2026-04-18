# Proposal Evaluation: DiffuMamba3 Experiment Ranking

## PureSSM Reality Check

Before evaluating individual proposals, the throughput constraint must be
front and center. Every proposal's time estimates assume Mamba3 Triton
speeds (~35-45k tok/s steady-state, or ~935k tok/s at peak). PureSSM
delivers ~1.3k tok/s -- roughly 700x slower than Triton peak, and about
27-35x slower than the steady-state throughput the proposals cite.

Wall-clock cost per step on PureSSM (quokka, bs=8, seq=1024 = 8192 tok/step):
- 8192 tok / 1300 tok/s = ~6.3 seconds per step

Wall-clock cost per step on PureSSM (small, bs=8, seq=512 = 4096 tok/step):
- 4096 tok / ~650 tok/s (estimated -- larger model, slower per token) = ~6.3 s/step
- The small config is 2.4x bigger with MIMO SSM, so throughput likely drops further.
  Conservative estimate: ~8-12 seconds per step.

This changes everything:

| Run                          | Steps | Triton est. | PureSSM est.       |
|------------------------------|-------|-------------|--------------------|
| 1x quokka, 1000 steps       | 1000  | ~8 min      | ~1.75 hours        |
| 1x quokka, 5000 steps       | 5000  | ~25 min     | ~8.75 hours        |
| 1x small, 3000 steps        | 3000  | ~30 min     | ~8-10 hours        |
| 1x quokka, 500 steps        | 500   | ~4 min      | ~53 minutes        |

A 24-hour GPU budget on PureSSM supports roughly **13-14 quokka 1000-step
runs**, or **2-3 quokka 5000-step runs**, or **2 small 3000-step runs**.
This is far fewer experiments than any proposal assumes.

---

## Proposal 1: Scaling & Convergence

**Goal:** Verify Muon-flat vs Adam-minsnr advantage persists at 5x training
budget (5000 steps) and 2.4x model scale (84M "small" config).

### Strengths

- Directly addresses the most important open question: is the 0.04 nat Muon
  advantage real or an artifact of short training and single seeds?
- Well-motivated by Muon scaling literature (Liu et al., Essential AI).
- Simple experimental design -- no code changes needed for Phase 1.
- Clear decision criteria with actionable thresholds.
- The control (adam_flat_5k) is a smart addition that disentangles the
  optimizer effect from the loss-weighting effect.

### Weaknesses

- **Severely underestimates PureSSM wall time.** Phase 1 alone (3x quokka
  5k = 15,000 steps) costs ~26 hours on PureSSM, not "~2.25 hrs." Phase 2
  (3x small 3k) costs ~24-30 hours. Total is ~50-56 hours -- more than
  double the 24-hour budget for just one proposal.
- The 84M "small" experiments are particularly dangerous: the MIMO SSM at
  d_model=512, 8 layers will be extremely slow on PureSSM, and VRAM may
  be tight (the proposal acknowledges this).
- Single-seed at 0.04 nat gap is acknowledged as inconclusive, but the
  proposal doesn't budget for multi-seed. At PureSSM speeds, even 2 seeds
  per condition is prohibitive.

### Feasibility on PureSSM (24 hours)

Cannot run as proposed. Feasible subset:
- 2x quokka 5000-step runs (muon_flat vs adam_minsnr): ~17.5 hours
- That leaves ~6.5 hours for one more run (adam_flat control or a 3000-step
  small run if it fits in VRAM).
- Phase 2 (small 84M) and Phase 3 (10k steps) are out of budget.

### Verdict

**High scientific value, low feasibility.** The question is the right one,
but PureSSM makes the proposed experiment matrix impossible. A drastically
cut version (2 runs, quokka only) is still worth doing.

---

## Proposal 2: Architecture Variants (Hybrid Attention + Soft Masking)

**Goal:** Test hybrid Mamba-attention blocks and soft masking feedback to
improve model quality.

### Strengths

- Well-grounded in recent literature (DiffuMamba-H, MaBERT, Nemotron-H).
- Hybrid attention is a genuine architectural improvement with strong
  empirical support across multiple papers.
- Soft masking is clever -- 3 parameters for a meaningful quality gain.
- Weight tying (Caduceus-style) is zero-cost to test and potentially useful.
- The modular design (drop-in BiAttentionBlock) is clean.

### Weaknesses

- **Massive experiment count.** 16 runs at 3 seeds each = up to 48 total
  runs. At ~1.75 hours per quokka 1000-step run on PureSSM, that is
  ~84 hours. The proposal estimates "~5.3 hours" -- off by 16x.
- **Implementation complexity is real.** ~300 LOC across BiAttentionBlock,
  SoftMaskFeedback, forward_with_embeddings, and config plumbing. Debugging
  on a 700x-slow backend will be painful.
- **SM doubles the forward pass during training.** On PureSSM, where each
  forward pass already takes ~6 seconds, this means ~12 seconds per step
  for SM variants. A 1000-step SM run costs ~3.5 hours.
- **The proposal uses batch_size=32.** The current validated runs use bs=8.
  At bs=32 with seq=1024, that is 32K tokens per step. On PureSSM this
  likely OOMs or runs at ~25 seconds per step. This alone invalidates the
  time estimates.
- **Confounds the optimizer question.** The proposal uses "Muon + minsnr"
  as the default, but the project's best result is Muon + flat. The memory
  file explicitly says "muon + minsnr is suboptimal." Running all these
  architecture experiments on a suboptimal optimizer config wastes compute.
- **Premature.** We do not yet know if the Muon advantage persists past
  1000 steps. Testing architecture variants before confirming the optimizer
  is putting the cart before the horse.

### Feasibility on PureSSM (24 hours)

Cannot run as proposed. Feasible subset:
- Fix batch_size to 8 (not 32).
- Drop 3-seed requirement to 1 seed.
- Run 4 variants at 1 seed each: baseline, hybrid-25, hybrid-33, tied.
  That is 4 x 1.75 hours = 7 hours.
- Skip soft masking entirely (2x forward pass cost is brutal on PureSSM).
- Skip hybrid-50 (diminishing returns at 4 layers, 50% attention).
- That leaves ~17 hours for other work.

### Verdict

**Medium scientific value, very low feasibility.** The ideas are sound but
the experiment matrix is 16x over budget. The proposal also has a batch
size error that would cause OOM. If hybrid attention is a priority, run a
single seed of hybrid-25 and hybrid-33 as cheap scouts after confirming
the optimizer question.

---

## Proposal 3: Training Dynamics (Loss Weighting + NS Tuning)

**Goal:** Find Muon-optimal loss weighting via gamma sweep, test scheduled
weighting, and probe Newton-Schulz iteration count interaction.

### Strengths

- **Directly explains the core finding.** The Muon+flat vs Muon+minsnr
  gap is the project's most interesting result. Understanding *why* via
  mechanistic experiments (NS steps, gradient scale variance) is high-value
  science.
- **Experiment 1 (gamma sweep) needs zero code changes.** All the CLI args
  already exist. This can run immediately.
- **Experiment 3 (NS steps) needs one CLI arg.** The plumbing already
  exists in the optimizer; just needs `--ns_steps` exposed. Trivial.
- **Small experiments.** All runs are quokka 1000 steps -- the cheapest
  unit of compute in this project.
- **Progressive disclosure.** Experiments are ordered by implementation
  cost, with clear stop conditions. Experiment 4 (momentum clip) is
  explicitly deferred unless earlier results warrant it.
- **Novel mechanistic insight.** No paper has studied NS iteration count
  as a function of loss weight scale variance. This could explain Muon's
  failure on image diffusion more broadly.

### Weaknesses

- **Time estimates are still wrong.** 7 runs at "~8 min each" is Triton
  speed. On PureSSM, 7 x 1.75 hours = ~12.25 hours for Experiment 1
  alone. Total for Experiments 1+3 (13 runs) = ~22.75 hours. This
  technically fits in 24 hours but leaves almost no margin.
- **Risk of confirming the obvious.** Experiment 1 may simply show flat is
  best for Muon (which we already know). The gamma sweep between 1.5-20
  is probing whether "slightly less flat" beats "completely flat," which
  is a fine-grained question that may not resolve at n=1.
- **Experiment 2 (scheduled weighting) may need >1000 steps to show an
  effect.** Annealing gamma over 1000 steps is very fast -- the schedule
  barely moves before training ends. But extending to 2000+ steps doubles
  the cost.
- **The NS steps interaction (Experiment 3) is speculative.** If the
  mechanism is momentum corruption (as hypothesized), ns_steps should
  interact. But if the issue is elsewhere (e.g., in the Adam auxiliary
  parameters, or in the embedding gradients that bypass Muon entirely),
  ns_steps will show no effect and we learn only that this path is a
  dead end.

### Feasibility on PureSSM (24 hours)

The most feasible of the three proposals, but still tight:
- Experiment 1 (7 gamma sweep runs): ~12.25 hours
- Experiment 3 (6 NS x weight runs): ~10.5 hours
- Total: ~22.75 hours -- barely fits, no room for Experiment 2 or 4.

Realistic plan: run Experiment 1 (7 runs, ~12 hours), then cherry-pick
2-3 runs from Experiment 3 based on results (~3.5-5.25 hours). Skip
Experiments 2 and 4. Total: ~15.5-17.5 hours, leaving buffer.

### Verdict

**High scientific value, moderate feasibility.** The gamma sweep is the
single highest-value experiment across all three proposals per GPU-hour.
Zero code changes, directly tests the core hypothesis, and even a null
result (flat is best) is informative. The NS steps experiment is a
worthwhile follow-up if time permits.

---

## Ranked Recommendation

### Rank 1: Training Dynamics (Proposal 3), Experiment 1 only

**Run the Muon gamma sweep first.** Seven runs, zero code changes, directly
probes the most interesting finding in the project. Even if flat wins
(which is the likely outcome), the shape of the gamma-vs-loss curve tells
us whether Muon is completely intolerant of any timestep reweighting or
whether there is a narrow sweet spot.

Modification: drop gamma=20 (too close to flat to distinguish) and add
gamma=1 (which is effectively a tighter clamp than 1.5 and should be
tested). Keep 7 runs total. Also, use `--no_time_cond` to match the
validated best config (muon_flat at 6.54 used --no_time_cond).

### Rank 2: Scaling & Convergence (Proposal 1), Phase 1 only (2 runs)

**Run muon_flat vs adam_minsnr at 5000 steps on quokka.** This is the
highest-impact binary question: does the advantage persist? Drop the
adam_flat control (it can wait). Two runs at 5000 steps costs ~17.5
hours on PureSSM, which is too expensive to run alongside the gamma
sweep in a single 24-hour window. Run it second, after the gamma sweep.

Modification: if the gamma sweep finds a better-than-flat gamma for Muon,
substitute that in place of muon_flat for the 5000-step run. This makes
the scaling test more informative.

### Rank 3: Training Dynamics (Proposal 3), Experiment 3 (NS steps)

**Run 2-3 cherry-picked NS step experiments** if the gamma sweep produces
interesting results. Specifically: ns_steps=3 + minsnr_gamma5 (tests
whether fewer NS steps rescue Muon+minsnr) and ns_steps=7 + flat (tests
whether tighter orthogonalization helps). Skip ns_steps=5 variants since
that is the current default and we already have those numbers.

### Rank 4: Architecture Variants (Proposal 2), scout runs only

**Defer to a future 24-hour block.** If hybrid attention is a priority,
run two single-seed scouts (hybrid-25 and hybrid-33) at quokka 1000
steps. Skip soft masking entirely on PureSSM -- the 2x forward pass
overhead is unacceptable. Skip hybrid-50. Fix the batch size to 8. Fix
the optimizer config to muon+flat (not muon+minsnr). This costs ~3.5
hours and gives a preliminary signal.

### Not recommended at this time

- Proposal 1, Phase 2 (small 84M): Too slow on PureSSM (~8-10 hours per
  run). Wait for Mamba3 Triton to work on RDNA4.
- Proposal 2, Soft Masking: 2x forward pass on PureSSM is disqualifying.
- Proposal 3, Experiments 2 and 4: Implementation work for uncertain
  payoff. Experiment 2 (scheduled weighting) needs >1000 steps to show
  an effect. Experiment 4 (momentum clip) is acknowledged as speculative.

---

## Suggested 24-Hour Plan

The plan below maximizes learning per GPU-hour on PureSSM at ~1.3k tok/s.
All runs use quokka config (31.5M, d=384, 4 layers, seq=1024), bs=8,
`--no_time_cond`, cosine LR, Muon lr=0.02 / Adam lr=3e-4.

### Block 1: Gamma Sweep (hours 0-12.5)

Seven sequential runs, 1000 steps each, ~1.75 hours each.

```
Run 1: muon, --loss_weight flat                        (control, replicate 6.54)
Run 2: muon, --loss_weight minsnr --minsnr_gamma 1.0
Run 3: muon, --loss_weight minsnr --minsnr_gamma 1.5
Run 4: muon, --loss_weight minsnr --minsnr_gamma 2.0
Run 5: muon, --loss_weight minsnr --minsnr_gamma 3.0
Run 6: muon, --loss_weight minsnr --minsnr_gamma 5.0   (replicate 6.97)
Run 7: muon, --loss_weight minsnr --minsnr_gamma 10.0
```

Expected learning: shape of the gamma-vs-loss curve for Muon. If
monotonically increasing (flat best), the "Muon needs uniform gradients"
hypothesis is strongly confirmed. If U-shaped with minimum at gamma 1.5-3,
there is a Muon-optimal weighting that beats flat.

### Block 2: Analyze + Decide (hour 12.5)

Review gamma sweep results. Three possible paths:

**Path A** (flat is best): The gradient-uniformity hypothesis is confirmed.
Proceed to Block 3A -- run 2 NS step experiments to probe the mechanism.

**Path B** (gamma X < flat): We found a Muon-optimal weighting. Proceed to
Block 3B -- run one 5000-step validation with the new best config.

**Path C** (results are noisy / no clear winner): Replicate the top 2
configs with a second seed each.

### Block 3A: NS Steps Probe (hours 13-19.5, if Path A)

Three runs, 1000 steps each:
```
Run 8: muon, flat, --ns_steps 3     (does rough orthogonalization help?)
Run 9: muon, flat, --ns_steps 7     (does tight orthogonalization help?)
Run 10: muon, minsnr gamma=5, --ns_steps 3  (does ns=3 rescue Muon+minsnr?)
```
Note: --ns_steps requires a one-line CLI arg addition to train.py. The
optimizer already reads ns_steps from the param group dict.

Remaining ~4.5 hours: run one architecture scout (hybrid-25, 1000 steps).

### Block 3B: Long Training Validation (hours 13-22, if Path B)

One 5000-step run of the best Muon config from the sweep:
```
Run 8: muon, --loss_weight minsnr --minsnr_gamma [best], 5000 steps
```
This costs ~8.75 hours. With the remaining ~2.5 hours, run one NS step
experiment (ns_steps=3 + best_gamma) to check mechanism.

### Block 3C: Replication (hours 13-17, if Path C)

Two replication runs of the top 2 configs from the sweep (second seed
each). Costs ~3.5 hours. Then proceed to whichever of Path A or B the
replications support.

### Summary

| Block | Duration | Runs | Key Question Answered |
|-------|----------|------|-----------------------|
| Gamma sweep | ~12.5 hrs | 7 | Is flat truly optimal for Muon, or is there a sweet spot? |
| NS steps or long training | ~7-9 hrs | 2-3 | Mechanism (A) or durability (B) of the best config |
| Scout | ~2-4 hrs | 0-1 | Is hybrid attention worth pursuing? |
| **Total** | **~22-24 hrs** | **9-11 runs** | |

---

## Modifications to Proposed Experiments

### All proposals
- Use `--no_time_cond` consistently (validated as the best wall-clock
  tradeoff per memory finding).
- Use `bs=8` (not bs=32; the latter may OOM or be catastrophically slow
  on PureSSM).
- Multiply all time estimates by 10-15x for PureSSM reality.

### Proposal 1 (Scaling)
- Cut Phase 2 (small 84M) entirely until Triton works.
- Cut adam_flat control from Phase 1 (run it later if time).
- Cut Phase 3 (10k steps) entirely.
- Consider reducing 5000 steps to 3000 if the gamma sweep consumes too
  much budget.

### Proposal 2 (Architecture)
- Fix batch_size from 32 to 8.
- Fix optimizer config from muon+minsnr to muon+flat.
- Cut soft masking (2x forward pass cost on PureSSM).
- Cut 3-seed requirement to 1 seed.
- Cut hybrid-50 (50% attention on 4 layers is not a meaningful Mamba test).
- Run as scouts only, not as a primary experiment block.

### Proposal 3 (Training Dynamics)
- Drop gamma=20 from sweep (too close to flat).
- Add gamma=1.0 (test the extreme tight-clamp regime).
- Add `--ns_steps` CLI arg to train.py (one line).
- Skip Experiments 2 (scheduled weighting) and 4 (momentum clip).
- Cherry-pick Experiment 3 runs based on Experiment 1 results rather
  than running the full grid.

---

## What to Do After This 24-Hour Block

Priority depends on results, but likely next steps:

1. **Get Mamba3 Triton working on RDNA4.** This is the single highest-
   leverage engineering task. It converts 1 GPU-day of experiments from
   ~11 runs to ~300+ runs. Everything else is marginal optimization
   while this bottleneck exists.

2. **Multi-seed validation** of whatever config wins the gamma sweep +
   long training. At least 3 seeds to get error bars.

3. **Architecture experiments** (hybrid attention) once throughput allows
   the full matrix at 3 seeds.

4. **Scale to small (84M)** once both optimizer and architecture are
   settled at quokka scale.
