# Round 2 Proposal Evaluation

## Context: What Has Changed Since Round 1

Round 1 operated on PureSSM at ~1.3k tok/s. Now we have Mamba3 Triton at
~58k tok/s -- a 45x speedup. A 1k-step quokka run costs ~2.5 min, not
~1.75 hours. This changes feasibility calculations entirely.

More importantly, the gamma sweep (Round 1, Proposal 3 Exp 1) has been
completed and produced a surprising result: gamma=1.5 beat flat by >1 nat.
But an NS-steps probe showed the ranking REVERSED at ns=3 and ns=7. All
prior data is n=1. This inconsistency is the central problem.

A 2x2x3-seed 5k-step replication ({Adam,Muon} x {gamma=1.5,gamma=5} x 3
seeds) is ALREADY RUNNING and will land in ~2 hours. This is the most
important experiment in the project right now.

---

## Proposal Summaries

### Proposal A: Statistical Replication (round2_replication.md)

Three-phase design: (1) 6-seed variance pilot at 1k steps, (2) 5-8 seed
main paired replication of 4 configs at 1k steps, (3) 4-seed ns x gamma
interaction probe. Heavy statistical machinery: paired t-tests, BCa
bootstrap CIs, sign-flip permutation tests, Bonferroni correction. Total
budget: 40-60 runs, ~2-3 hours.

### Proposal B: Optimizer Dynamics (round2_optimizer_dynamics.md)

Four-phase deep-dive into the ns x gamma interaction: (1) 12-run 3-seed
replication of the 2x2 grid, (2) SVD instrumentation of Muon's momentum
buffer to measure gradient spectral shape, (3) offline NS convergence curve
analysis, (4) three targeted fixes (spectral-norm init, adaptive NS,
NorMuon row-norm). Total budget: 24 instrumented runs, ~75 min + 1 hr
implementation.

### Proposal C: Scale Up (round2_scaleup.md)

Four experiments: (1) scale to small 84M at 5k steps, (2) extend to 10k
steps, (3) text generation with Gen PPL and MAUVE evaluation, (4) 3-seed
reproducibility at small. Total budget: 14 training runs + 3 eval runs,
~1.75 hours.

---

## Evaluation

### 1. Overlap with the running replication

The running experiment is: {Adam, Muon} x {gamma=1.5, gamma=5} x 3 seeds,
5k steps each. That is 12 runs.

**Proposal A** duplicates the running experiment almost exactly. Phase 1
(6-seed pilot at 1k steps) is partially redundant -- the running 5k runs
will produce 1k-step checkpoints along the way. Phase 2 (main replication)
at 1k steps with more seeds addresses a different training horizon, but the
5k results are strictly more informative if we have them. Phase 3 (ns x
gamma probe) is the only non-overlapping component.

**Proposal B** Phase 1 (12-run 3-seed replication of the ns x gamma grid)
partially overlaps. The running experiment covers ns=5 only. Proposal B
explicitly varies ns={3,5}, so only the ns=5 x {gamma=1.5,gamma=5} cells
overlap. Phases 2-4 (SVD instrumentation, convergence curves, NS fixes)
are entirely novel.

**Proposal C** has zero overlap. It operates at a different scale (84M vs
35.9M), different step counts (5k-10k), and includes text generation. None
of this is covered by the running experiment.

**Verdict:** Proposal A has the most redundancy. Proposal C has the least.

### 2. Scientific value

The central question is: given n=1 inconsistency (gamma rankings reverse
with ns), what should we do?

There are two schools of thought:

(a) **Nail down variance first.** We cannot interpret anything until we
know the noise floor. This is Proposal A's thesis. It is correct in
principle. But the running 5k replication already addresses this. If the
3-seed 5k results show a consistent Muon > Adam ranking across seeds,
the headline finding is confirmed regardless of the gamma reversal at 1k
steps with different ns values. The gamma reversal may be a 1k-step
artifact that washes out by 5k.

(b) **Demonstrate practical value.** n=1 results at 35.9M with no text
output are scientifically hollow. A model that generates text, evaluated
by standard metrics, with a scaling check at 84M, is a qualitatively
different kind of evidence. This is Proposal C's thesis. It is the right
thesis IF the running replication confirms the Muon advantage.

(c) **Understand the mechanism.** The ns x gamma reversal, if real, is the
most interesting finding in the project. It would mean Muon's loss-weight
sensitivity is mediated by the NS iteration count -- a novel result with
implications for Muon on any non-standard loss. This is Proposal B's
thesis. But it is gated on Phase 1 confirming the interaction is real,
and it requires significant implementation work (SVD instrumentation, three
NS variants) for what might be noise.

**Verdict:** Proposal C has the highest expected value because it converts
a training observation into a demonstration. Proposal B has the highest
ceiling if the interaction is real, but also the highest risk. Proposal A
is mostly superseded by the running experiment.

### 3. Sequencing: what to do AFTER the 5k replication lands

The running replication will produce one of three outcomes:

**Outcome 1: Muon clearly beats Adam across seeds (gap > 0.15 nats, same
sign in all 3 seeds).** This is the best case. The right next step is
Proposal C: scale up, generate text, ground the result. The gamma reversal
at 1k steps becomes a curiosity, not a blocker.

**Outcome 2: Results are mixed (gap inconsistent across seeds, or
gamma=1.5 vs gamma=5 rankings flip).** This means variance is high even
at 5k steps. The right next step is Proposal A Phase 3 (ns x gamma
interaction probe) to determine whether the inconsistency is about ns or
about seed noise. But do it with 5 seeds, not 3, since we already know
3 seeds may not resolve it.

**Outcome 3: Adam matches or beats Muon.** The headline finding was noise.
Pivot away from optimizer research entirely. Proposal C's Experiment 3
(text generation) still has value as infrastructure for future work, but
the optimizer angle is dead.

This sequencing logic means the proposals should be ranked by their value
conditional on the most likely outcome, weighted by outcome probability.
Given the 5k-step results (muon_gamma1.5=5.52 vs adam_minsnr=5.95, a 0.43
nat gap), Outcome 1 is most likely. That favors Proposal C.

### 4. Feasibility

At 58k tok/s, all three proposals are feasible in a single session:

| Proposal | Training runs | Wall time | Implementation work |
|----------|--------------|-----------|---------------------|
| A | 40-60 | 2-3 hrs | ~30 min (analysis script) |
| B | 24 | ~75 min training | ~1 hr (SVD instrumentation + NS variants) |
| C | 14 + 3 eval | ~1.75 hrs | ~45 min (sample_and_eval.py) |

All are feasible. Proposal A is the most expensive in runs (and in wall
time, due to the large number of sequential 1k-step runs). Proposal B has
the most implementation work. Proposal C requires a new script but the
individual runs are fast.

### 5. Minimum viable next step

The project needs one of:
- (a) Confirm Muon beats Adam with significance.
- (b) Demonstrate practical value (text generation).

The running replication addresses (a). If it succeeds, (b) is the obvious
next move. Proposal C is the only proposal that addresses (b).

If the replication fails (Outcome 2 or 3), then (a) is still unresolved
and needs more data. But the right response is targeted additional seeds
at 5k steps, not the elaborate Phase 1-2-3 pipeline of Proposal A.

---

## Ranked Recommendation

### Rank 1: Scale Up (Proposal C)

**Conditional on the running replication confirming Muon > Adam.**

This is the highest-leverage use of the next few hours. The project has
spent its entire life inside a 35.9M model at 1-5k steps with no text
output. Showing that the result scales to 84M and that the model actually
generates text transforms this from a training-dynamics observation into
a meaningful result.

**What to run:**
- Experiment 1 (3 runs, 84M, 5k steps): ~18 min. This is the gating
  experiment. If the Muon gap disappears at 84M, stop.
- Experiment 2 (2 runs, 84M, 10k steps): ~24 min. Only if Exp 1 shows
  a gap.
- Experiment 3 (text generation): ~30 min. The most novel contribution.
  Even rough text quality comparison between Muon and Adam checkpoints is
  new evidence.
- Experiment 4 (3-seed validation): ~35 min. Strengthens Exp 1 but is
  lower priority than text generation. Can be deferred.

**What to cut:** Experiment 4 can wait. If time is short, run Experiments
1-3 only (~72 min).

**What to keep:** The sample_and_eval.py script. This is infrastructure
that pays dividends in every future experiment.

### Rank 2: Optimizer Dynamics (Proposal B), Phases 1+2 only

**Conditional on the running replication showing the gamma reversal
persists at 5k steps across seeds.** This is unlikely but would be the
most scientifically interesting outcome.

**What to run:**
- Phase 1 (12 runs, 3 seeds x 2 ns x 2 gamma): ~30 min. Confirms or
  refutes the ns x gamma interaction.
- Phase 2 (4 instrumented runs): ~16 min. Produces gradient spectrum data
  regardless of Phase 1 outcome.

**What to cut:** Phases 3-4. The NS convergence curve (Phase 3) is offline
analysis that can happen any time. Phase 4 (spectral-norm init, adaptive
NS, NorMuon) is implementation-heavy and only justified if Phase 1
conclusively confirms the interaction. Do not implement three NS variants
on spec.

**What to keep:** The SVD instrumentation from Phase 2. Even if the
interaction is noise, knowing the gradient spectral shape under different
gamma values is independently useful for understanding Muon on diffusion.

### Rank 3: Statistical Replication (Proposal A), Phase 3 only

**The rest of Proposal A is superseded by the running experiment.**

**What to run:** Phase 3 (ns x gamma interaction probe, 16 runs, ~40 min)
if and only if the running replication suggests the interaction matters.
Specifically: if gamma=1.5 and gamma=5 show different rankings across
seeds at 5k steps.

**What to cut:** Phase 1 (variance pilot) -- the running 5k experiment
gives us variance data at the more informative training horizon. Phase 2
(main replication at 1k steps) -- the 5k results dominate.

**What to keep:** The statistical framework (paired t-test + BCa bootstrap
+ permutation test). Apply this analysis to the running replication's
data when it lands, regardless of which proposal runs next.

---

## Concrete Plan: After the 5k Replication Arrives

### Step 0: Analyze the replication (~15 min)

When the 12-run replication ({Adam,Muon} x {gamma=1.5,gamma=5} x 3 seeds)
finishes:

1. Compute per-cell means and standard deviations.
2. Compute paired deltas: muon - adam, paired by seed, for each gamma.
3. Run paired t-tests on the deltas.
4. Determine which outcome we are in (1, 2, or 3 from above).

### If Outcome 1 (Muon wins clearly): Run Proposal C

Hour 0-0.5: Implement sample_and_eval.py.
Hour 0.5-0.8: Run Experiment 1 (3 runs at small 84M, 5k steps).
Decision gate: if Muon gap >= 0.1 nats at 84M, continue.
Hour 0.8-1.2: Run Experiment 2 (2 runs at small 84M, 10k steps).
Hour 1.2-1.7: Run Experiment 3 (text generation + eval).
Hour 1.7-2.0: Write up results.

Total: ~2 hours. Produces: scaling evidence + text samples + quantitative
generation quality comparison. This is a complete story.

### If Outcome 2 (mixed results): Targeted replication

Hour 0-0.5: Run 2 additional seeds of the most ambiguous cell (likely
gamma=1.5) at 5k steps.
Hour 0.5-1.0: If the gamma reversal appears in the 5k data, run Proposal
B Phase 1 (ns x gamma interaction probe, 12 runs at 1k steps).
Hour 1.0-1.5: Run Proposal B Phase 2 (4 instrumented SVD runs) regardless
of Phase 1 outcome -- the spectral data is useful.
Hour 1.5-2.0: Analyze and decide whether to pursue NS fixes.

Total: ~2 hours. Produces: stronger variance estimates + mechanism data.

### If Outcome 3 (Muon does not win): Pivot

Hour 0-0.3: Confirm with 2 more seeds that Adam really matches Muon.
Hour 0.3-1.0: Run Proposal C Experiment 3 (text generation) using the
best Adam checkpoint. The generation infrastructure is valuable regardless
of which optimizer wins.
Hour 1.0-2.0: Begin architecture experiments (hybrid attention scout at
84M). The optimizer question is settled; move to architecture.

Total: ~2 hours. Produces: generation infrastructure + architecture scout.

---

## Summary Table

| Proposal | Rank | Primary value | Overlap with running exp | When to run |
|----------|------|---------------|--------------------------|-------------|
| C (Scale Up) | 1 | Practical demonstration | None | After replication, if Muon wins |
| B (Dynamics) | 2 | Mechanistic understanding | Partial (ns=5 cells) | If gamma reversal persists |
| A (Replication) | 3 | Statistical rigor | Heavy (Phases 1-2 redundant) | Phase 3 only, if interaction matters |

The running 5k replication is the right gating experiment. Do not duplicate
it. Wait for its results, then execute the conditional plan above. The most
likely path (Outcome 1) leads to Proposal C, which transforms the project
from "interesting optimizer observation" to "working diffusion LM with
scaling evidence."
