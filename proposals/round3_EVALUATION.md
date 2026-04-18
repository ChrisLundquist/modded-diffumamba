# Round 3 Proposal Evaluation

## Baseline Reference

Current best: Muon-VS + out_proj, gamma=1.5, cosine schedule, warmup=400,
10k steps, batch_size=8. From `best_10k_summary.json`:

| Config     | s42    | s137   | s2024  | Mean   | Std    |
|------------|--------|--------|--------|--------|--------|
| vs_outproj | 5.2868 | 5.2396 | 5.2707 | 5.2657 | 0.024  |
| vs_no_out  | 5.3418 | 5.2882 | 5.3260 | 5.3187 | 0.028  |
| adam       | 5.7317 | 5.6739 | 5.7281 | 5.7112 | 0.033  |

Run time per 10k run: ~1380s (~23 min). Per 5k run: ~12 min.
Budget: 6 hours = 360 min. Conservative (15 runs at 24 min = 360 min).

Note: sweep_best_10k.py does NOT pass --no_time_cond, meaning the 10k
baseline has time conditioning ON. The CLAUDE.md says --no_time_cond is
preferred, but the actual validated 10k number (5.27) was with time cond
ON. Any comparison must match this. The proposals are inconsistent on this
point: round3_lr_schedule.md uses --no_time_cond, round3_data_reg.md uses
--no_time_cond, round3_loss_noise.md does not. New experiments should NOT
use --no_time_cond to match the 10k baseline, OR rerun the baseline control
arm. Since the time-cond effect is marginal (~0.013 nats), this is not
fatal, but it is a confound to track.


## Per-Experiment Ranking

Ranked by expected nats improvement per GPU-hour.

### 1. FineWeb-Edu vs FineWeb (from round3_data_reg.md, Phase 1)

**Expected improvement:** 0.05-0.15 nats
**GPU time:** 7 runs x 12 min = 84 min (3 seeds x 2 conditions + 1 Edu-val check)
**Code changes:** ZERO. --data_dir data/fineweb-edu-10B just works.
Already confirmed: Edu shards exist (10 train + 1 val, correct naming convention).
**Nats/hour:** 0.04-0.12 nats/hr
**Independence:** Fully independent; can run in parallel with anything.
**Reusable controls:** No. Our 10k baselines trained on FineWeb. New FineWeb-Edu
runs need their own FineWeb control arm (same steps, same config) to get a
paired comparison. BUT we can run at 5k steps to screen cheaply (6 runs, 72 min).

**Verdict: RUN FIRST.** Highest leverage, zero code changes, independent.
The data quality multiplies all other gains. If Edu wins, we adopt it
for all subsequent experiments and save ourselves from optimizing on
inferior data.

**Risk note:** The proposal correctly identifies that Edu might hurt raw
NLL on FineWeb val (domain shift). The B' run (Edu val) disambiguates.
Val on FineWeb val is the correct metric since that is our established
benchmark.

### 2. Muon-VS LR Screen (from round3_lr_schedule.md, Phase 1a)

**Expected improvement:** 0.05-0.25 nats (best case 0.15-0.25)
**GPU time:** 5 runs x 12 min = 60 min (screen), then 6 runs x 12 min = 72 min
(validate top LR + control)
**Code changes:** ZERO for the LR screen. Only --muon_lr changes.
**Nats/hour:** 0.05-0.25 nats/hr (wide range; screen resolves this)
**Independence:** Fully independent; can run in parallel with Edu.

**Verdict: RUN SECOND (or parallel with #1).** The strongest theoretical
argument: Muon-VS normalizes gradient variance like Adam, which is exactly
what enables higher LR in Adam vs SGD. NorMuon already demonstrated
lr=0.06 works with per-neuron normalization. VS's per-element normalization
is strictly more granular. The 0.02 LR was swept on base Muon at n=1 on
500 steps -- it was never validated for VS.

**Key concern:** The proposal budgets 132 min for Phase 1 alone (screen +
validation). This is correct: we must validate the winning LR at 3 seeds
because n=1 LR screens are noisy (see memory: "n=1 results unreliable").
Budget 132 min for the full LR investigation.

### 3. Muon-VS x Loss Weighting (from round3_loss_noise.md, Phase 1)

**Expected improvement:** -0.03 to +0.05 nats (60% chance of null result)
**GPU time:** 6 new runs x 24 min = 144 min (reusing 3 existing vs_outproj)
**Code changes:** ZERO. Only --loss_weight and --minsnr_gamma flags.
**Nats/hour:** 0.00-0.02 nats/hr
**Independence:** Fully independent.

**Verdict: RUN, but lowest priority among the three main experiments.**
The proposal is intellectually compelling (does VS make gamma irrelevant?)
but the expected value is low. The gamma sweep already showed gamma barely
matters (0.025 nats) under base Muon. VS plausibly flattens the landscape
further, making the answer "still barely matters." The 60% null probability
is honest and well-calibrated.

However, there is a hidden efficiency gain: if loss weighting is confirmed
irrelevant under VS, we can drop gamma from the hyperparameter space and
simplify future sweeps. This has operational value beyond the nat count.

**Critical issue:** The proposal runs at 10k steps (24 min each), eating
144 min for 6 new runs. At 5k steps it would be 72 min. Since we already
know gamma barely matters at 5k, running at 10k adds information only if
the effect emerges with more training. Given 60% null probability, the
extra precision is not worth the 2x cost. **Recommendation: run at 5k
steps to screen, validate at 10k only if there is signal.**

### 4. WSD Schedule (from round3_lr_schedule.md, Phase 2a)

**Expected improvement:** -0.05 to +0.05 nats (most likely equivalent)
**GPU time:** 2 runs x 12 min = 24 min
**Code changes:** ~15 lines in train.py (add WSD branch, promote
cooldown_frac to CLI arg). Low risk.
**Nats/hour:** ~0.00 nats/hr (operational benefit, not loss benefit)
**Independence:** Depends on LR screen result (uses winning LR).

**Verdict: RUN IF TIME PERMITS.** The proposal correctly notes that WSD's
value is operational (anytime stopping), not raw loss. At 5k steps, WSD
has only 1k-4k steps at peak LR before decay begins -- it may look worse
than cosine simply because cosine's smooth decay is better suited to short
training. WSD's advantage grows with training length.

**Recommendation:** Only worth running if we plan to extend to 50k+ steps
in a future round. At 5k-10k steps, cosine is fine. Skip for now unless
we finish the higher-priority experiments early.

### 5. Weight Decay Sweep (from round3_data_reg.md, Phase 2)

**Expected improvement:** 0.00-0.05 nats (most likely confirms current WD)
**GPU time:** 9 runs x 12 min = 108 min (5k steps)
**Code changes:** ZERO. Only --muon_wd flag.
**Nats/hour:** 0.00-0.03 nats/hr
**Independence:** Independent.

**Verdict: INTERESTING THEORETICALLY, LOW PRIORITY PRACTICALLY.** The Muon
spectral norm theory (arXiv 2506.15054) makes a specific testable
prediction: WD=0 should hurt due to unconstrained spectral growth. This is
a genuine scientific question. But the expected loss improvement is near
zero (the current WD=0.01 is likely already fine). The main practical
outcome is confirming the default.

**Recommendation:** Run 3 conditions (WD=0, 0.01, 0.1) x 1 seed as a quick
screen (36 min). If WD=0 diverges or shows > 0.1 nat penalty, that is an
interesting finding worth reporting. If all three are within 0.03 nats,
WD does not matter at sub-epoch and we move on.

### 6. Batch Size Sweep (from round3_data_reg.md, Phase 3)

**Expected improvement:** Unclear, highly confounded
**GPU time:** 6 runs, variable time (36 min)
**Code changes:** ZERO.
**Independence:** Independent.

**Verdict: SKIP.** The proposal acknowledges the design is confounded:
different batch sizes at fixed steps see different token counts. The
interesting comparison (fixed tokens, variable BS) requires 3.5 hours by
itself. The budget-cut version (5k steps, 2 seeds) is underpowered.

More importantly, batch_size is constrained by VRAM. bs=16 with quokka at
seq_len=1024 may OOM on 16GB. bs=4 is viable but doubles wall-clock time
per token (more optimizer steps per token). The question "is bs=4 better
per-token?" is interesting for a scaling paper but not for squeezing loss
in a 6-hour GPU window.

### 7. Noise Schedule/Eps (from round3_loss_noise.md, Phase 2)

**Expected improvement:** 0.00-0.02 nats
**GPU time:** 6 runs x 24 min = 144 min
**Code changes:** ~20-40 lines (CosineNoise class, CLI args).
**Independence:** Depends on Phase 1 loss weighting result.

**Verdict: SKIP in this round.** The ELBO is theoretically
schedule-invariant. Any effect is through training dynamics, which the
proposal estimates at < 0.02 nats. The cosine schedule's Fisher-Rao
optimality is for inference, not training. The eps change (1e-3 vs 1e-4)
affects < 0.1% of the timestep range.

At 144 min, this is the most expensive experiment relative to its expected
payoff. The code changes are modest but add unnecessary risk for a likely
null result. If we ever move to longer training (where schedule choice
matters more for timestep sampling), revisit then.

### 8. Warmup Length / Dropout / Grad Clip (lowest priority items)

- **Warmup sweep:** 36 min for ~0.01 nat expected effect. The proposal
  itself rates this LOW priority and recommends cutting it. Agree.
- **Dropout:** Expected to hurt (MDLM masking already regularizes, sub-epoch
  regime). Not worth the code change.
- **Grad clip removal:** Quick 2-run check, but if it works, the gain is
  just "one less hyperparameter," not better loss. Low priority.


## The 6-Hour Plan

### Timeline (15 runs, ~360 min)

**Hour 0:00-1:12 -- Block A: FineWeb-Edu Screen (5k steps)**

Run FineWeb-Edu vs FineWeb at 5k steps, 3 seeds. Zero code changes.

| Run | Data       | Seed | Steps | Time  | Notes          |
|-----|------------|------|-------|-------|----------------|
| A1  | FineWeb    | 42   | 5000  | 12min | Control        |
| A2  | FineWeb    | 137  | 5000  | 12min | Control        |
| A3  | FineWeb    | 2024 | 5000  | 12min | Control        |
| A4  | FineWeb-Edu| 42   | 5000  | 12min | Treatment      |
| A5  | FineWeb-Edu| 137  | 5000  | 12min | Treatment      |
| A6  | FineWeb-Edu| 2024 | 5000  | 12min | Treatment      |

Both validated on FineWeb val (--val_data_path data/fineweb10B for Edu runs).
Total: 6 runs, ~72 min.

**Decision gate A:** If Edu wins by > 0.03 nats (paired), adopt for all
subsequent runs. If Edu loses or tie, keep FineWeb.

**Hour 1:12-2:12 -- Block B: Muon-VS LR Screen (5k steps, seed=42)**

Screen 5 LRs at 1 seed. Zero code changes.

| Run | muon_lr | Seed | Steps | Time  |
|-----|---------|------|-------|-------|
| B1  | 0.01    | 42   | 5000  | 12min |
| B2  | 0.02    | 42   | 5000  | 12min | (reuse A1 if same config) |
| B3  | 0.04    | 42   | 5000  | 12min |
| B4  | 0.06    | 42   | 5000  | 12min |
| B5  | 0.08    | 42   | 5000  | 12min |

Data: winner of Block A.
Total: 4-5 runs, ~48-60 min (B2 may be reusable from A1 if data is FineWeb).

**Decision gate B:** Identify top LR. If 0.02 still wins, skip Block C.
If a higher LR wins, proceed to Block C.

**Hour 2:12-3:24 -- Block C: LR Validation (5k steps, 3 seeds)**

Validate top LR against 0.02 control. 3 seeds each.

| Run | muon_lr | Seed | Steps | Time  |
|-----|---------|------|-------|-------|
| C1  | LR*     | 42   | 5000  | 12min | (reuse B screen if same seed) |
| C2  | LR*     | 137  | 5000  | 12min |
| C3  | LR*     | 2024 | 5000  | 12min |
| C4  | 0.02    | 42   | 5000  | 12min | (reuse B2 or A1)             |
| C5  | 0.02    | 137  | 5000  | 12min | (reuse if available)         |
| C6  | 0.02    | 2024 | 5000  | 12min | (reuse if available)         |

Total: 3-6 new runs depending on reuse, ~36-72 min.

If LR=0.02 won the screen (no higher LR is better), skip Block C entirely
and proceed to Block D.

**Hour 3:24-4:48 -- Block D: Loss Weighting under VS (5k steps, 3 seeds)**

Test whether VS makes gamma irrelevant. Zero code changes.

| Run | loss_weight | gamma | Seed | Steps | Time  |
|-----|-------------|-------|------|-------|-------|
| D1  | minsnr      | 1.5   | 42   | 5000  | 12min | (reuse from Block B/C) |
| D2  | minsnr      | 1.5   | 137  | 5000  | 12min | (reuse from Block B/C) |
| D3  | minsnr      | 1.5   | 2024 | 5000  | 12min | (reuse from Block B/C) |
| D4  | minsnr      | 5.0   | 42   | 5000  | 12min |
| D5  | minsnr      | 5.0   | 137  | 5000  | 12min |
| D6  | minsnr      | 5.0   | 2024 | 5000  | 12min |
| D7  | elbo        | --    | 42   | 5000  | 12min |
| D8  | elbo        | --    | 137  | 5000  | 12min |
| D9  | elbo        | --    | 2024 | 5000  | 12min |

Control (D1-D3) reused from Block B/C. 6 new runs, ~72 min.

Uses winner LR from Blocks B/C and winner data from Block A.

**Hour 4:48-5:24 -- Block E: Weight Decay Quick Screen (5k, 1 seed)**

Quick 1-seed screen of WD=0.0, 0.01, 0.1. Zero code changes.

| Run | muon_wd | Seed | Steps | Time  |
|-----|---------|------|-------|-------|
| E1  | 0.0     | 42   | 5000  | 12min |
| E2  | 0.01    | 42   | 5000  | 12min | (reuse from earlier) |
| E3  | 0.1     | 42   | 5000  | 12min |

Total: 2 new runs, ~24 min. Validates theory; divergence at WD=0 is
interesting even at n=1.

**Hour 5:24-6:00 -- Block F: Buffer / WSD / Analysis**

Use remaining ~36 min for either:
- (a) Analysis and writeup of results from Blocks A-E.
- (b) If results are clear and time remains: WSD screen (2 runs, 24 min).
  Requires ~15 lines of code in train.py.
- (c) If LR or Edu results are borderline: additional validation runs.

### Run Accounting

| Block | New runs | Reused | Time (new) | Cumulative |
|-------|----------|--------|------------|------------|
| A     | 6        | 0      | 72 min     | 72 min     |
| B     | 4-5      | 0-1    | 48-60 min  | 132 min    |
| C     | 0-6      | 0-6    | 0-72 min   | 204 min    |
| D     | 6        | 3      | 72 min     | 276 min    |
| E     | 2        | 1      | 24 min     | 300 min    |
| F     | 0-2      | 0      | 0-24 min   | 324 min    |

**Total: 18-25 runs, 216-324 min (3.6-5.4 hours).** Well within 6 hours
even in the worst case. The variance comes from whether Block C runs in
full (LR screen found a winner) or is skipped (LR=0.02 still optimal).


## The 3-Hour Plan (If Budget is Halved)

Drop Blocks D, E, F. Keep A and B/C.

| Block | Runs  | Time     | Purpose                    |
|-------|-------|----------|----------------------------|
| A     | 6     | 72 min   | FineWeb-Edu vs FineWeb     |
| B     | 5     | 60 min   | LR screen (1 seed)         |
| C     | 3-6   | 36-72 min| LR validation (if needed)  |

Total: 14-17 runs, 168-204 min (~2.8-3.4 hours).

These are the two highest-leverage experiments. If forced to pick ONE,
pick the LR screen (Block B + C): the expected improvement is larger
(up to 0.25 nats) and has a stronger theoretical basis. FineWeb-Edu is
safer (likely ~0.05-0.10 nats) but has a ceiling.

Actually: if we truly have only 3 hours, run Blocks A and B in parallel
planning. Start Block A, and while analyzing A results, plan Block B
with the winning data source. No dead time.


## What to Cut (Summary)

| Experiment              | Verdict      | Why                                              |
|-------------------------|--------------|--------------------------------------------------|
| FineWeb-Edu             | RUN          | Zero code, highest leverage, independent         |
| VS LR screen            | RUN          | Zero code, strong theory, never tuned            |
| VS LR validation        | RUN          | Needed to trust the screen result                |
| Loss weighting x VS     | RUN (5k)     | Zero code, answers a real question, cheap at 5k  |
| WD screen               | RUN (1 seed) | Zero code, tests theory, 24 min total            |
| WSD schedule            | IF TIME      | Needs code changes, operational not loss benefit  |
| Warmup sweep            | CUT          | Proposal itself rates LOW; ~0.01 nat expected    |
| Batch size sweep        | CUT          | Confounded design, VRAM-constrained, underpowered |
| Cosine noise schedule   | CUT          | ELBO schedule-invariant, high cost, ~0.00 nats   |
| Noise eps               | CUT          | Affects < 0.1% of timestep range                 |
| Dropout                 | CUT          | Expected to hurt (sub-epoch, MDLM masks)         |
| Grad clip removal       | CUT          | No loss benefit, just parameter simplification    |


## Cross-Proposal Notes

### Confound: --no_time_cond

The proposals use --no_time_cond but the 10k baseline (sweep_best_10k.py)
does NOT. The vs_outproj 5.27 number is WITH time conditioning. Either:
- (a) All new runs match the baseline: do NOT use --no_time_cond.
- (b) Rerun the baseline control arm with --no_time_cond in every block.

Recommendation: (a). Time conditioning costs ~0.013 nats and some
wall-clock time. Since all comparisons are within-block (paired), the
absolute level does not matter. But for cross-block comparisons and for
eventual "Round 3 champion" claims, consistency with the 10k baseline
is important. Do NOT use --no_time_cond.

### Confound: warmup_steps

The 10k baseline uses warmup=400. The round3_data_reg.md proposal uses
warmup=50. This is a 8x difference and will affect results. Use warmup=400
for all Round 3 experiments to match the baseline.

### Interaction: LR x Data

If FineWeb-Edu changes the gradient scale (higher info density = stronger
gradients), the optimal LR might shift. Run the LR screen AFTER the data
decision (Block A before Block B). This is reflected in the plan above.

### Interaction: LR x Loss Weight

The LR screen (Block B) and loss weighting test (Block D) should use the
same LR. If Block B finds a new optimal LR, Block D should use it. This
is why D comes after B/C in the timeline.

### Zero-Code Advantage

Blocks A, B, C, D, E require ZERO code changes. Only Block F (WSD)
needs code. This is a strong operational advantage: no implementation
risk, no debugging time, all GPU time is productive.


## What We Learn Even From Null Results

1. **LR=0.02 is optimal for VS:** VS's variance normalization does NOT
   change the effective LR landscape for MDLM. The benefit of VS is
   purely in gradient quality (direction), not in enabling higher LR.
   This is still a contribution: it clarifies the mechanism.

2. **Loss weighting is irrelevant under VS:** Confirms the theory that
   VS normalizes per-timestep gradient scale. Practical win: drop gamma
   from the hyperparameter space. Use ELBO (theoretically principled, no
   gamma to tune).

3. **FineWeb-Edu is neutral on NLL:** Education filtering helps benchmarks
   but not raw language modeling. Data quality at our scale is about token
   count, not filtering.

4. **WD=0.01 is correct:** Muon's spectral norm constraint is active and
   helpful even at sub-epoch. This validates the theory.

All of these are publishable findings in a DiffuMamba paper. Null results
that confirm theory are valuable.
