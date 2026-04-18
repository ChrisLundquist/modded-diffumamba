# Round 3: Learning Rate and Schedule Optimization for Muon-VS

## Title

**LR Frontier and Schedule Search: Unlocking Muon-VS on Masked Diffusion
with WSD, Warmup Tuning, and Per-Group Adam LR**

## Hypothesis

The current Muon-VS config inherits base Muon's lr=0.02, which was swept
on base Muon at 500 steps (n=1). Three independent improvements likely
compound:

1. **Muon-VS tolerates higher LR than base Muon.** The variance
   normalization before Newton-Schulz (`M_tilde / sqrt(Gamma)`) acts as
   an implicit preconditioner, damping high-variance gradient directions.
   This is analogous to how Adam's second-moment scaling enables higher
   LR than SGD. The Muon-VS paper (arXiv 2601.14603) used the same LR
   as base Muon but explicitly notes they did NOT tune LR for VS --
   they "inherit the base Muon hyperparameters" to isolate the VS
   contribution. NorMuon (modded-nanogpt PR #144) already showed lr=0.06
   works with per-neuron normalization; VS's per-element normalization
   may enable similar or higher LR.

2. **WSD schedule beats cosine for flexible stopping.** Literature
   consensus (2024-2026): WSD with ~10-20% decay fraction matches or
   beats cosine at any compute budget, and enables anytime stopping
   during the stable phase. For a research project with frequent
   early-stop decisions, this is operationally valuable. Falcon-Mamba-7B
   used WSD successfully. The decay-to-zero variant ("D2Z") with linear
   decay is the current best practice (arXiv 2502.15938).

3. **Warmup was never swept.** Current 400-step warmup at 10k steps is
   4% of training. Literature says 0.5-2% is typical for small models.
   For a 31.5M model, 100-200 steps may suffice and wastes less budget
   at peak LR. However, Muon's NS orthogonalization may need longer
   warmup because the momentum buffer EMA needs time to converge --
   this is worth measuring.

## Method

### Budget accounting

Hardware: ~58k tok/s, ~12 min per 5k run, ~24 min per 10k run.
Total budget: ~3 hours = 180 min.

All experiments use: quokka config, bs=8, 3 seeds (42, 137, 2024),
`--muon_variant vs --muon_out_proj --loss_weight minsnr --minsnr_gamma 1.5
--no_time_cond`.

The schedule is sequential: each phase's results inform the next.

---

### Phase 1: Muon-VS LR frontier (5k steps, 3 seeds x 4 LRs)
**12 runs, ~12 min each = ~144 min ... too expensive at 3 seeds.**

**Revised: screen at 5k steps, 1 seed, 5 LRs; then validate top 2
at 5k steps, 3 seeds.**

#### Phase 1a: LR screen (1 seed, 5 runs, ~60 min)

```
seed=42, 5000 steps, cosine schedule, warmup=200

muon_lr = {0.01, 0.02, 0.04, 0.06, 0.08}
adam_lr  = 3e-4 (held constant)
```

Rationale for range:
- 0.01: below current optimum, sanity check
- 0.02: current best (control)
- 0.04: 2x current, modest increase
- 0.06: NorMuon's working point; VS may reach it
- 0.08: aggressive, tests stability ceiling

**Decision rule:** Pick the two LRs with lowest val_loss. If 0.02
still wins, VS doesn't benefit from higher LR -- skip Phase 1b
and keep 0.02.

If training diverges (loss > 8.0 at step 500), record as failed
and don't waste remaining steps.

#### Phase 1b: LR validation (3 seeds, 2 runs each, ~60 min total)

Validate the top-2 LRs from Phase 1a with seeds {42, 137, 2024}
at 5k steps. Report mean +/- std and paired t-test vs lr=0.02
control (which already has 3-seed data from 10k baseline, but
we need 5k-step comparisons).

Actually: also run lr=0.02 x 3 seeds as the within-experiment
control. **Total: 3 LRs x 3 seeds = 9 runs, ~108 min.**

This is over budget if we also do Phases 2-3. **Compromise:**
run the control (0.02) and top-1 LR only if the screen shows a
clear winner. 2 LRs x 3 seeds = 6 runs, ~72 min.

**Phase 1 total: 5 + 6 = 11 runs, ~132 min = 2.2 hr.**

This is tight. We need to cut scope elsewhere.

---

### REVISED BUDGET PLAN

Given 3-hour budget, we cannot do all three phases at 3 seeds.
Prioritize by expected impact:

| Phase | Question | Runs | Time | Priority |
|-------|----------|------|------|----------|
| 1a | VS LR screen | 5 x 1 seed | 60 min | HIGH |
| 2a | WSD vs cosine screen | 2 x 1 seed | 24 min | MED |
| 3a | Warmup screen | 3 x 1 seed | 36 min | LOW |
| Val | Validate winners | 3-6 x 3 seeds | 36-72 min | HIGH |

**Total screening: 10 runs = 120 min (2 hr).**
**Validation budget: 60 min = 5 runs x 3 seeds or 3 runs x 3 seeds.**

Strategy: screen everything at 1 seed first (120 min), then spend
remaining 60 min validating the single most impactful finding.

---

### Phase 1a: Muon-VS LR screen (5 runs, ~60 min)

```bash
# All share: --config quokka --batch_size 8 --max_steps 5000
#   --val_every 250 --warmup_steps 200 --lr_schedule cosine
#   --optimizer muon --muon_variant vs --muon_out_proj
#   --loss_weight minsnr --minsnr_gamma 1.5 --no_time_cond
#   --seed 42

python train.py [common] --muon_lr 0.01 --adam_lr 3e-4   # LR_0.01
python train.py [common] --muon_lr 0.02 --adam_lr 3e-4   # LR_0.02 (control)
python train.py [common] --muon_lr 0.04 --adam_lr 3e-4   # LR_0.04
python train.py [common] --muon_lr 0.06 --adam_lr 3e-4   # LR_0.06
python train.py [common] --muon_lr 0.08 --adam_lr 3e-4   # LR_0.08
```

**Expected outcome:** If VS acts like NorMuon, the optimum shifts
from 0.02 to 0.04-0.06 (0.1-0.3 nat improvement). If VS doesn't
help, 0.02 remains optimal and we learn VS needs no LR adjustment.

**Early stopping rule:** If val_loss at step 2500 is > 7.0, abort
(probable divergence).

### Phase 2a: WSD vs cosine screen (2 runs, ~24 min)

Requires adding WSD schedule to `get_lr_multiplier()`.

```python
# New schedule option in get_lr_multiplier:
if schedule == "wsd":
    # Warmup phase
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    # Decay phase (last cooldown_frac of training)
    decay_start = int(max_steps * (1 - cooldown_frac))
    if step >= decay_start:
        # Linear decay to zero (D2Z, arXiv 2502.15938)
        t = (step - decay_start) / max(1, max_steps - decay_start)
        return 1.0 * (1 - t)
    # Stable phase: constant peak LR
    return 1.0
```

Implementation: add `"wsd"` to the `lr_schedule` choices and add the
`elif schedule == "wsd"` branch. Also add `--cooldown_frac` arg
(default 0.2 for WSD, 0.6 for linear -- existing code already has
`cooldown_frac=0.6` hardcoded, should be promoted to CLI arg).

```bash
# Use Phase 1a's winning LR (call it LR*)

# WSD with 20% decay (literature standard)
python train.py [common] --muon_lr LR* --lr_schedule wsd \
  --cooldown_frac 0.2 --seed 42

# WSD with 10% decay (aggressive, more time at peak LR)
python train.py [common] --muon_lr LR* --lr_schedule wsd \
  --cooldown_frac 0.1 --seed 42
```

Control is Phase 1a's winning cosine run (same seed, same LR).

**Expected outcome:** WSD matches cosine within ~0.05 nats at 5k
steps. The operational benefit (anytime stopping) matters more than
raw loss. If WSD is worse by > 0.1 nat, stick with cosine.

### Phase 3a: Warmup length screen (3 runs, ~36 min)

Use Phase 1a's winning LR and Phase 2a's winning schedule.

```bash
python train.py [common] --muon_lr LR* --lr_schedule SCHED* \
  --warmup_steps 50  --seed 42    # 1% of training
python train.py [common] --muon_lr LR* --lr_schedule SCHED* \
  --warmup_steps 100 --seed 42    # 2% of training
python train.py [common] --muon_lr LR* --lr_schedule SCHED* \
  --warmup_steps 400 --seed 42    # 8% of training (current)
```

Note: warmup=200 is the control from Phase 1a (already run).

**Expected outcome:** Warmup 100-200 is the sweet spot. Very short
warmup (50) may cause early instability with Muon-VS because the
variance buffer `var_buf` needs ~50-100 steps to get a reliable
estimate (EMA with beta=0.95 has effective window ~20 steps, but
early steps have high bias correction noise). Very long warmup (400)
wastes steps at low LR. Predicting 100 or 200 optimal.

### Phase 4: Validation (3 seeds, ~36-72 min)

Run the single best config from Phases 1-3 with 3 seeds at 5k steps,
paired against the current best (Muon-VS, lr=0.02, cosine, warmup=400).

```bash
# Current best (control)
for seed in 42 137 2024; do
  python train.py [common] --muon_lr 0.02 --lr_schedule cosine \
    --warmup_steps 400 --seed $seed
done

# Proposed best
for seed in 42 137 2024; do
  python train.py [common] --muon_lr LR* --lr_schedule SCHED* \
    --warmup_steps WARM* --seed $seed
done
```

**6 runs x 12 min = 72 min.** If tight on budget, drop to 5k steps
for the validation (saves no time -- already at 5k).

Alternatively, if one of Phases 2a/3a shows no effect, skip its
validation and reallocate those runs to Phase 1b (LR validation
at more points).

**Decision criteria:**
- Adopt new config if mean improvement > 0.05 nats AND paired t-test
  p < 0.10 (one-tailed). We use a relaxed threshold because at n=3
  we have low power; a trend at p < 0.10 justifies adoption given
  no cost penalty.
- If improvement is < 0.05 nats: stay with current config (not worth
  the complexity).

---

## Implementation Changes Required

### 1. Add WSD schedule (train.py)

Modify `get_lr_multiplier()` to support `schedule="wsd"` and add
`--cooldown_frac` CLI argument. Approximately 15 lines of code.

```python
# In get_lr_multiplier, add:
elif schedule == "wsd":
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    decay_start = int(max_steps * (1 - cooldown_frac))
    if step >= decay_start:
        t = (step - decay_start) / max(1, max_steps - decay_start)
        return 1.0 * (1 - t)  # linear D2Z
    return 1.0

# In parse_args, change choices:
p.add_argument("--lr_schedule", choices=["cosine", "linear", "wsd"])
p.add_argument("--cooldown_frac", type=float, default=0.2,
               help="Fraction of training for cooldown (WSD/linear)")
```

### 2. Promote cooldown_frac to CLI

Currently hardcoded as 0.6 in `get_lr_multiplier` default. Thread the
new `--cooldown_frac` arg through to the call site at line 702-703.

### 3. Sweep script (sweep_lr_schedule.py)

Write a sweep script following the `sweep_arch_5k.py` pattern, with
sequential phase execution and per-phase summary stats.

---

## Expected Outcome

### Best case (estimated probability: 30%)
Muon-VS optimal LR is 0.04-0.06, giving ~0.15-0.25 nat improvement
over lr=0.02. WSD matches cosine. Warmup=100-200 is fine. Combined
validated improvement: val_loss ~5.05-5.15 at 5k steps (vs current
5.27). This would be a significant finding: Muon-VS unlocks higher
LR for masked diffusion, analogous to NorMuon for AR language models.

### Middle case (estimated probability: 50%)
Muon-VS optimal LR is 0.02-0.03, giving ~0.05 nat improvement. WSD
is equivalent to cosine. Warmup 200 is fine. Net improvement: ~0.05
nats. Modest but confirms current config is near-optimal and WSD is
a viable alternative for future longer runs.

### Worst case (estimated probability: 20%)
LR=0.02 remains optimal for VS (the variance normalization doesn't
actually change the effective LR landscape for MDLM's cross-entropy
gradient). WSD is slightly worse than cosine at 5k steps (not enough
time in stable phase). Warmup 400 is actually needed for VS's variance
buffer. No improvement. Still valuable: confirms current config with
tighter evidence and establishes that VS's benefit is purely in
gradient quality, not LR range.

## Risk / Cost

| Risk | Severity | Mitigation |
|------|----------|------------|
| LR screen is n=1, top pick is noise | HIGH | Phase 4 validates with 3 seeds before adopting |
| WSD needs > 5k steps to show benefit | MED | WSD's advantage grows with training length; if neutral at 5k, revisit at 10k |
| Warmup interacts with LR (confound) | MED | Screen warmup with the winning LR, not independently |
| VS variance buffer instability at high LR | LOW | Early stopping rule at step 2500; worst case is 12 min wasted per diverged run |
| Budget overrun from Phase 1b expansion | MED | Hard stop at 3 hours; if behind, drop Phase 3a (warmup) since it has lowest expected impact |
| Implementation bugs in WSD schedule | LOW | Unit-test the schedule function before running (5 min check: print LR curve for 5k steps) |

### Budget summary

| Phase | Runs | Time | Cumulative |
|-------|------|------|------------|
| 1a: LR screen | 5 | 60 min | 60 min |
| 2a: WSD screen | 2 | 24 min | 84 min |
| 3a: Warmup screen | 3 | 36 min | 120 min |
| 4: Validation | 6 | 72 min | 192 min |

**Total: 16 runs, ~192 min (3.2 hr).** Slightly over budget. To fit
in 3 hours, either:
- (a) Drop Phase 3a (warmup) entirely: saves 36 min -> 156 min.
  Justification: warmup=200 vs 400 is likely < 0.02 nats.
- (b) Run Phase 4 at 3k steps instead of 5k: saves ~24 min -> 168 min.
  Not recommended: 3k-step results are less reliable.
- **(c) Recommended: drop Phase 3a, use saved time as buffer for
  potential Phase 1b expansion.** Total: 13 runs, ~156 min (2.6 hr).

## Execution Order

1. Implement WSD schedule in train.py (~15 min)
2. Verify WSD with a quick print-LR-curve sanity check
3. Run Phase 1a (LR screen): 5 sequential runs
4. Analyze Phase 1a: pick top LR
5. Run Phase 2a (WSD screen): 2 runs with winning LR
6. Analyze Phase 2a: pick schedule
7. *If time permits:* Run Phase 3a (warmup): 3 runs
8. Run Phase 4 (validation): 6 runs with best config vs control
9. Paired t-test analysis, adopt or reject

## Literature

- Muon-VS: [Variance-Adaptive Muon (arXiv 2601.14603)](https://arxiv.org/abs/2601.14603)
  -- VS inherits base Muon's LR; LR was not independently tuned for VS.
- WSD schedule: [Understanding WSD (arXiv 2410.05192)](https://arxiv.org/abs/2410.05192)
  -- Decay ~10% of training, matches cosine oracle at all budgets.
- Linear D2Z: [Why linear decay to zero works best (arXiv 2502.15938)](https://arxiv.org/abs/2502.15938)
  -- Linear decay-to-zero is compute-optimal under proper peak LR.
- WSD for Mamba: [Falcon-Mamba with WSD](https://huggingface.co/tiiuae/falcon-mamba-7b-pre-decay)
  -- Production Mamba model trained with WSD schedule.
- NorMuon at lr=0.06: [modded-nanogpt PR #144](https://github.com/KellerJordan/modded-nanogpt/pull/144)
  -- Per-neuron normalization enables 3x higher LR for Muon.
- Warmup theory: [WSD schedules survey (EmergentMind)](https://www.emergentmind.com/topics/warmup-stable-decay-wsd-schedules)
  -- 0.5-2% warmup typical; absolute count matters more than percentage.
- Muon LR landscape: [Keller Jordan blog](https://kellerjordan.github.io/posts/muon/)
  -- "Sweep in logspace"; LR=0.02 is default but not universal.
