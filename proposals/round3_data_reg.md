# Round 3: Data Quality and Regularization Optimization

## Title

**Data Diet and Regularization Audit: FineWeb-Edu, Weight Decay
Disentanglement, and Batch Size Efficiency for Sub-Epoch MDLM Training**

## Motivation

We have optimized architecture (all-Mamba, additive merge) and optimizer
(Muon + MinSNR gamma=1.5), but have never touched data quality or
regularization. At 10k steps x 8 x 1024 = 82M tokens, we are deep in the
sub-epoch regime (0.08 epochs of 1B tokens) -- the model sees each token
at most once. This changes the regularization calculus fundamentally:

1. **Weight decay may be counterproductive.** Quokka (arXiv 2510.03280)
   found "WD has little benefit at one epoch, useful in multi-epoch."
   But Muon complicates this: recent theory (arXiv 2506.15054) proves
   that Muon + decoupled WD implicitly enforces a spectral norm constraint
   on weight matrices, with the constraint radius = 1/lambda. Setting
   WD=0 removes this implicit regularization entirely, which may hurt
   even in the sub-epoch regime. The interaction between Muon's spectral
   constraint and sub-epoch training has never been tested.

2. **FineWeb-Edu may be strictly better at our token budget.** HuggingFace
   ablations show FineWeb-Edu gives +12% relative MMLU and +24% relative
   ARC over FineWeb at 350B tokens (arXiv 2406.17557). At our smaller 1B
   token budget, the education filter removes low-quality text that wastes
   gradient updates -- each of our ~82M token views should count. The
   FineWeb-Edu shards are already downloaded and shard-format compatible.

3. **Batch size has a known per-FLOP efficiency frontier.** arXiv 2507.07101
   shows that small batch sizes are (a) more robust to hyperparameter
   misspecification, (b) equal or better per-FLOP, and (c) achieve nearly
   optimal loss even at batch size 1 for 30M-scale models. Our current
   batch_size=8 was inherited from early experiments, never validated. The
   interaction with Muon's momentum dynamics (beta=0.95) is untested.

4. **MDLM has built-in regularization through masking.** Recent work
   (arXiv 2601.22450, arXiv 2510.04071) shows that masked diffusion's
   random token masking acts as an implicit regularizer, suppressing
   memorization. Adding dropout on top may be redundant or even harmful
   at sub-epoch training. This needs empirical confirmation at our scale.

## Hypotheses

**H1 (Data quality):** FineWeb-Edu will improve val_loss by 0.05-0.15
nats over FineWeb at 5k steps, because education-filtered text has
higher information density per token for a general LM.

**H2 (Weight decay):** WD=0.01 for Muon is near-optimal due to Muon's
spectral norm constraint mechanism. WD=0 will hurt (unconstrained spectral
growth), and WD=0.1 will over-constrain. For Adam params, WD barely
matters at sub-epoch.

**H3 (Batch size):** Smaller batch (bs=4) will match or beat bs=8 in
val_loss at fixed token count, per the small-batch efficiency finding.
bs=16 will be slightly worse per-token but faster wall-clock.

**H4 (Dropout):** Dropout will hurt at sub-epoch training. MDLM's masking
already provides sufficient regularization; adding dropout reduces the
effective model capacity without preventing overfitting (there is none).

## Method

All experiments use the validated best config as baseline:
```
--config quokka --optimizer muon --muon_variant base
--muon_lr 0.02 --adam_lr 3e-4 --loss_weight minsnr --minsnr_gamma 1.5
--lr_schedule cosine --warmup_steps 50 --no_time_cond
--val_every 250 --val_steps 10 --log_every 50
```

### Phase 1: FineWeb-Edu vs FineWeb (6 runs, ~24 min)

**The highest-leverage experiment.** Data quality is multiplicative with
all other improvements.

```
Conditions:
  A) --data_dir data/fineweb10B         (baseline, FineWeb)
  B) --data_dir data/fineweb-edu-10B    (FineWeb-Edu)

Steps: 5000
Batch size: 8, seq_len: 1024
Seeds: 42, 137, 2024
Val: both conditions evaluated on FineWeb val (data/fineweb10B val shard)
     to ensure comparable metric
Total: 2 x 3 = 6 runs x ~4 min = ~24 min
```

**Critical detail:** Both conditions must use the same val set. Train on
Edu, validate on FineWeb. This measures whether Edu training transfers
to general text, not just whether the model fits Edu-style text better.

Also run one FineWeb-Edu condition with Edu val to check for domain gap:
```
  B') --data_dir data/fineweb-edu-10B --val_data_path data/fineweb-edu-10B
```
(1 extra run, ~4 min. Same seed=42 as B for direct comparison.)

**Decision gate:** If FineWeb-Edu wins by >0.03 nats (mean, consistent
across seeds), adopt it as the default dataset for all subsequent phases.

### Phase 2: Weight Decay Sweep (9 runs, ~36 min)

Test WD independently for Muon and Adam param groups.

```
Conditions (Muon WD sweep, Adam WD fixed at 0.01):
  C) --muon_wd 0.0   --adam_wd 0.01
  D) --muon_wd 0.01  --adam_wd 0.01   (baseline)
  E) --muon_wd 0.1   --adam_wd 0.01

Steps: 5000, batch_size: 8, seq_len: 1024
Seeds: 42, 137, 2024
Data: winner of Phase 1 (or FineWeb if Phase 1 is inconclusive)
Total: 3 x 3 = 9 runs x ~4 min = ~36 min
```

We only sweep Muon WD first because (a) the spectral norm constraint
theory makes specific predictions, and (b) Muon controls ~80% of params.
If we find that Muon WD matters, we do a follow-up Adam WD sweep in
Phase 4.

**Predictions:**
- WD=0: val_loss ~0.05-0.10 worse than WD=0.01 (spectral norm grows,
  updates become unstable late in training)
- WD=0.01: near-optimal (current default)
- WD=0.1: val_loss ~0.02-0.05 worse (over-constrains, limits capacity)

### Phase 3: Batch Size Sweep (9 runs, ~36 min)

```
Conditions:
  F) --batch_size 4   (4 x 1024 = 4096 tok/step, 20k steps for 82M tok)
  G) --batch_size 8   (8 x 1024 = 8192 tok/step, 10k steps for 82M tok)
  H) --batch_size 16  (16 x 1024 = 16384 tok/step, 5k steps for 82M tok)

Fixed total tokens: 82M (adjust max_steps to match)
  F) --max_steps 20000 --warmup_steps 100 --val_every 1000
  G) --max_steps 10000 --warmup_steps 50  --val_every 500     (baseline at 10k)
  H) --max_steps 5000  --warmup_steps 25  --val_every 250

Seeds: 42, 137, 2024
Data: winner of Phase 1
Total: 3 x 3 = 9 runs
```

**Time estimate:** All conditions process 82M tokens. At ~58k tok/s,
each condition takes ~23.5 min. 9 runs = ~3.5 hours -- OVER BUDGET.

**Budget-saving modification:** Run Phase 3 at 41M tokens (half):
```
  F) --batch_size 4  --max_steps 10000  --warmup_steps 50  --val_every 500
  G) --batch_size 8  --max_steps 5000   --warmup_steps 50  --val_every 250
  H) --batch_size 16 --max_steps 2500   --warmup_steps 25  --val_every 250

Time per run: ~12 min
Total: 9 x ~12 min = ~108 min... still over budget.
```

**Final budget fit:** Run at 5k steps each (unequal tokens), seeds 42 and
137 only. This tests per-step efficiency, not per-token:
```
  F) --batch_size 4  --max_steps 5000  (20M tok)
  G) --batch_size 8  --max_steps 5000  (41M tok, baseline)
  H) --batch_size 16 --max_steps 5000  (82M tok)

Seeds: 42, 137
Total: 3 x 2 = 6 runs x ~4-8 min = ~36 min
```

This gives us: (a) per-step efficiency (same steps, different BS), and
(b) a first look at whether more tokens/step helps. If bs=4 matches bs=8
at fewer tokens, the small-batch finding extends to Muon+MDLM.

**Note on beta2 / token half-life:** arXiv 2507.07101 emphasizes that
Adam's beta2 must be adjusted when changing batch size to maintain a fixed
"token half-life." For Muon params this does not apply (no second moment),
but our Adam auxiliary group uses beta2=0.999. At bs=4 vs bs=16, the
token half-life changes by 4x. We keep beta2 fixed for this initial sweep
to isolate the batch-size effect; if batch size matters, we investigate
the beta2 interaction separately.

### Phase 4: Quick Checks (4 runs, ~16 min)

Only run if time remains after Phases 1-3.

**4a. Dropout (2 runs):** Add 0.1 dropout after each MLP in BiMamba3Block.
Requires a small model.py edit. Run at 5k steps, seed=42, FineWeb winner.
Compare dropout vs no-dropout.

**4b. Gradient clipping (2 runs):** Run with --grad_clip 0 (no clipping)
vs baseline --grad_clip 1.0. If loss diverges without clipping, clipping
is essential; if it matches, we can remove it (one less hyperparameter).

```
Total Phase 4: 4 runs x ~4 min = ~16 min
```

## Implementation Plan

### Code changes needed:

1. **Data loading for FineWeb-Edu:** Already works. The glob pattern
   `*fineweb_{split}_*.bin` matches `edu_fineweb_train_*.bin` and
   `edu_fineweb_val_*.bin` in `data/fineweb-edu-10B/`. Just pass
   `--data_dir data/fineweb-edu-10B`. Verify with a quick test.

2. **Dropout (Phase 4a only):** Add `--dropout` flag to train.py and
   `dropout` field to DiffuMamba3Config. In BiMamba3Block.forward(),
   add `F.dropout(self.mlp(h), p=self.dropout, training=self.training)`
   after the MLP. ~10 lines of code.

3. **Sweep script:** Write `sweep_data_reg.py` following the pattern of
   `sweep_best_10k.py`. Phases run sequentially; each phase's results
   inform the next phase's data choice.

### Execution order:

```
Phase 1 (FineWeb-Edu):    ~28 min  (including B' run)
Phase 2 (Weight decay):   ~36 min
Phase 3 (Batch size):     ~36 min
Phase 4 (Quick checks):   ~16 min  (if time permits)
                          --------
Total:                    ~116 min (~2 hours)
Budget remaining:         ~1 hour buffer for analysis + reruns
```

## Expected Outcomes

### Phase 1 (Data Quality)

| Condition | Expected val_loss | Confidence |
|-----------|------------------|------------|
| FineWeb (baseline) | 5.52 +/- 0.06 | Known (5k validated) |
| FineWeb-Edu | 5.40 +/- 0.08 | Medium -- Edu helps on benchmarks, but our metric is raw NLL on general text |

**Risk:** FineWeb-Edu may actually hurt raw perplexity on general FineWeb
val set because Edu-filtered text has a different distribution (more
formal/educational, less conversational/web). The MMLU/ARC gains from Edu
may not translate to lower NLL. This is the main uncertainty.

### Phase 2 (Weight Decay)

| Condition | Expected val_loss | Reasoning |
|-----------|------------------|-----------|
| Muon WD=0.0 | 5.55-5.65 | Loses spectral norm constraint, weights may grow |
| Muon WD=0.01 (baseline) | 5.52 | Established |
| Muon WD=0.1 | 5.52-5.55 | Slight over-constraint, but small effect at sub-epoch |

**Key insight from literature:** Muon WD=0 is qualitatively different from
Adam WD=0. For Adam, WD=0 in sub-epoch is fine (Quokka). For Muon, WD=0
removes the implicit spectral norm bound, which is a structural property
of the optimizer, not just a regularizer. We predict Muon needs WD>0
even at sub-epoch.

### Phase 3 (Batch Size)

| Condition | Tokens seen | Expected val_loss | Wall time |
|-----------|-------------|-------------------|-----------|
| bs=4, 5k steps | 20M | 5.70-5.80 | ~4 min |
| bs=8, 5k steps | 41M | 5.52 | ~4 min |
| bs=16, 5k steps | 82M | 5.40-5.45 | ~8 min |

The per-step comparison conflates batch size with token count. The
interesting question is: if we run bs=4 for 10k steps (= same tokens as
bs=8 at 5k), does it match or beat? The 2-seed screening tells us whether
there is a batch-size effect worth investigating at equal tokens.

### Phase 4 (Quick Checks)

- **Dropout:** Expected to hurt by 0.02-0.05 nats. MDLM masking already
  regularizes; dropout adds noise to a model that is not overfitting.
- **No grad clip:** Likely matches baseline. If training is stable without
  clipping, we remove it. If it diverges, clipping stays.

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FineWeb-Edu hurts NLL on FineWeb val | Medium | 30% | Run B' with Edu val to disentangle domain shift from quality |
| Phase 3 over budget | Medium | 40% | Already cut to 6 runs (2 seeds). Can drop bs=16 if needed |
| WD=0 causes divergence | Low | 10% | If loss explodes, log it as "WD=0 diverges" and proceed |
| Dropout code change introduces bugs | Low | 5% | Only in Phase 4; test on CPU first |
| n=2 seeds insufficient for Phase 3 | Medium | 50% | Phase 3 is a screen. Any promising finding gets 3-seed replication in a follow-up round |
| Muon momentum interacts with batch size | Medium | 40% | Not tested here. If bs matters, follow-up to sweep momentum x batch |

### Known limitation: confounds in Phase 3

Running all batch sizes at the same step count means larger batches see
more tokens. This is deliberate: we want to know "does bs=16 at 5k steps
beat bs=8 at 5k steps?" (practical wall-clock question) AND infer the
per-token efficiency from the learning curves. A rigorous per-token
comparison requires matching total tokens, which is Phase 3.5 (follow-up).

### Budget guard rails

- Phase 1 is gating: if FineWeb-Edu is clearly better or clearly worse,
  we know immediately. No ambiguity expected (3 seeds, paired).
- Phase 2 is gating: if WD=0 diverges, we know in <1 min (loss NaN).
  Skip remaining WD=0 seeds, reallocate time.
- Phase 3 is a screen (2 seeds). Any interesting finding triggers a
  follow-up round, not an in-session deep dive.

## Decision Tree

```
Phase 1 result
  |
  +-- Edu wins (>0.03 nats): adopt Edu for all subsequent phases
  |
  +-- Edu loses or tie: keep FineWeb
  |
  v
Phase 2 result
  |
  +-- WD=0.01 best: keep default, confirms theory
  |
  +-- WD=0 best: surprising! Muon spectral constraint not needed
  |   at sub-epoch. Test at 20k steps to check if WD becomes
  |   needed with more training.
  |
  +-- WD=0.1 best: increase WD. Test WD=0.2, 0.5 in follow-up.
  |
  v
Phase 3 result
  |
  +-- bs=4 competitive per-step: small batch wins, follow up
  |   with per-token comparison (bs=4 at 10k steps vs bs=8 at 5k)
  |
  +-- bs=8 clearly best per-step: default is correct
  |
  +-- bs=16 clearly best per-step: scale up, check VRAM
  |
  v
Combine best settings from each phase for a final 3-seed 10k-step
validation run (the "Round 3 champion" config).
```

## Literature Support

### FineWeb-Edu Quality
Penedo et al. (arXiv 2406.17557) demonstrate that education-quality
filtering via an LLM-trained classifier (threshold=3/5) yields dramatic
benchmark improvements over unfiltered FineWeb. The 1.82B ablation model
on 350B tokens shows MMLU 33% -> 37% and ARC 46% -> 57%. Ultra-FineWeb
(2025) further extends this with +3.61 points over FineWeb base.

### Muon Spectral Norm Theory
Chandra et al. (arXiv 2506.15054) prove that Muon with decoupled weight
decay lambda solves a constrained problem where weight matrices are bounded
by spectral norm <= 1/lambda. Setting lambda=0 removes the bound entirely.
Setting lambda=0.1 constrains spectral norm <= 10, while lambda=0.01
gives <= 100. The practical question is which radius is right for our
31.5M model's weight matrices.

### Small Batch Efficiency
Marek et al. (arXiv 2507.07101) show that batch_size=1 can match large-batch
training for models up to 1.3B when the optimizer's token half-life is
properly maintained. They find that small batches are "more robust to
hyperparameter misspecification." The "critical batch size" framework
(arXiv 2505.23971) provides a complementary view: below the critical batch
size, all batch sizes perform similarly per-token; above it, diminishing
returns set in. For a 30M model, the critical batch size is likely well
below 1024 tokens, suggesting our bs=8 (8192 tokens) may already be in
the diminishing-returns regime.

### MDLM Implicit Regularization
Gu & Kim (arXiv 2601.22450) analyze MDLM's implicit regularizer, showing
that the masking objective penalizes model confidence on "unidentifiable"
inputs (heavily masked sequences), naturally preventing memorization.
Yang et al. (arXiv 2510.04071) confirm that random masking is "the
dominant factor" in MDLM's data efficiency advantage over AR models.
This suggests that explicit regularization (dropout, weight decay for
overfitting prevention) may be unnecessary for sub-epoch MDLM training.

### Quokka Scaling and Data Hunger
Quokka (arXiv 2510.03280) establishes that DLMs need 2.2-6.7x more data
than Chinchilla predicts for AR models (i.e., 44-134 tokens/parameter
vs Chinchilla's 20). For our 31.5M model, compute-optimal training
requires 1.4B-4.2B tokens. We are currently at 82M tokens (10k steps),
which is 2-6% of compute-optimal. Even at 1B tokens available, we are
data-starved. This makes data quality (FineWeb-Edu) especially important:
every token must count.
