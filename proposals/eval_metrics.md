# Proposal: Sample-Quality-Aligned Evaluation Metrics for DiffuMamba3

**Status:** proposed
**Author:** research advisor
**Date:** 2026-04-17

---

## 1. The Divergence Phenomenon

We have direct evidence that our north-star metric (ELBO-weighted val NLL on
masked positions) is no longer tracking what we care about:

| Model                         | Params | Steps | val_loss | Sample quality                              |
|-------------------------------|--------|-------|----------|---------------------------------------------|
| quokka (4L × 384d)            | 31.5M  | 10k   | 5.07     | Grammatical fragments, local coherence      |
| big (10L × 640d)              | 111.7M | 50k   | **4.75** | Repetitive collapse ("I, I, and I", "family, family, family") |

The larger, longer-trained model wins on val_loss by 0.32 nats while producing
qualitatively worse generations. Every "best config" we've reported
(`project_muon_definitive`, `final 10k`, arch/depth/width sweeps) has been
ranked on this metric. **If the metric is wrong past a certain capability
threshold, the entire ranking stack is suspect near and above that
threshold.**

Three mechanisms plausibly drive this:

1. **ELBO upweights low-noise positions (1/t).** Low-noise ≈ easy in-context
   copying. Models can farm this loss without learning global distributional
   structure. Min-SNR clamp (γ=1.5) only partially mitigates.
2. **Per-position CE ≠ joint fluency.** Val loss averages over mask positions
   that are conditioned on mostly-unmasked context. This is a much easier
   problem than the actual generation task (start fully masked → end fully
   unmasked). A model can ace per-token denoising and still collapse at
   generation time.
3. **We use bf16 Gumbel-max sampling.** Zheng et al. (arXiv 2409.02908)
   showed this truncates Gumbel noise, effectively lowering sampling
   temperature. Fp32 gen-ppl drops ~4× (126 → 31) but unigram entropy falls
   from 5.66 to 5.17 — a signature of mode collapse. Our bf16 path has
   strictly worse precision than fp32, so our sampler is probably running at
   an even more degenerate effective temperature than any reported number.
   (Code: `model.py:643`.)

Recent literature (2025–2026) converges on the same diagnosis for MDLMs:
val NELBO is informative *within* a diffusion family but misleading
*across* families and at scale ("Scaling Beyond MDLM", arXiv 2602.15014),
and pure gen-ppl is degenerate unless paired with an entropy constraint
("Generative Frontiers", arXiv 2604.02718).

---

## 2. Proposed Metric Stack

We propose a **three-tier metric stack**, matching three compute regimes.

### Tier 0 — zero-generation cheap sentinels (every val step, <5% overhead)

These don't require generation. They're tripwires for pathologies that val_loss hides.

**0.a. Per-bucket val loss.** Split the val batches into mask-rate buckets
(e.g. t ∈ [0.05, 0.3), [0.3, 0.7), [0.7, 0.95]) and report mean loss per
bucket, in addition to the aggregate. Repetition collapse shows up first as
an anomaly in the high-mask bucket (where the model has little context and
must "invent" tokens) while the low-mask bucket keeps improving. Cost: zero
— we already sample t, just bin it.

**0.b. Flat-weighted val loss.** Also report val loss with `weight=1`
("flat"), not just ELBO (1/t). Flat weighting gives equal influence to the
hard high-mask regime that gen-time cares about. Cost: zero (reuse same
forward).

**0.c. Prior-KL on output distribution.** Compute mean `KL(p_θ(·|x_t) ||
unigram_emp)` on val batches, where `unigram_emp` is the empirical GPT-2
token frequency over FineWeb-Edu val (precomputed once). A degenerate model
concentrates mass on a handful of tokens → KL to the broad empirical prior
blows up. Cheap; detects mode collapse before generation ever runs.

### Tier 1 — cheap generation metrics (every N×val, +5–15% wall-clock)

Generate a small batch, measure self-consistency + basic diversity. **No
reference LM needed.**

**1.a. Unigram entropy H_1 over generated tokens.** The MDLM literature's
consensus fluency floor. Human OpenWebText has H_1 ≈ 5.43 (range [5.37, 5.55]
per Generative Frontiers). Healthy generations should land here. A model
producing "the family, family, family" falls off a cliff — this metric
catches it immediately. Compute: aggregate all tokens from N generations,
compute discrete entropy over the vocabulary. O(NL) CPU.

**1.b. Distinct-n (n=2, 3, 4) + repetition-4.** We already have
`repetition_4gram` in `sample_and_categorize.py` — promote it to training.
Cost: negligible.

**1.c. Self-BLEU-4 across samples.** Low self-BLEU = diverse samples, high
self-BLEU = mode collapse to one or two sentence templates. Cost: O(N²)
string work for small N, still dominated by generation.

### Tier 2 — reference-LM metric (every 2–5k steps, ~50–100s added)

**2.a. Generative perplexity under frozen GPT-2 small.** Sample from our
model, score under GPT-2-124M (frozen, bf16, no grad). Use **GPT-2 small**
(not large — fits in our VRAM budget) as the reference. For the reference
range, we can compare against the gap between our gen-ppl and GPT-2-scored
FineWeb-Edu val text gen-ppl from the same reference.

**Crucial:** always report gen-ppl **next to H_1**. Gen-ppl alone rewards
mode collapse (the low-entropy degenerate sampler in Zheng et al. got 31
gen-ppl vs the honest 126). We treat a model as "improving on quality" only
if `gen_ppl ↓ AND H_1 stays ≥ 5.2`. This is the operational version of
Generative Frontiers without the full temperature sweep.

---

## 3. Fixing the Sampler Before Measuring Anything

**Before rolling out any of these metrics, fix the Gumbel sampler.**

In `model.py:643-644`, cast `q_xs` to fp32 (or fp64) for the Gumbel-max step
only, then cast back. Keep the model forward in bf16. This is ~2 lines and
costs essentially nothing but corrects the Zheng et al. precision bug. It is
*possible* that what we're calling "repetitive collapse" in the large model
is partly an effective-temperature artifact of bf16 sampling, not a real
model deficiency. We should verify this first; it's a fast win.

```python
# before Gumbel-max, promote precision
q_xs_f32 = q_xs.float()
gumbel = -(torch.rand_like(q_xs_f32) + 1e-10).log()
sampled = (q_xs_f32 / (gumbel + 1e-10)).argmax(dim=-1)
```

If fixing this alone makes the 10L×640d samples coherent at the same
checkpoint, a big chunk of the "divergence" is a sampler bug rather than a
training objective problem.

---

## 4. Integration into Training

```
every args.val_every steps:            existing val_loss (ELBO + flat + per-bucket + prior-KL)
every args.val_every × 5 steps:        tier-1 gen metrics (16 samples × 256 tok, steps=128)
every args.val_every × 10 steps OR end: tier-2 gen-ppl under GPT-2 small (64 samples × 512 tok)
```

**Save-best logic update.** Replace `if val_loss < best_val_loss` with a
composite rule:
```
ok_health = (H_1 >= 5.2) AND (distinct_4 >= 0.6)
is_better = ok_health AND (val_loss_flat < best_val_loss_flat)
```
Motivation: switch the ranking objective from ELBO-weighted to flat-weighted
(the denoising objective, not the SNR-upweighted one), gated by a sample-
health check. If a checkpoint improves val but kills diversity, we don't
save it.

Also log all metrics to wandb so we can retroactively re-rank.

---

## 5. Compute Budget

Quokka-size (31M) on RX 9070 XT:

- Forward pass: ~55k tok/s (PureSSM fallback) to ~200k tok/s (Mamba3 Triton).
- 16 samples × 256 tok × 128 steps = 524k forward-tokens per tier-1 eval →
  **~3–10 s** per tier-1 eval (dominated by the 128 iterative denoising
  steps, not the token count).
- 64 samples × 512 tok × 128 steps = 4.2M forward-tokens → **~20–80 s**
  per tier-2 eval.
- GPT-2-124M reference scoring: 64 × 512 tok forward, one pass, bf16 →
  <2 s on this GPU, ~300 MB VRAM. Well within 16 GB budget (training uses
  ~12 GB, GPT-2-small ≈ 0.3 GB weights + 0.5 GB activations at bs=4).
  Loaded once, kept in VRAM; no host-device thrash.

For a 10k-step run with `val_every=500`:
- 20 val events, 4 tier-1 events, 2 tier-2 events
- Added wall-clock: ~1 min total vs ~30–60 min total training → **~2–3%
  overhead**. Acceptable.

For the 111M model: scale all compute ~4×. Still under 10% overhead.

---

## 6. Retroactive Invalidation Risk

Applying the new stack retroactively to existing results would likely
**invalidate or reshuffle**:

- **"Best config at 10k steps" (FineWeb-Edu + lr=0.01, val=5.07).** The
  0.07-nat FineWeb-Edu advantage over plain FineWeb is smaller than the
  typical flat-vs-ELBO val-loss swing we'd see; the ranking may or may not
  hold.
- **Gamma sweep.** Already known to be noise-level (project memory:
  "gamma=1.5 win was n=1 noise"). Flat-weighted val loss may re-open this —
  gamma changes *which timesteps dominate* the aggregate, so the ranking can
  flip under flat weighting.
- **Muon-VS vs Mousse at 10k.** Mousse had lower val_loss but 2.4× wall-
  clock. If Mousse at matched wall-clock instead produced lower H_1, the
  "Mousse best raw loss" framing is misleading.
- **Depth vs width iso-param finding.** Depth = width at iso-params by
  val_loss, but the 10L×640d experiment suggests depth helps val_loss and
  hurts H_1 in a correlated way. May need a second look at 8L×320 vs
  4L×384.
- **All-Mamba vs hybrid attention.** Hybrid attention was −0.06 nats on val
  (sig). It might be better on gen-ppl / H_1 (attention blocks often help
  with long-range diversity). **Highest-priority retroactive check.**
- **Adam vs Muon (t=40, p<0.001, +0.34 nats).** The *magnitude* is large
  enough that even a wrong metric can't plausibly flip the sign, but the
  absolute gap might shrink under flat/gen-ppl metrics. The qualitative
  Muon-wins headline is robust; the 0.34-nat figure specifically might not
  be.

**What would NOT be invalidated:** backend findings (Mamba3 Triton works,
MIMO broken), wall-clock measurements, n=1 unreliability lessons. Those are
orthogonal to the loss metric.

---

## 7. Risks and Failure Modes

- **Metric gaming.** A sampler that degenerates to GPT-2-small's own
  distribution (e.g. memorizing reference-LM preferred tokens) will score
  well on gen-ppl. Mitigation: H_1 guardrail + distinct-n.
- **Reference-LM mismatch.** GPT-2 was trained on WebText; we train on
  FineWeb-Edu. Reference-LM perplexity on our own training data is a
  calibration curve, not a quality ceiling. We should measure
  GPT-2-gen-ppl on a held-out FineWeb-Edu sample once to set the "human
  floor."
- **Tier-1 noise at small N.** 16 samples is few. H_1 is stable (aggregates
  tokens), but self-BLEU has high variance. Use 3-seed bands for any final
  ranking decision; intra-run is just monitoring.
- **Gen-ppl at training-distribution mismatch.** If our model generates
  good FineWeb-Edu-style text, GPT-2 (WebText) may over-penalize it. This
  biases *against* topic-specialized generations. Tolerable for coarse
  ranking, not for publication claims.
- **Sampler bug fix may "break" our existing checkpoints' metrics.** That's
  the point — those numbers were a mirage. Accept that historical numbers
  need a recomputation pass before cross-referencing.
- **Compute creep.** If we increase N samples or num_steps to reduce
  variance, overhead climbs fast. Hard cap: tier-2 must stay <10% of
  training wall-clock.

---

## 8. Concrete Next Steps

In order, smallest first:

1. **Fix the Gumbel bf16 precision bug in `model.py:643`.** 2 lines. Rerun
   sample generation on the 10L×640d checkpoint. If repetition disappears,
   we've just saved a month of "objective re-design" work that wasn't
   needed.
2. **Add tier-0 metrics to `train.py` val loop.** Per-bucket loss +
   flat-weighted loss + prior-KL. No generation. ~30 lines.
3. **Add tier-1 gen metrics (H_1, distinct-n, self-BLEU-4) at every
   5×val_every.** ~80 lines.
4. **Precompute unigram frequency of FineWeb-Edu val once.** Cache as
   `data/unigram_emp_fwe.pt`. One-time cost.
5. **Load GPT-2-small at training start, run tier-2 every 10×val_every.**
   Pin to VRAM once; measure headroom on an 11.5 GB training config first.
6. **Re-rank the 10k-step 3-seed checkpoints under the new stack** as the
   first honest validation that the metric stack agrees with eyeball
   quality. If it does: promote to default. If it doesn't: iterate.

After step 1 we will know whether the problem is a sampler precision bug or
a genuine objective/sample-quality divergence — and those imply very
different follow-up work.

---

## References

- Sahoo et al., *Simple and Effective Masked Diffusion LMs*, NeurIPS 2024 (arXiv 2406.07524)
- Zheng et al., *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling*, arXiv 2409.02908 — Gumbel fp32/fp64 issue
- Nie et al., *LLaDA: Large Language Diffusion Models*, arXiv 2502.09992 — 8B MDLM, eval protocol
- Ye et al., *Dream 7B: Diffusion Large Language Models*, arXiv 2508.15487 — MAUVE + gen-ppl
- *Scaling Beyond Masked Diffusion Language Models*, arXiv 2602.15014 — 1000-sample Llama-2 gen-ppl, low-variance training loss
- *Generative Frontiers: Why Evaluation Matters for Diffusion Language Models*, arXiv 2604.02718 — entropy-perplexity frontier framework
- Pillutla et al., *MAUVE: Measuring the Gap Between Neural Text and Human Text*, NeurIPS 2021 (arXiv 2102.01454)
- Hyperparameter practicalities: `github.com/krishnap25/mauve` (5k sample default, GPT-2-Large backbone)
