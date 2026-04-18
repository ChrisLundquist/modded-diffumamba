# State Review 2026-04-18

Senior advisor candid feedback on DiffuMamba3 current state.

## 1. "We beat MDLM 169M" — this claim is weak, soften it.

The headline is not defensible as currently framed. Specific problems:

- **Sampler mismatch.** Our 54.3 uses top-k=50; MDLM's 82 is their reported
  number with their sampler (top-p=0.9, 1000 steps, their temperature default).
  You cannot compare gen-PPL numbers across different truncation regimes —
  top-k=50 at T=1.0 is an aggressive-truncation config that mechanically lowers
  gen-PPL by pushing mass onto GPT-2-likely tokens. Our no-truncation number
  is 348 at 30k, which is "worse than MDLM" under the same rule.
- **Tokenizer/corpus advantage.** We trained on FineWeb-Edu (cleaner, more
  GPT-2-aligned English) with GPT-2 tokenizer. MDLM used OpenWebText. Scoring
  under GPT-2 small rewards whichever training corpus is closer to WebText.
- **Compute asymmetry argument cuts both ways.** "100x less compute" sounds
  good but MDLM's number is at 1M steps where they've plateaued; early MDLM
  checkpoints would also hit ~54 under our sampler.
- **N=1.** Single seed, single sampler config, 16 samples × 128 tokens is
  too small a denominator for a paper-vs-paper claim.

**Recommendation:** rewrite as "our stack reaches gen-PPL 54 under top-k=50
at 250M tokens, within the range of reported MDLM numbers". Drop "beats"
language until you have matched-sampler, matched-eval, ≥256 samples, and
3 seeds.

## 2. Which prior findings are suspect (beyond sampler)?

Val_loss findings stand structurally (training ≠ sampling). But:

- **"lr=0.01 > 0.02" was tested only at 5k; at 50k the ordering could easily
  flip** — high-LR runs often win early and lose late. Treat as 5k-specific.
- **3-seed paired t-tests with t=40 and t=-37.8 are suspicious.** Those
  t-values imply between-seed std <0.01 nats on effects of 0.06 nats.
  Plausible for paired+shared-init but worth sanity-checking that "paired"
  really means shared data order and init — otherwise the DoF is overstated.
- **"Architecture screen" at n=1/1k is already invalidated by your own
  lesson**; the 5k re-runs (hybrid +0.06, gated +0.24) are thin (3 seeds,
  one scale, 31.5M). Do not generalize to 111M.
- **"FineWeb-Edu beats FineWeb on val"** is apples-to-oranges: the val set
  used is FineWeb-Edu's own val for the Edu run. A fair comparison requires
  evaluating both models on the *same* held-out set (ideally both).
- **Gamma findings and ELBO-is-bad** are probably real but measured at one
  scale only.

## 3. Is gen-PPL under GPT-2 small a good north star? No.

Known pathologies that apply to us:
- Rewards distributional mimicry of GPT-2's training corpus (OpenWebText),
  not semantic quality. A model that produces fluent-but-wrong English
  beats one that produces correct-but-technical English.
- Degenerate minimum: low-entropy repetitive text gets low gen-PPL if the
  repeated tokens are GPT-2-likely. Your H_1 guardrail partially fixes this.
- Top-k at eval time doubles the bias: you're measuring "how well does our
  model's top-50 look like GPT-2's top-50".

**Add, in order of ROI:**
1. Per-mask-rate val (essentially free, catches training-time collapse).
2. Self-BLEU or distinct-n at T=1, no truncation (which you already have
   partially — keep it).
3. MAUVE against a FineWeb-Edu held-out set (not GPT-2). Higher signal
   than gen-PPL for open-ended generation.
4. An honest zero-shot benchmark at the 111M scale (LAMBADA, HellaSwag
   normalized). MDLM and LLaDA both report these.

## 4. FineWeb vs FineWeb-Edu — right answer: the comparison is broken.

The current evidence does not let you pick a winner. Interpretation #3
(non-matched val) is certainly true. Interpretation #1 (GPT-2 bias) is
partially true. Interpretation #2 (Edu overfitting to narrow distribution)
is plausible but unvalidated.

**Resolution:** evaluate both models on *both* val sets, and on a neutral
third set (e.g., Wikitext). If Edu wins on Edu-val by 0.2 but loses on
FineWeb-val by 0.2, the effect is dataset mimicry, not general quality.
Also check zero-shot LAMBADA on both — that's the cheapest real-quality
referee.

## 5. Highest-leverage next experiments (ranked)

1. **Matched-eval protocol + re-score everything on disk** (~4h, no
   training). Implement per-mask-rate val, flat val, cross-dataset val,
   MAUVE, and add zero-shot LAMBADA. Re-rank all 10k seeds and the 10L×640d
   checkpoints. Expected impact: may flip FineWeb-Edu finding, will
   dramatically clarify which "wins" are real. This is the single highest-
   leverage thing you can do.
2. **Longer training at 111M with proper eval gating** (~24h on current
   hardware). The 30k > 50k regression in gen-PPL is probably a real
   training-side signal worth characterizing. Train to 100k with health
   checkpoints every 10k; plot gen-PPL, H_1, val trajectory together.
   Expected impact: decides whether GIDD is needed.
3. **Scale-matched Muon replication at 111M** (~12h). All your
   Muon-vs-Adam rigor is at 31.5M. A single 111M Adam-vs-Muon pair
   would either cement or weaken the headline. Even n=1 paired at this
   scale is worth it before scaling further.

Deprioritize: more architecture screens at 31.5M, more gamma sweeps,
more LR sweeps at 5k.

## 6. Repetition/coherence gap — structural, not just undertraining.

"Grammatical but semantically weird" is the signature failure of masked
diffusion at small/mid scale and limited context. Three forces:

- **MDLM has no autoregressive commitment** — each unmasking is conditioned
  on a partially-filled context where earlier commits may be locally
  plausible but globally incoherent. The model cannot "retract".
- **Bidirectional Mamba has weaker long-range precision than attention
  at small scale.** Your own hybrid-attention-hurts finding is at 31.5M;
  it probably reverses before 1B.
- **111M is below the known coherence threshold for MDLM.** LLaDA needed
  ~1B to produce semantically coherent paragraphs; MDLM 169M produces
  exactly the quality you're seeing.

So: partly undertraining, partly scale, partly architectural. Don't expect
full coherence below ~500M for this family.

## 7. GIDD — still worth doing, signal to motivate it:

Motivating signal: if your Experiment #2 shows gen-PPL plateauing or
regressing while val continues to drop past 30k-50k, that is the exact
pathology GIDD's hybrid noise targets. If instead gen-PPL tracks val
monotonically once you're past sampler bugs, GIDD is a nice-to-have,
not load-bearing. Run #1 and #2 first, then decide.

## Bottom line

The project is technically solid on optimizer findings. The "beats MDLM"
claim is over-indexed on a single sampler config at n=1 and should be
softened immediately. The highest-impact move is a 4-hour eval-protocol
overhaul before any more training.
