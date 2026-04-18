# Round 2: Scale Up and Demonstrate Practical Value

## Title

From Quokka to Small: Scaling Muon+MDLM to 84M, Adding Text Generation, and Grounding Results Against Published Baselines

## Status Quo

All project results come from n=1 runs on a 35.9M "quokka" model trained for
at most 5000 steps on FineWeb-10B. The headline finding -- Muon+gamma1.5 at
val_loss=5.52 vs Adam+minsnr at 5.95 (0.43 nat gap) -- is promising but lives
in a narrow, unvalidated regime. No text has been generated. No external
baseline contextualizes these numbers. The project currently demonstrates a
training dynamics observation, not a practical capability.

This proposal designs four experiments that, together, produce a minimum viable
demonstration: "Muon accelerates masked diffusion LM training at 84M scale,
and the resulting model generates coherent text."

---

## Experiment 1: Scale to Small (84M) -- Does the Muon Gap Persist?

### Hypothesis

The 0.43 nat Muon advantage observed at quokka (35.9M, 5k steps) will persist
at small (84M, 5k steps), with the gap holding at >= 0.2 nats. Muon's
compute-efficiency advantage is scale-stable in autoregressive LMs (Liu et al.
2025: no crossover up to 16B; Essential AI 2025: holds 100M-4B), and masked
diffusion cross-entropy is structurally similar to AR cross-entropy.

### Method

Three runs on the "small" config (84M params, d=512, 8 layers, seq=512):

```bash
# Run 1: Muon + minsnr gamma=1.5 (current best)
python train.py --config small --batch_size 8 --max_steps 5000 \
    --val_every 500 --log_every 100 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight minsnr --minsnr_gamma 1.5 \
    --muon_lr 0.02 --adam_lr 3e-4 --save_best --save_path best_small_muon.pt

# Run 2: Adam + minsnr gamma=5 (best Adam config)
python train.py --config small --batch_size 8 --max_steps 5000 \
    --val_every 500 --log_every 100 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight minsnr --minsnr_gamma 5.0 \
    --adam_lr 3e-4 --save_best --save_path best_small_adam.pt

# Run 3: Muon + flat (previous best before gamma sweep)
python train.py --config small --batch_size 8 --max_steps 5000 \
    --val_every 500 --log_every 100 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight flat \
    --muon_lr 0.02 --adam_lr 3e-4
```

Tokens per run: 5000 x 8 x 512 = 20.5M tokens.

At ~58k tok/s (Mamba3 Triton non-MIMO), each run takes ~5.9 minutes.
Total for Experiment 1: ~18 minutes.

If VRAM is tight at bs=8, fall back to bs=4, max_steps=10000 to keep total
tokens constant. The small config (84M params ~336MB) with gradient
checkpointing and bf16 should fit well within 16GB.

### Expected outcome

- Muon+gamma1.5 beats Adam+minsnr by >= 0.2 nats at small scale.
- If the gap narrows below 0.1 nats, Muon's advantage may be scale-dependent
  or an artifact of small models. If it reverses, the quokka finding was noise.

### Decision point

- Gap >= 0.2 nats, same sign as quokka: proceed to Experiments 2-4.
- Gap 0.05-0.2 nats: run 2 more seeds of the top 2 to get error bars.
- Gap < 0.05 or reversed: Muon advantage does not scale. Pivot to architecture
  exploration (hybrid attention) instead of optimizer research.

---

## Experiment 2: Longer Training (10k steps) -- Convergence Check

### Hypothesis

The Muon advantage at 84M persists to 10k steps with no crossover. This is
predicted by Essential AI (2025): "Muon strictly lower bounds AdamW on loss
curves, even to the end of training far beyond the Chinchilla-optimal budget
(no crossover)." However, their evidence is all autoregressive. A crossover
on masked diffusion would be a genuinely novel finding.

### Method

Extend the winner and runner-up from Experiment 1 to 10k steps:

```bash
# Run 4: Best Muon config from Exp 1, 10k steps
python train.py --config small --batch_size 8 --max_steps 10000 \
    --val_every 500 --log_every 100 --warmup_steps 400 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight minsnr --minsnr_gamma 1.5 \
    --muon_lr 0.02 --adam_lr 3e-4 --save_best --save_path best_small_muon_10k.pt

# Run 5: Best Adam config from Exp 1, 10k steps
python train.py --config small --batch_size 8 --max_steps 10000 \
    --val_every 500 --log_every 100 --warmup_steps 400 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight minsnr --minsnr_gamma 5.0 \
    --adam_lr 3e-4 --save_best --save_path best_small_adam_10k.pt
```

Tokens per run: 10000 x 8 x 512 = 40.96M tokens. ~11.8 min each.
Total for Experiment 2: ~24 minutes.

### Key metric

Plot val_loss vs step for both optimizers. Look for:
1. Constant gap (parallel curves): Muon has a permanent advantage.
2. Widening gap: Muon's advantage grows with compute -- strongest result.
3. Narrowing gap / crossover: Muon is faster early but Adam catches up.

---

## Experiment 3: Text Generation and Quality Evaluation

### Hypothesis

A 84M masked diffusion LM trained for 10k steps can generate locally coherent
text that is evaluable by standard metrics. We do not expect competitive
perplexity vs AR models at this scale and training budget (MDLM 169M trained
on full OpenWebText achieved gen-PPL ~115 via GPT-2 Large), but we expect to
produce readable English and to observe a quality difference between the Muon
and Adam checkpoints that tracks the val_loss gap.

### Background: How diffusion LM quality is evaluated

The standard protocol from MDLM (Sahoo et al. 2024) and follow-up work:

1. **Generative Perplexity (Gen PPL)**: Generate N unconditional samples
   (typically 256-1024 sequences of length 1024), then score each with a
   frozen reference AR model (GPT-2 Large or GPT-2 XL). Lower is better.
   This measures how well the generated text matches the distribution that
   the reference model learned.

2. **MAUVE score** (Pillutla et al. 2021): Compares the distribution of
   generated text to a corpus of human text using neural embeddings. Scores
   range 0-1 (higher is better). Captures both quality and diversity --
   a model that generates only one sentence scores low despite high quality.

3. **Qualitative inspection**: Read a handful of samples. At 84M/10k-steps,
   we expect semi-coherent text with grammatical English fragments, topical
   drift, and occasional repetition.

The recent paper "Generative Frontiers" (April 2026) provides a comprehensive
survey of evaluation methods for diffusion LMs and confirms Gen PPL + MAUVE
as the standard evaluation pair.

### Method

This requires a small addition to the codebase: a `sample_and_eval.py` script.

```python
# sample_and_eval.py (new file, ~80 lines)
#
# 1. Load a saved checkpoint
# 2. Generate N samples using model.sample()
# 3. Decode tokens to text via tiktoken
# 4. Compute Gen PPL using GPT-2 Large (from HuggingFace)
# 5. Compute MAUVE against held-out FineWeb val text
# 6. Print 5 random samples for qualitative inspection

# Key parameters:
#   --checkpoint best_small_muon_10k.pt
#   --config small
#   --num_samples 256
#   --sample_steps 128 (MDLM default denoising steps)
#   --temperature 1.0
#   --reference_model gpt2-large
```

Evaluation runs (generation + scoring, no training):

```bash
# Run 6: Evaluate Muon checkpoint
python sample_and_eval.py --checkpoint best_small_muon_10k.pt \
    --config small --num_samples 256 --sample_steps 128

# Run 7: Evaluate Adam checkpoint
python sample_and_eval.py --checkpoint best_small_adam_10k.pt \
    --config small --num_samples 256 --sample_steps 128

# Run 8: Evaluate Muon at different temperatures
python sample_and_eval.py --checkpoint best_small_muon_10k.pt \
    --config small --num_samples 256 --sample_steps 128 \
    --temperature 0.8
```

Generation cost: 256 samples x 128 denoising steps x 512 tokens = ~16.8M
forward passes. At ~58k tok/s, each sample set takes ~5 minutes. GPT-2 Large
scoring (4.7GB) fits in 16GB alongside the diffusion model if we generate
first, free VRAM, then score.

Total for Experiment 3: ~20 minutes (generation) + ~10 minutes (scoring).

### What "good" looks like at this scale

Based on published results:
- MDLM 169M trained on full OpenWebText reports gen-PPL ~115 and MAUVE ~0.8
  (with optimized sampling).
- Our 84M model trained on 41M tokens (vs MDLM's ~9B tokens) will produce
  much higher gen-PPL (likely 200-500+) and lower MAUVE (0.2-0.5).
- The key metric is not absolute quality but relative: does Muon's lower
  val_loss translate to lower gen-PPL and higher MAUVE?
- LLaDA 8B and Dream 7B achieve competitive gen quality, but those are
  1000x our parameter count. At 84M, we are firmly in the "can it form
  English sentences?" regime.

### Realistic expectations

At 84M params and 41M tokens of training:
- Grammatical English fragments: likely yes.
- Coherent paragraphs: unlikely.
- Factual content: no.
- The samples will look similar to a poorly-trained GPT-2 small: topical
  text with frequent non sequiturs and repetition.

The value is not the samples themselves but the protocol: establishing
that we can generate, measure, and compare, which unlocks future scaling.

---

## Experiment 4: Reproducibility -- 3-Seed Validation of Best Config

### Hypothesis

The val_loss rankings are robust to random seed variation. At n=1, the 0.43
nat gap at quokka and whatever gap we observe at small could be partly luck.
Three seeds per condition give us a mean and standard deviation.

### Method

Take the best Muon config and best Adam config from Experiments 1-2.
Re-run each 3 times at small/5k-steps with different seeds:

```bash
# Seeds: 42, 137, 2024
for SEED in 42 137 2024; do
  python train.py --config small --batch_size 8 --max_steps 5000 \
      --val_every 500 --log_every 100 --warmup_steps 200 \
      --no_time_cond --lr_schedule cosine \
      --optimizer muon --loss_weight minsnr --minsnr_gamma 1.5 \
      --muon_lr 0.02 --adam_lr 3e-4 \
      --seed $SEED  # NOTE: needs --seed arg added to train.py
done
```

This requires adding a `--seed` arg to train.py that sets
`torch.manual_seed(seed)` and `torch.cuda.manual_seed(seed)` before model
init. Approximately 5 lines of code.

6 runs total (3 Muon + 3 Adam) x ~5.9 min each = ~35 minutes.

### Statistical bar

With n=3, we can compute mean +/- std. If the means are separated by more
than 1 standard deviation of either group, the result is suggestive. If
separated by 2+ std, it is strong. A proper Welch t-test with n=3 has low
power, but it is better than n=1.

---

## Full Experiment Plan: Budget and Timeline

All times assume Mamba3 Triton at 58k tok/s on AMD RX 9070 XT.

| Exp | Description | Runs | Steps each | Minutes | Cumulative |
|-----|-------------|------|------------|---------|------------|
| 1 | Scale to small (84M), 5k steps | 3 | 5,000 | ~18 | 18 min |
| -- | Decision gate: proceed if Muon gap >= 0.05 | | | | |
| 2 | Longer training (10k steps) | 2 | 10,000 | ~24 | 42 min |
| 3 | Text generation + evaluation | 3 | N/A (inference) | ~30 | 72 min |
| 4 | 3-seed reproducibility | 6 | 5,000 | ~35 | 107 min |

**Total: ~1.75 hours of GPU time.** This fits comfortably in a single session.

If stuck on PureSSM (~1.3k tok/s), multiply all times by ~45x. The full plan
would take ~79 hours -- infeasible. On PureSSM, run only Experiment 1 (3 runs,
~13.5 hours) and defer the rest.

---

## Implementation Work Required

### Changes to existing code (minimal):

1. **train.py**: Add `--seed` argument (5 lines). Set `torch.manual_seed()`
   and `torch.cuda.manual_seed()` before model construction.

2. **autoresearch.py**: Add a `scaling_roundtwo` mode that runs Experiments
   1-2 programmatically, similar to the existing `opt_x_lossweight` mode.

### New code:

3. **sample_and_eval.py** (~80-120 lines): Load checkpoint, generate samples
   via `model.sample()`, decode with tiktoken, compute Gen PPL with a
   HuggingFace GPT-2 Large model, compute MAUVE using the `mauve-text`
   package, print random samples. Dependencies: `tiktoken`, `transformers`,
   `mauve-text`.

### Not required:

- No changes to model.py or ssm.py.
- No new model architectures.
- No data pipeline changes (FineWeb-10B already sufficient).

---

## Expected Outcomes and What They Mean

### Best case (Muon advantage scales + generates text):

- Small (84M) Muon beats Adam by >= 0.2 nats at 5k steps.
- Gap persists or widens at 10k steps (no crossover).
- Muon checkpoint produces lower Gen PPL and higher MAUVE than Adam.
- 3-seed validation shows the gap is > 1 std.
- **Conclusion**: "Muon accelerates masked diffusion LM training, and this
  translates to better generation quality. First demonstration at this scale."

### Mixed case (advantage shrinks but holds):

- Gap is 0.05-0.2 nats at small, consistent direction.
- Generation quality difference is noisy (Gen PPL within 10%).
- 3-seed shows overlap in distributions.
- **Conclusion**: "Muon provides a modest speedup for MDLM training. Effect
  is real but small at 84M -- worth testing at larger scale before claiming
  practical significance."

### Null case (advantage disappears):

- Gap < 0.05 nats or reverses at small.
- Val loss trajectories cross before 10k steps.
- **Conclusion**: "Muon's advantage on MDLM is specific to the quokka regime
  (35M params, 5k steps) and does not transfer to practical scales. The flat-
  weighting insight from round 1 was a local optimum."
- **Pivot**: Switch focus to architecture experiments (hybrid attention) or
  data-quality improvements rather than optimizer research.

---

## Risk Assessment

### Risk 1: VRAM on small config
**Probability: Low.** The small config is 84M params = ~168MB in bf16. With
gradient checkpointing, activations for 8 layers x bs=8 x seq=512 should be
well under 16GB. Mamba3 non-MIMO uses less activation memory than attention.
**Mitigation**: If OOM, reduce batch_size to 4 (double steps to keep tokens).

### Risk 2: GPT-2 Large does not fit alongside diffusion model for scoring
**Probability: Medium.** GPT-2 Large is 774M params (~1.5GB in fp16). The
diffusion model is 84M (~168MB bf16). Together: ~1.7GB of parameters. But
GPT-2 Large activation memory at seq=512 could be 2-4GB. Total ~6-8GB, which
should fit in 16GB.
**Mitigation**: Generate all samples first, free the diffusion model from GPU,
then load GPT-2 Large for scoring. Or score in fp32 on CPU (slower but safe).

### Risk 3: Text quality is too poor to evaluate meaningfully
**Probability: Medium.** At 84M / 41M training tokens, the model has seen
~0.5 tokens per parameter -- severely undertrained. Generated text may be
mostly gibberish, making Gen PPL and MAUVE unreliable.
**Mitigation**: If 5k-step samples are poor, use only the 10k-step checkpoint.
If still poor, extend the best config to 20k steps (an additional ~12 min).
The MDLM paper shows quality improves rapidly with training even at small scale.

### Risk 4: Muon's NS overhead matters more at small scale
**Probability: Low.** Newton-Schulz adds ~5 matrix multiplications per Muon
parameter group per step. At 84M with ~50-60M Muon params, this is negligible
vs the forward/backward pass. The 5k-step quokka results show Muon and Adam
take similar wall-clock time (687s vs 641s -- Muon is 7% slower, within noise
of different loss-weight compute).
**Mitigation**: Track tok/s for both optimizers at small scale.

### Risk 5: All results are still single-GPU, single-data-split
**Probability: Certain (known limitation).** This is a single RX 9070 XT
project. Results on FineWeb-10B may not transfer to other datasets. This is
acceptable for an autoresearch project -- the goal is to find promising
directions, not to publish definitive results.

---

## Literature Support

### Muon scaling evidence

- **"Muon is Scalable for LLM Training"** (Liu et al., Moonshot AI, Feb 2025):
  ~2x compute efficiency over AdamW at scales up to 16B params / 5.7T tokens.
  No crossover. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

- **"Practical Efficiency of Muon for Pretraining"** (Essential AI, May 2025):
  100M-4B models, "Muon strictly lower bounds AdamW, no crossover." 10-15%
  token savings. [arXiv:2505.02222](https://arxiv.org/abs/2505.02222)

- **"Muon and SOAP prove to be highly efficient also for diffusion models"**
  (from optimizer benchmark, 2025): Direct evidence that Muon works for
  diffusion, though this was on image diffusion with continuous noise, not
  masked/discrete diffusion on text.

### Masked diffusion LM evaluation standards

- **MDLM** (Sahoo et al., NeurIPS 2024): Established Gen PPL + MAUVE as the
  standard eval for masked diffusion LMs. 169M model on LM1B achieves PPL
  27.04. On OpenWebText, gen-PPL ~115 with standard sampling.
  [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)

- **"Scaling Up Masked Diffusion Models on Text"** (ICLR 2025): First scaling
  law for masked diffusion. MDMs scale comparably to AR models, with a
  relatively small compute gap. Trained models up to 1.1B parameters.
  [OpenReview](https://openreview.net/forum?id=WNvvwK0tut)

- **"Scaling Behavior of Discrete Diffusion LMs"** (Dec 2025): Scaling
  behavior depends on noise type; masked diffusion scales best among discrete
  diffusion variants. [arXiv:2512.10858](https://arxiv.org/abs/2512.10858)

- **"Generative Frontiers: Why Evaluation Matters for DLMs"** (Apr 2026):
  Comprehensive survey of evaluation methods for diffusion LMs. Confirms
  Gen PPL + MAUVE as standard. [arXiv:2604.02718](https://arxiv.org/abs/2604.02718)

### Diffusion LM quality at scale (context for expectations)

- **LLaDA 8B** (Feb 2025): Competitive with LLaMA3 8B on benchmarks. First
  diffusion LM to match AR quality at scale.
  [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)

- **Dream 7B** (Aug 2025): Outperforms on planning tasks (Countdown, Sudoku).
  [arXiv:2508.15487](https://arxiv.org/abs/2508.15487)

- **ADLM** (May 2025): Anchored diffusion improves test perplexity by 9.54%
  over MDLM on LM1B. [arXiv:2505.18456](https://arxiv.org/abs/2505.18456)

- **DiffuMamba** (Nov 2025): Scales Mamba-based diffusion LM to 1.3B,
  matching Transformer-based diffusion quality with 8.2x inference throughput.
  [arXiv:2511.15927](https://arxiv.org/abs/2511.15927)

### Mamba scaling context

- **Mamba-3** (Mar 2026): MIMO variant at 1.5B achieves 57.6% avg accuracy,
  2.2 points above Transformer. Complex-valued states enable state-tracking.
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)

- Mamba-3B "outperforms Transformers of the same size and matches Transformers
  twice its size" in pretraining and downstream eval (original Mamba paper,
  confirmed at multiple scales).

---

## What This Proposal Does NOT Cover (Future Work)

These are explicitly deferred, not forgotten:

1. **Transformer MDLM baseline**: An 84M Transformer-based MDLM (replacing
   Mamba blocks with standard attention) would contextualize whether our
   results are about the Mamba backbone or the Muon optimizer. This requires
   implementing a separate model class and is a full follow-up experiment.

2. **Base config (231M)**: VRAM may be tight. Defer until small-scale results
   justify the investment. If the Muon advantage holds at 84M, 231M is the
   natural next step.

3. **Hybrid attention (DiffuMamba-H)**: Interleaving 1 attention layer per 5
   Mamba blocks. Architecturally interesting but orthogonal to the optimizer
   question. Run after settling the optimizer.

4. **Multi-GPU / longer training**: Training to convergence on 1B+ tokens
   would take hours even with Triton. Meaningful for a paper but not for an
   autoresearch session.

5. **Downstream tasks**: LAMBADA, HellaSwag, etc. require prompt-conditional
   generation that masked diffusion handles differently from AR models. This
   is a research problem in itself (see DARE framework, April 2026).
