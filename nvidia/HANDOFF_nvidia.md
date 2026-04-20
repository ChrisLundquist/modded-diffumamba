# MDLM + Mamba3 Experiment Handoff

## STATUS UPDATE 2026-04-19: rep pathology is a sampler problem; PAPL is null

**Headline reframe**: transformer-MDLM has a real repetition pathology
(rep_4 ≈ 0.16 vs real text 0.012). After ~3 days of experiments:

1. **Mamba-MDLM doesn't have the pathology** (rep_4 ≈ 0.003, ROCm agent's data).
   Cross-architecture comparison at matched ~30M, FineWeb-Edu, MDLM objective.
   60× lower vanilla repetition than transformer.

2. **Training-side fix (PAPL, Peng 2025) is seed-unstable on transformer.**
   Phase-1 single-seed result at 30M scratch (PAPL τ=0.3 reduced rep_4 by 0.046)
   did NOT replicate at seed=43 (Δ=0.001). Mean across seeds ~0.02 — at the
   pre-registered threshold but with extreme between-seed variance. Inverse-PAPL
   (mechanistic control: same code, sign-flipped planner score) shows nearly
   the same effect, confirming most of the gain is "any per-position reweight"
   not PAPL's specific direction. PAPL is also null on Mamba (ROCm agent).

3. **Inference-side fix (ReMDM+Gumbel, Wang 2025 + Chang 2022) reliably reduces
   transformer rep_4 by ~50%** across seeds AND scales:
   - 30M vanilla s42: 0.162 → 0.086 (ReMDM n=24)
   - 30M vanilla s43: 0.168 → 0.125 (ReMDM n=24)
   - 30M vanilla s42 at n_steps=48: **rep_4 = 0.071** (best)
   - 125M @ 40k: 0.131 → 0.085 (ReMDM n=24)
   - 125M @ 50k: 0.146 → 0.094 (ReMDM n=24)

   Cost: ReMDM moves to a different Pareto point. Teacher_NLL increases by
   ~1 nat (3.6 → 4.5 on GPT-2; 3.5 → 4.4 on rhysjones-FWE). Higher entropy,
   more diverse, less "WebText-fluent" outputs.

4. **Mechanism (untested directly but well-supported by literature)**:
   bidirectional attention's softmax over keys produces sharp output
   distributions at masked positions; confidence-based unmasking commits
   to high-confidence stopwords first; cumulative repetition. Mamba's
   bidirectional structure (forward + backward scans averaged) routes info
   through fixed-dim hidden state, producing structurally smoother output
   distributions — no concentration to lock onto. ReMDM's revisability +
   Gumbel jitter (confidence_noise=4.5, Chang 2022 default) breaks the
   stopword attractor at inference for transformer; without Gumbel the
   sampler degenerates (rep_4 ≈ 0.75).

5. **Honest gap to ceiling**:
   - Real text rep_4 = 0.012
   - rhysjones AR (124M, FWE-trained) rep_4 = 0.022
   - Best transformer-MDLM (30M + ReMDM n=48) rep_4 = 0.071 — still 3-6× worse
   - Mamba-MDLM (ROCm agent, vanilla) rep_4 = 0.003 — beats real text on rep_4

6. **CRITICAL CAVEAT on the ReMDM "win" — qualitative samples show different
   failure mode, not better text** (2026-04-19, qualitative read of vanilla
   30M s43 samples on prompt #3 about net stock accounting):
   - vanilla+topk: *"sub assets plus the tax of all. tax tax tax tax tax tax
     tax tax tax tax tax tax tax tax tax tax tax tax tax..."* (high-confidence
     stopword collapse — what rep_4 metric flags)
   - vanilla+ReMDM (Gumbel 4.5): *"income stock. are generally for total. by
     which account is fixed to the and, stock the the paid. are the stock.
     on fixed net tax total. and1 is be income; equa"* (high-entropy
     fragmentation — short phrases, broken grammar, scattered punctuation)
   - REAL: coherent paragraph about asset accounting

   ReMDM trades **high-confidence repetition** for **high-entropy
   fragmentation**. Both are broken; rep_4 doesn't penalize fragmentation.
   The +1 nat teacher_NLL increase under ReMDM is real — the text really is
   less fluent, just less repetitive. ReMDM is a metric-artifact win on
   rep_4, not a quality win.

   The HONEST headline is "transformer-MDLM at 30M produces broken text in
   either failure mode; Mamba-MDLM doesn't have this issue at the same scale."

   **125M + ReMDM read (2026-04-19, prompts #3/17/50)**: still fragmented
   ("125M@50k+remdm: '- is of\n, and not\n- We can.\n- is does.\n-----'").
   Worse: rep pathology partially leaks through the Gumbel noise on prompts
   with strong context cues — "hol hol hol hol hol my (...) hol hol hol hol"
   on prompt #50 (about golf). The bigger model's sharper logits resist the
   noise injection more than the 30M's. ReMDM is fundamentally a band-aid.

7. **What ReMDM tells us about the mechanism** (interpretation, 2026-04-19):
   The fact that adding annealed Gumbel noise to the confidence ranking
   produces MORE diverse but FRAGMENTED output (rather than coherent diverse
   output) is itself evidence that the underlying issue is **the model's
   training-time output distribution is too peaky for confidence-based
   sampling at any noise level**. Either:
   - Greedy → repetition (high confidence locks onto stopwords)
   - Noisy → fragmentation (noise breaks repetition by sampling unrelated
     tokens that don't compose)
   No middle ground exists in the topk/ReMDM family.

   Implication: the right fix is to **train the model to produce smoother
   output distributions at masked positions** — either architectural
   (something Mamba-like) or via training-side regularization (output entropy
   penalty, label smoothing). Sampler-side fixes have hit a ceiling.

## Status: 125M D_modern on 10B complete — generation worse than a Markov chain

### Summary (2026-04-18)
Scaled up to GPT-2-small size (125M) on 10B FineWeb-Edu tokens, single epoch.
Val 4.499 MinSNR gamma=5 (best of all our models).

| Model | MinSNR γ=5 | Proper ELBO | 1/t ELBO | Params | Data |
|-------|------------|-------------|----------|--------|------|
| **125M D_modern (10B, 1ep)** | **4.499** | **24.16** | **15.50** | 123.6M | 10B |
| 30M D_modern (1B, 7ep) | 5.250 | 25.24 | 17.83 | 29.9M | 7B eff |
| 30M Mamba3 (1B, 7ep) | 5.666 | 25.57 | 19.22 | 25.3M | 7B eff |
| Pub baseline GELU (1B, 7ep) | 5.349 | — | — | 35.5M | 7B eff |
| Pub baseline SwiGLU (1B, 7ep) | 5.317 | — | — | 35.5M | 7B eff |

### Critical Caveat (2026-04-18): ELBO gains don't translate to generation

Multi-seed generation eval (10 seeds, top-k=50 sampling, fixed sampler):

| Model | Avg unique ratio | Quality |
|-------|------------------|---------|
| 125M D_modern (best ELBO) | **29%** (worst) | fragmented, repetitive |
| 30M D_modern | 47% | noisy but more diverse |
| 30M Mamba3 (worst ELBO) | **62%** (best) | most topical/coherent |

**ELBO-generation decoupling is real and strong**: the 125M has 0.77 nats better ELBO
than 30M Mamba3 but produces qualitatively worse text. Our generations are **worse than
a 4-gram Markov chain on 10B tokens** for local fluency.

MDLM's training objective (Min-SNR weighted ELBO) rewards confident prediction of
easy tokens (stopwords in context). This gets optimized but doesn't transfer to
generation coherence. The "0.08 nat Muon win" and "0.20 nat embedding LR win" are
ELBO improvements that don't translate to useful text.

Architecture ranking at 5K steps (3-seed probe, ELBO metric):
| Config | Mean | Std |
|--------|------|-----|
| D modern (RoPE+SwiGLU) | **5.943** | 0.007 |
| E combo (D + U-Net) | 5.959 | 0.025 |
| A baseline (GPT-2 style) | 6.192 | 0.005 |
| B deep (12L x 320d) | 6.219 | 0.005 |
| C U-Net | 6.223 | 0.006 |
| Mamba3 (Adam, gamma=5) | ~6.42 | — |

---

## Environment

- **GPU:** RTX 5090 32GB (Blackwell, sm_120), 400W power limit
- **OS:** WSL2 Ubuntu 24.04 on Windows
- **Python:** .venv at `/mnt/d/code/gpt-slide/.venv` (--system-site-packages)
- **PyTorch:** 2.10/2.11 + CUDA 13.0 (system torch)
- **mamba-ssm:** Built from source (v2.3.1 + 19 commits, has Mamba3)
  - Source at `/tmp/mamba_build` (may not survive reboot)
  - Installed to `.venv` with `CUDA_HOME=/usr/local/cuda-12.8`
  - **Patched** `cpp_extension.py` to allow CUDA 12.8 nvcc with PyTorch CUDA 13.0
- **causal-conv1d:** Built from source at `/tmp/causal_conv1d_build`
- **TileLang:** 0.1.8, **patched** `__slots__` fix for MIMO kernel
  - Fix at `.venv/lib/python3.12/site-packages/tilelang/3rdparty/tvm/python/tvm/runtime/support.py` line 153
  - PR submitted upstream
- **Data:** 1.1B FineWeb-Edu tokens at `/home/clundquist/muon_data/fineweb_1B.npy` (uint16, 2.1GB)
  - Also on Windows mount: `nvidia/data_cache/fineweb_1B.npy` (84x slower I/O)
  - Train: first 1B tokens, Val: last 100M tokens, GPT-2 tokenizer

### Known Patches (will be lost if packages reinstalled)
1. `cpp_extension.py` CUDA version check: RuntimeError → warning for major version mismatch
2. `tilelang/support.py` __slots__ fix for TVMDerivedObject
3. Both causal-conv1d and mamba-ssm built from git HEAD, not pip release

---

## What We Found

### Confirmed Findings
1. **Cosine LR hurts MDLM long training.** Transformer MDLM "converged" at val 6.17 (epoch 2) with cosine. Constant LR recovered 0.29 nats → val 5.88 at epoch 7.3. Data reshuffling between epochs doesn't matter.

2. **Muon CAN work for Mamba3 on NVIDIA — it's a config issue, not hardware.**
   Our original config (in+out_proj, lr=0.01, Adam 1.5e-4, gamma=5) lost to Adam by 0.23 nats.
   Other agent's config (in_proj ONLY, lr=0.02, Adam 3e-4, gamma=1.5, wd=0.01) BEATS Adam by 0.011 nats on our same NVIDIA hardware. Key fixes:
   - **Exclude out_proj from Muon** — NS orthogonalization distorts SSM output projection
   - **gamma ≤ 2** — heavy timestep reweighting (gamma=5) creates gradient scale variance that conflicts with NS's Frobenius normalization
   - **Higher auxiliary Adam LR** (3e-4 vs 1.5e-4) — non-Muon params (21.7M) were undertrained
   - **Muon lr=0.02** (2x our 0.01) and **wd=0.01**
   NS precision is NOT the issue — bf16/fp16/fp32 all give identical results (tested).

3. **d_state=32 matches d_state=64 in quality, 2x throughput.** Eliminating PCIe memory spill (34.6GB → 31.2GB) doubled tok/s from 120K to 250K.

4. **Mamba3 beats UNDERTRAINED transformer at convergence.** Mamba3 val 5.6659 at epoch 7.34 vs transformer val 5.88 at epoch 7.3 — 0.214 nats gap (0.460 under proper ELBO). BUT the transformer used AdamW lr=1.5e-4 for embeddings while Mamba3 used Adam lr=1e-3. Fixing this to Adam lr=1e-3 for the transformer makes it 0.22 nats better than Mamba3 at 5K steps. **Convergence comparison invalid until transformer is rerun with fixed embedding LR.**

5. **Mamba3 does not overfit at epoch 7.3.** Improvement rate decelerating (0.023→0.008 nats/2K steps) but still monotonically decreasing with no U-shape, unlike the transformer which showed U-shape at epoch 7.9.

6. **Proper ELBO confirmed wider gap (on old baseline).** 1000-timestep eval: Mamba3 25.572 nats vs Transformer 26.032 nats = 0.460 nats gap. Min-SNR metric underestimated gap. But this comparison uses the undertrained transformer.

7. **AdaLN time conditioning: marginal (+0.03 nats).** Not worth the 12% param overhead and 10% throughput cost at 25M scale.

8. **MIMO kernel doesn't work on RTX 5090.** Shared memory limit is 100KB, MIMO needs 107KB+. Works on datacenter GPUs (A100/H100). TileLang nvcc path fix also needed (must use CUDA 12.8 not 12.0).

9. **Transformer embedding LR is critical.** Adam lr=1e-3 for embeddings gives 0.20 nats over AdamW lr=1.5e-4. Embeddings are 19.3M of the 30.3M params — the dominant learning signal.

10. **Adam lr=1e-3 diverges on transformer attention/MLP.** Val 9.93 at 5K steps. Transformers need Muon or lower LR for non-embedding weights. Adam lr=3e-4 works (val 6.75) but still trails Muon (val 6.20).

11. **D_modern's 0.242-nat advantage is 95% RoPE, not SwiGLU.** Ablation: RoPE-only gives +0.230 nats; SwiGLU-only gives -0.012 nats (slightly harmful). Small interaction +0.024. Since published MDLM already uses RoPE, our D_modern was just catching up to the literature's position encoding. SwiGLU is NOT a novel contribution for MDLM.

12. **U-Net skip connections don't help for text MDLM.** +0.031 nats (worse) as standalone, neutral when combined with RoPE+SwiGLU. Unlike image diffusion where U-Net is standard.

13. **Depth (12L x 320d) doesn't help at 30M scale.** +0.027 nats worse than 6L x 384d, and 20% slower. The narrower embedding (320 vs 384) hurts more than extra depth helps.

14. **Lower ELBO does not guarantee better factual recall.** Cloze probes (fill-in-the-blank) show Mamba3 outperforms D_modern on factual retrieval despite 0.33 nats worse ELBO.

15. **Naive iterative demasking degenerates at 25M params.** Greedy demasking collapses into repetition.

16. **Sampler bug discovery (2026-04-18)**: Initial nucleus sampling had a critical bug — ordering positions by post-nucleus max probability biased unmasking toward peaky positions (stopwords), creating self-reinforcing repetition. Fix: use pre-nucleus distribution for ordering. Top-k=50 with temp=1.0 is the most robust sampler. The buggy sampler inflated Mamba3's apparent advantage (transformers have peakier distributions, suffered more from the bug).

17. **ELBO-generation decoupling confirmed (10 seeds, fixed sampler, 2026-04-18)**:
   - 125M D_modern (best ELBO 4.50): 29% unique word ratio — worst generation
   - 30M D_modern (ELBO 5.25): 47% unique
   - 30M Mamba3 (worst ELBO 5.67): 62% unique — best generation
   The 4x larger, 0.77-nats-better-ELBO model produces worse text than the worst-ELBO
   baseline. MDLM optimizes for confident-prediction-of-easy-tokens, not coherence.

18. **Our 125M MDLM is worse than a 4-gram Markov chain for local fluency.** A Markov
   model on 10B tokens would produce "the United States of America", natural commas,
   common phrases. Ours produces "the children-no-no those". This is a fundamental
   MDLM/iterative-demasking limitation at our scale, not a bug we can fix.

### Soft Findings (single seed, not validated)
- Muon helps transformer MDLM — confirmed, but embedding LR matters more
- Min-SNR gamma=5 works for MDLM; gamma=1.5 marginally better (~0.06 nats for Adam, critical for Muon)
- 25M is compute-optimal for 1B tokens (Quokka confirmed)
- Muon param routing for SSMs: in_proj only, NOT out_proj (confirmed cross-agent)
- RoPE is already standard in MDLM literature. SwiGLU for MDLM IS novel — everyone uses GELU.
- Muon's NS precision (bf16/fp16/fp32) makes zero difference — the issue is config, not numerics.

---

## Key Checkpoints

### Transformer MDLM (30.3M params, Muon + Min-SNR, cosine then constant LR)
| Checkpoint | Val | Epoch | Notes |
|-----------|-----|-------|-------|
| `mdlm_converge/checkpoint_15000.pt` | 6.17 | 1.97 | Cosine LR "converged" |
| `mdlm_extend_25m/checkpoint_50000.pt` | 5.88 | 6.55 | Best (constant LR) |
| `mdlm_extend_25m/checkpoint_60000.pt` | 5.89 | 7.86 | U-shape starting |

### 50M Transformer MDLM (resumed at lr=0.01)
| Checkpoint | Val | Epoch | Notes |
|-----------|-----|-------|-------|
| `mdlm_resume_50m/checkpoint_15000.pt` | 6.06 | 1.97 | Marginal gain over 25M |

### Mamba3 MDLM (25.3M params, Adam 1e-3 constant, d_state=32)
| Checkpoint | Val | Epoch | Notes |
|-----------|-----|-------|-------|
| `mamba3_converge/checkpoint_5000.pt` | ~6.57 | 0.66 | |
| `mamba3_converge/checkpoint_10000.pt` | 6.15 | 1.31 | Resume point |
| `mamba3_converge/checkpoint_15000.pt` | ~6.02 | 1.97 | |
| `mamba3_converge/checkpoint_20000.pt` | 5.91 | 2.62 | |
| `mamba3_converge/checkpoint_25000.pt` | 5.86 | 3.15 | Beat transformer here |
| `mamba3_converge/checkpoint_30000.pt` | 5.80 | 3.93 | |
| `mamba3_converge/checkpoint_35000.pt` | 5.77 | 4.46 | |
| `mamba3_converge/checkpoint_40000.pt` | 5.74 | 5.24 | |
| `mamba3_converge/checkpoint_45000.pt` | 5.72 | 5.77 | |
| `mamba3_converge/checkpoint_50000.pt` | 5.69 | 6.55 | |
| `mamba3_converge/checkpoint_55000.pt` | 5.67 | 7.08 | |
| `mamba3_converge/checkpoint_56000.pt` | **5.6659** | 7.34 | **Best, final** |

### Mamba3 Recipe Comparison (various optimizers, 1 epoch probes)
| Config | Steps | Best Val | Checkpoints |
|--------|-------|----------|-------------|
| B: Adam 3e-3 cosine | 4000 | 6.76 | `mamba3_recipes/B_*/ckpt_{1-4}000.pt` |
| C: Adam 1e-3 constant | 4000 | 6.56 | `mamba3_recipes/C_*/ckpt_{1-4}000.pt` |
| AdaLN test | 4000 | 6.53 | `adaln_test/mamba3_with_adaln/ckpt_{1-4}000.pt` |

---

## Code Structure

```
nvidia/
├── src/
│   ├── gpt2.py              # GPT-2 transformer baseline (MDLM with causal=False)
│   ├── transformer_v2.py    # D_modern: RoPE + SwiGLU + U-Net + 6-way AdaLN + QK-norm
│   ├── muon.py              # Muon optimizer (NS orthogonalization, weight_decay support)
│   ├── hybrid_model.py      # DiffuMambaH + QuokkaBlock (matches main repo architecture)
│   ├── adaln.py             # AdaLN primitives (zero-init modulation)
│   └── data.py              # TokenDataset, DataLoader helpers
├── training/
│   ├── train_dmodern_30m.py       # D_modern 30M (val 5.272 MinSNR γ=5)
│   ├── train_dmodern_125m.py      # D_modern 125M on 10B (val 4.499 MinSNR γ=5)
│   ├── train_published_baseline.py # Faithful MDLM reproduction (AdamW+cosine)
│   ├── train_mamba3_30m.py        # Mamba3 30M (val 5.666 MinSNR γ=5)
│   ├── sweep_variants_3seeds.py   # A/B/C/D/E variants × 3 seeds (RoPE/SwiGLU/U-Net/depth)
│   ├── sweep_muon_config.py       # Cross-agent Muon config match (5 configs)
│   ├── sweep_rope_swiglu_ablation.py  # 2×2 RoPE/SwiGLU isolation (RoPE = 95%)
│   ├── sweep_ns_precision.py      # bf16/fp16/fp32 NS test (all identical)
│   └── sweep_transformer_lr.py    # Transformer LR sensitivity study
├── probes/
│   ├── generation_multi_seed.py   # 10-seed generation + category analysis (MAIN finding)
│   ├── sampler_ablation.py        # Greedy/nucleus/top-k comparison (top-k=50 best)
│   └── cloze_basic.py             # Fill-in-the-blank factual probe
├── eval/
│   ├── standard_nll.py            # 3 metrics: proper ELBO, MinSNR γ=5, 1/t ELBO
│   ├── profile_vram.py            # VRAM profiler (validates scale-up)
│   └── download_fineweb_10b.py    # 10B tokens with chunked .npy save
├── README.md                      # user-facing overview
├── CLAUDE.md                      # agent context + "killed vs defensible" table
└── HANDOFF_nvidia.md              # this file
```

---

## To Extend Mamba3 Further (from 56K)

```python
# In mdlm_mamba3_converge.py, change:
RESUME_CKPT = 'outputs/mamba3_converge/checkpoint_56000.pt'
TOTAL_STEPS = 65000  # ~epoch 8.5
```

Script already has resume logic (MICRO_BATCH=16, GRAD_ACCUM=8). Just update RESUME_CKPT and TOTAL_STEPS.

---

## What the Reviewer Said We Need for Publication

1. ~~**Standard eval metric**~~ ✓ DONE — proper ELBO (1000 timesteps) implemented and run
2. ~~**Tuned AdamW baseline**~~ ✓ DONE — LR sweep found Muon + Adam 1e-3 embed is 0.20 nats better than original. **Now needs convergence run.**
3. **Multiple seeds** — at least 3 on headline comparisons. BLOCKED on fair convergence comparison.
4. **Published baseline reproduction** — train with MDLM paper's recipe to anchor numbers. Lower priority.

## Experiments Completed (2026-04-13)

### Standard NLL Eval (DONE, updated with D_modern)
Proper ELBO (1000 timesteps, no Min-SNR):

| Model | Proper ELBO | MinSNR | vs best |
|-------|-------------|--------|---------|
| D_modern | 25.239 | 5.251 | — |
| Mamba3 | 25.572 | 5.666 | +0.333 |
| Old transformer | 26.031 | 5.891 | +0.793 |

### Cloze Probes (DONE)
D_modern vs Mamba3 on 8 fill-in-the-blank tests. Mamba3 wins more (7 of 12 masked
positions) despite worse ELBO. ELBO ≠ factual recall.

### Published MDLM Baseline (Config 1 DONE, Config 2 IN PROGRESS)
Faithful reproduction of Sahoo et al.: RoPE + GELU + 6-way AdaLN + AdamW 4e-4 + cosine.
Config 1 (GELU): best val **5.349** at step 54K (35.5M params).
Config 2 (SwiGLU): in progress.
D_modern beats published baseline by **0.077 nats** with 5.6M fewer params.
Gap is Muon+constant LR vs AdamW+cosine — the optimizer is the real improvement.

### Muon Config Match (DONE)
| Config | Val @ 5K | vs Adam |
|--------|----------|---------|
| Hybrid (in+out, lr=0.02, Adam 3e-4) | **3.641** | **-0.063** |
| Muon-VS (in only, lr=0.02) | 3.683 | -0.021 |
| Their Muon (in only, lr=0.02) | 3.693 | -0.011 |
| Adam lr=1e-3 | 3.704 | — |
| Our Muon (in+out, lr=0.01) | 3.739 | +0.035 |

Key: Muon works on NVIDIA with correct config. Not a hardware issue.

### RoPE vs SwiGLU Ablation (DONE)
RoPE: +0.230 nats (95%), SwiGLU: -0.012 nats (harmful alone), Interaction: +0.024.
D_modern's advantage is all RoPE. SwiGLU is not a novel contribution.

### Muon+Gamma Probe (DONE)
2x2 grid on Mamba3, 5K steps, d_state=32:

| Config | Val (g5) | Val (g1.5) |
|--------|----------|------------|
| Adam+gamma1.5 | **6.359** | **3.686** |
| Adam+gamma5 | 6.417 | 3.721 |
| Muon+gamma1.5 | 6.493 | 3.741 |
| Muon+gamma5 | 6.645 | 3.813 |

### Muon NS Precision Test (DONE — negative result)
bf16, fp16, fp32 all give identical Muon results on Mamba3 (within 0.002 nats).
Precision is NOT the cause of Muon underperformance.

| Config | Val @ 5K |
|--------|----------|
| Adam lr=1e-3 | 6.409 |
| Muon bf16 NS | 6.644 |
| Muon fp16 NS | 6.643 |
| Muon fp32 NS | 6.642 |

### Muon Config Match (IN PROGRESS — 3 of 5 done)
Testing other agent's exact Muon config on our NVIDIA hardware. All gamma=1.5.

| Config | Val @ 5K | vs Adam |
|--------|----------|---------|
| Their Muon (in only, lr=0.02, Adam 3e-4) | **3.693** | **-0.011** |
| Adam lr=1e-3 | 3.704 | — |
| Our Muon (in+out, lr=0.01, Adam 1.5e-4) | 3.739 | +0.035 |
| Hybrid (in+out + their LRs) | pending | |
| Muon-VS (in only + variance scaling) | pending | |

**Key finding: Muon works on NVIDIA with correct config. Not a hardware issue.**

### Transformer LR Sweep (DONE)
3 configs at 5K steps:

| Config | Val @ 5K |
|--------|----------|
| A: Adam lr=1e-3 (all params) | 9.93 (diverged) |
| B: Adam lr=3e-4 (all params) | 6.75 |
| **C: Muon + Adam lr=1e-3 (embed)** | **6.20** |
| Original baseline (Muon + AdamW 1.5e-4) | ~6.40 |

## Most Impactful Next Experiments (ranked)

**The project has hit a crisis point**: ELBO improvements don't translate to generation quality.
The 125M model (our best) is worse than a Markov chain at producing fluent text. Decide the path forward
before more training.

### Path A: Fix Generation (highest value if it works)
1. **Better MDLM sampling** — Gillespie algorithm, non-greedy position scheduling, longer inference budget (many more demasking steps). Current procedure may just be too crude.
2. **AR baseline comparison** — train an equivalent 125M AR model on same 10B data. Shows whether the gap is MDLM-specific or model-specific.
3. **Downstream benchmarks** — MAUVE / HellaSwag / LAMBADA. Does ELBO even correlate with those?

### Path B: Address Reviewer-Level Confounds (solidifies current findings)
4. **2x2 {Muon, AdamW} × {constant, cosine} at 125M/10B, 3 seeds** — disentangles Muon vs schedule at scale. The definitive Muon claim.
5. **Mamba3 + Quokka-style block at 125M** — first clean head-to-head at matched scale. Script ready (`src/hybrid_model.py` QuokkaBlock).
6. **Moonlight RMS scaling probe** — would make Muon LR transferable across scales, fixes the ad-hoc 0.02→0.01 retune.

### Path C: Accept Limitations, Pivot Framing
7. **Write up as "MDLM training dynamics study"** — acknowledge generation limitation upfront.
   Defensible contributions: Muon config sensitivity for SSMs (cross-agent validated),
   embedding LR sensitivity, NS precision negative result, ELBO-generation decoupling.
   NOT defensible: "Muon wins" headlines, SwiGLU novelty (killed by ablation).

**Recommendation**: Do #3 (downstream benchmarks, cheap) to see if ANY metric correlates
with ELBO at our scale. If yes, push on Path B. If no, pivot to Path A or C.

## Coordination with Other Agent

The other agent (9070 XT, 16GB) uses Mamba3 Triton backend (different from our mamba-ssm).

Key cross-agent findings:
- **Muon vs Adam on Mamba3: RESOLVED — configuration issue, not hardware.** Our original Muon config lost to Adam by 0.23 nats. Other agent's config (in_proj only, lr=0.02, gamma=1.5, Adam 3e-4, wd=0.01) beats Adam by 0.011 nats on our same NVIDIA hardware. The "hardware difference" was actually 4 config differences compounding. Key insight from other agent: gamma≤2 is essential for Muon because heavy timestep reweighting creates gradient scale variance incompatible with NS normalization. NS precision (bf16/fp16/fp32) is irrelevant (tested, identical within 0.002 nats). Hybrid and Muon-VS tests pending.
- **NS steps x gamma interaction**: They found a stark ns×gamma interaction at 1K steps (n=1). We see a modest gamma effect (0.06-0.15 nats) but no reversal at 5K steps.
- Cosine vs constant LR: **constant wins for long training** (our finding)
- AdaLN: **marginal at 25M** (both agents now agree)
- **EVAL METRICS DIFFER ACROSS AGENTS.** We use Min-SNR gamma=5 weighted ELBO; they use 1/t ELBO. Absolute numbers are NOT comparable. They are running gamma=5 eval for cross-comparison.
- **Quokka architecture**: Their Mamba3 blocks include SwiGLU MLP sublayers, AdaLN, sum merge (not average). 4 layers vs our 6 — MLPs eat the param budget. Their best Mamba3 (Muon-VS, gamma=1.5, cosine): val 5.27 on 1/t ELBO (not comparable to our 5.27 Min-SNR).

## Prior Art (from review agent, 2026-04-13)

- **DiffuMamba/DiffuApriel (Singh et al., Nov 2025)**: Mamba-2 + MDLM at 240M-1.3B. Mamba trails at 240M, leads at 1.3B. Partially scoops our idea.
- **Mamba-3 (Lahoti et al., ICLR 2026)**: Complex-valued recurrence. We are first to use Mamba-3 as diffusion backbone.
- **WSD schedules (Hu et al., 2024; LLaDA)**: Constant-then-cooldown matches cosine. Supports our constant LR finding.
- **Muon (Shah et al., May 2025)**: Tested only on Transformers. Our SSM optimizer study is novel.

**Remaining defensible novelty**:
1. **Muon config sensitivity for SSMs** — gamma≤2 essential, param routing matters, cross-agent resolution showing config not hardware. **Cross-agent validated.**
2. **Muon NS precision negative result** — bf16/fp16/fp32 identical. Cleanly rules out precision as cross-hardware issue.
3. **Embedding LR sensitivity** — single hyperparameter flipped "Mamba wins" → "Transformer wins" at 30M. Scale-dependent (embeddings 64% of 30M, 31% of 125M).
4. **ELBO-generation decoupling** — multi-seed probe shows best-ELBO model generates worst text. Fundamental MDLM limitation, not a bug.

**Killed**:
- SwiGLU for MDLM (ablation: all gain is RoPE, which MDLM papers already use)
- "Mamba3 > Transformer" (artifact of undertrained embedding LR)
- "Muon beats AdamW" as headline (n=1, confounded with schedule, small effect)
- Generation quality as a win for any of our models (all worse than 4-gram Markov)

**Not yet reviewed**: 125M AR baseline, proper downstream benchmarks, Moonlight Muon scaling.
