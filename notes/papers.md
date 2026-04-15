# Research Papers

## Core: Masked Diffusion Language Models

### MDLM — Simple and Effective Masked Diffusion Language Models
- **Authors:** Sahoo et al. (2024)
- **arXiv:** [2406.07524](https://arxiv.org/abs/2406.07524)
- **Key idea:** Absorbing-state discrete diffusion for text. Tokens are progressively replaced with [MASK]; the model learns to predict original tokens from partially masked sequences. Simplified training objective = reweighted cross-entropy over masked positions.
- **Why it matters:** Establishes the masked diffusion framework we're building on. Clean, simple, effective.

### SEDD — Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
- **Authors:** Lou et al. (2024)
- **arXiv:** [2310.16834](https://arxiv.org/abs/2310.16834)
- **Key idea:** Score entropy loss for discrete diffusion — estimates ratios of data distribution rather than predicting tokens directly. Alternative to MDLM's cross-entropy approach.
- **Why it matters:** Competing framework to MDLM. Worth comparing approaches.

### D3PM — Structured Denoising Diffusion Models in Discrete State-Spaces
- **Authors:** Austin et al. (2021)
- **arXiv:** [2107.03006](https://arxiv.org/abs/2107.03006)
- **Key idea:** Foundational work on discrete diffusion with transition matrices (uniform, absorbing, discretized Gaussian). Introduced the absorbing-state process that MDLM simplifies.
- **Why it matters:** Theoretical foundation for all discrete diffusion LMs.

### Soft-Masked Diffusion Language Models
- **Authors:** Hersche, Moor-Smith, Hofmann, Rahimi (ICLR 2026)
- **arXiv:** [2510.17206](https://arxiv.org/abs/2510.17206)
- **Key idea:** Instead of binary masking (keep/replace), soft masking blends [MASK] embedding with top-k predicted token embeddings from the previous step. Preserves partial information across denoising steps.
- **Why it matters:** Drop-in improvement to any masked diffusion LM. Applied to Dream-7B models with consistent gains. We should implement this.

---

## Core: DiffuMamba (Most Directly Relevant)

### DiffuMamba — High-Throughput Diffusion LMs with Mamba Backbone
- **Authors:** Singh, Ostapenko, Noël, Belilovsky, Scholak (2025-2026)
- **arXiv:** [2511.15927](https://arxiv.org/abs/2511.15927)
- **Key idea:** Replaces the Transformer backbone in masked diffusion LMs with **bidirectional Mamba-2**. Two independent Mamba layers (forward + backward), combined via additive integration.
- **Architecture:**
  - Bidirectional Mamba-2 mixers (forward + reverse passes, additive merge)
  - MLP refinement with residual connections per block
  - DiffuMamba-H: hybrid with 1 attention layer per 5 Mamba blocks (~20% attention)
  - Mamba state dim: 128, MLP expansion: 2x (half of transformer's 4x)
- **Training:**
  - Absorbing-state masked diffusion (MDLM-style)
  - Adam (lr 1e-4 → 1e-6), weight decay 0.1, betas (0.9, 0.95), cosine annealing
  - DCLM dataset, GPT-2 tokenizer, 1024 context length
  - Log-linear noise schedule with antithetic sampling
  - Scales: 240M, 0.5B, 1.3B params
- **Results (1.3B):**
  - DiffuMamba-H: PPL 20.17 vs DiffuTran 22.72
  - 8.2x inference throughput (full-sequence), 2.3x with block caching
  - Matches or beats transformer-based diffusion on all 7 zero-shot benchmarks
- **Limitations:** No joint training with block cache reuse explored. Eval at batch size 1 only. No Mamba-3 or Muon optimizer explored.
- **Code:** Not publicly released as of Feb 2026.

**This is our direct baseline.** Our project improves on this by:
1. Upgrading Mamba-2 → Mamba-3 (complex states, MIMO, half state size)
2. Replacing Adam → Muon (faster convergence from nanogpt speedrun)
3. Adding soft masking (from Hersche et al.)
4. nanogpt-clean single-file implementation
5. Autoresearch methodology for systematic hyperparameter search

---

## Backbone: Mamba Family

### Mamba — Linear-Time Sequence Modeling with Selective State Spaces
- **Authors:** Gu & Dao (2023)
- **arXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)
- **Key idea:** Selective state space model — input-dependent state transitions enable content-based reasoning while maintaining linear complexity. O(N) in sequence length vs O(N²) for transformers.
- **Code:** [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

### Mamba-2 — Transformers are SSMs (State Space Duality)
- **Authors:** Dao & Gu (2024)
- **arXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)
- **Key idea:** Shows duality between structured state spaces and attention. Leads to more efficient algorithms and better hardware utilization. Used as backbone in DiffuMamba.

### Mamba-3 — Improved Sequence Modeling using State Space Principles
- **Authors:** Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu (ICLR 2026)
- **arXiv:** [2603.15569](https://arxiv.org/abs/2603.15569)
- **Key improvements over Mamba-2:**
  1. **More expressive recurrence** from SSM discretization
  2. **Complex-valued state updates** for richer state tracking
  3. **MIMO (multi-input, multi-output)** formulation — better accuracy without extra decode latency
- **Results (1.5B):** +1.8 points downstream accuracy over Gated DeltaNet. Matches Mamba-2 perplexity with **half the state size**.
- **Code:** `modules/mamba3.py` in [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
  - Install: `MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation`
  - Usage: `Mamba3(d_model=dim, d_state=128, headdim=64, is_mimo=True, mimo_rank=4, chunk_size=16, dtype=torch.bfloat16)`
- **Why it matters for us:** Half the state size = fits better on 16GB VRAM. MIMO = potential synergy with bidirectional processing. Complex states = better state tracking for denoising.

### Bidirectional Mamba Approaches
- **Vision Mamba (Vim):** Forward + backward SSM scanning, concat or add outputs. [github.com/hustvl/Vim](https://github.com/hustvl/Vim) (ICML 2024)
- **Caduceus:** Bidirectional Mamba for DNA sequences. [caduceus-dna.github.io](https://caduceus-dna.github.io/)
- **DiffuMamba approach:** Two independent Mamba layers (fwd + bwd), additive merge — simplest and most effective for diffusion LMs
- **BiMamba:** Generic bidirectional wrapper. [github.com/CiaoHe/bi-mamba](https://github.com/CiaoHe/bi-mamba)

---

## Optimizer: Muon

### Muon Optimizer — MomentUm Orthogonalized by Newton-Schulz
- **Authors:** Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, Jeremy Bernstein
- **Blog:** [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- **Code:** [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon) (2.4k stars)
- **PyTorch native:** `torch.optim.Muon` (v2.9+)
- **No formal paper** — Keller Jordan: "I only trust speedruns"

**How it works:**
1. Compute SGD-momentum (Nesterov) on gradients
2. Post-process momentum matrix with Newton-Schulz (NS) iteration to orthogonalize
3. NS approximates UV^T from SVD — finds the nearest semi-orthogonal matrix
4. Only applies to 2D weight matrices; embeddings/biases/layernorm use AdamW

**Why it helps:** Adam updates have very high condition numbers — nearly low-rank. Orthogonalization amplifies underrepresented "rare directions" that may be important for learning.

**nanogpt speedrun impact:** Single largest contributor — 1.35x faster. Persisted through 12 consecutive speed records. GPT-2 124M training: 45min → ~90sec.

**Scaling validated:**
- Moonlight (Moonshot AI): 3B/16B MoE on 5.7T tokens — 2x compute efficiency vs AdamW
- Kimi K2: 1T params, 15.5T tokens, zero loss spikes (used MuonClip variant)
- Essential AI: validated to 4B scale, retains efficiency at large batch sizes

**Key variants:** AdaMuon, NorMuon, Newton-Muon, Flash-Muon (Triton kernels), MuonClip

### !! CRITICAL WARNING: Muon + Diffusion Models !!

**Standard Muon FAILS for image diffusion training.** In the "Speedrunning ImageNet Diffusion" benchmark ([arxiv 2512.12386](https://arxiv.org/abs/2512.12386)), Muon performed **catastrophically badly** — plateaued early, FID 48.70 (very poor). Adam consistently overtook Muon by 400K iterations.

**HOWEVER**, masked diffusion LMs may be different:
- MDLM training objective = **cross-entropy over masked positions** (not noise prediction)
- Structurally more like MLM/language model training (where Muon excels)
- DiffuMamba uses Adam — nobody has tried Muon on masked diffusion LMs yet
- **This is a key research question for our project**

**Hypothesis:** Muon should work for masked diffusion LMs because the loss landscape resembles language modeling, not continuous diffusion. Testing this is one of our novel contributions.

---

## Latest Diffusion LM Papers (2025-2026) — The Field is Exploding

### LLaDA — Large Language Diffusion Models (Feb 2025)
- **arXiv:** [2502.09992](https://arxiv.org/abs/2502.09992)
- **Key:** First large-scale masked diffusion LM at **8B params** with full pre-train + SFT pipeline.
  Competitive with LLaMA3 8B on in-context learning. Solves the "reversal curse" that trips AR models.
- **Why it matters:** Proves masked diffusion scales to the full LLM paradigm.

### Scaling Beyond MDLM (Feb 2025)
- **arXiv:** [2602.15014](https://arxiv.org/abs/2602.15014) (same group as MDLM)
- **Key:** First scaling laws for discrete diffusion. Masked diffusion can be ~12% more FLOPs-efficient
  with simple CE objective. At 1.7B, uniform diffusion beats AR on GSM8K despite worse perplexity.
- **Code:** [github.com/s-sahoo/scaling-dllms](https://github.com/s-sahoo/scaling-dllms)

### Efficient-DLM — AR to Diffusion Conversion (Dec 2025)
- **arXiv:** [2512.14067](https://arxiv.org/abs/2512.14067)
- **Key:** Convert pretrained AR models to diffusion LMs. Block-wise attention (causal across blocks,
  bidirectional within). 8B model achieves +5.4% accuracy, 4.5x throughput vs Dream 7B.

### Corrective Diffusion Language Models (Dec 2025)
- **arXiv:** [2512.15596](https://arxiv.org/abs/2512.15596)
- **Key:** Standard MDLMs can't revise incorrect tokens (only unmask). Post-training principle
  that teaches error correction. Important for iterative decoding quality.

### dUltra — Ultra-Fast Diffusion LMs via RL (Dec 2025)
- **arXiv:** [2512.21446](https://arxiv.org/abs/2512.21446)
- **Key:** RL framework (GRPO) for learning efficient unmasking strategies. Most MDLMs decode
  <5 tokens per step even with sophisticated sampling. Unmasking planner head fixes this.

### Dream-Coder 7B (Sep 2025)
- **arXiv:** [2509.01142](https://arxiv.org/abs/2509.01142)
- **Key:** Diffusion LM for code. Emergent any-order generation — sketch-first for algorithms,
  left-to-right for simple completions. 21.4% pass@1 on LiveCodeBench.

### April 2026 — Active Papers

| Paper | Focus |
|-------|-------|
| DMax (2604.08302) | Aggressive parallel decoding |
| DualDiffusion (2604.05250) | Speculative decoding for MDLMs |
| DARE (2604.04215) | RLHF alignment for diffusion LMs |
| Dependency-Guided Decoding (2604.02560) | Speed |
| Model Scheduling (2604.02340) | Not all denoising steps are equal |

---

## Muon Optimizer Variants (2025-2026) — Active Research Area

### AdaMuon — Adaptive Muon Optimizer
- **Authors:** (Multiple authors)
- **arXiv:** [2507.11005](https://arxiv.org/abs/2507.11005) (July 2025)
- **Key idea:** Augments Muon with element-wise second-moment modulation that captures orthogonal gradient updates, plus RMS-aligned rescaling. Two mutually dependent modules.
- **Results:** Consistently outperforms original Muon. Claims >40% training efficiency gain over Adam in large-scale scenarios.
- **Relevance:** Interaction with Min-SNR loss weighting is untested. The element-wise adaptivity could help or hurt on masked diffusion.

### NorMuon — Making Muon More Efficient and Scalable
- **Authors:** (Multiple authors)
- **arXiv:** [2510.05491](https://arxiv.org/abs/2510.05491) (October 2025)
- **Key idea:** Neuron-wise adaptive learning rates computed from accumulated second-order statistics, applied after NS orthogonalization. Combines advantages of Muon (low condition number) and AdamW (uniform neuron norms).
- **Results:** 21.74% better training efficiency than Adam, 11.31% over Muon at 1.1B scale.
- **Relevance:** Neuron-wise normalization may help with timestep-varying gradient scales in masked diffusion.

### Mousse — Rectifying the Geometry of Muon with Curvature-Aware Preconditioning
- **Authors:** (Multiple authors)
- **arXiv:** [2603.09697](https://arxiv.org/abs/2603.09697) (March 2026)
- **Key idea:** Preconditions gradient with Shampoo's Kronecker-factored curvature before NS orthogonalization. Addresses Muon's isotropic trust region assumption which ignores curvature disparities.
- **Results:** 12% fewer training steps, 3% wall-clock overhead vs standard Muon. Tested 160M-800M.
- **Relevance:** Most principled Muon improvement. Curvature-aware preconditioning could be especially valuable for masked diffusion where loss landscape varies with timestep.

### Newton-Muon — The Newton-Muon Optimizer
- **Authors:** Zhehang Du, Weijie Su
- **arXiv:** [2604.01472](https://arxiv.org/abs/2604.01472) (April 2026)
- **Key idea:** Standard Muon neglects right preconditioning from input second moments. Newton-Muon adds this via a surrogate quadratic model using gradient, output curvature, and input data matrix.
- **Results:** 6% fewer iteration steps, 4% less wall-clock time on GPT-2 pretraining.
- **Relevance:** Very recent. Minimal validation so far.

### Variance-Adaptive Muon — NSR-Modulated and Variance-Scaled Momentum
- **Authors:** (Multiple authors)
- **arXiv:** [2601.14603](https://arxiv.org/abs/2601.14603) (January 2026)
- **Key idea:** Two variants: Muon-NSR (noise-to-signal ratio modulation) and Muon-VS (parameter-free variance scaling). Both apply variance statistics before NS step.
- **Results:** 1.36x fewer iterations to target loss on LLaMA-1.2B vs well-tuned Muon.
- **Relevance:** Muon-VS is parameter-free — attractive for autoresearch. The 1.36x claim on LLaMA is strong.

### MuonAll — Muon Variant for Efficient Finetuning
- **arXiv:** [2511.06086](https://arxiv.org/abs/2511.06086) (November 2025)
- **Key idea:** Extends Muon to 1D parameters by reshaping to diagonal matrices, running NS, reshaping back. Removes dependency on AdamW for non-2D params.
- **Results:** "At par" with AdamW across benchmarks for finetuning. NOT shown to be better than Muon+AdamW hybrid.
- **Relevance:** Low. At-par results mean no improvement expected.

---

## Noise Schedule & Inference Optimization

### Optimal Inference Schedules for Masked Diffusion Models
- **Authors:** Sitan Chen, Kevin Cong, Jerry Li (Harvard)
- **arXiv:** [2511.04647](https://arxiv.org/abs/2511.04647) (November 2025)
- **Key idea:** Rigorously quantifies statistical errors from parallel token sampling. Derives optimal unmasking schedule depending on data distribution and target error.
- **Results:** Cosine schedule arises as Fisher-Rao-geodesic optimal. Log-linear is a reasonable approximation. Optimal schedule is data-dependent.
- **Relevance:** Mostly about inference scheduling, not training. Confirms our log-linear training schedule is reasonable.

### Scaling Behavior of Discrete Diffusion Language Models
- **arXiv:** [2512.10858](https://arxiv.org/abs/2512.10858) (December 2025)
- **Key idea:** Scaling behavior of discrete diffusion LMs "strongly depends on noise type and is considerably different from autoregressive language models." Examines batch size and LR interactions.
- **Relevance:** Important context for our scaling experiments. Batch size affects diffusion differently than AR.

### Small Batch Size Training for Language Models
- **arXiv:** [2507.07101](https://arxiv.org/abs/2507.07101) (July 2025)
- **Key idea:** Small batches train stably, are more robust to hyperparameters, and achieve "equal or better per-FLOP performance." Gradient accumulation is "wasteful" for single-GPU.
- **Relevance:** Argues against gradient accumulation for our setup. Our batch_size=8 may already be near-optimal.

---

## Hybrid Mamba-Attention (Encoder-Specific)

### Nemotron-H — Family of Accurate, Efficient Hybrid Mamba-Transformer Models
- **arXiv:** [2504.03624](https://arxiv.org/abs/2504.03624) (April 2025)
- **Key finding:** 7-8% attention optimal for CAUSAL decoders. 4 attn layers out of 52 total at 8B. Up to 3x inference speedup.
- **Relevance:** Sets the baseline for causal models. Our bidirectional model likely needs MORE attention.

### Nemotron-3 Super — Open Hybrid MoE for Agentic Reasoning
- **From:** NVIDIA, March 2026
- **Key finding:** 120B hybrid Mamba-Attention MoE with 5x throughput. Extends Nemotron-H approach to MoE.

### CRITICAL: Encoder vs Decoder Attention Requirements
Literature consensus for optimal attention ratio:
- **Causal decoder:** 7-12% (Nemotron-H, Jamba)
- **Bidirectional encoder:** 20-33% (DiffuMamba-H, MaBERT)
- Our model is bidirectional (encoder-like), so we should target the higher range.

---

## Methodology: Autoresearch

### Karpathy Autoresearch Concept
- AI-assisted systematic exploration of hyperparameters and architecture choices
- Train -> evaluate -> analyze -> propose changes -> repeat
- The "test" in our context: achieve target perplexity/accuracy as fast as possible
- Metrics: wall-clock time to reach target perplexity, final perplexity at fixed compute budget
