# Wide Exploration: Research-Backed Evaluation

> Generated: 2026-04-12
> Context: Muon beats Adam by 0.34 nats (t=40, p<0.001) at 5k steps on quokka 31.5M.
> Hardware: AMD RX 9070 XT, ~58k tok/s (Mamba3 Triton), ~500 screen runs/day budget.
> Strategy: Go wide -- many cheap 1k-step screens, validate winners at 5k with 3 seeds.

---

## A. Architecture Variants

### A1. Hybrid Attention Ratios

**Literature:**

The optimal attention ratio for Mamba hybrids depends heavily on whether the
model is an ENCODER (bidirectional) or a DECODER (causal). This is a critical
distinction the literature makes clear:

| Paper | Architecture | Ratio | Context |
|-------|-------------|-------|---------|
| NVIDIA Nemotron-H (2504.03624) | Causal decoder, 8B-56B | 7-8% attention | 4/52 attn layers at 8B, 10/118 at 56B |
| Jamba (ICLR 2025) | Causal decoder, 52B | 1:7 (~12.5%) | Every 8th layer is attention |
| DiffuMamba-H (2511.15927) | Bidi diffusion, 240M-1.3B | 1:5 (~20%) | 1 attn per 5 Mamba blocks |
| MaBERT (2603.03001) | Bidi encoder (MLM), 12L | 2:1 (~33%) | MMTMMTMMTMMT pattern |
| MambaVision (CVPR 2025) | Vision encoder | ~25% | Final stages use attention |

**Key insight: encoder models need MORE attention than decoders.** MaBERT
explicitly tested 8 interleaving patterns on a 12-layer encoder and found that
MMTMMTMMTMMT (33% attention) beats all other patterns including all-Mamba,
all-Transformer, and sparser attention schedules. Single-family architectures
"consistently underperform mixed schedules."

For causal decoders, 7-12% attention suffices (NVIDIA, Jamba). But for
bidirectional models -- which is what we are building -- the evidence points
to 20-33%. DiffuMamba-H's 20% is likely conservative. MaBERT's 33% is the
only result from a controlled ENCODER ablation.

However, our quokka config has only 4 layers, which limits pattern options:
- 25% = 1 attn layer = [A,M,M,M] or [M,M,M,A]
- 50% = 2 attn layers = [M,A,M,A] or [A,M,A,M]
- 33% is impossible with 4 layers (would need 6+ layers)

At 8 layers (small config), we can test 25% [M,M,A,M,M,A,M,M] and 37.5%
[M,A,M,M,A,M,M,A] which brackets MaBERT's 33%.

**Expected effect size:** DiffuMamba-H at 1.3B beat pure DiffuMamba on 5/7
benchmarks and improved PPL from ~22 to ~20 (a substantial gain). At our
scale (31.5M), the effect will be smaller but should still be measurable.
Expect 0.05-0.2 nat improvement at 1k steps.

**Implementation:** ~150 LOC for BiAttentionBlock. Already designed in the
architecture_variants proposal. PyTorch SDPA dispatches to flash attention
on RDNA4 via Triton backend.

**Priority: HIGH.** This is the single most architecturally impactful
direction. The literature is clear that some attention helps for encoder
models, and we have zero attention layers currently. The question is how
much, not whether.

**Recommended screen runs:**
1. baseline (all Mamba, 4L) -- control
2. hybrid-25 attn-first: [A,M,M,M] -- DiffuMamba-H style
3. hybrid-25 attn-last: [M,M,M,A] -- test position sensitivity
4. hybrid-50: [M,A,M,A] -- upper bound
5. At small 8L: hybrid-25 [M,M,A,M,M,A,M,M] and hybrid-37 [M,A,M,M,A,M,M,A]

Cost: 5 screen runs at quokka = ~12 minutes. Negligible.

---

### A2. Weight Tying Forward/Backward

**Literature:**

Caduceus (Schiff et al., ICML 2024) directly ablated this. Their BiMamba ties
forward and backward projection weights, and the ablation showed that "the
parameter efficient implementation of bi-directionality leads to better
pre-training loss" compared to naive untied bidirectional Mamba. The reason:
weight tying enables deeper models at the same parameter count, and depth
matters more than per-layer capacity for SSMs.

JanusDNA (2505.17257, 2025) builds on Caduceus and also uses weight-tied
bidirectional Mamba, confirming the approach scales.

**Expected effect size:** Caduceus showed better MLM loss with tying. For us,
tying halves the Mamba parameters per block (~50% of block params), freeing
budget for either more depth or wider d_model. At fixed param count, the
wider/deeper model should win by 0.05-0.15 nats.

**Implementation:** Trivial -- 2 lines in BiMamba3Block.__init__:
`self.mamba_bwd = self.mamba_fwd` and reverse the input in forward.

**Priority: HIGH.** Near-zero implementation cost, directly validated by
Caduceus ablation, and gives us more params to play with. Should screen
both (a) tied at same architecture, and (b) tied + wider d_model.

**Recommended screen runs:**
1. baseline untied (current)
2. tied same d_model -- pure tying effect
3. tied + wider d_model (e.g., d_model=448 at ~same param count) -- reinvest savings

Cost: 3 screen runs = ~7 minutes.

---

### A3. Multiplicative Gating vs Additive Merge

**Literature:**

BiGS (Wang et al., 2022, arXiv 2212.10544) used multiplicative gating
(element-wise product of forward and backward SSM outputs) and matched
BERT on GLUE. DiffuMamba (2511.15927) uses additive merge (h_fwd + h_bwd)
and is the current SOTA for diffusion LMs with Mamba.

No direct ablation comparing additive vs multiplicative for Mamba-based
bidirectional models exists in the literature. BiGS used S4 (not Mamba),
so the comparison is not apples-to-apples. The multiplicative approach is
motivated by GLU-style gating -- it allows one direction to gate the other,
potentially being more expressive. However, additive merge is simpler and
what the more recent papers (DiffuMamba, Vim) converged on.

**Expected effect size:** Unknown. Could go either way. The BiGS paper does
not ablate this choice against additive. This is a genuine unknown.

**Implementation:** ~5 lines: change `h_fwd + h_bwd` to `h_fwd * h_bwd`.
Need to be careful with initialization -- multiplicative gating can cause
gradient issues if both directions produce large values.

**Priority: MEDIUM.** Very cheap to test (~1 line change), but no literature
guidance on expected direction. Worth a screen run purely because it's free.

**Recommended screen runs:**
1. additive (current baseline)
2. multiplicative (element-wise product)
3. gated: `sigmoid(W_gate @ h_fwd) * h_bwd + (1 - sigmoid(W_gate @ h_fwd)) * h_fwd`
   (learnable gate, ~d_model extra params)

Cost: 3 screen runs = ~7 minutes.

---

### A4. MLP Expansion Ratio

**Literature:**

DiffuMamba uses 2x MLP expansion (following the convention that Mamba's
internal expansion already provides capacity, unlike Transformers which need
4x MLP). Mamba-3 (ICLR 2026) continues using 2x. Nemotron-H interleaves
separate MLP layers with standard expansion. Jamba uses MoE MLP layers
at 8x expansion but only activates a fraction.

The Mamba-3 paper states they follow "standard Transformer conventions"
for the interleaved MLP layers, which typically means 4x expansion with
SwiGLU (net ~2.67x effective expansion due to the gate). Our 2x SwiGLU
gives ~1.33x effective expansion, which is conservative.

No systematic ablation of MLP expansion ratio exists for Mamba diffusion
models. For Transformers, Scaling MLPs (Fedus et al., 2021) showed that
wider MLPs help up to a point, but the optimal ratio depends on the
attention-to-MLP parameter budget balance.

**Expected effect size:** Small. At our scale (31.5M, 4 layers), the MLP
is already a minority of parameters. Going from 2x to 4x would roughly
double MLP parameters but only increase total params by ~15-20%. The gain
is likely 0.02-0.05 nats -- within noise at 1k steps.

**Implementation:** 1 line: change `mlp_expansion=2` to `mlp_expansion=4`
in config.

**Priority: LOW.** Cheap to test but small expected effect. Better to
focus on attention ratio and weight tying first.

**Recommended screen runs:** 1 run with 4x expansion. Cost: ~2.5 minutes.

---

### A5. Width vs Depth at Fixed Param Count

**Literature:**

The Goomba Lab "Tradeoffs of SSMs and Transformers" blog (2025) tested
this for byte-level LMs and found SSMs benefit more from depth than width.
For Transformers, Scaling Laws (Kaplan et al., 2020) found depth matters
more than width at small scale, but the relationship inverts at large scale.

For Mamba specifically, the state dimension and number of scans benefit from
depth because each layer refines the hidden state. The DiffuMamba paper
tested at 240M, 0.5B, and 1.3B but always scaled both width and depth
together, never ablating the ratio.

**Expected effect size:** Moderate at our scale. Going from 4L/384d to
6L/320d (roughly iso-param) should show a measurable difference. But the
direction is uncertain -- Mamba's recurrent structure may favor depth more
than Transformers do, but our 4-layer model is already very shallow.

**Implementation:** Zero code changes, just config.

**Priority: MEDIUM.** Easy to test, informative for scaling decisions, but
the result may not transfer to larger scales.

**Recommended screen runs:**
1. 4L x 384d (current quokka)
2. 6L x 320d (~same params)
3. 3L x 448d (~same params)
4. 8L x 256d (~same params)

Cost: 4 screen runs = ~10 minutes.

---

## B. Optimizer Variants

### B1. AdaMuon

**Literature:**

AdaMuon (arXiv 2507.11005, July 2025) augments Muon with element-wise
second-moment modulation and RMS-aligned rescaling. Tested at 160M-1.3B
on standard LM pretraining.

Key claims:
- "Consistently outperforms the original Muon"
- "More than 40% training efficiency gain over Adam in large-scale scenarios"
- Captures "orthogonal gradient updates to ensure update-level adaptivity"

**BUT:** These results are on standard causal LM pretraining with cross-entropy
loss. Our setting is masked diffusion, which has different gradient statistics
(timestep-varying masking rates, Min-SNR weighting). The interaction between
AdaMuon's element-wise adaptivity and our gamma=1.5 Min-SNR weighting is
completely untested.

**Expected effect size:** If AdaMuon's gains transfer to masked diffusion,
expect 5-15% faster convergence (0.05-0.15 nat improvement at same step
count). But given Muon's known sensitivity to loss weighting in our setting
(the gamma=1.5 finding), AdaMuon's additional adaptivity could help OR hurt.

**Implementation:** ~30 LOC in the optimizer. Need to add second-moment
tracking and RMS rescaling before NS orthogonalization.

**Priority: MEDIUM-HIGH.** The paper claims substantial gains and it's
reasonably easy to implement. But the interaction with Min-SNR loss
weighting is uncertain.

**Recommended screen runs:**
1. Muon baseline (gamma=1.5)
2. AdaMuon (gamma=1.5)
3. AdaMuon (gamma=5) -- test if AdaMuon changes optimal gamma

Cost: 3 screen runs + ~30 min implementation = worthwhile.

---

### B2. NorMuon

**Literature:**

NorMuon (arXiv 2510.05491, October 2025) adds neuron-wise adaptive learning
rates computed from accumulated second-order statistics. Tested at up to 1.1B.

Key claims:
- "21.74% better training efficiency than Adam"
- "11.31% improvement over Muon"
- "Uniform neuron norms" combining advantages of Muon and AdamW

NorMuon's neuron-wise normalization is conceptually interesting for our
setting: if different neurons learn at different rates due to the timestep-
varying masking, NorMuon could naturally adapt. The "uniform neuron norms"
property aligns with the EGD/equalization principle that seems to benefit
our masked diffusion training.

**Expected effect size:** Similar to AdaMuon -- 5-15% convergence speedup
if it transfers. NorMuon's neuron-wise approach may interact better with
Min-SNR weighting than AdaMuon's element-wise approach, but this is
speculative.

**Implementation:** ~40 LOC. Track neuron-wise second moments, normalize
after NS iterations.

**Priority: MEDIUM.** Similar rationale to AdaMuon. The 11% gain over Muon
is attractive, but unvalidated on diffusion. Test after AdaMuon if AdaMuon
shows promise.

---

### B3. MuonClip

**Literature:**

MuonClip (Kimi K2, arXiv 2507.20534, July 2025) adds QK-clip to prevent
attention logit explosion during training. Used to train Kimi K2 (1T params,
15.5T tokens) with zero loss spikes.

Key mechanism: After each Muon step, check max QK attention score. If it
exceeds threshold t, rescale W_q by eta^alpha and W_k by eta^(1-alpha)
where eta = t / max_score.

**Relevance to our setting:** We currently have NO attention layers. MuonClip
is only relevant if we add hybrid attention (A1). Even then, at 31.5M params
and 5k steps, attention logit explosion is unlikely. MuonClip matters at
scale (billions of params, millions of steps).

**Expected effect size:** Zero at current scale. Potentially important at
base (231M) or larger with hybrid attention.

**Implementation:** ~20 LOC, but needs attention layers first.

**Priority: LOW.** Not relevant until we (a) add attention layers AND (b)
scale up significantly. Defer.

---

### B4. Per-Layer Muon LR

**Literature:**

LeRaC (Learning Rate Curriculum, 2024) assigns higher LR to layers closer
to the input, based on the finding that "shallower layers converge faster
than deeper layers." LARS/LAMB (layer-wise adaptive) are standard for large
batch training.

For Muon specifically, no paper has tested per-layer LR. However, Muon
already has built-in muP-like scaling (the `max(1, m/n)^0.5` factor after
NS orthogonalization), which provides some implicit per-layer adaptation.

**Expected effect size:** Small. Muon's orthogonalization already equalizes
update magnitudes across layers. Adding per-layer LR on top may have
diminishing returns. Expect < 0.05 nat difference.

**Implementation:** ~10 LOC: scale LR by layer index in the param group
construction.

**Priority: LOW.** Easy to implement but small expected effect given Muon's
built-in scaling. Not worth a screen run over higher-priority items.

---

### B5. MuonAll (Extend to All Parameters)

**Literature:**

MuonAll (arXiv 2511.06086, November 2025) extends Muon to 1D parameters
by reshaping them to diagonal matrices, running NS, and reshaping back.
Tested for finetuning up to 500M params.

Key finding: "MuonAll performing at par with AdamW across major benchmarks"
for finetuning. Note: "at par" not "better than." The paper shows MuonAll
matches but does not clearly beat the Muon+AdamW hybrid for pretraining.

**Expected effect size:** Minimal. Our auxiliary Adam parameters (embeddings,
norms, AdaLN, biases) are a small fraction of total params (~3-5M out of
31.5M). Optimizing them with Muon instead of Adam is unlikely to make a
meaningful difference. The embedding layer specifically should stay with
Adam per Keller Jordan's recommendation.

**Implementation:** ~30 LOC for diagonal reshaping.

**Priority: LOW.** The literature shows at-par performance, not improvement.
Not worth the implementation effort for a null result.

---

### B-EXTRA. Newer Muon Variants Worth Tracking

Three additional Muon variants have appeared in 2026 that are more promising
than the ones in the original proposal list:

**Mousse** (arXiv 2603.09697, March 2026): Curvature-aware preconditioning.
Preconditions gradient with Shampoo's Kronecker-factored curvature before
NS orthogonalization. Claims 12% fewer steps at 3% wall-clock overhead.
160M-800M scale. **This is the most principled improvement over base Muon.**

**Newton-Muon** (arXiv 2604.01472, April 2026): Right preconditioning from
input second moments. 6% fewer iterations, 4% less wall-clock time on
GPT-2 pretraining. Very recent, minimal validation.

**Variance-Adaptive Muon** (arXiv 2601.14603, January 2026): NSR-modulated
and variance-scaled momentum before orthogonalization. 1.36x faster
convergence on LLaMA-1.2B. Two variants: Muon-NSR (with hyperparameter)
and Muon-VS (parameter-free).

**Recommendation:** If testing Muon variants, prioritize Mousse (most
principled, 12% gain) and Muon-VS (parameter-free, 1.36x on LLaMA).
Skip AdaMuon and NorMuon unless Mousse fails.

---

## C. Training Recipe

### C1. Noise Schedule Alternatives

**Literature:**

The log-linear schedule (current) gives `move_chance(t) = (1-eps)*t` --
masking probability is linear in t. Alternatives:

**Cosine schedule** (from MaskGIT): Slows unmasking at the start of reverse
generation. The cosine schedule arises as the Fisher-Rao-geodesic optimal
schedule in the space of probability distributions (Chen et al., arXiv
2511.04647, November 2025). This is a theoretical optimality result for
INFERENCE scheduling, not training.

**Geometric schedule** (from SEDD, Lou et al.): First used for continuous
diffusion, adapted for discrete. No direct comparison to log-linear for
masked diffusion LMs.

**Key paper:** "Optimal Inference Schedules for Masked Diffusion Models"
(Chen, Cong, Li, arXiv 2511.04647) rigorously quantifies the error from
parallel token sampling and derives optimal unmasking schedules. The result:
the optimal schedule depends on the data distribution, and heuristics like
cosine and log-linear are both reasonable approximations.

**For training:** "Scaling Beyond MDLM" (arXiv 2602.15014) compared ELBO
(1/t weighting) vs MaskGIT (uniform weighting) and found MaskGIT converges
faster initially but ELBO achieves better final performance. Our Min-SNR
gamma=1.5 is a principled middle ground.

**Expected effect size:** Small for training schedule changes. The noise
schedule primarily affects inference quality. For training, our Min-SNR
gamma=1.5 already addresses the main issue (gradient conflict across
timesteps). Changing the noise schedule itself is likely < 0.05 nats.

**Implementation:** ~20 LOC for cosine/geometric alternatives.

**Priority: LOW for training, MEDIUM for inference.** The training noise
schedule is already well-tuned via Min-SNR. Cosine inference schedule is
worth testing when we have text generation working (the 5090 agent's domain).

---

### C2. WSD Schedule (Warmup-Stable-Decay)

**Literature:**

WSD (Warmup-Stable-Decay) has become popular for LLM pretraining (ICLR
2025 paper, arXiv 2410.05192). Key properties:
- Constant LR during "stable" phase enables indefinite training without
  pre-specified compute budget
- Sharp loss decline during decay phase
- "River valley" loss landscape theory explains why it works
- Eliminates logarithmic slowdowns, achieves lower final risk than cosine
- Optimal under functional scaling laws (arXiv 2602.06797)

**Relevance:** We currently use cosine schedule. WSD's advantage is that
it does not require knowing max_steps in advance -- you train at constant
LR and decay when you want to stop. This is ideal for autoresearch where
we want to branch off checkpoints at different step counts.

However, at 1k-5k steps with warmup=200, the schedule differences are small.
WSD's advantages emerge at longer training horizons (100k+ steps).

**Expected effect size:** < 0.03 nats at 5k steps. Cosine and WSD are
nearly identical for short runs. The benefit is operational (no need to
set max_steps) rather than performance.

**Implementation:** ~15 LOC. Trivial.

**Priority: LOW for short runs, HIGH for infrastructure.** If we plan to
do any long training (10k+ steps), WSD is strictly better than cosine for
operational flexibility. Worth implementing but not worth a screen run --
just switch the default.

---

### C3. Gradient Accumulation

**Literature:**

"Small Batch Size Training for Language Models" (arXiv 2507.07101, 2025)
challenges the conventional wisdom that larger batches are better:
- "Small batch sizes train stably, are consistently more robust to
  hyperparameter choices"
- "Equal or better per-FLOP performance than larger batch sizes"
- Recommends "using the smallest batch size that maximizes model throughput"
- Gradient accumulation is "wasteful" unless bandwidth-limited on multi-GPU

"Scaling Behavior of Discrete Diffusion Language Models" (arXiv 2512.10858)
found that diffusion LM scaling behavior "strongly depends on noise type
and is considerably different from autoregressive language models."

**Relevance:** We use batch_size=8 at seq_len=1024 (8192 tokens/step).
Gradient accumulation to simulate batch_size=32 or 64 would 4-8x the
effective batch. But the literature suggests this may not help and could
hurt (less frequent updates at same compute).

For diffusion models specifically, the batch size affects the diversity
of timesteps seen per update. Larger batches sample more timesteps per
step, which could reduce variance. But our antithetic sampling already
achieves good timestep coverage at batch_size=8.

**Expected effect size:** Likely neutral or slightly negative per the
recent literature. The critical batch size for 31.5M params is probably
well below 8192 tokens.

**Implementation:** ~5 LOC (accumulate gradients, step every N batches).

**Priority: LOW.** The literature suggests this is actively harmful for
single-GPU training. Not worth a screen run. If anything, we should try
SMALLER batches (bs=4).

---

### C4. Time Conditioning ON (Re-test on Mamba3 Triton)

**Literature:**

DiffuMamba uses time conditioning (concatenated timestep token). MDLM
defaults to time conditioning OFF. Our prior test showed time conditioning
ON was +1.3% better, but that was on PureSSM backend.

The HANDOFF notes that time conditioning is currently OFF because AdaLN
receives zeros, making it a no-op. Re-testing on Mamba3 Triton is needed
to see if the +1.3% holds with the faster backend.

**Expected effect size:** 0.05-0.10 nats based on the prior PureSSM result
(+1.3%). May differ on Mamba3 Triton due to different numerical behavior.

**Implementation:** Zero -- just remove `--no_time_cond` flag.

**Priority: MEDIUM.** Free to test (no code changes), and the prior result
was marginally positive. Worth one screen run to confirm or reject.

**Recommended screen runs:**
1. time_cond OFF (current best)
2. time_cond ON

Cost: 2 screen runs = ~5 minutes.

---

## Priority Rankings (Consolidated)

### Tier 1: Screen immediately (high expected value, low cost)

| Direction | Est. effect | Impl. cost | Screen cost | Rationale |
|-----------|------------|------------|-------------|-----------|
| A1. Hybrid attention ratios | 0.05-0.20 nat | ~150 LOC | 5 runs, 12 min | Literature unanimously supports; encoder needs >20% attn |
| A2. Weight tying fwd/bwd | 0.05-0.15 nat | 2 LOC | 3 runs, 7 min | Caduceus ablation directly validates; near-free |
| A3. Multiplicative gating | unknown | 5 LOC | 3 runs, 7 min | Unknown direction but ~free to test |
| C4. Time cond ON re-test | 0.05-0.10 nat | 0 LOC | 2 runs, 5 min | Prior +1.3% on PureSSM; free to verify |

**Total Tier 1: ~13 runs, ~31 minutes, ~155 LOC implementation.**

### Tier 2: Screen if Tier 1 results are clear (moderate value)

| Direction | Est. effect | Impl. cost | Screen cost | Rationale |
|-----------|------------|------------|-------------|-----------|
| B-extra. Mousse optimizer | 0.10-0.20 nat | ~60 LOC | 3 runs, 7 min | 12% fewer steps; curvature-aware, principled |
| B-extra. Muon-VS | 0.10-0.15 nat | ~30 LOC | 3 runs, 7 min | 1.36x faster on LLaMA; parameter-free |
| B1. AdaMuon | 0.05-0.15 nat | ~30 LOC | 3 runs, 7 min | 40% over Adam claimed; interaction w/ gamma unknown |
| A5. Width vs depth | 0.05-0.15 nat | 0 LOC | 4 runs, 10 min | Informative for scaling; SSMs may favor depth |

**Total Tier 2: ~13 runs, ~31 minutes, ~120 LOC implementation.**

### Tier 3: Lower priority (small expected effect or deferred)

| Direction | Est. effect | Impl. cost | Screen cost | Rationale |
|-----------|------------|------------|-------------|-----------|
| A4. MLP expansion 4x | 0.02-0.05 nat | 1 LOC | 1 run, 2.5 min | Small effect; conservative 2x may be fine |
| C2. WSD schedule | < 0.03 nat | 15 LOC | 1 run, 2.5 min | Operational benefit > performance; implement for infra |
| B2. NorMuon | 0.05-0.15 nat | ~40 LOC | 3 runs, 7 min | Test only if Mousse/AdaMuon fail |
| C1. Noise schedule | < 0.05 nat | 20 LOC | 2 runs, 5 min | More relevant for inference; training already tuned |

### Not recommended at this time

| Direction | Reason |
|-----------|--------|
| B3. MuonClip | No attention layers yet; only matters at scale |
| B4. Per-layer Muon LR | Muon already has built-in scaling; tiny expected effect |
| B5. MuonAll | Literature shows at-par, not better; waste of screen run |
| C3. Gradient accumulation | Literature says actively harmful for single-GPU |

---

## Recommended Day Plan

### Block 1: Implementation (1 hour)

1. Implement BiAttentionBlock for hybrid attention (A1) -- ~150 LOC
2. Implement weight tying flag (A2) -- 2 LOC
3. Implement multiplicative gating option (A3) -- 5 LOC
4. Total: ~160 LOC, all in model.py

### Block 2: Tier 1 Screens (45 minutes)

Run all 13 Tier 1 screen experiments (1k steps each, quokka, single seed):

```
# A1: Hybrid attention
baseline                          # control
hybrid-25-first [A,M,M,M]        # attn at position 0
hybrid-25-last  [M,M,M,A]        # attn at position 3
hybrid-50       [M,A,M,A]        # 50% attention

# A2: Weight tying
tied-same-d                       # tied at d=384
tied-wider      d=448             # reinvest param savings

# A3: Merge strategy
multiplicative                    # h_fwd * h_bwd
gated                             # learned gate

# C4: Time conditioning
time-cond-on                      # re-test without --no_time_cond
time-cond-off                     # control

# Combined best guesses
hybrid-25-first + tied            # combine A1 winner + A2
hybrid-25-first + time-cond-on    # combine A1 + C4
```

### Block 3: Analyze + Tier 2 (1.5 hours)

Rank Tier 1 results. If hybrid attention helps:
- Implement Mousse optimizer (60 LOC)
- Run Mousse + best architecture from Tier 1
- Run width/depth sweep at quokka scale

### Block 4: Validate winners at 5k steps, 3 seeds (2 hours)

Take the top 2-3 configs from Tier 1+2 and validate:
- 5k steps, 3 seeds each = 6-9 runs at ~12 min each = 72-108 min
- Compute paired t-tests vs Muon baseline

---

## Key Open Questions for the 5090 Agent

The "deep" work on the 5090 should focus on what we cannot do cheaply:

1. **Hybrid attention at scale (84M, 231M):** Does the optimal ratio change
   with scale? MaBERT tested at one scale only.
2. **Mousse/Muon-VS + longer training (10k+ steps):** Do the optimizer gains
   compound or plateau?
3. **WSD schedule for 10k+ runs:** Operational benefit of not needing max_steps.
4. **Text generation quality:** Does hybrid attention improve sample quality
   disproportionately to perplexity gains?

---

## Literature Summary Table

| Paper | arXiv | Key finding for us |
|-------|-------|-------------------|
| Nemotron-H | 2504.03624 | 7-8% attention optimal for causal decoders |
| MaBERT | 2603.03001 | 33% attention (MMTMMTMMTMMT) optimal for ENCODER |
| DiffuMamba-H | 2511.15927 | 20% attention for bidi diffusion; 8.2x throughput |
| Caduceus | 2403.03234 | Weight-tied BiMamba beats untied at same params |
| BiGS | 2212.10544 | Multiplicative gating works for bidi SSM + MLM |
| AdaMuon | 2507.11005 | Element-wise adaptivity on Muon; 40% over Adam |
| NorMuon | 2510.05491 | Neuron-wise norms; 11% over Muon at 1.1B |
| Mousse | 2603.09697 | Curvature-aware Muon; 12% fewer steps, 3% overhead |
| Newton-Muon | 2604.01472 | Right preconditioning; 6% fewer steps |
| Muon-VS | 2601.14603 | Variance-scaled; 1.36x on LLaMA-1.2B, no hyperparams |
| MuonAll | 2511.06086 | All-param Muon; at-par, not better |
| MuonClip | 2507.20534 | QK-clip for stability; zero spikes at 1T scale |
| Optimal schedules | 2511.04647 | Inference schedule >> training schedule for quality |
| Scaling DLLMs | 2512.10858 | Diffusion scaling differs from AR; batch size matters |
| Small batch | 2507.07101 | Small batches better per-FLOP; grad accum wasteful |
| WSD theory | 2410.05192 | River-valley explanation; flexible stopping |
