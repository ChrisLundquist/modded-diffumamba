# Architecture Variants Experiment: Hybrid Attention + Soft Masking

## Title

Hybrid Mamba-Attention Blocks and Soft Masking for DiffuMamba3

## Hypothesis

**H1 (Hybrid attention):** Inserting 1 bidirectional self-attention layer for every
2-4 Mamba blocks will improve validation loss by 2-5% relative, at modest throughput
cost. The attention layers provide exact global token-token interaction that SSMs can
only approximate, and this is especially valuable for masked diffusion where every
position must attend to arbitrary unmasked context.

**H2 (Soft masking):** Adding the SM feedback module from Hersche et al. (ICLR 2026)
will improve validation loss by 0.5-1.5% relative, at near-zero parameter cost
(3 extra scalars). SM gives the model a richer input signal by blending [MASK]
embeddings with top-k predictions from the previous denoising step, rather than
discarding all information when a mask is retained.

**H3 (Combined):** The two interventions are complementary -- hybrid attention improves
the backbone's representational capacity, while SM improves the quality of the input
signal at each denoising step. Combined gain should exceed either alone.

**Why we expect this to work:**
- DiffuMamba paper shows DiffuMamba-H (20% attention) matches full-Transformer
  diffusion quality while keeping 4.3x throughput advantage.
- MaBERT (March 2026) found the optimal encoder pattern is MMTMMTMMTMMT (2:1
  Mamba:Transformer), consistently beating both pure-Mamba and pure-Transformer
  encoders on GLUE benchmarks.
- NVIDIA Nemotron-H validated that even 7% attention (1 attn per ~13 Mamba layers)
  preserves quality at 3x throughput gain.
- SM paper shows 1.51 perplexity reduction on OpenWebText (169M model) with only
  3 extra parameters and continued pretraining. MAUVE improvements up to +0.30.

## Method

### Experiment 1: Hybrid Attention Ratio Sweep

Test four configurations on the `quokka` config (31.5M base), 1000 steps, 3 seeds each.
All use Muon (lr=0.02), minsnr (gamma=5), time conditioning ON.

| Variant | Pattern (4 layers) | Attn layers | Attn % | Notes |
|---------|-------------------|-------------|--------|-------|
| `baseline` | MMMM | 0 | 0% | Current DiffuMamba3 |
| `hybrid-25` | MMMA | 1 | 25% | DiffuMamba-H style |
| `hybrid-33` | MMA MMA... | 1-2 | 33% | MaBERT-optimal ratio |
| `hybrid-50` | MAMA | 2 | 50% | Upper bound cost |

For the quokka config (4 layers), the patterns map to:
- `baseline`: [Mamba, Mamba, Mamba, Mamba]
- `hybrid-25`: [Attn, Mamba, Mamba, Mamba] (attention first, following DiffuMamba-H)
- `hybrid-33`: [Mamba, Mamba, Attn, Mamba] (closest to 2:1 MaBERT ratio)
- `hybrid-50`: [Mamba, Attn, Mamba, Attn]

**Implementation in model.py:**

Add a new `BiAttentionBlock` class alongside `BiMamba3Block`:

```python
class BiAttentionBlock(nn.Module):
    """Bidirectional self-attention block with AdaLN conditioning.

    Standard multi-head attention (no causal mask) for masked diffusion.
    Drop-in replacement for BiMamba3Block at selected layer positions.
    """
    def __init__(self, d_model: int, cond_dim: int, n_heads: int = 8,
                 mlp_expansion: int = 2, dtype=torch.bfloat16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # QKV projection (no bias, following LLaMA/LLaDA convention)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Conditioning and MLP (same as BiMamba3Block)
        self.adaln_attn = AdaLN(d_model, cond_dim)
        self.adaln_mlp = AdaLN(d_model, cond_dim)
        self.mlp = SwiGLU(d_model, expansion=mlp_expansion)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h, gate = self.adaln_attn(x, c)

        # QKV split
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (NO causal mask -- bidirectional)
        # Uses PyTorch's F.scaled_dot_product_attention which dispatches to
        # flash attention on ROCm via Triton backend
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, n_heads, L, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        h = self.out_proj(attn_out)

        x = x + gate * h

        # MLP with AdaLN (identical to BiMamba3Block)
        h, gate = self.adaln_mlp(x, c)
        x = x + gate * self.mlp(h)
        return x
```

Add `attention_pattern` to `DiffuMamba3Config`:

```python
@dataclass
class DiffuMamba3Config:
    # ... existing fields ...
    attention_pattern: list[int] = field(default_factory=list)
    # List of layer indices that should use attention instead of Mamba.
    # Empty = all Mamba (baseline). Example: [0] = attention at layer 0.
    attn_n_heads: int = 8  # heads for attention layers
```

Modify `DiffuMamba3.__init__` to build mixed blocks:

```python
self.blocks = nn.ModuleList()
for i in range(c.n_layers):
    if i in c.attention_pattern:
        self.blocks.append(BiAttentionBlock(
            d_model=c.d_model, cond_dim=c.cond_dim,
            n_heads=c.attn_n_heads, mlp_expansion=c.mlp_expansion,
            dtype=c.dtype,
        ))
    else:
        self.blocks.append(BiMamba3Block(...))  # existing code
```

**Parameter budget control:** The attention block at d_model=384 with 8 heads uses
~1.77M params (qkv + out_proj + MLP + AdaLN), comparable to a BiMamba3Block at
~2.0M params per layer. The 2x MLP expansion is kept the same across both block
types to keep parameter counts aligned, following DiffuMamba's approach of halving
MLP expansion for the Mamba-containing model to match parameter budgets.

### Experiment 2: Soft Masking (SM)

Test SM on top of both baseline and best hybrid from Experiment 1.

**Implementation in model.py:**

Add SM feedback as a thin wrapper around the forward pass:

```python
class SoftMaskFeedback(nn.Module):
    """Soft masking module from Hersche et al. (ICLR 2026).

    Blends [MASK] embedding with top-k predicted token embeddings
    weighted by prediction confidence. Only 3 trainable parameters.

    Applied during iterative sampling: after each denoising step,
    retained masks get enriched with partial prediction information.
    During training: uses a two-pass approach (detached first pass
    to get predictions, then SM-augmented second pass with gradients).
    """
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        # Learnable SM parameters (constrained via sigmoid/softplus)
        self.omega_s = nn.Parameter(torch.tensor(0.0))  # -> sigmoid -> [0,1]
        self.omega_a = nn.Parameter(torch.tensor(1.0))  # -> softplus -> >=0
        self.omega_b = nn.Parameter(torch.tensor(-1.0)) # -> -softplus -> <=0

    def compute_lambda(self, log_probs: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """Compute per-position blending weight lambda from prediction confidence.

        Args:
            log_probs: (B, L, V) log probabilities from model
            mask: (B, L) bool, True where token is [MASK]
        Returns:
            lam: (B, L) in [0, 1], blending weight (0 = pure mask, 1 = pure pred)
        """
        # Entropy of prediction distribution at masked positions
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

        # Confidence = negative entropy (higher = more confident)
        omega_s = torch.sigmoid(self.omega_s)
        omega_a = F.softplus(self.omega_a)
        omega_b = -F.softplus(-self.omega_b)  # negative

        lam = omega_s * torch.sigmoid(omega_a * (-entropy - omega_b))
        # Zero out lambda for unmasked positions
        lam = lam * mask.float()
        return lam  # (B, L)

    def blend_embeddings(self, tok_emb_layer: nn.Embedding,
                         x_t: torch.Tensor, log_probs: torch.Tensor,
                         mask_token_id: int) -> torch.Tensor:
        """Replace [MASK] embeddings with SM-blended embeddings.

        Args:
            tok_emb_layer: embedding layer
            x_t: (B, L) current token ids (with [MASK])
            log_probs: (B, L, V) detached log probs from first pass
            mask_token_id: the [MASK] token id
        Returns:
            blended_emb: (B, L, D) embeddings with SM applied to masked positions
        """
        mask = (x_t == mask_token_id)  # (B, L)
        lam = self.compute_lambda(log_probs, mask)  # (B, L)

        # Top-k predictions
        topk_logp, topk_ids = log_probs.topk(self.k, dim=-1)  # (B, L, k)
        topk_probs = topk_logp.exp()
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize

        # Weighted average of top-k token embeddings
        topk_embs = tok_emb_layer(topk_ids)  # (B, L, k, D)
        pred_emb = (topk_probs.unsqueeze(-1) * topk_embs).sum(dim=2)  # (B, L, D)

        # Original embeddings
        orig_emb = tok_emb_layer(x_t)  # (B, L, D)

        # Blend: (1-lambda)*mask_emb + lambda*pred_emb, only at masked positions
        lam = lam.unsqueeze(-1)  # (B, L, 1)
        blended = torch.where(mask.unsqueeze(-1),
                              (1 - lam) * orig_emb + lam * pred_emb,
                              orig_emb)
        return blended
```

Modify `DiffuMamba3.compute_loss` for two-pass SM training:

```python
def compute_loss(self, x_0):
    # ... existing t sampling, masking code ...

    if self.config.use_soft_mask:
        # Pass 1: get predictions (no gradient)
        with torch.no_grad():
            log_probs_detached = self(x_t, sigma_cond)

        # SM-augmented embeddings for pass 2
        blended_emb = self.sm_module.blend_embeddings(
            self.tok_emb, x_t, log_probs_detached, self.config.mask_token_id)

        # Pass 2: forward with SM embeddings (with gradient)
        log_probs = self.forward_with_embeddings(blended_emb, sigma_cond)
    else:
        log_probs = self(x_t, sigma_cond)

    # ... rest of loss computation unchanged ...
```

Add `forward_with_embeddings` method that takes pre-computed embeddings instead of
running `tok_emb(x_t)`:

```python
def forward_with_embeddings(self, h: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Forward pass starting from pre-computed embeddings (for SM)."""
    B, L, D = h.shape
    pos = torch.arange(L, device=h.device)
    h = h + self.pos_emb(pos)
    c = self.sigma_map(sigma)
    # ... same block loop and output as forward() ...
```

**Config additions:**

```python
use_soft_mask: bool = False
soft_mask_k: int = 3  # top-k for SM blending
```

### Experiment 3: Combined (Best Hybrid + SM)

Run best hybrid variant from Exp 1 with SM enabled.

### Experiment 4: Ablation -- Weight Tying for Bidirectional Mamba

Test Caduceus-style weight tying between forward and backward Mamba scans. This halves
the Mamba parameters per block, freeing budget for a wider model or more layers.

**Implementation:** In `BiMamba3Block.__init__`, set `self.mamba_bwd = self.mamba_fwd`
(share weights). The forward pass remains `h_fwd = mamba_fwd(h)` and
`h_bwd = mamba_fwd(h.flip(1)).flip(1)` -- same module, reversed input.

Add `tie_bidi_weights: bool = False` to config.

### Measurement Protocol

All experiments use:
- Config: `quokka` (d_model=384, n_layers=4, seq_len=1024)
- Steps: 1000 (sufficient for ranking at this scale per prior autoresearch runs)
- Batch size: 32
- Optimizer: Muon (lr=0.02) + AdamW (lr=3e-4)
- Loss weight: minsnr (gamma=5)
- Time conditioning: ON
- Seeds: 3 per variant (report mean +/- std)
- Metric: validation loss (ELBO-weighted, for cross-experiment comparability)
- Secondary: wall-clock time per step (to track throughput cost of attention)

### Experiment Matrix (Total: ~16 runs)

| # | Variant | Attn pattern | SM | Tied | Estimated time |
|---|---------|--------------|-----|------|----------------|
| 1 | baseline | [] | off | no | 3x ~15 min |
| 2 | hybrid-25 | [0] | off | no | 3x ~17 min |
| 3 | hybrid-33 | [2] | off | no | 3x ~17 min |
| 4 | hybrid-50 | [1,3] | off | no | 3x ~19 min |
| 5 | baseline+SM | [] | on | no | 3x ~25 min |
| 6 | best_hybrid+SM | [best] | on | no | 3x ~28 min |
| 7 | baseline+tied | [] | off | yes | 3x ~15 min |

## Expected Outcome

**If H1 confirmed (hybrid helps):**
- hybrid-25 or hybrid-33 improves val loss by 0.1-0.3 nats over baseline
- hybrid-50 may overfit or show diminishing returns (too much attention for 4 layers)
- Wall-clock overhead of attention layers is < 30% (PyTorch SDPA dispatches to
  flash attention on ROCm via Triton backend, which works on RDNA4)

**If H1 rejected:**
- All hybrid variants within noise of baseline (< 0.05 nat difference)
- Would suggest the PureSSM already captures sufficient global context for this
  task/scale, or that the quokka config is too small to benefit

**If H2 confirmed (SM helps):**
- SM variants improve by 0.05-0.15 nats over their non-SM counterparts
- Larger gain on MAUVE/generation quality than on raw perplexity
- omega_s parameter rises from ~0 toward ~1 during training (model learns to use SM)

**If H2 rejected:**
- SM overhead (2x forward pass during training) not worth the quality gain
- Possible cause: 1000 steps insufficient for SM parameters to converge

**If weight tying helps:**
- Tied variant matches or beats untied at same layer/width count
- Opens path to wider models within same VRAM budget

**Decision rule for next step:**
- If hybrid-25 or hybrid-33 wins by > 0.1 nat: adopt as default, scale to `small` config
- If SM wins by > 0.05 nat: adopt, note training cost (2x passes)
- If both win: test combined at `small` scale (84M) for 5000 steps

## Risk / Cost

| Risk | Mitigation |
|------|-----------|
| Attention on RDNA4 slow or broken | PyTorch SDPA with Triton backend confirmed working on RDNA4 per CLAUDE.md. Fallback: F.scaled_dot_product_attention with `enable_flash=False` uses math backend. |
| SM 2x training cost too expensive | SM adds one extra forward pass (no backward). At quokka scale this is ~60% wall-clock overhead. If SM wins, can explore single-pass approximations (cache predictions from step N-1). |
| 4-layer quokka too small to see hybrid benefit | MaBERT tested on 12-layer models. If quokka results are inconclusive, re-run hybrid-33 on `small` (8 layers) where [2, 5] pattern gives ~25% attention. |
| Parameter count mismatch biases comparison | Attention block params (~1.77M at d=384) are close to BiMamba3Block (~2.0M). Difference is < 15%, within noise for this comparison. For strict iso-param, can adjust d_model slightly. |
| SM omega parameters need longer training to converge | Paper shows omega_s rises during continued pretraining. 1000 steps may be short. If SM shows flat omega_s, extend to 2500 steps for that variant only. |

**Implementation complexity:** ~150 LOC for BiAttentionBlock + config changes.
~100 LOC for SoftMaskFeedback module. ~50 LOC for weight tying option. Total ~300 LOC.

**GPU hours:** ~16 runs x ~20 min avg = ~5.3 hours on RX 9070 XT. Fits in a single
overnight autoresearch session.

## Literature Support

### Hybrid Mamba-Attention

- **DiffuMamba-H** (Arriaga et al., Nov 2025, arXiv 2511.15927): 1 attention layer
  per 5 Mamba blocks (~20%). Matches Transformer-based diffusion quality with 4.3x
  throughput gain. Uses additive merge for bidirectional Mamba, halved MLP expansion.
  Tested up to 1.3B params.

- **MaBERT** (March 2026, arXiv 2603.03001): Hybrid encoder interleaving Mamba and
  Transformer layers. Tested 8 patterns on 12-layer encoder. Optimal: MMTMMTMMTMMT
  (2:1 Mamba:Transformer = 33% attention). Single-family architectures (all-Mamba or
  all-Transformer) consistently underperform mixed schedules. 2.4x training speedup
  over Transformer-only at 4096 context.

- **NVIDIA Nemotron-H** (2025): 7% attention (92% Mamba-2 blocks) at 8B scale. 3x
  throughput over pure Transformer with competitive quality. Shows that even very
  sparse attention is valuable.

- **Jamba** (AI21, ICLR 2025): 1:7 attention:Mamba ratio in 52B model. Validates
  hybrid approach at very large scale.

### Soft Masking

- **Hersche et al.** (ICLR 2026, arXiv 2510.17206): SM blends [MASK] embedding with
  top-k predicted token embeddings weighted by entropy-based confidence. Only 3 extra
  parameters (omega_s, omega_a, omega_b). On 169M model: perplexity drops from 22.88
  to 21.63 with SM (1.25 ppl reduction). MAUVE improves up to +0.30. Finetuning
  Dream-7B and Dream-Coder-7B with SM improves HumanEval and MBPP, especially in
  high-throughput (few denoising steps) regimes. Best k: 3 for language, 1 for code.

### Bidirectional Mamba Design

- **Caduceus** (Schiff et al., ICML 2024): Weight-tied bidirectional Mamba for DNA.
  Tying forward/backward scan weights halves Mamba parameters with minimal quality loss,
  enabled by the shared dynamics of forward/reverse processing.

- **DiffuMamba** baseline: Uses additive merge (h_fwd + h_bwd) for bidirectional,
  which we already implement. Alternative: multiplicative gating (BiGS style) was
  not tested in DiffuMamba paper.

### Broader Context

- **LLaDA** (Feb 2025, arXiv 2502.09992): 8B masked diffusion LM using bidirectional
  Transformer (LLaMA-3 without causal mask), RoPE, RMSNorm, SwiGLU. Competitive with
  LLaMA-3 8B. Validates that standard Transformer components work well for masked
  diffusion when made bidirectional.

- **Dream-7B** (2025, arXiv 2508.15487): 7B diffusion LM. Outperforms autoregressive
  models on planning tasks (Countdown, Sudoku). Architecture uses standard Transformer.

- **Flash Attention on RDNA4:** PyTorch SDPA dispatches to Triton flash attention
  backend on ROCm (FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"). Supports bidirectional
  (no causal mask) attention, MQA/GQA, and RoPE. Confirmed working on gfx1201.

Sources:
- [DiffuMamba (arXiv 2511.15927)](https://arxiv.org/abs/2511.15927)
- [MaBERT (arXiv 2603.03001)](https://arxiv.org/abs/2603.03001)
- [Soft-Masked Diffusion LMs (arXiv 2510.17206)](https://arxiv.org/abs/2510.17206)
- [SM GitHub (IBM)](https://github.com/IBM/soft-masked-diffusion-language-models)
- [Caduceus (arXiv 2403.03234)](https://arxiv.org/abs/2403.03234)
- [LLaDA (arXiv 2502.09992)](https://arxiv.org/abs/2502.09992)
- [Dream-7B (arXiv 2508.15487)](https://arxiv.org/abs/2508.15487)
- [NVIDIA Nemotron-H](https://research.nvidia.com/labs/adlr/nemotronh/)
- [Jamba (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf)
- [MambaVision (CVPR 2025)](https://github.com/NVlabs/MambaVision)
- [Flash Attention ROCm](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
