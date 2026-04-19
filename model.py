"""
DiffuMamba3: Masked Diffusion Language Model with Bidirectional Mamba-3 Backbone

Architecture:
  Input tokens → Embed → [BiMamba3Block × N] → RMSNorm → LM Head → Logits

Each BiMamba3Block:
  - AdaLN (conditioned on sigma)
  - Forward Mamba-3 scan → h_fwd
  - Backward Mamba-3 scan → h_bwd
  - h = h_fwd + h_bwd  (additive merge, following DiffuMamba)
  - Residual connection
  - AdaLN + SwiGLU MLP + residual

Diffusion: MDLM-style absorbing-state masked diffusion (Sahoo et al. 2024).
  - Forward: tokens → [MASK] at rate governed by noise schedule
  - Loss: -log p_theta(x0|xt) * dsigma/expm1(sigma), per masked position
  - The SUBS parameterization: model predicts clean token distribution,
    mask token logit is killed, unmasked positions forced to copy.

References:
  - MDLM: https://arxiv.org/abs/2406.07524
  - DiffuMamba: https://arxiv.org/abs/2511.15927
  - Mamba-3: https://arxiv.org/abs/2603.15569
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# SSM backend: Mamba-3 → Mamba-2 (Triton SSD) → PureSSM (pure PyTorch)
#
# mamba_ssm's __init__.py unconditionally imports selective_scan_cuda which
# doesn't exist without a CUDA build. Import the modules directly to bypass.
# ---------------------------------------------------------------------------

SSM_BACKEND = "pure"
SSM_MIMO_OK = False  # whether Mamba3 MIMO path works (separate tilelang kernel)


def _probe_ssm(cls, label, **kwargs):
    """Try a small forward pass to verify the backend works at runtime."""
    import torch
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = cls(**kwargs).to(device)
        x = torch.randn(1, 16, kwargs["d_model"], device=device)
        _ = m(x)
        del m, x
        return True
    except Exception as e:
        print(f"[SSM] {label} probe failed: {e}")
        return False


try:
    from mamba_ssm.modules.mamba3 import Mamba3
    if _probe_ssm(Mamba3, "Mamba3", d_model=64, d_state=32, headdim=32, expand=2,
                  chunk_size=16):
        SSM_BACKEND = "mamba3"
        # MIMO uses a separate tilelang kernel that may not work on all GPUs
        SSM_MIMO_OK = _probe_ssm(Mamba3, "Mamba3-MIMO", d_model=64, d_state=32,
                                 headdim=32, expand=2, is_mimo=True, mimo_rank=4,
                                 chunk_size=16)
        if not SSM_MIMO_OK:
            print("[SSM] Mamba3 available (non-MIMO only)")
    else:
        Mamba3 = None
except Exception as e:
    print(f"[SSM] Mamba3 import failed: {e}")
    Mamba3 = None

if SSM_BACKEND == "pure":
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        if _probe_ssm(Mamba2, "Mamba2", d_model=64, d_state=32, headdim=32,
                      expand=2, chunk_size=16):
            SSM_BACKEND = "mamba2"
        else:
            Mamba2 = None
    except Exception as e:
        print(f"[SSM] Mamba2 import failed: {e}")
        Mamba2 = None

from ssm import PureSSM


# ---------------------------------------------------------------------------
# Noise schedule (log-linear, from MDLM)
# ---------------------------------------------------------------------------

class LogLinearNoise:
    """Log-linear noise schedule from MDLM.

    sigma(t) = -log(1 - (1-eps)*t)    (cumulative noise)
    dsigma(t) = (1-eps) / (1-(1-eps)*t) (rate)
    move_chance(t) = (1-eps)*t          (masking probability, linear in t)
    """
    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)

    def dsigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def move_chance(self, t: torch.Tensor) -> torch.Tensor:
        """Probability that a token is masked at time t."""
        return (1 - self.eps) * t

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sigma, dsigma) for timestep t."""
        return self.sigma(t), self.dsigma(t)


# ---------------------------------------------------------------------------
# Timestep embedding (sinusoidal → MLP, from MDLM/DiT)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by MLP, from MDLM's DiT backbone."""
    def __init__(self, d_out: int, freq_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_out, bias=True),
            nn.SiLU(),
            nn.Linear(d_out, d_out, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float → (B, d_out)"""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Cast to model weight dtype (bf16 on GPU, fp32 on CPU)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (AdaLN) — zero-initialized, from DiT
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive LayerNorm with scale, shift, gate — zero-initialized."""
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 3 * d_model, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, D), c: (B, cond_dim) → (normed_x, gate)"""
        shift, scale, gate = self.modulation(c).unsqueeze(1).chunk(3, dim=-1)
        normed = self.norm(x) * (1 + scale) + shift
        return normed, gate


# ---------------------------------------------------------------------------
# MLP variants
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU MLP: gate * SiLU(proj) with 2x expansion (modded-nanogpt style)."""
    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeluMLP(nn.Module):
    """Standard GELU MLP (DiffuMamba style): GELU(proj) with 2x expansion."""
    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def build_mlp(d_model: int, expansion: int = 2, mlp_type: str = "swiglu"):
    if mlp_type == "gelu":
        return GeluMLP(d_model, expansion=expansion)
    return SwiGLU(d_model, expansion=expansion)


# ---------------------------------------------------------------------------
# BiMamba3 Block
# ---------------------------------------------------------------------------

def _build_ssm(d_model, d_state, headdim, expand, is_mimo, mimo_rank, chunk_size):
    """Build a single SSM layer using the best available backend."""
    if SSM_BACKEND == "mamba3":
        use_mimo = is_mimo and SSM_MIMO_OK
        return Mamba3(
            d_model=d_model, d_state=d_state, headdim=headdim,
            expand=expand, is_mimo=use_mimo,
            **(dict(mimo_rank=mimo_rank) if use_mimo else {}),
            chunk_size=chunk_size,
        )
    elif SSM_BACKEND == "mamba2":
        return Mamba2(
            d_model=d_model, d_state=d_state, headdim=headdim,
            expand=expand, chunk_size=chunk_size,
        )
    else:
        return PureSSM(d_model=d_model, d_state=d_state, expand=expand)


class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3 block with AdaLN conditioning.

    Following DiffuMamba: two independent scans (forward + backward), merged.
    Gate from AdaLN modulates the residual.
    """
    def __init__(self, d_model: int, cond_dim: int, d_state: int = 128,
                 headdim: int = 64, expand: int = 2, is_mimo: bool = True,
                 mimo_rank: int = 4, chunk_size: int = 16,
                 mlp_expansion: int = 2, tie_weights: bool = False,
                 merge: str = "add", mlp_type: str = "swiglu"):
        super().__init__()
        self.merge = merge

        self.mamba_fwd = _build_ssm(d_model, d_state, headdim, expand,
                                     is_mimo, mimo_rank, chunk_size)
        if tie_weights:
            self.mamba_bwd = self.mamba_fwd  # Caduceus-style weight sharing
        else:
            self.mamba_bwd = _build_ssm(d_model, d_state, headdim, expand,
                                         is_mimo, mimo_rank, chunk_size)

        if merge == "gate":
            self.merge_gate = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.merge_gate.weight)

        # Conditioning and MLP
        self.adaln_mamba = AdaLN(d_model, cond_dim)
        self.adaln_mlp = AdaLN(d_model, cond_dim)
        self.mlp = build_mlp(d_model, expansion=mlp_expansion, mlp_type=mlp_type)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D), c: (B, cond_dim) → (B, L, D)"""
        h, gate = self.adaln_mamba(x, c)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)

        if self.merge == "mul":
            merged = h_fwd * h_bwd
        elif self.merge == "gate":
            g = torch.sigmoid(self.merge_gate(h_fwd))
            merged = g * h_fwd + (1 - g) * h_bwd
        else:  # "add"
            merged = h_fwd + h_bwd

        x = x + gate * merged

        h, gate = self.adaln_mlp(x, c)
        x = x + gate * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# Bidirectional Attention Block (for hybrid Mamba-attention models)
# ---------------------------------------------------------------------------

class BiAttentionBlock(nn.Module):
    """Bidirectional self-attention block with AdaLN conditioning.

    Drop-in replacement for BiMamba3Block at selected layer positions.
    Uses F.scaled_dot_product_attention (dispatches to flash attn via Triton on RDNA4).
    """
    def __init__(self, d_model: int, cond_dim: int, nheads: int = None,
                 mlp_expansion: int = 2, mlp_type: str = "swiglu", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads or max(1, d_model // 64)
        self.head_dim = d_model // self.nheads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.adaln_attn = AdaLN(d_model, cond_dim)
        self.adaln_mlp = AdaLN(d_model, cond_dim)
        self.mlp = build_mlp(d_model, expansion=mlp_expansion, mlp_type=mlp_type)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D), c: (B, cond_dim) → (B, L, D)"""
        B, L, D = x.shape
        h, gate = self.adaln_attn(x, c)

        qkv = self.qkv(h).reshape(B, L, 3, self.nheads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, nheads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Bidirectional: no causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + gate * self.out_proj(attn_out)

        h, gate = self.adaln_mlp(x, c)
        x = x + gate * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffuMamba3Config:
    # Model
    vocab_size: int = 50304       # GPT-2 (50257) padded to multiple of 64
    d_model: int = 768
    n_layers: int = 12
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    is_mimo: bool = True
    mimo_rank: int = 4
    chunk_size: int = 16          # 64 // mimo_rank for MIMO
    mlp_expansion: int = 2
    mlp_type: str = "swiglu"      # "swiglu" or "gelu" (DiffuMamba uses gelu)
    max_seq_len: int = 1024
    cond_dim: int = 128           # timestep conditioning dim

    # Architecture options
    attn_layers: list = None      # layer indices using attention (None = all Mamba)
    tie_bidi_weights: bool = False  # Caduceus-style fwd/bwd weight tying
    merge: str = "add"            # "add", "mul", or "gate" for bidi merge

    # Diffusion
    mask_token_id: int = 50257    # first unused id after GPT-2 vocab
    noise_eps: float = 1e-3       # noise schedule eps
    sampling_eps: float = 1e-3    # min t during training
    time_conditioning: bool = True  # MDLM defaults False; DiffuMamba uses True
    antithetic_sampling: bool = True
    gradient_checkpointing: bool = True  # recompute blocks during backward to save VRAM
    loss_weight: str = "minsnr"    # "elbo" (1/t), "flat" (1), "minsnr" (clamped 1/t)
    minsnr_gamma: float = 5.0      # clamp value for Min-SNR (Hang et al. ICCV 2023)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class DiffuMamba3(nn.Module):
    """Masked Diffusion Language Model with Bidirectional Mamba-3."""

    def __init__(self, config: DiffuMamba3Config):
        super().__init__()
        self.config = config
        c = config

        # Embeddings
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)

        # Timestep conditioning
        self.sigma_map = TimestepEmbedder(c.cond_dim)

        # Build blocks: Mamba by default, attention at specified layers
        attn_set = set(c.attn_layers) if c.attn_layers else set()
        self.blocks = nn.ModuleList()
        for i in range(c.n_layers):
            if i in attn_set:
                self.blocks.append(BiAttentionBlock(
                    d_model=c.d_model, cond_dim=c.cond_dim,
                    mlp_expansion=c.mlp_expansion, mlp_type=c.mlp_type,
                ))
            else:
                self.blocks.append(BiMamba3Block(
                    d_model=c.d_model, cond_dim=c.cond_dim,
                    d_state=c.d_state, headdim=c.headdim, expand=c.expand,
                    is_mimo=c.is_mimo, mimo_rank=c.mimo_rank,
                    chunk_size=c.chunk_size, mlp_expansion=c.mlp_expansion, mlp_type=c.mlp_type,
                    tie_weights=c.tie_bidi_weights, merge=c.merge,
                ))

        # Output: AdaLN + linear
        self.out_adaln = AdaLN(c.d_model, c.cond_dim)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Noise schedule
        self.noise = LogLinearNoise(eps=c.noise_eps)

        # Init: global init, then re-apply SSM-specific and merge gate inits
        self.apply(self._init_weights)
        for block in self.blocks:
            for attr in ['mamba_fwd', 'mamba_bwd']:
                ssm = getattr(block, attr, None)
                if ssm is not None and hasattr(ssm, '_init_weights'):
                    ssm._init_weights()
            # Re-zero merge_gate after _init_weights overwrites it
            if hasattr(block, 'merge_gate'):
                nn.init.zeros_(block.merge_gate.weight)

    def _init_weights(self, module):
        if isinstance(module, AdaLN):
            # Preserve zero-init for AdaLN modulation (blocks start as identity)
            nn.init.zeros_(module.modulation.weight)
            nn.init.zeros_(module.modulation.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass: masked tokens + noise level → log probabilities.

        Args:
            x_t:   (B, L) token ids (with [MASK] tokens)
            sigma: (B,) noise level sigma(t), or zeros if time_conditioning=False

        Returns:
            log_probs: (B, L, vocab_size) — log probabilities for SUBS parameterization
        """
        B, L = x_t.shape

        # Embeddings
        pos = torch.arange(L, device=x_t.device)
        h = self.tok_emb(x_t) + self.pos_emb(pos)

        # Timestep conditioning → c (sigma_map MLP already has SiLU)
        c = self.sigma_map(sigma)  # (B, cond_dim)

        # Mamba-3 blocks (with optional gradient checkpointing for VRAM savings)
        # Note: checkpointing is incompatible with Mamba3/Mamba2 custom autograd
        use_ckpt = (self.config.gradient_checkpointing
                    and self.training
                    and SSM_BACKEND == "pure")
        for block in self.blocks:
            if use_ckpt:
                h = checkpoint(block, h, c, use_reentrant=False)
            else:
                h = block(h, c)

        # Output with final AdaLN
        h, gate = self.out_adaln(h, c)
        logits = self.lm_head(h * gate)  # (B, L, V)

        # SUBS parameterization: kill mask logit, normalize, force unmasked to copy
        logits = self._subs_parameterization(logits, x_t)
        return logits

    def _subs_parameterization(self, logits: torch.Tensor,
                                x_t: torch.Tensor) -> torch.Tensor:
        """MDLM SUBS parameterization.

        1. Set mask token logit to -inf (model cannot predict [MASK])
        2. Normalize to log probabilities
        3. For unmasked positions, force output = delta(x_t) (deterministic copy)
        """
        # Kill mask token logit
        logits[:, :, self.config.mask_token_id] = torch.finfo(logits.dtype).min

        # Log-softmax
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # For unmasked positions: force deterministic copy
        unmasked = (x_t != self.config.mask_token_id)
        if unmasked.any():
            log_probs[unmasked] = torch.finfo(log_probs.dtype).min
            # Set the known token's log prob to 0 (prob = 1)
            log_probs[unmasked, x_t[unmasked]] = 0.0

        return log_probs

    # ------------------------------------------------------------------
    # Diffusion: forward process
    # ------------------------------------------------------------------

    def q_xt(self, x_0: torch.Tensor,
             move_chance: torch.Tensor) -> torch.Tensor:
        """Forward process: independently mask each token.

        Args:
            x_0: (B, L) clean token ids
            move_chance: (B, 1) masking probability per sample

        Returns:
            x_t: (B, L) noised token ids
        """
        mask = torch.rand_like(x_0, dtype=torch.float32) < move_chance
        return torch.where(mask, self.config.mask_token_id, x_0)

    # ------------------------------------------------------------------
    # Diffusion: training loss
    # ------------------------------------------------------------------

    def _sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps with antithetic/stratified sampling (from MDLM).

        Returns t in [sampling_eps, 1].
        """
        u = torch.rand(n, device=device)
        if self.config.antithetic_sampling:
            offset = torch.arange(n, device=device).float() / n
            u = (u / n + offset) % 1
        t = (1 - self.config.sampling_eps) * u + self.config.sampling_eps
        return t

    def compute_loss(self, x_0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """MDLM training loss: continuous-time ELBO.

        L = E_t[ -log p_theta(x_0 | x_t) * dsigma(t) / (exp(sigma(t)) - 1) ]

        For log-linear schedule this simplifies to:
            L = E_t[ -log p_theta(x_0 | x_t) / t ]  (per position)

        Unmasked positions contribute 0 via the SUBS parameterization.
        """
        B, L = x_0.shape
        device = x_0.device

        # Sample timesteps
        t = self._sample_t(B, device)  # (B,)

        # Compute noise level
        sigma, dsigma = self.noise(t)

        # Forward process: mask tokens
        move_chance = self.noise.move_chance(t).unsqueeze(1)  # (B, 1)
        x_t = self.q_xt(x_0, move_chance)

        # Timestep conditioning
        if self.config.time_conditioning:
            sigma_cond = sigma
        else:
            sigma_cond = torch.zeros_like(sigma)

        # Model prediction
        log_probs = self(x_t, sigma_cond)  # (B, L, V)

        # Gather log p(x_0 | x_t) for the true token at each position
        log_p_x0 = torch.gather(log_probs, dim=-1,
                                index=x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Loss weighting across timesteps
        # "elbo": dsigma/expm1(sigma) = 1/t — standard MDLM (upweights low-noise)
        # "flat": weight=1 — uniform across timesteps
        # "minsnr": clamp(1/t, max=gamma) — Min-SNR (Hang et al. ICCV 2023)
        #   Clips extreme weights at low-noise timesteps to reduce gradient conflict
        lw = self.config.loss_weight
        if lw == "flat":
            weight = torch.ones_like(sigma)
        elif lw == "minsnr":
            weight = dsigma / torch.expm1(sigma)  # 1/t
            weight = torch.clamp(weight, max=self.config.minsnr_gamma)
        else:  # "elbo"
            weight = dsigma / torch.expm1(sigma)

        # Loss per position, weighted
        loss_per_pos = -log_p_x0 * weight.unsqueeze(1)  # (B, L)

        # Average over all positions (unmasked positions contribute ~0 via SUBS)
        loss = loss_per_pos.mean()

        metrics = {
            "loss": loss.item(),
            "mask_rate": (x_t == self.config.mask_token_id).float().mean().item(),
            "t_mean": t.mean().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def compute_loss_decomp(self, x_0: torch.Tensor,
                             alpha: float = 1.0,
                             tau: float = 0.3,
                             minsnr_gamma: float = 1.5) -> dict:
        """PAPL-style val decomposition (Peng 2025; port of nvidia eval_mdlm_decomp).

        Computes two masked-position NLLs from the SAME forward pass:
          uniform_nll_masked: standard MDLM NLL averaged over masked positions,
            Min-SNR gamma clamp. Differs from our historical val_loss (which
            averages over ALL positions with SUBS=0 at unmasked) — this one
            follows nvidia/PAPL convention and is directly comparable to their
            published numbers.
          planner_w_nll_masked: same, but each masked position is reweighted by
            w_i = softmax(log p(x_0^i | x_t) / tau) over masked positions, with
            planner weight `1 + alpha * w`. This is PAPL's training objective.

        Reviewer signature of successful sampler-aware training: planner_w drops
        meaningfully while uniform moves little — means the model has
        reallocated capacity toward positions the self-planner would visit.
        """
        self.eval()
        B, L = x_0.shape
        t = self._sample_t(B, x_0.device)
        sigma, dsigma = self.noise(t)
        move_chance = self.noise.move_chance(t).unsqueeze(1)
        x_t = self.q_xt(x_0, move_chance)
        sigma_cond = sigma if self.config.time_conditioning else torch.zeros_like(sigma)
        log_probs = self(x_t, sigma_cond)  # (B, L, V), SUBS already applied

        gt_lp = torch.gather(log_probs, dim=-1,
                             index=x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Min-SNR weight (per sample): clamp(1/t, max=gamma)
        ew = torch.clamp(dsigma / torch.expm1(sigma), max=minsnr_gamma)  # (B,)

        mask = (x_t == self.config.mask_token_id)  # (B, L)
        mf = mask.float()
        mask_count = mf.sum(dim=1).clamp(min=1.0)  # (B,)
        per_token_loss = -gt_lp  # (B, L) — NLL; 0 at unmasked via SUBS

        # Uniform: average over masked positions, then Min-SNR weighted
        u_per_sample = (per_token_loss * mf).sum(dim=1) / mask_count
        uniform_nll = (u_per_sample * ew).mean().item()

        # Planner-weighted: w_i = softmax(gt_lp / tau) over masked positions only
        scores = (gt_lp / tau).masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=1) * mf  # zero at unmasked
        papl_w = 1.0 + alpha * w
        p_per_sample = (per_token_loss * mf * papl_w).sum(dim=1) / mask_count
        planner_w_nll = (p_per_sample * ew).mean().item()

        return {
            "uniform_nll_masked": uniform_nll,
            "planner_w_nll_masked": planner_w_nll,
            "papl_gap": uniform_nll - planner_w_nll,  # sampler-alignment signal
        }

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size: int = 1, seq_len: int = None,
               num_steps: int = 128, device: str = "cuda",
               temperature: float = 1.0, top_p: float = 1.0,
               top_k: int = 0) -> torch.Tensor:
        """Generate text by iterative unmasking (DDPM caching update from MDLM).

        Start fully masked, step from t=1 to t≈0, progressively unmask.

        Args:
            temperature: softmax temperature on token distribution (1.0 = off)
            top_p: nucleus sampling threshold (1.0 = off, 0.9 typical)
            top_k: top-k truncation (0 = off)
        """
        if seq_len is None:
            seq_len = self.config.max_seq_len
        eps = self.config.sampling_eps

        # Start fully masked
        x = torch.full((batch_size, seq_len), self.config.mask_token_id,
                        dtype=torch.long, device=device)

        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i]
            t_batch = t.expand(batch_size)

            # Move chances for current and next step
            move_chance_t = self.noise.move_chance(t)   # scalar
            move_chance_s = self.noise.move_chance(t - dt).clamp(min=0)  # scalar (next = cleaner)

            # Get model predictions (with caching)
            if p_x0_cache is None:
                sigma = self.noise.sigma(t_batch)
                if self.config.time_conditioning:
                    sigma_cond = sigma
                else:
                    sigma_cond = torch.zeros_like(sigma)
                log_probs = self(x, sigma_cond)
                p_x0 = log_probs.exp()  # (B, L, V)
            else:
                p_x0 = p_x0_cache

            # Apply temperature to token distribution (NOT to mask-retention prob)
            if temperature != 1.0:
                p_x0 = p_x0 ** (1.0 / temperature)
                p_x0 = p_x0 / (p_x0.sum(dim=-1, keepdim=True) + 1e-8)

            # Top-k: zero out all but the top-k token probabilities
            if top_k > 0:
                topk_vals, _ = p_x0.topk(top_k, dim=-1)
                threshold = topk_vals[..., -1:].expand_as(p_x0)
                p_x0 = torch.where(p_x0 < threshold, torch.zeros_like(p_x0), p_x0)
                p_x0 = p_x0 / (p_x0.sum(dim=-1, keepdim=True) + 1e-8)

            # Top-p (nucleus): zero out tokens outside the top-p probability mass
            if top_p < 1.0:
                sorted_p, sorted_idx = p_x0.sort(dim=-1, descending=True)
                cum_p = sorted_p.cumsum(dim=-1)
                # Keep tokens up to and including the one that crosses top_p
                sorted_mask = cum_p - sorted_p > top_p
                mask = torch.zeros_like(sorted_mask).scatter(-1, sorted_idx, sorted_mask)
                p_x0 = p_x0.masked_fill(mask, 0.0)
                p_x0 = p_x0 / (p_x0.sum(dim=-1, keepdim=True) + 1e-8)

            # Build normalized transition distribution for masked positions
            # p(unmask to token v) = p_x0[v] * (1 - move_chance_s / move_chance_t)
            # p(stay masked)       = move_chance_s / move_chance_t
            unmask_prob = 1.0 - move_chance_s / (move_chance_t + 1e-8)
            q_xs = p_x0 * unmask_prob
            q_xs[:, :, self.config.mask_token_id] = move_chance_s / (move_chance_t + 1e-8)

            # Sample using Gumbel-max trick — fp32 for Gumbel (bf16 truncates noise)
            # See Zheng et al. 2024 (arXiv 2409.02908): bf16 Gumbel → low-temp collapse
            q_xs_f = q_xs.float()
            gumbel = -(torch.rand_like(q_xs_f) + 1e-10).log()
            sampled = (q_xs_f / (gumbel + 1e-10)).argmax(dim=-1)

            # Only update masked positions
            is_masked = (x == self.config.mask_token_id)
            x_new = torch.where(is_masked, sampled, x)

            # Cache invalidation: if tokens changed, recompute next step
            if torch.equal(x_new, x) and not self.config.time_conditioning:
                p_x0_cache = p_x0
            else:
                p_x0_cache = None
            x = x_new

        # Final noise removal: argmax for any remaining masks
        is_masked = (x == self.config.mask_token_id)
        if is_masked.any():
            sigma = self.noise.sigma(torch.tensor([eps], device=device).expand(batch_size))
            sigma_cond = sigma if self.config.time_conditioning else torch.zeros_like(sigma)
            log_probs = self(x, sigma_cond)
            x = torch.where(is_masked, log_probs.argmax(dim=-1), x)

        return x


# ---------------------------------------------------------------------------
# Predefined configs
# ---------------------------------------------------------------------------

CONFIGS = {
    # Tiny: ~8M params, for testing / debugging
    "tiny": DiffuMamba3Config(
        d_model=128, n_layers=4, d_state=32, headdim=32, expand=2,
        is_mimo=False, mlp_expansion=2, max_seq_len=256, cond_dim=64,
    ),
    # Quokka: ~36M params, Quokka-optimal for 1B tokens (40-100 tok/param)
    "quokka": DiffuMamba3Config(
        d_model=384, n_layers=4, d_state=32, headdim=32, expand=2,
        is_mimo=False, mlp_expansion=2, max_seq_len=1024, cond_dim=64,
    ),
    # Small: ~84M params, fast autoresearch experiments on 9070 XT
    "small": DiffuMamba3Config(
        d_model=512, n_layers=8, d_state=64, headdim=64, expand=2,
        is_mimo=True, mimo_rank=2, chunk_size=32,
        mlp_expansion=2, max_seq_len=512, cond_dim=128,
    ),
    # Base: ~130M params, comparable to GPT-2 small / DiffuMamba 240M
    "base": DiffuMamba3Config(
        d_model=768, n_layers=12, d_state=128, headdim=64, expand=2,
        is_mimo=True, mimo_rank=4, chunk_size=16,
        mlp_expansion=2, max_seq_len=1024, cond_dim=128,
    ),
    # Large: ~350M params
    "large": DiffuMamba3Config(
        d_model=1024, n_layers=24, d_state=128, headdim=64, expand=2,
        is_mimo=True, mimo_rank=4, chunk_size=16,
        mlp_expansion=2, max_seq_len=1024, cond_dim=128,
    ),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Force PureSSM for CPU smoke test (Mamba3/Mamba2 need Triton/GPU)
    SSM_BACKEND = "pure"

    # Quick smoke test
    print("=== DiffuMamba3 Model Configs ===")
    for name, cfg in CONFIGS.items():
        model = DiffuMamba3(cfg)
        n = count_parameters(model)
        print(f"  {name:>6}: {n/1e6:>7.1f}M params  "
              f"(d={cfg.d_model}, L={cfg.n_layers}, seq={cfg.max_seq_len})")

    # Test forward + loss
    print("\n=== Smoke Test (tiny config, CPU) ===")
    cfg = CONFIGS["tiny"]
    model = DiffuMamba3(cfg)
    x_0 = torch.randint(0, 50257, (4, cfg.max_seq_len))  # use real vocab range
    loss, metrics = model.compute_loss(x_0)
    print(f"  loss={metrics['loss']:.4f}, mask_rate={metrics['mask_rate']:.3f}, "
          f"t_mean={metrics['t_mean']:.3f}")

    # Test sampling
    tokens = model.sample(batch_size=2, seq_len=32, num_steps=8, device="cpu")
    print(f"  sample shape: {tokens.shape}, "
          f"masks remaining: {(tokens == cfg.mask_token_id).sum().item()}")
    print("\nAll tests passed.")
