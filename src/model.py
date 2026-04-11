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
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# SSM backend: Mamba-3 (custom kernels) → PureSSM (pure PyTorch fallback)
# ---------------------------------------------------------------------------

try:
    from mamba_ssm.modules.mamba3 import Mamba3
    SSM_BACKEND = "mamba3"
except ImportError:
    SSM_BACKEND = "pure"

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
# SwiGLU MLP (2x expansion, following DiffuMamba)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# BiMamba3 Block
# ---------------------------------------------------------------------------

class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3 block with AdaLN conditioning.

    Following DiffuMamba: two independent scans (forward + backward), additive merge.
    Gate from AdaLN modulates the residual.
    """
    def __init__(self, d_model: int, cond_dim: int, d_state: int = 128,
                 headdim: int = 64, expand: int = 2, is_mimo: bool = True,
                 mimo_rank: int = 4, chunk_size: int = 16,
                 mlp_expansion: int = 2, dtype=torch.bfloat16):
        super().__init__()

        # Build SSM layers: Mamba-3 (fast, needs custom kernels) or PureSSM (portable)
        if SSM_BACKEND == "mamba3":
            ssm_kwargs = dict(
                d_model=d_model, d_state=d_state, headdim=headdim,
                expand=expand, is_mimo=is_mimo, mimo_rank=mimo_rank,
                chunk_size=chunk_size, dtype=dtype,
            )
            self.mamba_fwd = Mamba3(**ssm_kwargs)
            self.mamba_bwd = Mamba3(**ssm_kwargs)
        else:
            # PureSSM: proper selective SSM in pure PyTorch (works on any GPU)
            ssm_kwargs = dict(d_model=d_model, d_state=d_state, expand=expand)
            self.mamba_fwd = PureSSM(**ssm_kwargs)
            self.mamba_bwd = PureSSM(**ssm_kwargs)

        # Conditioning and MLP
        self.adaln_mamba = AdaLN(d_model, cond_dim)
        self.adaln_mlp = AdaLN(d_model, cond_dim)
        self.mlp = SwiGLU(d_model, expansion=mlp_expansion)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D), c: (B, cond_dim) → (B, L, D)"""
        # Bidirectional Mamba scan with AdaLN
        h, gate = self.adaln_mamba(x, c)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)
        x = x + gate * (h_fwd + h_bwd)

        # MLP with AdaLN
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
    max_seq_len: int = 1024
    cond_dim: int = 128           # timestep conditioning dim
    dtype: torch.dtype = torch.bfloat16

    # Diffusion
    mask_token_id: int = 50257    # first unused id after GPT-2 vocab
    noise_eps: float = 1e-3       # noise schedule eps
    sampling_eps: float = 1e-3    # min t during training
    time_conditioning: bool = True  # MDLM defaults False; DiffuMamba uses True
    antithetic_sampling: bool = True


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

        # Mamba-3 blocks
        self.blocks = nn.ModuleList([
            BiMamba3Block(
                d_model=c.d_model, cond_dim=c.cond_dim,
                d_state=c.d_state, headdim=c.headdim, expand=c.expand,
                is_mimo=c.is_mimo, mimo_rank=c.mimo_rank,
                chunk_size=c.chunk_size, mlp_expansion=c.mlp_expansion,
                dtype=c.dtype,
            )
            for _ in range(c.n_layers)
        ])

        # Output: AdaLN + linear
        self.out_adaln = AdaLN(c.d_model, c.cond_dim)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Noise schedule
        self.noise = LogLinearNoise(eps=c.noise_eps)

        # Init
        self.apply(self._init_weights)

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

        # Mamba-3 blocks
        for block in self.blocks:
            h = block(h, c)

        # Output with final AdaLN
        h, _ = self.out_adaln(h, c)
        logits = self.lm_head(h)  # (B, L, V)

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

        # ELBO weighting: dsigma / expm1(sigma), shape (B,)
        weight = dsigma / torch.expm1(sigma)  # = 1/t for log-linear

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

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size: int = 1, seq_len: int = None,
               num_steps: int = 128, device: str = "cuda",
               temperature: float = 1.0) -> torch.Tensor:
        """Generate text by iterative unmasking (DDPM caching update from MDLM).

        Start fully masked, step from t=1 to t≈0, progressively unmask.
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
            move_chance_s = self.noise.move_chance(t - dt)  # scalar (next = cleaner)

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

            # Build normalized transition distribution for masked positions
            # p(unmask to token v) = p_x0[v] * (1 - move_chance_s / move_chance_t)
            # p(stay masked)       = move_chance_s / move_chance_t
            unmask_prob = 1.0 - move_chance_s / (move_chance_t + 1e-8)
            q_xs = p_x0 * unmask_prob
            q_xs[:, :, self.config.mask_token_id] = move_chance_s / (move_chance_t + 1e-8)

            # Apply temperature
            if temperature != 1.0:
                q_xs = q_xs ** (1.0 / temperature)

            # Sample using Gumbel-max trick
            gumbel = -(torch.rand_like(q_xs) + 1e-10).log()
            sampled = (q_xs / (gumbel + 1e-10)).argmax(dim=-1)

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
    # Tiny: ~2M params, for testing / debugging
    "tiny": DiffuMamba3Config(
        d_model=128, n_layers=4, d_state=32, headdim=32, expand=2,
        is_mimo=False, mlp_expansion=2, max_seq_len=256, cond_dim=64,
    ),
    # Small: ~45M params, fast autoresearch experiments on 9070 XT
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
