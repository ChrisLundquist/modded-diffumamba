"""Adaptive Layer Normalization (AdaLN) for time-conditioned MDLM.

From DiT (Peebles & Xie 2023): modulates layer norm output with
learned shift, scale, and gate derived from timestep embedding.

Zero-initialized so blocks start as identity — training is stable
from step 0 regardless of time conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP → conditioning vector.

    Input: t (B,) or (B,1) float in [0, 1]
    Output: (B, cond_dim)
    """
    def __init__(self, cond_dim=384, max_period=10000):
        super().__init__()
        self.cond_dim = cond_dim
        half = cond_dim // 2
        # Sinusoidal frequencies (fixed, not learned)
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)
        # MLP to project sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t):
        """t: (B,) or (B,1) float timestep."""
        if t.dim() == 2:
            t = t.squeeze(1)  # (B,)
        # Sinusoidal embedding
        args = t[:, None] * self.freqs[None, :]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, cond_dim)
        return self.mlp(emb)


class AdaLNModulation(nn.Module):
    """Projects conditioning vector to (shift, scale, gate) for one block.

    Zero-initialized so the block starts as identity:
    - shift=0, scale=1, gate=0 → output = 0*block(1*norm(x)+0) = 0
    - With residual: x + gate*block_output = x + 0 = x (identity)
    """
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 3 * d_model)
        # Zero-init so gate starts at 0 (identity residual)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c):
        """c: (B, cond_dim) conditioning vector.
        Returns: shift (B,1,D), scale (B,1,D), gate (B,1,D)
        """
        out = self.proj(c)  # (B, 3*D)
        shift, scale, gate = out.chunk(3, dim=-1)  # each (B, D)
        # Unsqueeze for broadcasting over sequence length
        shift = shift.unsqueeze(1)  # (B, 1, D)
        scale = scale.unsqueeze(1)  # (B, 1, D)
        gate = gate.unsqueeze(1)    # (B, 1, D)
        return shift, scale, gate


def adaln_modulate(x_normed, shift, scale, gate):
    """Apply AdaLN modulation to normalized input.

    Args:
        x_normed: (B, T, D) output of LayerNorm
        shift: (B, 1, D) additive shift
        scale: (B, 1, D) multiplicative scale (added to 1)
        gate: (B, 1, D) output gate (multiplies block output)

    Returns:
        modulated: (B, T, D) = (1 + scale) * x_normed + shift
        gate: (B, 1, D) to multiply block output before residual
    """
    modulated = (1 + scale) * x_normed + shift
    return modulated, gate
