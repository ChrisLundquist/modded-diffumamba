"""DiffuMamba-H style hybrid: bidirectional Mamba + sparse attention.

Architecture: interleave Mamba and attention blocks.
Mamba blocks use forward+flip+backward scan for bidirectionality.
Optional AdaLN time conditioning (from DiT).
Includes Quokka-style blocks (BiMamba + SwiGLU MLP + AdaLN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from gpt2 import CausalSelfAttention, MLP
from adaln import TimestepEmbedding, AdaLNModulation, adaln_modulate

# Try Mamba3, fall back to Mamba1
try:
    from mamba_ssm.modules.mamba3 import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block: forward scan + backward scan, averaged."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, version=1,
                 use_adaln=False, cond_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln = AdaLNModulation(cond_dim or d_model, d_model)
        if version == 3 and HAS_MAMBA3:
            self.mamba = Mamba3(d_model=d_model, d_state=d_state, headdim=64, expand=expand)
        else:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x, cond=None, **kwargs):
        residual = x
        h = self.norm(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate = self.adaln(cond)
            h, gate = adaln_modulate(h, shift, scale, gate)
        else:
            gate = None
        # Sequential bidirectional scan (batched version had memory issues)
        fwd = self.mamba(h)
        bwd = self.mamba(h.flip(1)).flip(1)
        out = (fwd + bwd) / 2
        if gate is not None:
            return residual + gate * out
        return residual + out


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP matching param count of 2x expansion GELU MLP."""
    def __init__(self, d_model, expand=2, bias=False):
        super().__init__()
        # For 2x expansion GELU: 2 * d_model * (expand*d_model) params
        # SwiGLU has 3 matrices: hidden = 2/3 * expand * d_model to match
        hidden = ((int(2 / 3 * expand * d_model) + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=bias)
        self.w2 = nn.Linear(d_model, hidden, bias=bias)
        self.w3 = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class QuokkaBlock(nn.Module):
    """Quokka-style: BiMamba + SwiGLU MLP + 6-way AdaLN + sum merge.

    Block structure (from other agent):
      AdaLN → fwd_Mamba + bwd_Mamba → sum → gated residual →
      AdaLN → SwiGLU MLP → gated residual
    """
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, version=3,
                 mlp_expand=2, use_adaln=True, cond_dim=None, headdim=32):
        super().__init__()
        self.use_adaln = use_adaln

        # Mamba sublayer
        self.norm1 = nn.LayerNorm(d_model)
        if use_adaln:
            self.adaln1 = AdaLNModulation(cond_dim or d_model, d_model)
        if version == 3 and HAS_MAMBA3:
            self.mamba = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand)
        else:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # MLP sublayer
        self.norm2 = nn.LayerNorm(d_model)
        if use_adaln:
            self.adaln2 = AdaLNModulation(cond_dim or d_model, d_model)
        self.mlp = SwiGLUMLP(d_model, expand=mlp_expand)

    def forward(self, x, cond=None, **kwargs):
        # Mamba sublayer with bidirectional sum merge
        h = self.norm1(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate1 = self.adaln1(cond)
            h, gate1 = adaln_modulate(h, shift, scale, gate1)
        else:
            gate1 = None
        fwd = self.mamba(h)
        bwd = self.mamba(h.flip(1)).flip(1)
        mamba_out = fwd + bwd  # sum, not average (DiffuMamba style)
        x = x + (gate1 * mamba_out if gate1 is not None else mamba_out)

        # MLP sublayer
        h = self.norm2(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate2 = self.adaln2(cond)
            h, gate2 = adaln_modulate(h, shift, scale, gate2)
        else:
            gate2 = None
        mlp_out = self.mlp(h)
        x = x + (gate2 * mlp_out if gate2 is not None else mlp_out)
        return x


class AttentionBlock(nn.Module):
    """Bidirectional attention block (same as GPT2 block with causal=False)."""
    def __init__(self, n_embd, n_head, dropout=0.0,
                 use_adaln=False, cond_dim=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout=dropout)
        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln1 = AdaLNModulation(cond_dim or n_embd, n_embd)
            self.adaln2 = AdaLNModulation(cond_dim or n_embd, n_embd)

    def forward(self, x, cond=None, **kwargs):
        # Attention sublayer
        h = self.ln1(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate1 = self.adaln1(cond)
            h, gate1 = adaln_modulate(h, shift, scale, gate1)
        else:
            gate1 = None
        attn_out = self.attn(h, causal=False)
        x = x + (gate1 * attn_out if gate1 is not None else attn_out)

        # MLP sublayer
        h = self.ln2(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate2 = self.adaln2(cond)
            h, gate2 = adaln_modulate(h, shift, scale, gate2)
        else:
            gate2 = None
        mlp_out = self.mlp(h)
        x = x + (gate2 * mlp_out if gate2 is not None else mlp_out)
        return x


class DiffuMambaH(nn.Module):
    """DiffuMamba-H: hybrid Mamba+Attention for masked diffusion.

    Args:
        vocab_size: Vocabulary size
        n_embd: Embedding dimension
        n_head: Attention heads (for attention blocks)
        n_mamba: Number of Mamba blocks
        n_attn: Number of attention blocks
        attn_positions: Which layer indices are attention (rest are Mamba).
                        If None, uses bookend placement.
        max_seq_len: Maximum sequence length
        d_state: Mamba state dimension
        d_conv: Mamba conv width
        expand: Mamba expansion factor
    """
    def __init__(self, vocab_size=50258, n_embd=384, n_head=6,
                 n_mamba=4, n_attn=2, attn_positions=None,
                 max_seq_len=1024, d_state=16, d_conv=4, expand=2,
                 mamba_version=1, use_adaln=False, block_style='bimamba',
                 headdim=64, mlp_expand=2):
        super().__init__()
        n_total = n_mamba + n_attn
        self.use_adaln = use_adaln

        if attn_positions is None:
            # Bookend: attention at first and last positions
            attn_positions = {0, n_total - 1}
        self.attn_positions = set(attn_positions)

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)

        # Time conditioning (optional)
        if use_adaln:
            self.time_emb = TimestepEmbedding(cond_dim=n_embd)

        self.blocks = nn.ModuleList()
        for i in range(n_total):
            if i in self.attn_positions:
                self.blocks.append(AttentionBlock(n_embd, n_head,
                                                  use_adaln=use_adaln, cond_dim=n_embd))
            elif block_style == 'quokka':
                self.blocks.append(QuokkaBlock(n_embd, d_state, d_conv, expand,
                                               version=mamba_version, mlp_expand=mlp_expand,
                                               use_adaln=use_adaln, cond_dim=n_embd,
                                               headdim=headdim))
            else:
                self.blocks.append(BiMambaBlock(n_embd, d_state, d_conv, expand,
                                                version=mamba_version,
                                                use_adaln=use_adaln, cond_dim=n_embd))

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()
        self.config = dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
                           n_mamba=n_mamba, n_attn=n_attn,
                           attn_positions=sorted(self.attn_positions),
                           d_state=d_state, d_conv=d_conv, expand=expand,
                           use_adaln=use_adaln, block_style=block_style)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() == 2 and 'mamba' not in name and 'adaln' not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx, causal=False, t=None):
        """Forward pass. causal arg ignored (always bidirectional).

        Args:
            idx: (B, T) token ids
            causal: ignored
            t: (B,) or (B,1) timestep for AdaLN conditioning (optional)
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Compute time conditioning if AdaLN is enabled
        cond = None
        if self.use_adaln and t is not None:
            cond = self.time_emb(t)  # (B, cond_dim)

        for block in self.blocks:
            x = block(x, cond=cond)
        x = self.ln_f(x)
        return self.lm_head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups(self):
        """Split params for Muon (2D attention/projection weights) vs AdamW (rest).

        Mamba internal params (in_proj, out_proj are 2D but interact with SSM
        dynamics) — we put in_proj and out_proj in Muon, everything else in AdamW.
        """
        muon_params = []
        adamw_params = []
        muon_names = []
        adamw_names = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_2d = p.dim() == 2
            is_embedding = 'emb' in name
            is_lm_head = 'lm_head' in name
            is_norm = 'norm' in name or 'ln' in name
            # Mamba SSM-specific params: keep in AdamW
            is_mamba_ssm = any(k in name for k in ['dt_proj', 'A_log', 'D', 'conv1d',
                                                     'x_proj', 'dt_bias'])

            if is_2d and not is_embedding and not is_lm_head and not is_norm and not is_mamba_ssm:
                muon_params.append(p)
                muon_names.append(name)
            else:
                adamw_params.append(p)
                adamw_names.append(name)

        return muon_params, adamw_params, muon_names, adamw_names
