"""Transformer variants for MDLM architecture search.

Supports: RoPE, SwiGLU, U-Net skip connections, 6-way AdaLN, variable depth/width.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaln import TimestepEmbedding, AdaLNModulation, adaln_modulate


def rotary_embedding(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, n_head, T, head_dim)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, use_rope=False, use_qk_norm=False, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        if use_qk_norm:
            # RMSNorm per-head (as in Chameleon/Gemma 2)
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, causal=True, rope_cos=None, rope_sin=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope and rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class GELUMLP(nn.Module):
    def __init__(self, n_embd, bias=False):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class SwiGLUMLP(nn.Module):
    def __init__(self, n_embd, bias=False):
        super().__init__()
        # Match param count to 4x GELU MLP: 3*hidden ≈ 8*n_embd → hidden = 8/3*n_embd
        hidden = ((int(8 / 3 * n_embd) + 63) // 64) * 64
        self.w1 = nn.Linear(n_embd, hidden, bias=bias)
        self.w2 = nn.Linear(n_embd, hidden, bias=bias)
        self.w3 = nn.Linear(hidden, n_embd, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, use_rope=False, use_swiglu=False, bias=False,
                 use_adaln=False, cond_dim=None, use_qk_norm=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head, use_rope=use_rope,
                              use_qk_norm=use_qk_norm, bias=bias)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd, bias) if use_swiglu else GELUMLP(n_embd, bias)
        self.use_adaln = use_adaln
        if use_adaln:
            # 6-way AdaLN: separate shift/scale/gate for attention and MLP
            self.adaln_attn = AdaLNModulation(cond_dim or n_embd, n_embd)
            self.adaln_mlp = AdaLNModulation(cond_dim or n_embd, n_embd)

    def forward(self, x, causal=True, rope_cos=None, rope_sin=None, cond=None):
        # Attention sublayer
        h = self.ln1(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate1 = self.adaln_attn(cond)
            h, gate1 = adaln_modulate(h, shift, scale, gate1)
        else:
            gate1 = None
        attn_out = self.attn(h, causal=causal, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + (gate1 * attn_out if gate1 is not None else attn_out)

        # MLP sublayer
        h = self.ln2(x)
        if self.use_adaln and cond is not None:
            shift, scale, gate2 = self.adaln_mlp(cond)
            h, gate2 = adaln_modulate(h, shift, scale, gate2)
        else:
            gate2 = None
        mlp_out = self.mlp(h)
        x = x + (gate2 * mlp_out if gate2 is not None else mlp_out)
        return x


class TransformerV2(nn.Module):
    """Transformer with optional RoPE, SwiGLU, U-Net skips.

    Args:
        use_rope: Replace learned position embeddings with RoPE
        use_swiglu: Replace GELU MLP with SwiGLU
        use_unet: Add U-Net skip connections (layer i → layer n-1-i)
    """
    def __init__(self, vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
                 max_seq_len=1024, bias=False,
                 use_rope=False, use_swiglu=False, use_unet=False, use_adaln=False,
                 use_qk_norm=False):
        super().__init__()
        self.use_rope = use_rope
        self.use_unet = use_unet
        self.use_adaln = use_adaln
        self.n_layer = n_layer

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        if not use_rope:
            self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        else:
            self.pos_emb = None

        if use_adaln:
            self.time_emb = TimestepEmbedding(cond_dim=n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, use_rope=use_rope, use_swiglu=use_swiglu, bias=bias,
                  use_adaln=use_adaln, cond_dim=n_embd, use_qk_norm=use_qk_norm)
            for _ in range(n_layer)
        ])

        if use_unet and n_layer >= 4:
            n_skips = n_layer // 2
            self.skip_projs = nn.ModuleList([
                nn.Linear(n_embd, n_embd, bias=False)
                for _ in range(n_skips)
            ])
        else:
            self.skip_projs = None

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()
        self.config = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                           n_embd=n_embd, use_rope=use_rope, use_swiglu=use_swiglu,
                           use_unet=use_unet, use_adaln=use_adaln)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() == 2 and 'adaln' not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx, causal=True, t=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        rope_cos, rope_sin = None, None
        if self.use_rope:
            head_dim = x.shape[-1] // self.blocks[0].attn.n_head
            rope_cos, rope_sin = rotary_embedding(T, head_dim, idx.device)

        cond = None
        if self.use_adaln and t is not None:
            cond = self.time_emb(t)

        n = self.n_layer
        half = n // 2

        if self.use_unet and self.skip_projs is not None:
            skips = []
            for i in range(half):
                x = self.blocks[i](x, causal=causal, rope_cos=rope_cos, rope_sin=rope_sin, cond=cond)
                skips.append(x)
            for i in range(half, n):
                x = self.blocks[i](x, causal=causal, rope_cos=rope_cos, rope_sin=rope_sin, cond=cond)
                skip_idx = n - 1 - i
                if skip_idx < len(skips):
                    x = x + self.skip_projs[skip_idx](skips[skip_idx])
        else:
            for block in self.blocks:
                x = block(x, causal=causal, rope_cos=rope_cos, rope_sin=rope_sin, cond=cond)

        x = self.ln_f(x)
        return self.lm_head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups(self):
        muon_params, adamw_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_2d = p.dim() == 2
            is_emb = 'emb' in name
            is_head = 'lm_head' in name
            is_norm = 'ln' in name
            if is_2d and not is_emb and not is_head and not is_norm:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return muon_params, adamw_params
