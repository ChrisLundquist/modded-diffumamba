"""124M GPT-2 style transformer for Muon experiments."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x, causal=True):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_head, head_dim)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd, bias=False, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias=False, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, bias=bias)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, x, causal=True):
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 small (124M parameters).

    Args:
        vocab_size: Vocabulary size (default 50257 for GPT-2 tokenizer)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        max_seq_len: Maximum sequence length
        bias: Use bias in linear layers and layernorms
        dropout: Dropout rate (0.0 = no dropout)
    """
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
                 max_seq_len=1024, bias=False, dropout=0.0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                           n_embd=n_embd, max_seq_len=max_seq_len, bias=bias,
                           dropout=dropout)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, bias=bias, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() == 1 and 'ln' not in name and 'bias' not in name:
                nn.init.zeros_(p)

    def forward(self, idx, causal=True):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, causal=causal)
        x = self.ln_f(x)
        return self.lm_head(x)

    def param_count(self):
        """Count parameters (excluding position embeddings for standard reporting)."""
        return sum(p.numel() for p in self.parameters())

    def param_groups(self):
        """Split parameters into Muon-eligible (2D hidden weights) and AdamW (rest).

        Returns:
            muon_params: list of 2D weight tensors suitable for Muon
            adamw_params: list of all other parameters (embeddings, norms, biases, lm_head)
        """
        muon_params = []
        adamw_params = []
        muon_names = []
        adamw_names = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Muon targets: 2D weight matrices in attention and MLP
            # Exclude embeddings (tok_emb, pos_emb), layernorm, lm_head, biases
            is_2d_weight = p.dim() == 2
            is_embedding = 'emb' in name
            is_lm_head = 'lm_head' in name
            is_norm = 'ln' in name
            is_bias = 'bias' in name

            if is_2d_weight and not is_embedding and not is_lm_head and not is_norm:
                muon_params.append(p)
                muon_names.append(name)
            else:
                adamw_params.append(p)
                adamw_names.append(name)

        return muon_params, adamw_params, muon_names, adamw_names
