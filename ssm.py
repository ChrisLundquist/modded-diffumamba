"""
ssm.py — Pure-PyTorch Selective State Space Model (no custom CUDA kernels)

A hardware-agnostic implementation of the Mamba selective SSM that runs on
any backend (CUDA, ROCm, CPU). Uses the chunk-wise parallel scan strategy
from Mamba-2/SSD for reasonable GPU efficiency without custom kernels.

The core recurrence per head:
    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t * h_t + D * x_t

Where A_t, B_t, C_t are input-dependent (projected from x via in_proj).
This is the "selective" part — the model decides what to remember/forget
based on the current input.

For bidirectional usage, instantiate two PureSSM modules (fwd + bwd) and
merge their outputs, exactly as DiffuMamba does with Mamba-2.

Performance notes:
  - At seq_len=1024, d_state=64, the scan is fast enough for autoresearch
  - For production, install mamba_ssm for custom Triton/CUDA kernels
  - torch.compile helps significantly (2-3x speedup on the scan loop)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureSSM(nn.Module):
    """Selective State Space Model in pure PyTorch.

    Implements the Mamba-style selective scan with:
    - Input-dependent A (decay), B (input gate), C (output gate)
    - Gated output (SiLU gate, like Mamba-1)
    - No custom CUDA kernels — runs on any device

    Args:
        d_model: model dimension
        d_state: SSM state dimension per head (default 64)
        expand: expansion factor for inner dimension (default 2)
        d_conv: local convolution width (default 4)
        nheads: number of SSM heads (default: d_inner // d_state)
        chunk_size: chunk size for parallel scan (default 64)
    """
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2,
                 d_conv: int = 4, nheads: int = None, chunk_size: int = 32,
                 **kwargs):  # absorb extra kwargs for API compat
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.chunk_size = chunk_size

        if nheads is None:
            self.nheads = max(1, self.d_inner // d_state)
        else:
            self.nheads = nheads
        self.headdim = self.d_inner // self.nheads

        # Input projection: x, z (gate), B, C, dt
        d_proj = self.d_inner * 2 + self.nheads * (d_state * 2 + 1)
        self.in_proj = nn.Linear(d_model, d_proj, bias=False)

        # Short convolution (causal in unidirectional, but we use non-causal
        # padding since this will be used bidirectionally)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )

        # Learnable log(A) parameter — initialized like Mamba
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).repeat(self.nheads, 1))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.nheads))

        # dt (timestep) bias
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))
        # dt bias: log-uniform in [0.001, 0.1]
        with torch.no_grad():
            dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_bias.copy_(dt.log())

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project
        proj = self.in_proj(x)  # (B, L, d_proj)

        # Split projections
        d = self.d_inner
        n = self.nheads
        s = self.d_state

        x_inner = proj[:, :, :d]                       # (B, L, d_inner)
        z = proj[:, :, d:2*d]                           # (B, L, d_inner) — gate
        bc_dt = proj[:, :, 2*d:]                        # (B, L, n*(2s+1))

        # Reshape bc_dt
        bc_dt = bc_dt.view(B, L, n, 2*s + 1)
        B_proj = bc_dt[:, :, :, :s]                     # (B, L, n, s)
        C_proj = bc_dt[:, :, :, s:2*s]                  # (B, L, n, s)
        dt_proj = bc_dt[:, :, :, 2*s].squeeze(-1)       # (B, L, n)

        # Conv + SiLU activation on x
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Reshape x for heads: (B, L, n, headdim)
        x_heads = x_conv.view(B, L, self.nheads, self.headdim)

        # Compute A decay and dt
        A = -torch.exp(self.A_log)  # (n, s) — negative for stability
        dt = F.softplus(dt_proj + self.dt_bias)  # (B, L, n) — positive

        # Selective scan
        y_heads = self._scan(x_heads, A, B_proj, C_proj, dt)  # (B, L, n, headdim)

        # D skip connection
        y_heads = y_heads + self.D.view(1, 1, self.nheads, 1) * x_heads

        # Reshape and gate
        y = y_heads.reshape(B, L, self.d_inner)
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _scan(self, x: torch.Tensor, A: torch.Tensor,
              B: torch.Tensor, C: torch.Tensor,
              dt: torch.Tensor) -> torch.Tensor:
        """Selective scan — chunked parallel for GPU efficiency.

        Two-pass approach:
        Pass 1: Process all chunks in parallel (folded into batch dim) with
                zero initial state. Record final states and cumulative dA products.
                Sequential depth: chunk_size (not L).
        Pass 2: Propagate inter-chunk initial states (sequential over n_chunks).
                Then re-run all chunks in parallel with correct initial states.
                Sequential depth: n_chunks + chunk_size.

        Total sequential depth: ~2*chunk_size + L/chunk_size ≈ 2*sqrt(L) optimal.
        For L=1024, chunk_size=64: depth = 128+16 = 144 vs 1024 for flat scan.
        The key win: each sequential step processes B*n_chunks items in parallel.

        Args:
            x:  (B, L, n, headdim) — input values
            A:  (n, s)             — state decay (negative)
            B:  (B, L, n, s)       — input gate
            C:  (B, L, n, s)       — output gate
            dt: (B, L, n)          — timestep (discretization)

        Returns:
            y:  (B, L, n, headdim) — output values
        """
        B_sz, L, n, hdim = x.shape
        s = A.shape[1]
        device = x.device
        C_size = self.chunk_size

        # Discretize: dA = exp(A * dt), dB = B * dt
        dt_exp = dt.unsqueeze(-1)  # (B, L, n, 1)
        dA = torch.exp(A.view(1, 1, n, s) * dt_exp)  # (B, L, n, s)
        dB = B * dt_exp  # (B, L, n, s)

        # For short sequences, just do flat scan
        if L <= C_size:
            return self._scan_flat(x, dA, dB, C)

        # Pad to multiple of chunk_size
        n_chunks = (L + C_size - 1) // C_size
        pad = n_chunks * C_size - L
        if pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dA = F.pad(dA, (0, 0, 0, 0, 0, pad), value=1.0)
            dB = F.pad(dB, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))

        # Reshape: fold chunk dim into batch → (B*n_chunks, C_size, n, ...)
        x_c = x.reshape(B_sz * n_chunks, C_size, n, hdim)
        dA_c = dA.reshape(B_sz * n_chunks, C_size, n, s)
        dB_c = dB.reshape(B_sz * n_chunks, C_size, n, s)
        C_c = C.reshape(B_sz * n_chunks, C_size, n, s)

        # Pass 1: parallel intra-chunk scan (zero initial state)
        # All B*n_chunks chunks processed simultaneously, sequential over C_size
        BN = B_sz * n_chunks
        h = torch.zeros(BN, n, s, hdim, device=device, dtype=x.dtype)
        # We need final states and cumulative dA products
        y_raw = torch.zeros(BN, C_size, n, hdim, device=device, dtype=x.dtype)
        dA_prod = torch.ones(BN, n, s, device=device, dtype=dA.dtype)

        for t in range(C_size):
            h = dA_c[:, t, :, :, None] * h + \
                dB_c[:, t, :, :, None] * x_c[:, t, :, None, :]
            y_raw[:, t] = (C_c[:, t, :, :, None] * h).sum(dim=2)
            dA_prod = dA_c[:, t] * dA_prod

        # Extract final states per chunk: (B, n_chunks, n, s, hdim)
        chunk_finals = h.reshape(B_sz, n_chunks, n, s, hdim)
        chunk_dA_prods = dA_prod.reshape(B_sz, n_chunks, n, s)

        # Pass 2a: propagate initial states across chunks (sequential over n_chunks)
        # Build as a list to avoid in-place indexed assignment on autograd graph
        h_init_list = [torch.zeros(B_sz, n, s, hdim, device=device, dtype=x.dtype)]
        for ci in range(1, n_chunks):
            h_next = chunk_dA_prods[:, ci-1, :, :, None] * h_init_list[ci-1] + \
                     chunk_finals[:, ci-1]
            h_init_list.append(h_next)
        h_inits = torch.stack(h_init_list, dim=1)  # (B, n_chunks, n, s, hdim)

        # Pass 2b: correction — add contribution of initial state to outputs
        # For chunk ci with initial state h_init, the correction to output at
        # intra-chunk position t is: sum_s(C[t,s] * (cumulative_dA[0..t] * h_init)[s])
        # We can compute this efficiently by scanning again with h_init
        # But only if h_init is nonzero (skip chunk 0)
        if n_chunks > 1:
            # Fold h_inits into batch for parallel correction scan
            # Skip chunk 0 (zero init), process chunks 1..n_chunks-1
            h_init_flat = h_inits[:, 1:].reshape(B_sz * (n_chunks - 1), n, s, hdim)
            dA_corr = dA_c.reshape(B_sz, n_chunks, C_size, n, s)[:, 1:].reshape(
                B_sz * (n_chunks - 1), C_size, n, s)
            C_corr = C_c.reshape(B_sz, n_chunks, C_size, n, s)[:, 1:].reshape(
                B_sz * (n_chunks - 1), C_size, n, s)

            # Scan the initial state through the chunk's dA sequence
            h_corr = h_init_flat  # (B*(n_chunks-1), n, s, hdim)
            y_correction = torch.zeros(B_sz * (n_chunks - 1), C_size, n, hdim,
                                       device=device, dtype=x.dtype)
            for t in range(C_size):
                h_corr = dA_corr[:, t, :, :, None] * h_corr
                y_correction[:, t] = (C_corr[:, t, :, :, None] * h_corr).sum(dim=2)

            # Add corrections to y_raw for chunks 1..n_chunks-1
            y_raw_reshaped = y_raw.reshape(B_sz, n_chunks, C_size, n, hdim).clone()
            y_raw_reshaped[:, 1:] = y_raw_reshaped[:, 1:] + y_correction.reshape(
                B_sz, n_chunks - 1, C_size, n, hdim)
            y_raw = y_raw_reshaped.reshape(BN, C_size, n, hdim)

        # Reshape back: (B, L_padded, n, hdim) → trim to (B, L, n, hdim)
        y = y_raw.reshape(B_sz, n_chunks * C_size, n, hdim)
        return y[:, :L]

    def _scan_flat(self, x: torch.Tensor, dA: torch.Tensor,
                   dB: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Flat sequential scan (for short sequences or reference).

        Args:
            x:  (B, L, n, headdim) — input values
            dA: (B, L, n, s)       — discretized decay
            dB: (B, L, n, s)       — discretized input gate
            C:  (B, L, n, s)       — output gate

        Returns:
            y:  (B, L, n, headdim) — output values
        """
        B_sz, L, n, hdim = x.shape
        s = dA.shape[-1]
        device = x.device

        y = torch.zeros(B_sz, L, n, hdim, device=device, dtype=x.dtype)
        h = torch.zeros(B_sz, n, s, hdim, device=device, dtype=x.dtype)

        for t in range(L):
            h = dA[:, t, :, :, None] * h + \
                dB[:, t, :, :, None] * x[:, t, :, None, :]
            y[:, t] = (C[:, t, :, :, None] * h).sum(dim=2)

        return y

    def _scan_simple(self, x, A, B, C, dt):
        """Simple sequential scan (reference implementation, slower).

        Useful for correctness testing.
        """
        B_sz, L, n, hdim = x.shape
        s = A.shape[1]
        device = x.device

        dt_exp = dt.unsqueeze(-1)
        dA = torch.exp(A.view(1, 1, n, s) * dt_exp)
        dB = B * dt_exp

        h = torch.zeros(B_sz, n, s, hdim, device=device, dtype=x.dtype)
        y = torch.zeros_like(x)

        for t in range(L):
            h = dA[:, t, :, :, None] * h + \
                dB[:, t, :, :, None] * x[:, t, :, None, :]
            y[:, t] = torch.einsum('bns,bnsd->bnd', C[:, t], h)

        return y


class BiPureSSM(nn.Module):
    """Bidirectional PureSSM — two independent scans, additive merge.

    Drop-in replacement for BiMamba3Block's SSM layers.
    """
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2,
                 **kwargs):
        super().__init__()
        self.ssm_fwd = PureSSM(d_model, d_state=d_state, expand=expand, **kwargs)
        self.ssm_bwd = PureSSM(d_model, d_state=d_state, expand=expand, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h_fwd = self.ssm_fwd(x)
        h_bwd = self.ssm_bwd(x.flip(1)).flip(1)
        return h_fwd + h_bwd


if __name__ == "__main__":
    print("=== PureSSM Tests ===\n")

    torch.manual_seed(42)
    B, L, D = 2, 128, 256

    # Test basic forward pass
    ssm = PureSSM(d_model=D, d_state=64, expand=2)
    x = torch.randn(B, L, D)
    y = ssm(x)
    print(f"PureSSM: input {x.shape} → output {y.shape}")
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert not torch.isnan(y).any(), "NaN in output"
    print("  Shape and NaN check: OK")

    # Test that scan and scan_simple give same results
    ssm2 = PureSSM(d_model=D, d_state=32, expand=2, chunk_size=32)
    x2 = torch.randn(B, 64, D)
    # Run both scan paths
    proj = ssm2.in_proj(x2)
    d = ssm2.d_inner
    n = ssm2.nheads
    s = ssm2.d_state
    x_inner = proj[:, :, :d]
    z = proj[:, :, d:2*d]
    bc_dt = proj[:, :, 2*d:].view(B, 64, n, 2*s + 1)
    B_p = bc_dt[:, :, :, :s]
    C_p = bc_dt[:, :, :, s:2*s]
    dt_p = bc_dt[:, :, :, 2*s]
    x_conv = ssm2.conv1d(x_inner.transpose(1, 2))[:, :, :64].transpose(1, 2)
    x_conv = F.silu(x_conv)
    x_heads = x_conv.view(B, 64, n, ssm2.headdim)
    A = -torch.exp(ssm2.A_log)
    dt = F.softplus(dt_p + ssm2.dt_bias)

    y_chunk = ssm2._scan(x_heads, A, B_p, C_p, dt)
    y_simple = ssm2._scan_simple(x_heads, A, B_p, C_p, dt)
    diff = (y_chunk - y_simple).abs().max().item()
    print(f"  Chunk scan vs simple scan max diff: {diff:.6f}")
    assert diff < 1e-4, f"Scans diverge: {diff}"
    print("  Scan consistency: OK")

    # Test bidirectional
    bi = BiPureSSM(d_model=D, d_state=64, expand=2)
    y_bi = bi(x)
    print(f"\nBiPureSSM: input {x.shape} → output {y_bi.shape}")
    assert y_bi.shape == x.shape
    assert not torch.isnan(y_bi).any()
    print("  Shape and NaN check: OK")

    # Test gradient flow
    x_grad = torch.randn(B, L, D, requires_grad=True)
    y_grad = bi(x_grad)
    loss = y_grad.sum()
    loss.backward()
    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print("  Gradient flow: OK")

    # Param count
    n_params = sum(p.numel() for p in bi.parameters())
    print(f"  BiPureSSM params: {n_params/1e6:.1f}M (d_model={D})")

    # Speed test
    import time
    bi = BiPureSSM(d_model=512, d_state=64, expand=2)
    x_bench = torch.randn(4, 512, 512)
    # Warmup
    for _ in range(3):
        _ = bi(x_bench)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = bi(x_bench)
    elapsed = (time.perf_counter() - t0) / 10
    print(f"\n  Speed (B=4, L=512, D=512, CPU): {elapsed*1000:.0f}ms/fwd")

    print("\nAll PureSSM tests passed.")
